"""A small ReAct-style agent loop: given a question and exactly one
retrieval tool (keyword or vector), let a local LLM decide when and how to
search, then answer. This is the harness both benchmark conditions run
through; the only thing that changes between conditions is which tool
object gets bound in.
"""

import time
from dataclasses import dataclass, field

import ollama

from tools import KeywordSearchTool, SearchHit, VectorSearchTool, format_hits

MODEL = "qwen2.5:3b-instruct"
MAX_TURNS = 4

SYSTEM_PROMPT = (
    "You are a research agent. You do not have SEC filings memorized. Use the search "
    "tool to find the answer before responding. Once you have it, answer in one short "
    "sentence with the specific number or fact."
)


@dataclass
class AgentResult:
    question: str
    condition: str  # "keyword" or "vector"
    final_answer: str
    tool_calls: list[dict] = field(default_factory=list)  # [{"query": str, "hits": [SearchHit,...]}]
    turns_used: int = 0
    wall_clock_s: float = 0.0
    hit_evidence: bool = False  # did any tool call return the gold (doc_name, page_num)?


def _tool_schema(tool_name: str, description: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "search query"}},
                    "required": ["query"],
                },
            },
        }
    ]


def run_agent(
    question: str,
    condition: str,
    tool: KeywordSearchTool | VectorSearchTool,
    gold_doc: str,
    gold_page: int,
) -> AgentResult:
    t0 = time.time()
    tools_schema = _tool_schema(tool.name, tool.description)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    result = AgentResult(question=question, condition=condition, final_answer="")
    empty_retries = 0

    for turn in range(MAX_TURNS):
        resp = ollama.chat(model=MODEL, messages=messages, tools=tools_schema)
        msg = resp["message"]
        result.turns_used = turn + 1

        if not msg.get("tool_calls") and not msg.get("content", "").strip():
            # Small models occasionally emit a genuinely empty turn (no tool
            # call, no content). Nudge once instead of scoring a non-answer.
            if empty_retries < 1:
                empty_retries += 1
                messages.append(
                    {"role": "user", "content": "Use the search tool to find the answer, then respond."}
                )
                continue
            result.final_answer = ""
            break

        if not msg.get("tool_calls"):
            result.final_answer = msg.get("content", "").strip()
            break

        # Take only the first tool call per turn (single-action ReAct step);
        # smaller models sometimes emit several near-duplicate calls at once.
        call = msg["tool_calls"][0]
        query = call["function"]["arguments"].get("query", "")
        hits: list[SearchHit] = tool.search(query)

        for h in hits:
            if h.doc_name == gold_doc and abs(h.page_num - gold_page) <= 1:
                result.hit_evidence = True

        result.tool_calls.append({"query": query, "hits": [(h.doc_name, h.page_num, h.score) for h in hits]})

        messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": msg["tool_calls"]})
        messages.append({"role": "tool", "content": format_hits(hits)})

        if turn == MAX_TURNS - 1:
            # Out of turns: force a final answer with whatever context we have.
            messages.append({"role": "user", "content": "Give your final answer now, in one sentence."})
            resp = ollama.chat(model=MODEL, messages=messages)
            result.final_answer = resp["message"].get("content", "").strip()

    result.wall_clock_s = time.time() - t0
    return result
