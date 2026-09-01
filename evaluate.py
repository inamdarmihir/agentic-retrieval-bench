"""Answer-correctness scoring for FinanceBench questions.

FinanceBench answers are almost all a specific number ("$1577.00", "63.4%",
"$2.5 billion") or a short factual phrase. There's no local LLM-as-judge
budget here that wouldn't just introduce a second model's own error into
the measurement, so this uses a plain, inspectable heuristic: extract the
numbers from both strings and compare with a small relative tolerance, and
fall back to substring containment for non-numeric answers. This is
deliberately conservative and stated as a heuristic, not a judged score -
see the README's Limitations section before treating this as a general
QA-accuracy metric.
"""

import re


def _extract_numbers(text: str) -> list[float]:
    text = text.replace(",", "")
    matches = re.findall(r"-?\$?\d+\.?\d*\s*(?:billion|million|thousand|%)?", text, flags=re.IGNORECASE)
    numbers = []
    for m in matches:
        m_clean = m.replace("$", "").strip()
        multiplier = 1.0
        low = m_clean.lower()
        if "billion" in low:
            multiplier = 1e9
            m_clean = re.sub(r"billion", "", m_clean, flags=re.IGNORECASE).strip()
        elif "million" in low:
            multiplier = 1e6
            m_clean = re.sub(r"million", "", m_clean, flags=re.IGNORECASE).strip()
        elif "thousand" in low:
            multiplier = 1e3
            m_clean = re.sub(r"thousand", "", m_clean, flags=re.IGNORECASE).strip()
        elif "%" in low:
            m_clean = m_clean.replace("%", "").strip()
        try:
            numbers.append(float(m_clean) * multiplier)
        except ValueError:
            continue
    return numbers


def is_correct(predicted: str, reference: str, rel_tol: float = 0.02) -> bool:
    ref_numbers = _extract_numbers(reference)
    pred_numbers = _extract_numbers(predicted)

    if ref_numbers:
        if not pred_numbers:
            return False
        for rn in ref_numbers:
            for pn in pred_numbers:
                if rn == 0:
                    if pn == 0:
                        return True
                    continue
                if abs(pn - rn) / abs(rn) <= rel_tol:
                    return True
        return False

    # Non-numeric reference answer: fall back to loose substring containment.
    ref_clean = reference.lower().strip()
    return ref_clean in predicted.lower()
