"""
VIBE – Graders
Scoring functions for all three difficulty levels.

Score ranges: 0.0 (wrong) → 1.0 (perfect)
Partial credit is given for "flag" when truth is ambiguous.
"""

from typing import Optional


# ── Label normalisation ───────────────────────────────────────────────────────

DECISION_MAP = {
    # Accept all surface variants the LLM might produce
    "safe":    "safe",
    "approve": "safe",
    "ok":      "safe",
    "allow":   "safe",
    "flag":    "flag",
    "escalate":"flag",
    "review":  "flag",
    "remove":  "remove",
    "delete":  "remove",
    "reject":  "remove",
    "toxic":   "remove",
    "hate":    "remove",
}

LABEL_MAP = {
    "0":        "safe",
    "1":        "remove",
    "safe":     "safe",
    "flag":     "flag",
    "remove":   "remove",
    "toxic":    "remove",
    "hateful":  "remove",
    "neutral":  "safe",
}


def normalise_decision(raw: str) -> str:
    return DECISION_MAP.get(str(raw).strip().lower(), "flag")


def normalise_label(raw) -> str:
    return LABEL_MAP.get(str(raw).strip().lower(), "flag")


# ── Core grader ───────────────────────────────────────────────────────────────

def grade(
    prediction:   str,
    ground_truth,
    threat_type:  Optional[str] = "general",
) -> float:
    """
    Returns a score in [0.0, 1.0].

    Scoring logic:
      - Exact match                        → 1.0
      - Predicted "flag", truth is toxic   → 0.5  (cautious partial credit)
      - Predicted "flag", truth is safe    → 0.4  (over-cautious but not wrong)
      - Wrong direction (remove↔safe)      → 0.0
    """
    pred  = normalise_decision(prediction)
    truth = normalise_label(ground_truth)

    if pred == truth:
        return 1.0

    # Partial credit cases
    if pred == "flag":
        if truth == "remove":
            return 0.5   # escalated instead of removing — cautious, partial credit
        if truth == "safe":
            return 0.4   # over-cautious, but not harmful

    if truth == "flag":
        if pred == "safe":
            return 0.3   # missed an ambiguous case
        if pred == "remove":
            return 0.6   # over-moderated an ambiguous case

    # Worst cases: missed harmful content or falsely removed safe content
    if pred == "safe" and truth == "remove":
        return 0.0   # missed hate/toxic — unacceptable
    if pred == "remove" and truth == "safe":
        return 0.1   # false positive — penalised but not zero

    return 0.0


# ── Per-difficulty graders ────────────────────────────────────────────────────

def grade_easy(prediction: str, ground_truth, **kwargs) -> float:
    """Task 1 — single comment, unambiguous label."""
    return grade(prediction, ground_truth)


def grade_medium(prediction: str, ground_truth, context_match: bool = True, **kwargs) -> float:
    """
    Task 2 — thread moderation.
    Bonus 0.1 if prediction is consistent with thread context.
    """
    base = grade(prediction, ground_truth)
    if context_match and base > 0:
        base = min(1.0, base + 0.1)
    return base


def grade_hard(
    prediction:      str,
    ground_truth,
    justification:   str = "",
    threat_type:     str = "general",
    **kwargs,
) -> float:
    """
    Task 3 — cross-platform / multi-modal.
    Bonus 0.15 if justification references relevant signals.
    """
    base = grade(prediction, ground_truth, threat_type=threat_type)

    # Bonus for quality justification (heuristic: >10 words and not a generic phrase)
    if justification:
        words = justification.strip().split()
        generic_phrases = {"content", "violates", "policy", "guidelines", "inappropriate"}
        unique_words    = {w.lower() for w in words} - generic_phrases
        if len(words) >= 10 and len(unique_words) >= 5:
            base = min(1.0, base + 0.15)

    return base
