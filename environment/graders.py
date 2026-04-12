"""
VIBE – Graders
Scoring functions for all three difficulty levels.
Score ranges: (0, 1) — strictly bounded, never 0.0 or 1.0.
"""

from typing import Optional


# ── Label normalisation ───────────────────────────────────────────────────────

DECISION_MAP = {
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
    "0":       "safe",
    "1":       "remove",
    "safe":    "safe",
    "flag":    "flag",
    "remove":  "remove",
    "toxic":   "remove",
    "hateful": "remove",
    "neutral": "safe",
}


def normalise_decision(raw: str) -> str:
    return DECISION_MAP.get(str(raw).strip().lower(), "flag")


def normalise_label(raw) -> str:
    return LABEL_MAP.get(str(raw).strip().lower(), "flag")


# ── Clamp helper ─────────────────────────────────────────────────────────────

def clamp_score(score: float) -> float:
    """Ensures score is strictly between (0, 1). 0.0 and 1.0 are NOT allowed."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 0.05
    if s <= 0.0:
        return 0.01
    if s >= 1.0:
        return 0.99
    return round(s, 4)


# ── Core grader ───────────────────────────────────────────────────────────────

def grade(
    prediction:  str,
    ground_truth,
    threat_type: Optional[str] = "general",
) -> float:
    """
    Returns a score strictly in (0, 1).

    Exact match             → 0.95
    Predicted flag, truth remove → 0.50
    Predicted flag, truth safe   → 0.40
    safe predicted, truth remove → 0.05
    remove predicted, truth safe → 0.10
    All other cases         → 0.05
    """
    pred  = normalise_decision(prediction)
    truth = normalise_label(ground_truth)

    if pred == truth:
        score = 0.95

    elif pred == "flag":
        if truth == "remove":
            score = 0.50
        else:
            score = 0.40

    elif truth == "flag":
        if pred == "remove":
            score = 0.60
        else:
            score = 0.30

    elif pred == "safe" and truth == "remove":
        score = 0.05

    elif pred == "remove" and truth == "safe":
        score = 0.10

    else:
        score = 0.05   # catch-all — never 0.0

    return clamp_score(score)


# ── Per-difficulty graders ────────────────────────────────────────────────────

def grade_easy(prediction: str, ground_truth, **kwargs) -> float:
    """Task 1 — single comment, unambiguous label."""
    return grade(prediction, ground_truth)


def grade_medium(
    prediction:    str,
    ground_truth,
    context_match: bool = True,
    **kwargs
) -> float:
    """
    Task 2 — thread moderation.
    +0.08 bonus if prediction is consistent with thread context.
    Max achievable: 0.95 + 0.08 = 1.03 → clamped to 0.99.
    """
    base = grade(prediction, ground_truth)
    if context_match and base > 0.05:
        base += 0.08
    return clamp_score(base)


def grade_hard(
    prediction:    str,
    ground_truth,
    justification: str = "",
    threat_type:   str = "general",
    **kwargs,
) -> float:
    """
    Task 3 — cross-cultural / ambiguous moderation.
    +0.12 bonus for a rich, specific justification.
    Max achievable: 0.95 + 0.12 = 1.07 → clamped to 0.99.
    """
    base = grade(prediction, ground_truth, threat_type=threat_type)

    if justification:
        words         = justification.strip().split()
        generic       = {"content", "violates", "policy", "guidelines", "inappropriate"}
        unique_words  = {w.lower() for w in words} - generic

        if len(words) >= 10 and len(unique_words) >= 5:
            base += 0.12

    return clamp_score(base)
