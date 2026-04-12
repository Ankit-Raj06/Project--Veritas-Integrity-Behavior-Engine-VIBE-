"""
VIBE – Graders
Scoring functions for all three difficulty levels.
REVISED: Uses safe_score to ensure all values are strictly in (0, 1).
"""
from typing import Optional

# ── Label normalisation ───────────────────────────────────────────────────────
DECISION_MAP = {
    "safe":     "safe",
    "approve":  "safe",
    "ok":       "safe",
    "allow":    "safe",
    "flag":     "flag",
    "escalate": "flag",
    "review":   "flag",
    "remove":   "remove",
    "delete":   "remove",
    "reject":   "remove",
    "toxic":    "remove",
    "hate":     "remove",
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


# ── Safety clamp ─────────────────────────────────────────────────────────────
def safe_score(score: float) -> float:
    """Ensure score is strictly within (0, 1) — never exactly 0.0 or 1.0."""
    return max(0.001, min(0.999, float(score)))


# ── Core grader ───────────────────────────────────────────────────────────────
def grade(
    prediction:  str,
    ground_truth,
    threat_type: Optional[str] = "general",
) -> float:
    """
    Returns a raw score. Callers are responsible for applying safe_score().
    Base scores are capped at 0.80 to leave headroom for bonuses.
    """
    pred  = normalise_decision(prediction)
    truth = normalise_label(ground_truth)

    score = 0.01  # floor instead of 0.0

    if pred == truth:
        score = 0.80  # headroom for medium/hard bonuses

    elif pred == "flag":
        if truth == "remove":
            score = 0.50
        elif truth == "safe":
            score = 0.40

    elif truth == "flag":
        if pred == "safe":
            score = 0.30
        elif pred == "remove":
            score = 0.60

    elif pred == "safe" and truth == "remove":
        score = 0.01  # worst-case floor

    elif pred == "remove" and truth == "safe":
        score = 0.10

    return score


# ── Per-difficulty graders ────────────────────────────────────────────────────
def grade_easy(prediction: str, ground_truth, **kwargs) -> float:
    """Task 1 — single comment, unambiguous label.
    Max possible: 0.80 | Min possible: 0.01
    """
    return safe_score(grade(prediction, ground_truth))


def grade_medium(
    prediction:    str,
    ground_truth,
    context_match: bool = True,
    **kwargs
) -> float:
    """Task 2 — thread moderation.
    Bonus 0.10 if prediction is consistent with thread context.
    Max possible: 0.90 | Min possible: 0.01
    """
    base = grade(prediction, ground_truth)
    if context_match and base > 0.01:
        base += 0.10
    return safe_score(base)


def grade_hard(
    prediction:    str,
    ground_truth,
    justification: str = "",
    threat_type:   str = "general",
    **kwargs,
) -> float:
    """Task 3 — cross-platform / multi-modal.
    Bonus 0.15 if justification references relevant signals.
    Max possible: 0.95 | Min possible: 0.01
    """
    base = grade(prediction, ground_truth, threat_type=threat_type)

    if justification:
        words = justification.strip().split()
        generic_phrases = {"content", "violates", "policy", "guidelines", "inappropriate"}
        unique_words = {w.lower() for w in words} - generic_phrases
        if len(words) >= 10 and len(unique_words) >= 5:
            base += 0.15

    return safe_score(base)
