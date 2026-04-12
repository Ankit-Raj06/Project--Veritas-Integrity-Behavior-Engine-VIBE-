"""
VIBE – Graders
Scoring functions for all three difficulty levels.
REVISED: Removed clamp_score; logic now uses safe boundaries.
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
    Returns a score strictly in (0,1).
    Base scores are capped at 0.8 to allow headroom for bonuses.
    """
    pred  = normalise_decision(prediction)
    truth = normalise_label(ground_truth)

    # Instead of 0.0, we start at a tiny floor
    score = 0.01

    if pred == truth:
        # Instead of 1.0, we return 0.80. 
        # This leaves 0.20 of 'room' for medium/hard bonuses.
        score = 0.80

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
        score = 0.01 # Floor instead of 0.0

    elif pred == "remove" and truth == "safe":
        score = 0.10

    return score


# ── Per-difficulty graders ────────────────────────────────────────────────────

def grade_easy(prediction: str, ground_truth, **kwargs) -> float:
    """Task 1 — single comment, unambiguous label."""
    # Max possible: 0.80 | Min possible: 0.01
    return grade(prediction, ground_truth)


def grade_medium(
    prediction: str,
    ground_truth,
    context_match: bool = True,
    **kwargs
) -> float:
    """
    Task 2 — thread moderation.
    Bonus 0.1 if prediction is consistent with thread context.
    """
    base = grade(prediction, ground_truth)

    if context_match and base > 0.01:
        base += 0.10

    # Max possible: 0.90 | Min possible: 0.01
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

    # Bonus for quality justification
    if justification:
        words = justification.strip().split()
        generic_phrases = {"content", "violates", "policy", "guidelines", "inappropriate"}
        unique_words = {w.lower() for w in words} - generic_phrases

        if len(words) >= 10 and len(unique_words) >= 5:
            base += 0.15

    # Max possible: 0.95 | Min possible: 0.01
    return base
