"""
VIBE – Graders
Scoring functions for all three difficulty levels.
Score ranges: (0,1) — strictly bounded
Partial credit is given for "flag" when truth is ambiguous.
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





# ── Clamp helper ─────────────────────────────────────────────────────────────



def clamp_score(score: float) -> float:

    """
    Ensures score is strictly between (0, 1)
    """

    if score <= 0.0:

        return 0.01

    if score >= 1.0:

        return 0.99

    return score





# ── Core grader ───────────────────────────────────────────────────────────────



def grade(
    prediction:   str,
    ground_truth,
    threat_type:  Optional[str] = "general",
) -> float:

    """
    Returns a score strictly in (0,1).
    Scoring logic:
      - Exact match                        → ~1.0
      - Predicted "flag", truth is toxic   → 0.5
      - Predicted "flag", truth is safe    → 0.4
      - Wrong direction (remove↔safe)      → ~0.0
    """



    pred  = normalise_decision(prediction)

    truth = normalise_label(ground_truth)



    # Default score

    score = 0.0



    if pred == truth:

        score = 1.0



    elif pred == "flag":

        if truth == "remove":

            score = 0.5

        elif truth == "safe":

            score = 0.4



    elif truth == "flag":

        if pred == "safe":

            score = 0.3

        elif pred == "remove":

            score = 0.6



    elif pred == "safe" and truth == "remove":

        score = 0.0



    elif pred == "remove" and truth == "safe":

        score = 0.1



    return clamp_score(score)





# ── Per-difficulty graders ────────────────────────────────────────────────────



def grade_easy(prediction: str, ground_truth, **kwargs) -> float:

    """Task 1 — single comment, unambiguous label."""

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



    if context_match and base > 0:

        base += 0.1



    return clamp_score(base)





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



    return clamp_score(base)
