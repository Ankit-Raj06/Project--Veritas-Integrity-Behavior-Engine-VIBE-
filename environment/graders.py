import math
from typing import Optional

_SCORE_MIN: float = 0.001
_SCORE_MAX: float = 0.999

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

def clamp_score(score: float) -> float:
    if math.isnan(score):
        return 0.05
    return min(max(score, _SCORE_MIN), _SCORE_MAX)

def grade(prediction: str, ground_truth, threat_type: Optional[str] = "general") -> float:
    pred  = normalise_decision(prediction)
    truth = normalise_label(ground_truth)

    if pred == truth:
        score = 0.95        # ← was 1.0, now safe

    elif pred == "flag":
        if truth == "remove":
            score = 0.5
        elif truth == "safe":
            score = 0.4
        else:
            score = 0.4

    elif truth == "flag":
        if pred == "safe":
            score = 0.3
        elif pred == "remove":
            score = 0.6
        else:
            score = 0.3

    elif pred == "safe" and truth == "remove":
        score = 0.05        # ← was 0.0, now safe

    elif pred == "remove" and truth == "safe":
        score = 0.1

    else:
        score = 0.05        # ← was 0.0 default, now safe

    return clamp_score(score)

def grade_easy(prediction: str, ground_truth, **kwargs) -> float:
    return grade(prediction, ground_truth)

def grade_medium(prediction: str, ground_truth, context_match: bool = True, **kwargs) -> float:
    base = grade(prediction, ground_truth)
    if context_match and base > 0:
        base += 0.1
    return clamp_score(base)

def grade_hard(prediction: str, ground_truth, justification: str = "", threat_type: str = "general", **kwargs) -> float:
    base = grade(prediction, ground_truth, threat_type=threat_type)
    if justification:
        words = justification.strip().split()
        generic_phrases = {"content", "violates", "policy", "guidelines", "inappropriate"}
        unique_words = {w.lower() for w in words} - generic_phrases
        if len(words) >= 10 and len(unique_words) >= 5:
            base += 0.15
    return clamp_score(base)
