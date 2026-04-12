def grade(
    prediction:  str,
    ground_truth,
    threat_type: Optional[str] = "general",
) -> float:
    pred  = normalise_decision(prediction)
    truth = normalise_label(ground_truth)

    score = 0.05  # slightly higher floor

    if pred == truth:
        score = 0.85  # slightly higher reward for correctness

    elif pred == "flag":
        if truth == "remove":
            score = 0.55
        elif truth == "safe":
            score = 0.45

    elif truth == "flag":
        if pred == "safe":
            score = 0.40
        elif pred == "remove":
            score = 0.65

    elif pred == "safe" and truth == "remove":
        score = 0.05  # less brutal penalty

    elif pred == "remove" and truth == "safe":
        score = 0.15  # still penalized but fair

    return score
