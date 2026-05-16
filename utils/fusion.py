from typing import Dict, Tuple


def weighted_fusion(predictions: Dict[str, float], weights: Dict[str, float] | None = None) -> Tuple[str, float]:
    if not predictions:
        return "Legitimate", 0.0

    if weights is None:
        weights = {
            "url": 0.35,
            "email": 0.25,
            "sms": 0.2,
            "qr": 0.15,
            "deepfake": 0.05,
        }

    weighted_sum = 0.0
    total_weight = 0.0

    for source, phishing_prob in predictions.items():
        weight = weights.get(source, 0.1)
        weighted_sum += phishing_prob * weight
        total_weight += weight

    final_prob = weighted_sum / total_weight if total_weight else 0.0
    final_label = "Phishing" if final_prob >= 0.5 else "Legitimate"

    return final_label, float(final_prob)
