from .classifier import DESCRIPTIONS, RECOMMENDATIONS, STAGES, classify_rule_based


def classify(features: dict) -> dict:
    return classify_rule_based(features)
