# core/confidence.py — Confidence zone logic and response helpers.

from config import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM
from utils.logger import logger


def get_confidence_zone(confidence: float) -> str:
    """
    Map raw confidence float → zone string.

    Zones:
      "high"   (≥ 0.75) → execute directly, say nothing about confidence
      "medium" (≥ 0.50) → execute but speak a short prefix so user knows what we understood
      "low"    (< 0.50) → stop, ask the user ONE yes/no question before doing anything
    """
    if confidence >= CONFIDENCE_HIGH:
        zone = "high"
    elif confidence >= CONFIDENCE_MEDIUM:
        zone = "medium"
    else:
        zone = "low"
    logger.debug(f"Confidence {confidence:.2f} → '{zone}' zone")
    return zone


def build_clarification_question(mode: str) -> str:
    """
    When confidence is low, ask one simple yes/no question.
    Keep it short — blind users just need to say yes or no.
    """
    questions = {
        "navigation_mode": "Did you want me to describe your surroundings? Please say yes or no.",
        "reading_mode":    "Did you want me to read something for you? Please say yes or no.",
        "currency_mode":   "Did you want me to identify the currency you are holding? Please say yes or no.",
        "unknown":         "I did not understand. Do you want a scene description, text reading, or currency check?",
    }
    q = questions.get(mode, questions["unknown"])
    logger.debug(f"Clarification question → '{q}'")
    return q


def build_medium_prefix(mode: str) -> str:
    """
    Short spoken prefix when confidence is medium.
    Tells the user what we think they asked — before giving the answer.
    Example: "I think you want a scene description."
    """
    prefixes = {
        "navigation_mode": "I think you want a scene description.",
        "reading_mode":    "I think you want me to read something.",
        "currency_mode":   "I think you want to check your currency.",
    }
    return prefixes.get(mode, "")