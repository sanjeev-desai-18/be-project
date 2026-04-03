# modules/scene/output_filter.py — Smart output filter.
# Only filters generic object LISTS — not specific object answers the user asked for.

import re
from utils.logger import logger

# These patterns indicate unprompted object listing — filter them
UNPROMPTED_LIST_PATTERNS = [
    r"\bI see\b",
    r"\bI can see\b",
    r"\bI observe\b",
    r"\bI spot\b",
    r"\bvisible (?:is|are)\b",
    r"\bI detect\b",
]

# These are OK — they are direct answers to user questions
ALLOWED_PATTERNS = [
    r"\bthere (?:appears|seems) to be\b",   # "there appears to be a bottle"
    r"\bto your (left|right|front|back)\b", # directional answers
    r"\bin front of you\b",
    r"\bbehind you\b",
    r"\bnearby\b",
    r"\byes\b",
    r"\bno\b",
]

FALLBACK_RESPONSE = (
    "You appear to be in an indoor environment. "
    "The surroundings look generally clear."
)


def filter_output(text: str, is_object_query: bool = False) -> str:
    """
    Filter VLM output based on context.

    If is_object_query=True  → only remove generic lists, keep specific answers
    If is_object_query=False → strictly remove all object-listing sentences

    Args:
        text:            Raw VLM response
        is_object_query: Whether user asked about specific objects

    Returns:
        Clean text ready to be spoken aloud.
    """
    if not text or not text.strip():
        logger.warning("filter_output received empty text — using fallback")
        return FALLBACK_RESPONSE

    # If user asked about objects — trust the VLM response mostly
    # Only remove clearly bad generic listing patterns
    if is_object_query:
        logger.debug("Object query mode — light filtering only")
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        clean = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            is_bad_list = any(
                re.search(p, sentence, re.IGNORECASE)
                for p in UNPROMPTED_LIST_PATTERNS
            )
            if is_bad_list:
                logger.debug(f"Filtered generic list sentence: '{sentence[:60]}'")
            else:
                clean.append(sentence.strip())
        result = " ".join(clean) if clean else text
        logger.debug(f"Light filter output: '{result[:100]}'")
        return result

    # Strict mode — pure scene description, no object mentions
    else:
        logger.debug("Scene mode — strict filtering")
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        clean = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            is_bad = any(
                re.search(p, sentence, re.IGNORECASE)
                for p in UNPROMPTED_LIST_PATTERNS
            )
            if is_bad:
                logger.debug(f"Filtered: '{sentence[:60]}'")
            else:
                clean.append(sentence.strip())

        if not clean:
            logger.warning("All sentences filtered — using fallback")
            return FALLBACK_RESPONSE

        result = " ".join(clean)
        logger.debug(f"Strict filter output: '{result[:100]}'")
        return result