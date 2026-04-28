"""
modules/currency/currency_logic.py

Speaks the detected currency every time a confirmed note is present.
No cooldown, no scene-change tracking — just speak on every confirmed frame.
The TTS worker's drop-oldest queue ensures the speaker is never flooded:
if it's still speaking, the new message replaces the pending one.
"""

import threading
from collections import Counter
from utils.logger import logger

_speaker      = None
_speaker_lock = threading.Lock()


def _get_speaker():
    global _speaker
    with _speaker_lock:
        if _speaker is None:
            try:
                from tts.speaker import Speaker
                _speaker = Speaker()
                logger.info("currency_logic: Speaker initialised")
            except Exception as e:
                logger.error(f"currency_logic: Speaker init failed: {e}")
        return _speaker


def _speak(text: str):
    spk = _get_speaker()
    if spk:
        spk.speak(text)
        logger.info(f"currency_logic: speak -> '{text}'")
    else:
        logger.warning(f"currency_logic: [NO TTS] {text}")


def reset_logic_state():
    logger.info("currency_logic: state reset")


def process_confirmed_notes(confirmed: list) -> None:
    """
    Called every frame. Speaks immediately whenever confirmed notes are visible.
    The TTS drop-oldest queue handles rate limiting — if the speaker is busy,
    the stale pending message is replaced with the latest one.
    """
    if not confirmed:
        return

    msg = _build_message([t["confirmed_cls"] for t in confirmed])
    _speak(msg)


# ── Message builder ───────────────────────────────────────────────────────────

_SPOKEN = {
    "10_rupees":   "10 rupee",
    "20_rupees":   "20 rupee",
    "50_rupees":   "50 rupee",
    "100_rupees":  "100 rupee",
    "200_rupees":  "200 rupee",
    "500_rupees":  "500 rupee",
    "2000_rupees": "2000 rupee",
}


def _label(cls: str) -> str:
    return _SPOKEN.get(cls, cls.replace("_", " "))


def _build_message(classes: list) -> str:
    counts = Counter(_label(c) for c in classes)
    items  = sorted(counts.items(), key=lambda x: x[0])

    phrases = []
    for denom, qty in items:
        phrases.append(f"a {denom}" if qty == 1 else f"{qty} {denom}")

    if len(phrases) == 1:
        note_phrase = phrases[0]
    elif len(phrases) == 2:
        note_phrase = f"{phrases[0]} and {phrases[1]}"
    else:
        note_phrase = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"

    total = sum(counts.values())
    noun  = "notes" if total > 1 else "note"

    return f"I can see {note_phrase} {noun}."
