"""
modules/currency/currency_logic.py
"""

import time
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
        logger.info(f"currency_logic: queued -> '{text}'")
    else:
        logger.warning(f"currency_logic: [NO TTS] {text}")


_lock             = threading.Lock()
_announced_ids    = set()
_last_speak_time  = 0.0

SPEAK_COOLDOWN = 2.5


def reset_logic_state():
    global _announced_ids, _last_speak_time
    with _lock:
        _announced_ids   = set()
        _last_speak_time = 0.0
    logger.info("currency_logic: state reset")


def process_confirmed_notes(confirmed: list) -> None:
    global _announced_ids, _last_speak_time

    if not confirmed:
        with _lock:
            _announced_ids = set()
        return

    visible_ids = {t["track_id"] for t in confirmed}

    with _lock:
        _announced_ids &= visible_ids
        new_ids = visible_ids - _announced_ids

    if not new_ids:
        return

    now = time.time()
    with _lock:
        since_last = now - _last_speak_time

    if since_last < SPEAK_COOLDOWN:
        logger.debug(f"currency_logic: cooldown {since_last:.1f}s / {SPEAK_COOLDOWN}s")
        return

    msg = _build_message(confirmed)

    with _lock:
        _announced_ids  = _announced_ids | visible_ids
        _last_speak_time = now

    _speak(msg)


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


def _build_message(tracks: list) -> str:
    counts = Counter(_label(t["confirmed_cls"]) for t in tracks)
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
