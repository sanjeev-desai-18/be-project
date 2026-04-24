"""
modules/currency/currency_logic.py
───────────────────────────────────
Processes confirmed note tracks and triggers TTS announcements.

Called by currency_detector.py with a list of track dicts that have:
  - confirmed_cls  : str   (majority-voted denomination, locked)
  - track_id       : int   (persistent ID for this physical note)
  - announce_count : int   (how many times TTS has already fired for it)

Each track is spoken up to ANNOUNCE_REPEAT times (set in currency_detector),
after which the detector stops passing it here. This file enforces an
inter-announcement cooldown so rapid back-to-back calls don't stack TTS.

Returns the list of track_ids that were actually spoken this call so the
detector can increment announce_count via mark_announced().
"""

import time
import threading
from utils.logger import logger

# ── Shared TTS speaker ────────────────────────────────────────────────────────
_speaker      = None
_speaker_lock = threading.Lock()


def _get_speaker():
    global _speaker
    if _speaker is None:
        try:
            from tts.speaker import Speaker
            _speaker = Speaker()
        except Exception as e:
            logger.error(f"Could not initialise Speaker in currency_logic: {e}")
    return _speaker


def speak(text: str):
    try:
        spk = _get_speaker()
        if spk:
            spk.speak(text)
        else:
            logger.warning(f"[NO TTS] {text}")
    except Exception as e:
        logger.error(f"TTS error in currency_logic: {e}")


# ── Cooldown between successive announcements ─────────────────────────────────
# Prevents TTS from firing twice in the same breath when multiple confirmed
# tracks exist at once. Each track gets its own last-spoken timestamp.
_track_last_spoken: dict[int, float] = {}   # track_id -> timestamp
_state_lock = threading.Lock()

# Minimum gap between the 1st and 2nd announcement for the same track.
# At 30 fps a confirmed track fires its 1st announcement immediately;
# the 2nd fires after this cooldown so the user hears them distinctly.
REPEAT_COOLDOWN = 4.0   # seconds between announcement 1 and announcement 2

# Minimum gap between any two TTS calls globally (avoids overlapping speech).
GLOBAL_COOLDOWN  = 1.5  # seconds
_last_any_spoken: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT — called from currency_detector._run() each frame
# ══════════════════════════════════════════════════════════════════════════════
def process_confirmed_notes(tracks: list) -> list[int]:
    """
    Decide which confirmed tracks to announce and speak them.

    Args:
        tracks: list of dicts from _NoteTracker.update(), already filtered to
                those with confirmed_cls != None and announce_count < ANNOUNCE_REPEAT.
                Each dict has at minimum:
                  track_id, confirmed_cls, announce_count, confidence

    Returns:
        List of track_ids that were spoken this call.
        The detector calls mark_announced(tid) for each returned id.
    """
    global _last_any_spoken

    if not tracks:
        return []

    now = time.time()
    announced: list[int] = []

    # Sort by track_id so multi-note announcements are deterministic
    for t in sorted(tracks, key=lambda x: x["track_id"]):
        tid   = t["track_id"]
        cls   = t["confirmed_cls"]
        cnt   = t["announce_count"]  # 0 = first time, 1 = second time

        with _state_lock:
            last_t     = _track_last_spoken.get(tid, 0.0)
            last_any   = _last_any_spoken

        # First announcement fires as soon as gate passes (no per-track delay).
        # Second announcement requires REPEAT_COOLDOWN since the first.
        if cnt > 0 and now - last_t < REPEAT_COOLDOWN:
            logger.debug(
                f"track #{tid}: repeat cooldown active "
                f"({now - last_t:.1f}s / {REPEAT_COOLDOWN}s)"
            )
            continue

        # Global cooldown — don't overlap with previous TTS call
        if now - last_any < GLOBAL_COOLDOWN:
            logger.debug(f"track #{tid}: global cooldown active")
            continue

        msg = _build_message(cls, cnt)
        logger.info(f"Currency TTS track #{tid} (ann #{cnt+1}): {msg}")
        speak(msg)

        with _state_lock:
            _track_last_spoken[tid] = now
            _last_any_spoken        = now

        announced.append(tid)

    return announced


# ── Message builder ───────────────────────────────────────────────────────────
def _build_message(cls: str, announce_count: int) -> str:
    """
    First announcement  : "500 rupees detected"
    Second announcement : "500 rupees confirmed"
    (Extra redundancy helps visually impaired users be certain of the value.)
    """
    human = cls.replace("_", " ")
    if announce_count == 0:
        return f"{human} detected"
    return f"{human} confirmed"
