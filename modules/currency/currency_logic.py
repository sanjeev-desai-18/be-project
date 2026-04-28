"""
modules/currency/currency_logic.py

Detection → Speech pipeline — ROBUST REAL-TIME version.

Architecture
────────────
Detection loop (30+ fps)
    │  calls process_confirmed_notes() every frame — always non-blocking
    ▼
Simple state machine:
    IDLE  → note detected   → speak() → ANNOUNCED
    ANNOUNCED → same note   → do nothing
    ANNOUNCED → diff note   → interrupt_and_speak() → ANNOUNCED (new label)
    ANNOUNCED → gone N frames → IDLE  (let speech finish naturally)
    IDLE  → same note back  → speak() → ANNOUNCED

Key properties
──────────────
• Speaks ONCE when a note first appears.
• Does NOT repeat while the same note stays in view.
• When the note CHANGES: interrupts old speech, speaks new immediately.
• When the note LEAVES: state resets after a generous debounce (~1s)
  so brief detection gaps don't cause speech to break/restart.
• NO cooldown timers — new note = instant speech.
"""

import threading
import time
from collections import Counter
from utils.logger import logger

# ── Tuning constants ──────────────────────────────────────────────────────────

# How many consecutive "no note" frames before we consider the note truly gone.
# At 30 fps → 30 frames = 1 second.  This is generous enough to ride out
# intermittent YOLO detection gaps without causing speech break/restart cycles.
GONE_FRAMES = 30

# ── State variables (protected by _state_lock) ───────────────────────────────

_state_lock     = threading.Lock()
_current_label  = None   # label key currently announced / None if idle
_gone_counter   = 0      # consecutive frames with no detection


# ── Speaker singleton ─────────────────────────────────────────────────────────

_speaker      = None
_speaker_lock = threading.Lock()


def _get_speaker():
    global _speaker
    with _speaker_lock:
        if _speaker is None:
            try:
                from tts.speaker import Speaker
                _speaker = Speaker()
                logger.info("currency_logic: Speaker initialised ✓")
            except Exception as e:
                logger.error(f"currency_logic: Speaker init failed: {e}")
        return _speaker


_last_speak_time = 0.0
REPEAT_INTERVAL  = 2.5  # seconds
STABLE_FRAMES    = 7    # frames required to confirm a label change (~230ms)

_candidate_label   = None
_candidate_counter = 0

# ── Public API (called from currency_detector.py every frame) ────────────────

def process_confirmed_notes(confirmed: list) -> None:
    """
    Called every detection frame (30+ fps).  Must be non-blocking.
    """
    global _current_label, _gone_counter, _last_speak_time
    global _candidate_label, _candidate_counter

    with _state_lock:
        if confirmed:
            _gone_counter = 0
            label = _summarise_label([t["confirmed_cls"] for t in confirmed])
            now = time.time()

            if _current_label is None:
                # Transition IDLE → ANNOUNCED — first detection (instant)
                _current_label = label
                _candidate_label = label
                _candidate_counter = 0
                _last_speak_time = now
                msg = _build_message_from_label(label)
                logger.info(f"currency_logic: DETECTED → '{msg}'")
                speaker = _get_speaker()
                if speaker:
                    speaker.speak(msg)

            elif label != _current_label:
                # Potential label change — require STABLE_FRAMES consecutive
                # matches to prevent YOLO 1-frame hallucinations from stuttering speech.
                if label == _candidate_label:
                    _candidate_counter += 1
                    if _candidate_counter >= STABLE_FRAMES:
                        _current_label = label
                        _last_speak_time = now
                        msg = _build_message_from_label(label)
                        logger.info(f"currency_logic: CHANGED → '{msg}'")
                        speaker = _get_speaker()
                        if speaker:
                            speaker.interrupt_and_speak(msg)
                else:
                    _candidate_label = label
                    _candidate_counter = 1

            else:
                # same label, still visible — reset candidate & repeat every interval
                _candidate_label = label
                _candidate_counter = 0
                if now - _last_speak_time >= REPEAT_INTERVAL:
                    _last_speak_time = now
                    msg = _build_message_from_label(label)
                    logger.info(f"currency_logic: REPEATED → '{msg}'")
                    speaker = _get_speaker()
                    if speaker:
                        speaker.speak(msg)

        else:
            # No note this frame
            if _current_label is not None:
                _gone_counter += 1
                if _gone_counter >= GONE_FRAMES:
                    # Note truly gone — reset to IDLE.
                    # We let any currently playing announcement finish naturally
                    # rather than cutting it off mid-sentence.
                    logger.info(
                        f"currency_logic: GONE after {_gone_counter} frames "
                        f"(was '{_current_label}')"
                    )
                    _current_label = None


def reset_logic_state():
    """Called by currency_detector.reset() when currency mode stops."""
    global _current_label, _gone_counter
    with _state_lock:
        _current_label = None
        _gone_counter  = 0
    logger.info("currency_logic: state reset ✓")


# ── Label helpers ─────────────────────────────────────────────────────────────

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


def _summarise_label(classes: list) -> str:
    """
    Build a stable string key from the list of confirmed class names.
    Used as the identity token for change-detection.
    e.g. ["500_rupees", "100_rupees"] → "100_rupees|500_rupees"
    """
    return "|".join(sorted(classes))


def _build_message_from_label(label: str) -> str:
    """Convert a label key (as produced by _summarise_label) to spoken text."""
    classes = label.split("|") if label else []
    return _build_message(classes)


def _build_message(classes: list) -> str:
    counts = Counter(_label(c) for c in classes)
    items  = sorted(counts.items(), key=lambda x: x[0])

    phrases = []
    for denom, qty in items:
        if qty == 1:
            phrases.append(f"a {denom} note")
        else:
            phrases.append(f"{qty} {denom} notes")

    if len(phrases) == 1:
        body = phrases[0]
    elif len(phrases) == 2:
        body = f"{phrases[0]} and {phrases[1]}"
    else:
        body = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"

    return f"I can see {body}."
