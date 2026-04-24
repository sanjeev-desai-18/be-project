"""
modules/currency/currency_logic.py
───────────────────────────────────
Processes YOLOv11 detection output and triggers TTS announcements.

Logic:
  - Collects all detected denominations in a frame
  - Announces each unique denomination with a per-class cooldown (3 s)
  - Summarises totals when multiple notes are visible ("2 notes: 500 and 100 rupees")
  - Thread-safe via a per-function state dict
"""

import time
import threading
from utils.logger import logger

# ── Shared TTS speaker (singleton for currency module) ────────────────────────
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
    """Speak text via the shared Speaker instance."""
    try:
        spk = _get_speaker()
        if spk:
            spk.speak(text)
        else:
            logger.warning(f"[NO TTS] {text}")
    except Exception as e:
        logger.error(f"TTS error in currency_logic: {e}")


# ── Per-class cooldown state ───────────────────────────────────────────────────
_last_spoken:    dict  = {}    # class_label -> last spoken timestamp
_last_summary:   float = 0.0   # timestamp of last multi-note summary
_state_lock               = threading.Lock()

SINGLE_COOLDOWN  = 3.0   # seconds before re-announcing the same denomination
SUMMARY_COOLDOWN = 5.0   # seconds before re-announcing a multi-note summary


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSOR — called from currency_detector.py every inference frame
# ══════════════════════════════════════════════════════════════════════════════
def process_predictions(result: dict):
    """
    Receive a detection result dict and decide what to speak.

    Args:
        result: {
            "predictions": [
                {"class": "500_rupees", "confidence": 0.87, ...},
                ...
            ],
            "image": {"width": ..., "height": ...}
        }
    """
    global _last_summary

    try:
        predictions = _extract_predictions(result)
        if not predictions:
            return

        now = time.time()

        # ── Filter by confidence (already done in detector, but double-check) ──
        valid = [p for p in predictions if p.get("confidence", 0) >= 0.50]
        if not valid:
            return

        # ── Collect unique labels detected this frame ─────────────────────────
        detected_labels = []
        seen = set()
        for p in valid:
            label = p.get("class") or p.get("class_name") or ""
            if label and label not in seen:
                seen.add(label)
                detected_labels.append(label)

        if not detected_labels:
            return

        # ── Multi-note summary ────────────────────────────────────────────────
        if len(detected_labels) > 1:
            if now - _last_summary >= SUMMARY_COOLDOWN:
                human_labels = [_human(l) for l in detected_labels]
                msg = f"{len(human_labels)} notes detected: {', '.join(human_labels)}"
                logger.info(f"Currency summary: {msg}")
                speak(msg)
                with _state_lock:
                    _last_summary = now
                    for label in detected_labels:
                        _last_spoken[label] = now
            return

        # ── Single note ───────────────────────────────────────────────────────
        label = detected_labels[0]
        with _state_lock:
            last_t = _last_spoken.get(label, 0.0)

        if now - last_t >= SINGLE_COOLDOWN:
            msg = f"{_human(label)} detected"
            logger.info(f"Currency: {msg}")
            speak(msg)
            with _state_lock:
                _last_spoken[label] = now

    except Exception as e:
        logger.error(f"Error in process_predictions: {e}", exc_info=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _extract_predictions(result) -> list:
    """
    Normalise various result formats into a flat list of prediction dicts.
    Handles: dict with 'predictions' key, plain list, or nested list.
    """
    if isinstance(result, dict):
        preds = (result.get("predictions") or
                 result.get("output") or
                 result.get("detections"))
        if preds is None:
            return []
        return _extract_predictions(preds)

    if isinstance(result, list):
        if not result:
            return []
        first = result[0]
        # Nested list of dicts
        if isinstance(first, dict):
            return result
        # List of lists — try inner
        if isinstance(first, list):
            return _extract_predictions(first)

    return []


def _human(label: str) -> str:
    """
    Convert snake_case label to natural speech.
    '500_rupees' → '500 rupees'
    '10_rupees'  → '10 rupees'
    """
    return label.replace("_", " ")


# ══════════════════════════════════════════════════════════════════════════════
# VLM-VERIFIED RESULT SPEAKER
# ══════════════════════════════════════════════════════════════════════════════
_last_vlm_spoken: float = 0.0
VLM_SPEAK_COOLDOWN = 5.0   # seconds between VLM-verified announcements


def speak_vlm_result(vlm_text: str):
    """
    Speak the VLM verification result via TTS.

    Called from the VLM verification thread in currency_detector.py.
    Enforces a cooldown to avoid rapid-fire announcements.
    """
    global _last_vlm_spoken

    if not vlm_text or not vlm_text.strip():
        logger.warning("speak_vlm_result: empty text — skipping")
        return

    now = time.time()
    with _state_lock:
        if now - _last_vlm_spoken < VLM_SPEAK_COOLDOWN:
            logger.debug("speak_vlm_result: cooldown active — skipping")
            return
        _last_vlm_spoken = now

    logger.info(f"VLM currency result: {vlm_text[:120]}")
    speak(vlm_text)

