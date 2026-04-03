# utils/audio_utils.py — Microphone helpers

import numpy as np
import sounddevice as sd
from utils.logger import logger

SAMPLE_RATE = 16000   # standard for Whisper


def list_microphones():
    """Print all available input devices — useful for debugging."""
    devices = sd.query_devices()
    logger.info("Available audio input devices:")
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            logger.info(f"  [{i}] {device['name']}")
    return devices


def check_microphone_available() -> bool:
    """
    Try opening the microphone briefly.
    Returns True if accessible, False if not found.
    """
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32'):
            pass
        logger.info("Microphone check passed ✓")
        return True
    except Exception as e:
        logger.error(f"Microphone not available: {e}")
        return False


def record_audio(duration: float) -> np.ndarray:
    """
    Record fixed-duration audio from mic.
    Used for quick testing only — main flow uses silence-detection in listener.py
    """
    logger.debug(f"Recording {duration}s...")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    return audio.flatten()