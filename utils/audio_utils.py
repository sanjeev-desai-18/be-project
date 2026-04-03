# utils/audio_utils.py — Microphone helpers

import numpy as np
import sounddevice as sd
from utils.logger import logger


def _find_usb_mic_device() -> tuple:
    """
    Return (device_index, sample_rate) for the first USB mic found.
    Probes rates in order: 44100, 48000, 16000.
    Returns (None, 44100) if nothing found.
    """
    try:
        for i, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) < 1:
                continue
            if "usb" in dev["name"].lower():
                logger.info(f"audio_utils: USB mic found at device [{i}] {dev['name']}")
                for rate in [44100, 48000, 16000]:
                    try:
                        sd.check_input_settings(device=i, samplerate=rate, channels=1)
                        logger.info(f"audio_utils: mic supports {rate} Hz")
                        return i, rate
                    except Exception:
                        continue
                # fallback to device default
                default_rate = int(dev.get("default_samplerate", 44100))
                logger.warning(f"audio_utils: using device default rate {default_rate} Hz")
                return i, default_rate
    except Exception as e:
        logger.warning(f"audio_utils: device discovery error: {e}")
    return None, 44100


_MIC_DEVICE, SAMPLE_RATE = _find_usb_mic_device()
logger.info(f"audio_utils: mic device={_MIC_DEVICE}  sample_rate={SAMPLE_RATE}")


def list_microphones():
    """Print all available input devices — useful for debugging."""
    devices = sd.query_devices()
    logger.info("Available audio input devices:")
    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            logger.info(f"  [{i}] {device['name']}")
    return devices


def check_microphone_available() -> bool:
    """
    Try opening the USB microphone briefly at its probed sample rate.
    """
    try:
        kwargs = dict(samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        if _MIC_DEVICE is not None:
            kwargs["device"] = _MIC_DEVICE
        with sd.InputStream(**kwargs):
            pass
        logger.info(f"Microphone check passed ✓ (device={_MIC_DEVICE}, rate={SAMPLE_RATE})")
        return True
    except Exception as e:
        logger.error(f"Microphone not available: {e}")
        return False


def record_audio(duration: float) -> np.ndarray:
    """
    Record fixed-duration audio from mic.
    Used for quick testing only — main flow uses silence-detection in listener.py.
    """
    logger.debug(f"Recording {duration}s at {SAMPLE_RATE} Hz...")
    kwargs = dict(
        frames=int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    if _MIC_DEVICE is not None:
        kwargs["device"] = _MIC_DEVICE
    audio = sd.rec(**kwargs)
    sd.wait()
    return audio.flatten()
