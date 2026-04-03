"""
modules/stt/listener.py
────────────────────────
Microphone capture on Raspberry Pi 5 + Groq Whisper STT.

Changes from laptop version:
  • Explicitly uses USB mic device 0 (Usb Audio Device: USB Audio hw:2,0)
  • Probes supported sample rate (tries 16000 first, then 44100, then 48000)
  • Robust silence detection tuned for Pi audio stack (ALSA)
  • listen_from_file() retained for compatibility (not used in Pi flow)
"""

import io
import numpy as np
import sounddevice as sd
import soundfile as sf
from groq import Groq

from utils.logger import logger
from config import GROQ_API_KEY, SILENCE_THRESHOLD

CHANNELS = 1
DTYPE    = "float32"

# ── Groq client ───────────────────────────────────────────────────────────────
_client = Groq(api_key=GROQ_API_KEY)
logger.info("Groq Whisper STT client ready ✓")


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE DISCOVERY — find USB mic and its supported sample rate
# ══════════════════════════════════════════════════════════════════════════════
def _find_usb_mic_device():
    """
    Returns (device_index, sample_rate) for the best available USB mic.
    Probes common rates in order: 16000, 44100, 48000.
    Falls back to (None, 44100) if nothing found.
    """
    try:
        devices = sd.query_devices()
        candidates = []

        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) < 1:
                continue
            name_lower = dev["name"].lower()
            if "usb" in name_lower:
                candidates.insert(0, i)   # USB devices get priority
            elif "microphone" in name_lower or "mic" in name_lower:
                candidates.append(i)

        if not candidates:
            logger.warning("No USB/mic device found — using system default")
            return None, 44100

        idx = candidates[0]
        logger.info(f"Selected mic device: [{idx}] {devices[idx]['name']}")

        # Probe supported sample rates
        for rate in [16000, 44100, 48000]:
            try:
                sd.check_input_settings(device=idx, samplerate=rate, channels=1)
                logger.info(f"Mic supports sample rate: {rate} Hz")
                return idx, rate
            except Exception:
                continue

        # If none passed the check, use device default rate
        default_rate = int(devices[idx].get("default_samplerate", 44100))
        logger.warning(f"Using device default rate: {default_rate} Hz")
        return idx, default_rate

    except Exception as e:
        logger.warning(f"Device discovery error: {e} — using system default at 44100")
        return None, 44100


_MIC_DEVICE, SAMPLE_RATE = _find_usb_mic_device()
logger.info(f"Mic device={_MIC_DEVICE}  sample_rate={SAMPLE_RATE}")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: listen()  — record until silence, transcribe via Groq Whisper
# ══════════════════════════════════════════════════════════════════════════════
def listen() -> str:
    """
    Record from USB microphone until silence is detected.
    Sends audio to Groq Whisper API and returns transcription.

    Strategy:
      1. Measure background noise for 0.5 s → dynamic speech threshold
      2. Record until silence returns after speech
      3. Hard cap at 10 s so it never gets stuck
    """
    logger.info("Listening... (speak now)")

    chunk_duration   = 0.3                          # seconds per chunk
    chunk_samples    = int(SAMPLE_RATE * chunk_duration)
    max_silence      = max(int(SILENCE_THRESHOLD / chunk_duration), 4)
    max_chunks       = int(10.0 / chunk_duration)   # 10-second hard cap

    open_kwargs = dict(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
    )
    if _MIC_DEVICE is not None:
        open_kwargs["device"] = _MIC_DEVICE

    # ── Step 1: measure background noise ─────────────────────────────────────
    noise_level = 0.01   # safe fallback
    try:
        with sd.InputStream(**open_kwargs) as stream:
            noise_chunks = []
            for _ in range(3):                      # ~0.9 s of background
                chunk, _ = stream.read(chunk_samples)
                noise_chunks.append(float(np.sqrt(np.mean(chunk.flatten() ** 2))))
            noise_level = max(np.mean(noise_chunks) * 2.5, 0.01)
            logger.debug(f"Background noise level: {noise_level:.5f}")
    except Exception as e:
        logger.warning(f"Noise measurement failed: {e} — using default threshold")

    speech_threshold = noise_level

    # ── Step 2: record until silence ─────────────────────────────────────────
    audio_chunks     = []
    silence_chunks   = 0
    started_speaking = False
    total_chunks     = 0

    try:
        with sd.InputStream(**open_kwargs) as stream:
            while total_chunks < max_chunks:
                chunk, _ = stream.read(chunk_samples)
                chunk     = chunk.flatten()
                volume    = float(np.sqrt(np.mean(chunk ** 2)))
                total_chunks += 1

                logger.debug(f"Volume RMS: {volume:.5f}  threshold: {speech_threshold:.5f}")

                is_speech = volume > speech_threshold

                if is_speech:
                    started_speaking = True
                    silence_chunks   = 0
                    audio_chunks.append(chunk)
                elif started_speaking:
                    silence_chunks += 1
                    audio_chunks.append(chunk)
                    if silence_chunks >= max_silence:
                        logger.debug("Silence detected — recording complete")
                        break

            if total_chunks >= max_chunks and started_speaking:
                logger.debug("Max recording duration reached — processing")

    except Exception as e:
        logger.error(f"Microphone stream error: {e}")
        return ""

    if not audio_chunks:
        logger.debug("No speech detected in this window")
        return ""

    full_audio = np.concatenate(audio_chunks)

    if len(full_audio) < SAMPLE_RATE * 0.3:
        logger.warning("Recording too short — skipping")
        return ""

    return _transcribe_numpy(full_audio, SAMPLE_RATE)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: listen_from_file()  — transcribe a saved audio file (kept for compat)
# ══════════════════════════════════════════════════════════════════════════════
def listen_from_file(path: str) -> str:
    """
    Transcribe a saved audio file via Groq Whisper.
    Not used in the Pi mic loop, retained for API compatibility.
    """
    logger.info(f"Transcribing file: {path}")
    try:
        ext  = path.rsplit(".", 1)[-1].lower() if "." in path else "wav"
        mime_map = {
            "webm": "audio/webm", "ogg": "audio/ogg",
            "mp4":  "audio/mp4",  "wav": "audio/wav",
            "mp3":  "audio/mpeg",
        }
        mime     = mime_map.get(ext, "audio/wav")
        filename = f"recording.{ext}"

        with open(path, "rb") as f:
            raw_bytes = f.read()

        if len(raw_bytes) < 1000:
            logger.warning("Audio file too small — likely empty")
            return ""

        transcription = _client.audio.transcriptions.create(
            file=(filename, raw_bytes, mime),
            model="whisper-large-v3",
            language=None,
            response_format="text",
        )
        transcript = transcription.strip() if transcription else ""
        if transcript:
            logger.info(f"Heard: '{transcript}'")
        return transcript

    except Exception as e:
        logger.error(f"listen_from_file error: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL: resample numpy audio → 16kHz → Groq Whisper
# ══════════════════════════════════════════════════════════════════════════════
def _transcribe_numpy(audio: np.ndarray, source_rate: int) -> str:
    target_rate = 16000

    if source_rate != target_rate:
        target_length = int(len(audio) * target_rate / source_rate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_length),
            np.arange(len(audio)),
            audio,
        )

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio, target_rate, format="WAV", subtype="PCM_16")
    wav_buffer.seek(0)

    logger.debug("Sending audio to Groq Whisper...")
    try:
        transcription = _client.audio.transcriptions.create(
            file=("audio.wav", wav_buffer, "audio/wav"),
            model="whisper-large-v3",
            language=None,            # auto-detect Hindi / English / Hinglish
            response_format="text",
        )
        transcript = transcription.strip() if transcription else ""
        if transcript:
            logger.info(f"Heard: '{transcript}'")
        else:
            logger.warning("Whisper returned empty transcript")
        return transcript

    except Exception as e:
        logger.error(f"Groq Whisper API failed: {e}")
        return ""
