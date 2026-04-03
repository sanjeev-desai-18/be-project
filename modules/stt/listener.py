"""
modules/stt/listener.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Microphone capture on Raspberry Pi 5 + Groq Whisper STT.

Latency improvements over original:
  â€¢ Background noise measured ONCE at startup â€” not on every listen() call (~0.9s saved)
  â€¢ Chunk size reduced 0.3s â†’ 0.15s â€” silence detected 150ms faster per chunk
  â€¢ Single stream open per listen() â€” eliminates double ALSA open/close overhead
  â€¢ SILENCE_THRESHOLD in config.py reduced to 0.8s (was 2.0s) â€” 1.2s saved at end of speech
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

# â”€â”€ Groq client â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_client = Groq(api_key=GROQ_API_KEY)
logger.info("Groq Whisper STT client ready âœ“")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# DEVICE DISCOVERY
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def _find_usb_mic_device():
    try:
        devices = sd.query_devices()
        candidates = []
        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) < 1:
                continue
            name_lower = dev["name"].lower()
            if "usb" in name_lower:
                candidates.insert(0, i)
            elif "microphone" in name_lower or "mic" in name_lower:
                candidates.append(i)

        if not candidates:
            logger.warning("No USB/mic device found â€” using system default")
            return None, 44100

        idx = candidates[0]
        logger.info(f"Selected mic device: [{idx}] {devices[idx]['name']}")

        for rate in [16000, 44100, 48000]:
            try:
                sd.check_input_settings(device=idx, samplerate=rate, channels=1)
                logger.info(f"Mic supports sample rate: {rate} Hz")
                return idx, rate
            except Exception:
                continue

        default_rate = int(devices[idx].get("default_samplerate", 44100))
        logger.warning(f"Using device default rate: {default_rate} Hz")
        return idx, default_rate

    except Exception as e:
        logger.warning(f"Device discovery error: {e} â€” using system default at 44100")
        return None, 44100


_MIC_DEVICE, SAMPLE_RATE = _find_usb_mic_device()
logger.info(f"Mic device={_MIC_DEVICE}  sample_rate={SAMPLE_RATE}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# NOISE BASELINE â€” measured ONCE at startup, reused every listen() call
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def _measure_noise_level() -> float:
    """
    Open mic for ~0.6s at startup to measure background noise floor.
    Result is reused for every subsequent listen() call â€” no per-call overhead.
    """
    chunk_samples = int(SAMPLE_RATE * 0.15)
    open_kwargs   = dict(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE)
    if _MIC_DEVICE is not None:
        open_kwargs["device"] = _MIC_DEVICE
    try:
        with sd.InputStream(**open_kwargs) as stream:
            readings = []
            for _ in range(4):   # 4 Ã— 0.15s = 0.6s total
                chunk, _ = stream.read(chunk_samples)
                readings.append(float(np.sqrt(np.mean(chunk.flatten() ** 2))))
        level = max(np.mean(readings) * 2.5, 0.01)
        logger.info(f"Startup noise baseline: {level:.5f}")
        return level
    except Exception as e:
        logger.warning(f"Startup noise measurement failed: {e} â€” using default 0.01")
        return 0.01


# Measured once here â€” shared across all listen() calls
_NOISE_BASELINE: float = _measure_noise_level()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PUBLIC: listen() â€” record until silence, transcribe via Groq Whisper
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def listen() -> str:
    """
    Record from USB microphone until silence is detected.
    Sends audio to Groq Whisper API and returns transcription.

    Improvements:
      - No noise measurement per call (done at startup)
      - Single stream open covers the full record cycle
      - 0.15s chunks for faster silence detection
    """
    logger.info("Listening... (speak now)")

    chunk_duration = 0.15                            # was 0.3 â€” finer silence detection
    chunk_samples  = int(SAMPLE_RATE * chunk_duration)
    max_silence    = max(int(SILENCE_THRESHOLD / chunk_duration), 3)
    max_chunks     = int(10.0 / chunk_duration)      # 10-second hard cap

    open_kwargs = dict(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE)
    if _MIC_DEVICE is not None:
        open_kwargs["device"] = _MIC_DEVICE

    speech_threshold = _NOISE_BASELINE

    audio_chunks     = []
    silence_chunks   = 0
    started_speaking = False
    total_chunks     = 0

    try:
        # Single stream open â€” covers full recording cycle
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
                        logger.debug("Silence detected â€” recording complete")
                        break

            if total_chunks >= max_chunks and started_speaking:
                logger.debug("Max recording duration reached â€” processing")

    except Exception as e:
        logger.error(f"Microphone stream error: {e}")
        return ""

    if not audio_chunks:
        logger.debug("No speech detected in this window")
        return ""

    full_audio = np.concatenate(audio_chunks)

    if len(full_audio) < SAMPLE_RATE * 0.3:
        logger.warning("Recording too short â€” skipping")
        return ""

    return _transcribe_numpy(full_audio, SAMPLE_RATE)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PUBLIC: listen_from_file() â€” retained for compatibility
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def listen_from_file(path: str) -> str:
    logger.info(f"Transcribing file: {path}")
    try:
        ext      = path.rsplit(".", 1)[-1].lower() if "." in path else "wav"
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
            logger.warning("Audio file too small â€” likely empty")
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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# INTERNAL: resample â†’ 16kHz â†’ Groq Whisper
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
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
            language=None,
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
