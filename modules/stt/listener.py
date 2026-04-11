"""
modules/stt/listener.py
────────────────────────────────────────────────────────────────────────────
Microphone capture on Raspberry Pi 5 + Groq Whisper STT.

Noise-rejection improvements over previous version:
  • response_format changed to "verbose_json" — gives per-segment
    no_speech_prob and avg_logprob so we can discard noise transcriptions
    that Whisper hallucinates from ambient sound.
  • Hard thresholds: no_speech_prob > 0.5 OR avg_logprob < -1.0 → discard.
  • Noise-word blocklist: single words / stock phrases Whisper commonly
    emits for silence ("thank you", "thanks", "bye", etc.) → discard.
  • Minimum audio duration raised to 1.0s (was 0.5s) — real commands are
    rarely shorter; brief noise bursts are often shorter.
  • Word-count filter kept at < 2 words → discard.

Latency improvements (unchanged from previous version):
  • Background noise measured ONCE at startup — not on every listen() call.
  • Chunk size 0.15s — silence detected faster.
  • Single stream open per listen() — no double ALSA open/close overhead.
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


# Common words / phrases Whisper hallucinates from silence or ambient noise.
# All entries must be lower-case; comparison is case-insensitive.
_NOISE_PHRASES = frozenset({
    "thank you", "thanks", "thank you.", "thanks.",
    "bye", "bye.", "goodbye", "goodbye.",
    "okay", "ok", "okay.", "ok.",
    "yeah", "yes", "no",
    "hmm", "um", "uh", "oh", "ah",
    "a", "the", "you",
    ".", ",", "...",
    "subtitles by", "subtitle by",
    "you.", "you,",
})


# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════
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
            logger.warning("No USB/mic device found — using system default")
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
        logger.warning(f"Device discovery error: {e} — using system default at 44100")
        return None, 44100


_MIC_DEVICE, SAMPLE_RATE = _find_usb_mic_device()
logger.info(f"Mic device={_MIC_DEVICE}  sample_rate={SAMPLE_RATE}")


# ═══════════════════════════════════════════════════════════════════════════════
# NOISE BASELINE — measured ONCE at startup, reused every listen() call
# ═══════════════════════════════════════════════════════════════════════════════
def _measure_noise_level() -> float:
    """
    Open mic for ~0.6s at startup to measure background noise floor.
    Result is reused for every subsequent listen() call — no per-call overhead.
    """
    chunk_samples = int(SAMPLE_RATE * 0.15)
    open_kwargs   = dict(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE)
    if _MIC_DEVICE is not None:
        open_kwargs["device"] = _MIC_DEVICE
    try:
        with sd.InputStream(**open_kwargs) as stream:
            readings = []
            for _ in range(4):   # 4 x 0.15s = 0.6s total
                chunk, _ = stream.read(chunk_samples)
                readings.append(float(np.sqrt(np.mean(chunk.flatten() ** 2))))
        level = max(np.mean(readings) * 2.5, 0.01)
        logger.info(f"Startup noise baseline: {level:.5f}")
        return level
    except Exception as e:
        logger.warning(f"Startup noise measurement failed: {e} — using default 0.01")
        return 0.01


# Measured once here — shared across all listen() calls
_NOISE_BASELINE: float = _measure_noise_level()


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC: listen() — record until silence, transcribe via Groq Whisper
# ═══════════════════════════════════════════════════════════════════════════════
def listen() -> str:
    """
    Record from USB microphone until silence is detected.
    Sends audio to Groq Whisper API and returns transcription.

    Returns empty string if the recording is likely noise rather than
    a real user command (confidence filtering applied).
    """
    logger.info("Listening... (speak now)")

    chunk_duration = 0.15                            # finer silence detection
    chunk_samples  = int(SAMPLE_RATE * chunk_duration)
    max_silence    = max(int(SILENCE_THRESHOLD / chunk_duration), 3)
    max_chunks     = int(10.0 / chunk_duration)      # 10-second hard cap

    open_kwargs = dict(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE)
    if _MIC_DEVICE is not None:
        open_kwargs["device"] = _MIC_DEVICE

    # Clamp threshold: at least 0.20 (reject quiet noise),
    # at most 0.32 (don't make detection too hard in noisy rooms)
    speech_threshold = float(np.clip(_NOISE_BASELINE, 0.20, 0.32))
    logger.debug(f"Speech threshold clamped to {speech_threshold:.3f}  (raw baseline {_NOISE_BASELINE:.5f})")

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

    # Raised from 0.5s to 1.0s: real commands are rarely shorter than 1s;
    # brief noise bursts that pass the volume threshold usually are.
    if len(full_audio) < SAMPLE_RATE * 1.0:
        logger.warning("Recording too short — skipping (likely noise)")
        return ""

    transcript = _transcribe_numpy(full_audio, SAMPLE_RATE)

    # Discard very short / nonsensical transcriptions (noise artefacts)
    if transcript and len(transcript.split()) < 2:
        logger.warning(f"Transcript too short ('{transcript}') -- discarding as noise")
        return ""

    return transcript


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC: listen_from_file() — retained for compatibility
# ═══════════════════════════════════════════════════════════════════════════════
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
            logger.warning("Audio file too small — likely empty")
            return ""

        transcription = _client.audio.transcriptions.create(
            file=(filename, raw_bytes, mime),
            model="whisper-large-v3",
            language="en",
            response_format="text",
        )
        transcript = transcription.strip() if transcription else ""
        if transcript:
            logger.info(f"Heard: '{transcript}'")
        return transcript

    except Exception as e:
        logger.error(f"listen_from_file error: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL: resample → 16kHz → Groq Whisper (verbose_json for confidence)
# ═══════════════════════════════════════════════════════════════════════════════
def _transcribe_numpy(audio: np.ndarray, source_rate: int) -> str:
    """
    Resample audio to 16 kHz, send to Groq Whisper with verbose_json format
    so we receive per-segment confidence scores.

    Filters applied (in order):
      1. no_speech_prob > 0.5  -> discard (Whisper thinks it is not speech)
      2. avg_logprob    < -1.0 -> discard (Whisper has very low confidence)
      3. Noise-word blocklist  -> discard common hallucination phrases
      4. Word count < 2        -> discard (single-word noise artefact)
    """
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

    logger.debug("Sending audio to Groq Whisper (verbose_json)...")
    try:
        transcription = _client.audio.transcriptions.create(
            file=("audio.wav", wav_buffer, "audio/wav"),
            model="whisper-large-v3",
            language="en",
            response_format="verbose_json",
        )

        # ── Confidence filtering ─────────────────────────────────────────────
        try:
            segments = getattr(transcription, "segments", None) or []
            if segments:
                avg_no_speech = sum(
                    getattr(s, "no_speech_prob", 0.0) for s in segments
                ) / len(segments)
                avg_logprob = sum(
                    getattr(s, "avg_logprob", 0.0) for s in segments
                ) / len(segments)

                logger.debug(
                    f"Whisper confidence — no_speech_prob: {avg_no_speech:.3f}, "
                    f"avg_logprob: {avg_logprob:.3f}"
                )

                if avg_no_speech > 0.5:
                    logger.warning(
                        f"High no_speech_prob ({avg_no_speech:.2f}) — "
                        "discarding as noise"
                    )
                    return ""

                if avg_logprob < -1.0:
                    logger.warning(
                        f"Low confidence avg_logprob ({avg_logprob:.2f}) — "
                        "discarding as noise"
                    )
                    return ""

        except Exception as conf_err:
            # If confidence parsing fails for any reason, proceed without it
            logger.debug(f"Confidence check skipped: {conf_err}")

        # ── Extract text ─────────────────────────────────────────────────────
        transcript = ""
        if hasattr(transcription, "text"):
            transcript = (transcription.text or "").strip()
        elif isinstance(transcription, str):
            transcript = transcription.strip()

        if not transcript:
            logger.warning("Whisper returned empty transcript")
            return ""

        # ── Noise-phrase blocklist ────────────────────────────────────────────
        normalised = transcript.lower().strip().rstrip(".,!?")
        if normalised in _NOISE_PHRASES:
            logger.warning(
                f"Transcript '{transcript}' matched noise-phrase blocklist — "
                "discarding"
            )
            return ""

        logger.info(f"Heard: '{transcript}'")
        return transcript

    except Exception as e:
        logger.error(f"Groq Whisper API failed: {e}")
        return ""
