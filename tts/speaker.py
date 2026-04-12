"""
tts/speaker.py — Text-to-Speech for Raspberry Pi 5
───────────────────────────────────────────────────
Supports:
  • piper      — RECOMMENDED. Offline neural TTS, <200ms latency, natural voice.
                 pip install piper-tts
                 Set PIPER_MODEL in config.py to the .onnx file path.
  • pyttsx3    — offline, espeak-ng. Robotic but instant.
  • gTTS       — Google Translate TTS. ~1-2s network delay, unnatural prosody.
  • ElevenLabs — highest quality, requires API key.

Thread-safe: a background worker thread drains a queue so speak() is
ALWAYS non-blocking — the detection / camera loop is never stalled.
"""

import io
import os
import queue
import threading

from utils.logger import logger
from config import (
    TTS_ENGINE, TTS_LANGUAGE, TTS_SLOW,
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    PYTTSX3_RATE, PYTTSX3_VOLUME,
    PIPER_MODEL, PIPER_LENGTH_SCALE,
    EDGE_TTS_VOICE, EDGE_TTS_RATE,
)

# ── pygame mixer — only needed for gTTS / ElevenLabs ─────────────────────────
_pygame_ready = False
if TTS_ENGINE in ("gtts", "elevenlabs"):
    try:
        import pygame
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=4096)
        _pygame_ready = True
        logger.info("pygame.mixer initialised for audio playback ✓")
    except Exception as e:
        logger.warning(f"pygame.mixer init failed: {e}")



# ── Piper TTS voice — loaded ONCE into memory ────────────────────────────────
# The ONNX model (~60MB) is loaded at worker-thread startup via _init_piper().
# After that every synthesis call is pure ONNX inference (~0.2s on Pi 5) —
# no subprocess spawn, no model reload from disk, no cold-start delay.
_piper_voice = None
_piper_ready = threading.Event()   # set once model is loaded; speak_stream waits on it

# ── Bluetooth keepalive stream ────────────────────────────────────────────────
# Bluetooth speakers go to standby after ~1s of silence and need ~800ms-1s to
# wake up. The first part of every utterance is lost during that wake-up window.
# Fix: keep a sounddevice OutputStream open permanently, draining near-silent
# samples. The device never sleeps, so speech plays from the very first sample.
_keepalive_stream = None
_keepalive_lock   = threading.Lock()
_KEEPALIVE_RATE   = 44100   # Hz — standard A2DP rate; avoids BT driver resampling artifacts


def _start_keepalive():
    """Start a silent background stream to keep the audio device awake."""
    global _keepalive_stream
    with _keepalive_lock:
        if _keepalive_stream is not None:
            return
        try:
            import sounddevice as sd
            import numpy as np
            # blocksize=512 @ 22050Hz = ~23ms per callback — very low CPU
            _keepalive_stream = sd.OutputStream(
                samplerate=_KEEPALIVE_RATE,
                channels=1,
                dtype="float32",
                blocksize=512,
            )
            _keepalive_stream.start()
            logger.info("Audio keepalive stream started ✓")
        except Exception as e:
            logger.warning(f"Keepalive stream failed to start: {e}")
            _keepalive_stream = None


def _stop_keepalive():
    """Stop the keepalive stream (called on shutdown)."""
    global _keepalive_stream
    with _keepalive_lock:
        if _keepalive_stream is not None:
            try:
                _keepalive_stream.stop()
                _keepalive_stream.close()
            except Exception:
                pass
            _keepalive_stream = None


def _init_piper():
    """
    Load PiperVoice into memory. Called once at TTS worker thread startup.

    Keeping the model in RAM means every speak() call goes:
      text → ONNX inference (~0.2s) → WAV bytes → aplay → audio out
    instead of:
      text → spawn subprocess → load 60MB ONNX → inference → audio out  (~2s)
    """
    global _piper_voice
    if _piper_voice is not None:
        return _piper_voice
    import os
    if not os.path.isfile(PIPER_MODEL):
        logger.error(f"Piper model not found: {PIPER_MODEL}")
        return None
    try:
        from piper.voice import PiperVoice
        _piper_voice = PiperVoice.load(PIPER_MODEL)
        logger.info(f"Piper voice loaded into memory: {PIPER_MODEL} ✓")
        _start_keepalive()   # warm up audio device immediately after model loads
        _piper_ready.set()
        return _piper_voice
    except Exception as e:
        logger.error(f"Piper load failed: {e}")
        _piper_ready.set()   # unblock waiters even on failure
        return None


def _synthesise_sentence(voice, text: str, syn_cfg) -> "np.ndarray":
    """
    Synthesise one sentence to float32 mono samples at the voice's native rate.
    Returns a 1-D numpy float32 array.
    """
    import io, wave
    import numpy as np

    wav_buf  = io.BytesIO()
    wav_file = wave.open(wav_buf, "wb")
    try:
        if syn_cfg is not None:
            voice.synthesize_wav(text, wav_file, syn_config=syn_cfg)
        else:
            original_ls = getattr(voice.config, "length_scale", 1.0)
            original_ns = getattr(voice.config, "noise_scale",  0.667)
            voice.config.length_scale = PIPER_LENGTH_SCALE
            # Clamp noise_scale: keeps natural variation while avoiding BT
            # SBC codec artifacts (amy default 0.833 is too high for BT)
            voice.config.noise_scale  = min(original_ns, 0.45)
            try:
                voice.synthesize(text, wav_file)
            finally:
                voice.config.length_scale = original_ls
                voice.config.noise_scale  = original_ns
    finally:
        try:
            wav_file.close()
        except Exception:
            pass

    wav_buf.seek(0)
    wav_r  = wave.open(wav_buf)
    sr     = wav_r.getframerate()
    ch     = wav_r.getnchannels()
    sw     = wav_r.getsampwidth()
    frames = wav_r.readframes(wav_r.getnframes())
    wav_r.close()

    dtype   = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    samples /= float(np.iinfo(dtype).max)
    return samples, sr   # 1-D float32, native sample rate


def _resample(samples: "np.ndarray", from_rate: int, to_rate: int) -> "np.ndarray":
    """High-quality resample using scipy. Falls back to linear interp if unavailable."""
    if from_rate == to_rate:
        return samples
    try:
        from scipy.signal import resample_poly
        import math
        g = math.gcd(from_rate, to_rate)
        return resample_poly(samples, to_rate // g, from_rate // g).astype("float32")
    except ImportError:
        import numpy as np
        n_out = int(len(samples) * to_rate / from_rate)
        return np.interp(
            np.linspace(0, len(samples) - 1, n_out),
            np.arange(len(samples)),
            samples,
        ).astype("float32")


def _speak_piper(text: str) -> None:
    """
    Synthesise text sentence-by-sentence, insert natural pauses between
    sentences, resample to 44100 Hz so Bluetooth SBC has no resampling
    artifacts, then play through the keepalive stream.
    """
    import re
    import numpy as np
    import sounddevice as sd

    voice = _init_piper()
    if voice is None:
        logger.warning("Piper unavailable — falling back to gTTS")
        _speak_gtts(text)
        return

    try:
        # SynthesisConfig for new piper API (ignored if import fails)
        try:
            from piper import SynthesisConfig
            syn_cfg = SynthesisConfig(length_scale=PIPER_LENGTH_SCALE,
                                      noise_scale=0.45)
        except ImportError:
            syn_cfg = None

        # Split on sentence-ending punctuation, keep delimiter
        # e.g. "Hello. How are you? Fine!" → ["Hello.", "How are you?", "Fine!"]
        raw_sentences = re.split(r'(?<=[.!?])\\s+', text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        if not sentences:
            sentences = [text.strip()]

        PLAY_RATE = _KEEPALIVE_RATE   # 44100 Hz — stream is already open at this rate
        PAUSE_MS  = 350               # ms of silence between sentences
        pause_samples = np.zeros(int(PLAY_RATE * PAUSE_MS / 1000), dtype="float32")

        # Synthesise all sentences and build one contiguous audio array
        chunks = []
        for i, sentence in enumerate(sentences):
            pcm, native_rate = _synthesise_sentence(voice, sentence, syn_cfg)
            # Resample from piper native rate (22050) to 44100 — high quality
            pcm_44 = _resample(pcm, native_rate, PLAY_RATE)
            chunks.append(pcm_44)
            if i < len(sentences) - 1:
                chunks.append(pause_samples)   # natural pause between sentences

        full_audio = np.concatenate(chunks).reshape(-1, 1)   # (N, 1) for mono stream

        # Ensure keepalive stream is open and warm
        _start_keepalive()
        stream = _keepalive_stream
        if stream is not None and stream.active:
            stream.write(full_audio)
        else:
            sd.play(full_audio, samplerate=PLAY_RATE)
            sd.wait()

        logger.debug(f"Piper playback complete: {len(sentences)} sentence(s), "
                     f"{len(full_audio)/PLAY_RATE:.1f}s")

    except Exception as e:
        logger.error(f"Piper speak failed: {e} — falling back to gTTS")
        _speak_gtts(text)

# ── pyttsx3 engine singleton ──────────────────────────────────────────────────
# Lives in the TTS worker thread — pyttsx3 is NOT thread-safe across threads,
# so we create and use it only inside _TTSWorker._run().
_pyttsx3_engine = None


def _init_pyttsx3():
    """Create and configure pyttsx3 engine. Called once inside worker thread."""
    global _pyttsx3_engine
    if _pyttsx3_engine is not None:
        return _pyttsx3_engine
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate",   PYTTSX3_RATE)
        engine.setProperty("volume", PYTTSX3_VOLUME)

        # Pick the first English voice available
        voices = engine.getProperty("voices")
        for v in voices:
            if "en" in v.id.lower() or "english" in (v.name or "").lower():
                engine.setProperty("voice", v.id)
                logger.info(f"pyttsx3 voice selected: {v.id}")
                break

        _pyttsx3_engine = engine
        logger.info("pyttsx3 engine initialised ✓")
        return engine
    except Exception as e:
        logger.error(f"pyttsx3 init failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHESIS HELPERS  (run inside the worker thread — blocking is fine here)
# ══════════════════════════════════════════════════════════════════════════════

def _speak_pyttsx3(text: str):
    """
    Speak using pyttsx3 / espeak-ng.
    Offline, zero network latency, no file I/O.
    First syllable plays in <200ms.
    """
    engine = _init_pyttsx3()
    if engine is None:
        logger.warning("pyttsx3 unavailable — falling back to gTTS")
        _speak_gtts(text)
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logger.error(f"pyttsx3 speak failed: {e} — falling back to gTTS")
        try:
            # Reset engine state in case it's corrupted
            engine.stop()
        except Exception:
            pass
        _speak_gtts(text)


def _synthesise_to_buffer(text: str):
    """
    Synthesise text via edge-tts (Microsoft neural TTS) to an in-memory
    MP3 BytesIO buffer. edge-tts is free, no API key, natural human voice.
    Falls back to gTTS if edge-tts is unavailable.
    """
    try:
        import asyncio
        import edge_tts
        communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE, rate=EDGE_TTS_RATE)
        buf = io.BytesIO()
        async def _run():
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
        asyncio.run(_run())
        buf.seek(0)
        if buf.getbuffer().nbytes > 100:
            return buf
        raise RuntimeError("edge-tts returned empty audio")
    except Exception as e:
        logger.warning(f"edge-tts failed ({e}) — falling back to gTTS")
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang=TTS_LANGUAGE, slow=TTS_SLOW)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            return buf
        except Exception as e2:
            logger.error(f"gTTS also failed: {e2}")
            return None


def _speak_gtts(text: str):
    """Synthesise via edge-tts (or gTTS fallback) and play through pygame."""
    buf = _synthesise_to_buffer(text)
    if buf is None:
        print(f"\n[SPEECH OUTPUT]: {text}\n")
        return
    try:
        if _pygame_ready:
            import pygame
            buf.seek(0)
            pygame.mixer.music.load(buf)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.time.wait(300)
        else:
            import tempfile
            buf.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(buf.read())
                tmp = f.name
            try:
                os.system(f"ffmpeg -y -i {tmp} /tmp/_tts_out.wav -loglevel quiet && aplay /tmp/_tts_out.wav")
            finally:
                try: os.unlink(tmp)
                except Exception: pass
    except Exception as e:
        logger.error(f"TTS playback failed: {e}")


def _speak_elevenlabs(text: str):
    try:
        from elevenlabs import ElevenLabs
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio_generator = client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=text,
            model_id="eleven_turbo_v2",
        )
        audio_bytes = b"".join(audio_generator)
        mp3_buffer = io.BytesIO(audio_bytes)

        if _pygame_ready:
            import pygame
            pygame.mixer.music.load(mp3_buffer)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.time.wait(300)
        else:
            logger.error("ElevenLabs: pygame not available")
            _speak_gtts(text)

    except Exception as e:
        logger.error(f"ElevenLabs failed: {e} — falling back to gTTS")
        _speak_gtts(text)


# ══════════════════════════════════════════════════════════════════════════════
# ASYNC TTS WORKER
# ══════════════════════════════════════════════════════════════════════════════

_SENTINEL = object()


class _TTSWorker:
    """
    Background thread that synthesises and plays speech without blocking callers.

    TWO queues with different strategies:

    _announcement_q  (maxsize=1, drop-oldest)
        For one-shot announcements: currency detections, ACK phrases, errors.
        If the worker is busy and a new announcement arrives, the stale pending
        one is dropped so the freshest message plays. This is the original
        behaviour — correct for currency mode where stale "500 rupees" is
        useless by the time it would play.

    _stream_q  (unbounded)
        For streamed sentences from scene/reading modules.
        Every sentence MUST play in order — nothing is ever dropped.
        The streaming loop enqueues sentences as fast as they arrive from the
        VLM; the worker plays them one by one in sequence.

    Priority: the worker always drains _stream_q first. Once _stream_q is
    empty it checks _announcement_q. This means a streaming response is never
    interrupted mid-sentence by a currency announcement.
    """

    def __init__(self):
        self._announcement_q = queue.Queue(maxsize=1)
        self._stream_q       = queue.Queue()          # unbounded — never drops
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="tts-worker"
        )
        self._thread.start()
        logger.info("TTS worker thread started ✓")

    # ── announcement path (drop-oldest) ──────────────────────────────────────

    def enqueue_announcement(self, text: str):
        """
        Non-blocking. Drops stale pending announcement if queue is full.
        Use for: currency detections, one-shot status messages, errors.
        """
        try:
            self._announcement_q.put_nowait(text)
        except queue.Full:
            try:
                self._announcement_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._announcement_q.put_nowait(text)
            except queue.Full:
                pass

    # ── stream path (ordered, no drop) ───────────────────────────────────────

    def enqueue_stream(self, text: str):
        """
        Non-blocking. Puts text onto the unbounded stream queue.
        Every item will be played in order — nothing is dropped.
        Use for: streamed sentences from scene/reading modules.
        """
        self._stream_q.put_nowait(text)

    # ── stop ─────────────────────────────────────────────────────────────────

    def stop(self):
        self._stream_q.put_nowait(_SENTINEL)
        self._thread.join(timeout=5)

    # ── worker loop ──────────────────────────────────────────────────────────

    def _run(self):
        import time as _time

        if TTS_ENGINE == "pyttsx3":
            _init_pyttsx3()
        elif TTS_ENGINE == "piper":
            _init_piper()

        # Prefetch buffer: holds (text, audio_bytes_or_None) tuples.
        # For gTTS we synthesise the NEXT sentence in a background thread
        # while the current one is playing, so the gap between sentences is
        # the max(remaining_playback, synthesis_time) instead of their sum.
        prefetch_result  = [None]   # [BytesIO | None]
        prefetch_text    = [None]   # [str | None]
        prefetch_done    = threading.Event()
        prefetch_thread  = [None]

        def _start_prefetch(text):
            prefetch_result[0] = None
            prefetch_text[0]   = text
            prefetch_done.clear()

            def _fetch():
                try:
                    prefetch_result[0] = _synthesise_to_buffer(text)
                except Exception as e:
                    logger.warning(f"Prefetch failed for '{text[:40]}': {e}")
                    prefetch_result[0] = None
                finally:
                    prefetch_done.set()

            t = threading.Thread(target=_fetch, daemon=True, name="tts-prefetch")
            t.start()
            prefetch_thread[0] = t

        def _play_buffer(buf):
            """Play a BytesIO MP3 buffer via pygame."""
            if buf is None:
                return
            try:
                buf.seek(0)
                import pygame
                pygame.mixer.music.load(buf)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                # Drain ALSA hardware buffer — prevents final syllable cut-off.
                pygame.time.wait(300)
            except Exception as e:
                logger.error(f"Prefetch playback failed: {e}")

        while True:
            # ── priority: stream queue (prefetch pipeline) ────────────────
            try:
                item = self._stream_q.get_nowait()
            except queue.Empty:
                item = None

            if item is not None:
                if item is _SENTINEL:
                    logger.info("TTS worker stopping")
                    break

                if TTS_ENGINE == "gtts" and _pygame_ready and TTS_ENGINE != "piper":
                    # Prefetch pipeline for gTTS:
                    # 1. Synthesise current sentence to buffer (or reuse prefetch).
                    # 2. While playing, kick off prefetch for next sentence.
                    # 3. When playback done, next buffer is (usually) ready.

                    # Check if this sentence was prefetched already
                    if prefetch_text[0] == item:
                        prefetch_done.wait(timeout=8.0)
                        buf = prefetch_result[0]
                    else:
                        buf = _synthesise_to_buffer(item)

                    # Peek at next item to start prefetching immediately
                    try:
                        next_item = self._stream_q.queue[0]  # non-destructive peek
                        if next_item is not _SENTINEL and isinstance(next_item, str):
                            _start_prefetch(next_item)
                    except (IndexError, AttributeError):
                        pass

                    # Play current sentence
                    _play_buffer(buf)

                else:
                    # pyttsx3 / elevenlabs — no prefetch needed
                    self._synthesise_and_play(item)

                continue

            # ── fallback: announcement queue ──────────────────────────────
            try:
                item = self._announcement_q.get_nowait()
                if item is _SENTINEL:
                    logger.info("TTS worker stopping")
                    break
                self._synthesise_and_play(item)
                continue
            except queue.Empty:
                pass

            _time.sleep(0.02)

    def _synthesise_and_play(self, text: str):
        try:
            if TTS_ENGINE == "piper":
                _speak_piper(text)
            elif TTS_ENGINE == "pyttsx3":
                _speak_pyttsx3(text)
            elif TTS_ENGINE == "elevenlabs":
                _speak_elevenlabs(text)
            else:
                _speak_gtts(text)
        except Exception as e:
            logger.error(f"TTS worker synthesise error: {e}", exc_info=True)


# ── Module-level singleton ────────────────────────────────────────────────────
_worker_lock = threading.Lock()

# Pre-start the worker immediately at import time so the Piper model is loaded
# into memory before any speak() call arrives. Without this, the first call
# races with model loading and the startup announcement is skipped or delayed.
_worker: _TTSWorker = _TTSWorker()
logger.info("TTS worker pre-started at import time ✓")


def _get_worker() -> _TTSWorker:
    return _worker


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

class Speaker:
    """
    Thread-safe TTS speaker. All methods return immediately.

    speak(text)
        One-shot announcement. Uses drop-oldest queue — stale messages are
        evicted if the worker is busy. Use for currency detections, ACKs,
        errors, and any single standalone message.

    speak_stream(text)
        Ordered streaming sentence. Uses unbounded queue — nothing is ever
        dropped. Every call will be played in the order it was enqueued.
        Use for each sentence yielded by vlm.describe_stream() so the full
        scene/reading response plays completely without gaps.

    Usage:
        sp = Speaker()

        # One-shot ACK (drop-oldest is fine — only one of these)
        sp.speak("Looking at your surroundings.")

        # Each streamed sentence (must ALL play in order)
        for sentence in vlm.describe_stream(frame, prompt):
            sp.speak_stream(sentence)
    """

    def speak(self, text: str):
        """One-shot announcement — drop-oldest strategy."""
        if not text or not text.strip():
            logger.warning("Speaker.speak() received empty text — skipping")
            return
        preview = text[:70] + "..." if len(text) > 70 else text
        logger.info(f"Queuing announcement: '{preview}'")
        _get_worker().enqueue_announcement(text)

    def speak_stream(self, text: str):
        """Ordered stream sentence — never dropped."""
        if not text or not text.strip():
            return
        # Wait up to 5s for piper model to finish loading so the first
        # announcement (startup, ACK) is never enqueued before the worker
        # is ready to process it.
        if TTS_ENGINE == "piper":
            _piper_ready.wait(timeout=5.0)
        preview = text[:70] + "..." if len(text) > 70 else text
        logger.info(f"Queuing stream sentence: '{preview}'")
        _get_worker().enqueue_stream(text)
