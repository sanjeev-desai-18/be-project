"""
tts/speaker.py — Text-to-Speech for Raspberry Pi 5
───────────────────────────────────────────────────
Supports:
  • pyttsx3    — offline, zero-latency (RECOMMENDED for lowest delay)
                 Uses espeak-ng on Pi. No network, no file write.
                 First word plays in <200ms.
  • gTTS       — Google TTS, online. Uses in-memory buffer (no disk write).
  • ElevenLabs — high quality, requires API key

Audio playback:
  • pyttsx3 uses espeak-ng directly via ALSA — no pygame needed
  • gTTS uses pygame.mixer with an in-memory BytesIO buffer
  • ElevenLabs uses pygame.mixer

Thread-safe: a background worker thread drains a queue so that speak() is
ALWAYS non-blocking — the detection / camera loop is never stalled.

Drop-oldest strategy: if a new message arrives while the worker is playing,
the pending-but-not-yet-started item is replaced. Queue never grows.
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
)

# ── pygame mixer — only needed for gTTS / ElevenLabs ─────────────────────────
_pygame_ready = False
if TTS_ENGINE in ("gtts", "elevenlabs"):
    try:
        import pygame
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=1024)
        _pygame_ready = True
        logger.info("pygame.mixer initialised for audio playback ✓")
    except Exception as e:
        logger.warning(f"pygame.mixer init failed: {e}")


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


def _speak_gtts(text: str):
    """
    Google TTS using an in-memory BytesIO buffer — no temp file written.
    Saves ~0.5-1s vs the original tts.save(path) approach.
    """
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
        else:
            # Fallback: write to temp file and use aplay
            import tempfile
            buf.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(buf.read())
                tmp = f.name
            try:
                wav = tmp.replace(".mp3", "_tmp.wav")
                os.system(f"ffmpeg -y -i {tmp} {wav} -loglevel quiet")
                os.system(f"aplay {wav}")
            finally:
                for p in (tmp, tmp.replace(".mp3", "_tmp.wav")):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass
    except Exception as e:
        logger.error(f"gTTS playback failed: {e}")


def _synthesise_to_buffer(text: str):
    """
    Synthesise text via gTTS into an in-memory BytesIO buffer and return it.
    Does NOT play anything. Used by the prefetch pipeline so the next
    sentence's audio is ready before the current sentence finishes playing.
    Returns None on failure.
    """
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=TTS_LANGUAGE, slow=TTS_SLOW)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf
    except Exception as e:
        logger.error(f"gTTS synthesis failed: {e}")
        return None


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

                if TTS_ENGINE == "gtts" and _pygame_ready:
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
            if TTS_ENGINE == "pyttsx3":
                _speak_pyttsx3(text)
            elif TTS_ENGINE == "elevenlabs":
                _speak_elevenlabs(text)
            else:
                _speak_gtts(text)
        except Exception as e:
            logger.error(f"TTS worker synthesise error: {e}", exc_info=True)


# ── Module-level singleton ────────────────────────────────────────────────────
_worker      = None
_worker_lock = threading.Lock()


def _get_worker() -> _TTSWorker:
    global _worker
    if _worker is None:
        with _worker_lock:
            if _worker is None:
                _worker = _TTSWorker()
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
        preview = text[:70] + "..." if len(text) > 70 else text
        logger.info(f"Queuing stream sentence: '{preview}'")
        _get_worker().enqueue_stream(text)
