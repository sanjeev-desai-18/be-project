"""
tts/speaker.py — Text-to-Speech for Raspberry Pi 5
───────────────────────────────────────────────────
Supports:
  • gTTS  — free, online, uses Google TTS
  • ElevenLabs — high quality, requires API key

Audio playback uses pygame.mixer (works reliably with ALSA / Bluetooth on Pi).
playsound is avoided because it has known issues with ALSA on Raspberry Pi OS.

Thread-safe: a background worker thread drains a queue so that speak() is
ALWAYS non-blocking — the detection / camera loop is never stalled waiting
for network (gTTS) or audio playback to finish.

If a new item arrives while the worker is still playing, the old item is
replaced (drop-oldest strategy) so announcements stay current and the queue
never grows unboundedly.
"""

import os
import queue
import threading
import tempfile

from utils.logger import logger
from config import (
    TTS_ENGINE, TTS_LANGUAGE, TTS_SLOW,
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
)

# ── Pygame mixer — initialise once at module load ─────────────────────────────
_pygame_ready = False
try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=1024)
    _pygame_ready = True
    logger.info("pygame.mixer initialised for audio playback ✓")
except Exception as e:
    logger.warning(f"pygame.mixer init failed: {e} — will fall back to subprocess aplay")


def _play_mp3(path: str):
    """Play an MP3 file via pygame or fallback to aplay (Pi-safe)."""
    if _pygame_ready:
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            return
        except Exception as e:
            logger.warning(f"pygame playback failed: {e} — trying aplay")

    # Fallback: convert to wav and use aplay (always available on Pi)
    try:
        wav_path = path.replace(".mp3", "_tmp.wav")
        os.system(f"ffmpeg -y -i {path} {wav_path} -loglevel quiet")
        os.system(f"aplay {wav_path}")
        try:
            os.unlink(wav_path)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"aplay fallback also failed: {e}")
        print(f"\n[SPEECH OUTPUT — no audio]: {path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# ASYNC TTS WORKER
# ══════════════════════════════════════════════════════════════════════════════

_SENTINEL = object()          # poison pill to stop the worker


class _TTSWorker:
    """
    Background thread that synthesises and plays speech without blocking callers.

    Strategy: queue size = 1.  If a new message arrives while the worker is
    busy, the pending-but-not-yet-played item is dropped and replaced.  This
    keeps announcements fresh (no stale backlog) and the queue never grows.
    """

    def __init__(self):
        self._q = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, daemon=True, name="tts-worker")
        self._thread.start()
        logger.info("TTS worker thread started ✓")

    # ── public ────────────────────────────────────────────────────────────────

    def enqueue(self, text):
        """
        Non-blocking enqueue.  Returns immediately.
        Drops the pending item if the queue is already full (drop-oldest).
        """
        try:
            self._q.put_nowait(text)
        except queue.Full:
            # Drain the stale pending item, then put the fresh one.
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(text)
            except queue.Full:
                pass  # extremely rare race; skip rather than block

    def stop(self):
        """Signal the worker to exit cleanly."""
        self.enqueue(_SENTINEL)
        self._thread.join(timeout=5)

    # ── internal ──────────────────────────────────────────────────────────────

    def _run(self):
        while True:
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is _SENTINEL:
                logger.info("TTS worker stopping")
                break

            text = item
            try:
                self._synthesise_and_play(text)
            except Exception as e:
                logger.error(f"TTS worker error: {e}", exc_info=True)

    def _synthesise_and_play(self, text):
        if TTS_ENGINE == "elevenlabs":
            _speak_elevenlabs(text)
        else:
            _speak_gtts(text)


# ── Module-level singleton worker (created once at import time) ───────────────
_worker = None
_worker_lock = threading.Lock()


def _get_worker():
    global _worker
    if _worker is None:
        with _worker_lock:
            if _worker is None:
                _worker = _TTSWorker()
    return _worker


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHESIS HELPERS  (run inside the worker thread — blocking is fine here)
# ══════════════════════════════════════════════════════════════════════════════

def _speak_gtts(text):
    temp_path = None
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=TTS_LANGUAGE, slow=TTS_SLOW)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name
        tts.save(temp_path)
        _play_mp3(temp_path)
    except Exception as e:
        logger.error(f"gTTS failed: {e}")
        print(f"\n[SPEECH OUTPUT]: {text}\n")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


def _speak_elevenlabs(text):
    temp_path = None
    try:
        from elevenlabs import ElevenLabs
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio_generator = client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=text,
            model_id="eleven_turbo_v2",
        )
        audio_bytes = b"".join(audio_generator)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        _play_mp3(temp_path)
    except Exception as e:
        logger.error(f"ElevenLabs failed: {e} — falling back to gTTS")
        _speak_gtts(text)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API  — unchanged interface, now non-blocking
# ══════════════════════════════════════════════════════════════════════════════

class Speaker:
    """
    Thread-safe TTS speaker.
    speak() returns immediately — synthesis and playback happen in the
    background TTS worker thread so the camera/inference loop is never blocked.

    Usage: Speaker().speak("500 rupees detected")
    """

    def speak(self, text):
        if not text or not text.strip():
            logger.warning("Speaker received empty text — skipping")
            return

        preview = text[:70] + "..." if len(text) > 70 else text
        logger.info(f"Queuing speech: '{preview}'")

        _get_worker().enqueue(text)
