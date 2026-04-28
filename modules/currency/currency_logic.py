"""
modules/currency/currency_logic.py

Detection → Speech pipeline with parallel execution.

Architecture
────────────
Detection loop (15 fps)
    │  puts label string into a single-slot "latest-wins" queue
    ▼
_SpeechWorker thread  (runs independently, never blocks detection)
    │  pulls the latest label, speaks it, then waits for REPEAT_DELAY
    ▼
Speaker.speak()  (non-blocking TTS enqueue)

Key properties
──────────────
• Detection thread is NEVER blocked by TTS synthesis or playback.
• Only the *most recent* confirmed label is spoken — stale labels are
  automatically overwritten in the single-slot queue before they play.
• Once the worker finishes speaking, it immediately checks for the
  latest enqueued label (no artificial sleep between detections).
• REPEAT_DELAY prevents re-announcing the same note every frame while
  still re-announcing if the note changes or disappears and reappears.
• "No note" is handled by a GONE_DEBOUNCE: the worker only announces
  silence after the note has been absent for that many seconds, avoiding
  spurious "gone" announcements from brief tracking gaps.
"""

import threading
import time
from collections import Counter
from utils.logger import logger

# ── Tuning constants ──────────────────────────────────────────────────────────

# Minimum seconds before the *same* label is spoken again.
# Lower  = more responsive to holds; Higher = less repetitive.
REPEAT_DELAY = 3.0

# How many seconds a note must be continuously absent before the worker
# considers it gone and resets its "last spoken" memory.
GONE_DEBOUNCE = 1.5

# ── Single-slot "latest-wins" queue ──────────────────────────────────────────

class _LatestQueue:
    """
    Thread-safe, single-slot queue.  put() always overwrites the stored
    value so the consumer always sees the most-recent item.
    get_nowait() returns the stored value (and clears it) or None.
    """
    def __init__(self):
        self._lock  = threading.Lock()
        self._value = None          # None means "nothing new"

    def put(self, value):
        with self._lock:
            self._value = value

    def get_nowait(self):
        with self._lock:
            v, self._value = self._value, None
            return v

    def clear(self):
        with self._lock:
            self._value = None


_latest_label: _LatestQueue = _LatestQueue()


# ── Speech worker ─────────────────────────────────────────────────────────────

class _SpeechWorker:
    """
    Dedicated thread that consumes _latest_label and drives the TTS speaker.

    State machine
    ─────────────
    _last_spoken_label  : label spoken most recently (None = nothing yet)
    _last_spoken_time   : wall-clock time of that speak() call
    _last_seen_time     : last time a non-None label arrived from detection
    """

    _POLL_INTERVAL = 0.05   # 50 ms poll — low CPU, still very responsive

    def __init__(self):
        self._last_spoken_label: str | None = None
        self._last_spoken_time:  float      = 0.0
        self._last_seen_time:    float      = 0.0

        self._stop_evt = threading.Event()
        self._thread   = threading.Thread(
            target=self._run, daemon=True, name="currency-speech-worker"
        )
        self._thread.start()
        logger.info("_SpeechWorker: started ✓")

    # ── public ────────────────────────────────────────────────────────────────

    def enqueue(self, label: str | None):
        """Called from the detection thread.  Always non-blocking."""
        _latest_label.put(label)

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=3.0)

    def reset(self):
        _latest_label.clear()
        self._last_spoken_label = None
        self._last_spoken_time  = 0.0
        self._last_seen_time    = 0.0
        logger.info("_SpeechWorker: state reset")

    # ── worker loop ───────────────────────────────────────────────────────────

    def _run(self):
        speaker = _get_speaker()

        while not self._stop_evt.is_set():
            label = _latest_label.get_nowait()
            now   = time.monotonic()

            if label is not None:
                # Detection is alive and has a confirmed note
                self._last_seen_time = now
                self._maybe_speak(label, now, speaker)

            else:
                # No confirmed note in this slot — check gone debounce
                if (self._last_spoken_label is not None
                        and (now - self._last_seen_time) >= GONE_DEBOUNCE):
                    # Note has been absent long enough — reset memory so
                    # the same denomination will be announced fresh when
                    # it reappears.
                    logger.debug(
                        f"_SpeechWorker: '{self._last_spoken_label}' gone "
                        f"after {now - self._last_seen_time:.1f}s — resetting memory"
                    )
                    self._last_spoken_label = None
                    self._last_spoken_time  = 0.0

            self._stop_evt.wait(self._POLL_INTERVAL)

    def _maybe_speak(self, label: str, now: float, speaker):
        """
        Decide whether to speak *label* right now.

        Speak if:
          (a) label changed from last spoken, OR
          (b) same label but REPEAT_DELAY has elapsed (user still holding it)
        """
        same_label    = (label == self._last_spoken_label)
        time_elapsed  = (now - self._last_spoken_time) >= REPEAT_DELAY

        if same_label and not time_elapsed:
            return  # Too soon — skip

        msg = _build_message_from_label(label)
        logger.info(
            f"_SpeechWorker: speaking '{msg}' "
            f"({'repeat' if same_label else 'new label'})"
        )

        self._last_spoken_label = label
        self._last_spoken_time  = now

        if speaker:
            speaker.speak(msg)
        else:
            logger.warning(f"_SpeechWorker: [NO TTS] {msg}")


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


# ── Worker singleton ──────────────────────────────────────────────────────────

_worker: _SpeechWorker | None = None
_worker_lock = threading.Lock()


def _get_worker() -> _SpeechWorker:
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = _SpeechWorker()
        return _worker


# ── Public API (called from currency_detector.py) ────────────────────────────

def process_confirmed_notes(confirmed: list) -> None:
    """
    Called every detection frame.  Non-blocking — just enqueues the latest
    label into the single-slot queue and returns immediately.

    The _SpeechWorker thread handles rate-limiting and TTS in parallel.
    """
    if confirmed:
        label = _summarise_label([t["confirmed_cls"] for t in confirmed])
        _get_worker().enqueue(label)
    else:
        # Signal "nothing visible" so the worker can run gone-debounce logic
        _get_worker().enqueue(None)


def reset_logic_state():
    """Called by currency_detector.reset() when currency mode stops."""
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.stop()
            _worker = None
    _latest_label.clear()
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
    Used as the identity token for change-detection in _SpeechWorker.
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
        phrases.append(f"a {denom}" if qty == 1 else f"{qty} {denom}")

    if len(phrases) == 1:
        note_phrase = phrases[0]
    elif len(phrases) == 2:
        note_phrase = f"{phrases[0]} and {phrases[1]}"
    else:
        note_phrase = ", ".join(phrases[:-1]) + f", and {phrases[-1]}"

    total = sum(counts.values())
    noun  = "notes" if total > 1 else "note"

    return f"I can see {note_phrase} {noun}."
