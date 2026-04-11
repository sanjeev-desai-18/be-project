# modules/reading/reading_module.py
#
# Changes from previous version:
#   1. run() now uses a 1.5s initial collection buffer before starting TTS —
#      same pipeline as scene_module.  The VLM stream is consumed in a
#      background thread; the main thread waits 1.5s, joins accumulated
#      sentences into one first chunk, speaks it, then streams the rest.
#
#      Result: no more cut-off words / mid-word TTS artefacts on the first
#      chunk.  The reading response can be long (medicine labels, receipts)
#      so the pipeline ensures continuous playback without gaps.
#
#   2. All other behaviour (camera capture, sharpness selection, Groq Vision
#      call, prompt, logging) is unchanged.

import base64
import threading
import queue as _queue
import time as _time

import cv2
import numpy as np

from utils.logger import logger
from utils.image_utils import frame_to_base64, resize_frame
from utils.camera_manager import camera_manager
from config import GROQ_API_KEY, VLM_MODEL, VLM_MAX_TOKENS


# Sentence boundaries for the streaming buffer
_SENTENCE_ENDS = frozenset({".", "!", "?", "\n"})
_CLAUSE_MIN_LEN = 40   # only split on comma/semicolon after this many chars

# How long (seconds) to buffer VLM output before speaking the first chunk.
_INITIAL_BUFFER_S = 1.5


_READING_PROMPT = """\
You are a reading assistant for a visually impaired user.
Read ALL visible text in this image completely and exactly.

Rules:
- Start with one short sentence of context, e.g. "This is a medicine label." or "This is a receipt."
- Then read every word exactly as written, top to bottom, left to right.
- Do NOT summarise or skip anything.
- Do NOT use bullet points, lists, or headings — speak it as natural continuous sentences.
- For medicine labels: read name, dosage, instructions, and warnings.
- For receipts: read every item, price, and total.
- If no text is visible, say exactly: I could not find any text. Please hold the document closer.
"""


class ReadingModule:

    def _sharpness_score(self, b64_image: str) -> float:
        try:
            img_bytes = base64.b64decode(b64_image)
            np_arr    = np.frombuffer(img_bytes, np.uint8)
            img       = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            return cv2.Laplacian(img, cv2.CV_64F).var()
        except Exception as e:
            logger.warning(f"Sharpness calculation failed: {e}")
            return 0.0

    def _capture_frames(self, count: int = 2) -> list:
        frames = []
        picam2 = camera_manager.acquire(mode="reading", warmup=0.4)
        try:
            for i in range(count):
                raw = picam2.capture_array("main")
                bgr = resize_frame(raw, max_width=1920)
                frames.append(frame_to_base64(bgr, quality=92))
                logger.debug(f"Reading frame {i+1}/{count} captured ✓")
                if i < count - 1:
                    _time.sleep(0.15)
        finally:
            camera_manager.release()
        return frames

    def _pick_sharpest(self, frames: list) -> str:
        if len(frames) == 1:
            return frames[0]
        scores     = [self._sharpness_score(f) for f in frames]
        best_index = scores.index(max(scores))
        logger.debug(
            f"Sharpness scores: {[round(s, 1) for s in scores]} "
            f"— using frame {best_index}"
        )
        return frames[best_index]

    def run(self, speaker=None) -> str:
        """
        Capture frames, pick the sharpest, and read all visible text.

        Args:
            speaker: optional Speaker instance.  When provided, uses the
                     two-phase buffered pipeline:

                     Phase 1 — silent collection (up to 1.5 s):
                       VLM tokens are buffered into sentences and collected
                       without speaking.  This ensures TTS never starts on
                       a partial word or incomplete sentence.

                     Phase 2 — speak + pipeline:
                       All buffered sentences are joined and spoken as one
                       clean first chunk.  Remaining sentences are spoken
                       immediately as they arrive from the VLM.

        Returns:
            Full reading text (used for logging / state).
            If speaker was provided, audio has already been queued — caller
            should set spoken=True in state to skip tts_node.
        """
        logger.info("ReadingModule.run() | capturing frames via camera_manager")

        try:
            frames = self._capture_frames(2)
        except Exception as e:
            logger.error(f"Camera error: {e}")
            return "I could not access the camera. Please check it is connected."

        if not frames:
            return "I could not capture any frames from the camera."

        best_frame = self._pick_sharpest(frames)
        logger.info("Best frame selected for reading ✓")

        logger.info("Sending frame to Groq Vision (streaming)...")

        full_text = ""

        # ── Build the Groq streaming generator ───────────────────────────────
        def _make_stream():
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            return client.chat.completions.create(
                model=VLM_MODEL,
                max_tokens=2048,
                stream=True,
                timeout=30,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{best_frame}"}},
                        {"type": "text", "text": _READING_PROMPT}
                    ]
                }]
            )

        def _iter_sentences(stream):
            """
            Consume the raw Groq stream and yield complete sentences / clauses,
            using the same buffering logic as the original reading_module.
            """
            buffer = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta is None:
                    continue
                buffer += delta

                while True:
                    idx = -1
                    for i, ch in enumerate(buffer):
                        if ch in _SENTENCE_ENDS:
                            idx = i
                            break
                        if ch in {",", ";"} and i >= _CLAUSE_MIN_LEN:
                            idx = i
                            break
                    if idx == -1:
                        break
                    sentence = buffer[:idx + 1].strip()
                    buffer   = buffer[idx + 1:].lstrip()
                    if sentence:
                        yield sentence

            remainder = buffer.strip()
            if remainder:
                yield remainder

        if speaker:
            # ── Streaming pipeline with initial buffer ────────────────────
            sentence_q = _queue.Queue()

            def _generate():
                try:
                    stream = _make_stream()
                    for sentence in _iter_sentences(stream):
                        sentence_q.put(sentence)
                except Exception as e:
                    logger.error(f"Groq Vision streaming failed in ReadingModule: {e}")
                finally:
                    sentence_q.put(None)  # sentinel

            gen_thread = threading.Thread(target=_generate, daemon=True,
                                          name="reading-vlm-gen")
            gen_thread.start()

            # Phase 1: collect for _INITIAL_BUFFER_S before speaking
            initial_sentences = []
            deadline = _time.monotonic() + _INITIAL_BUFFER_S
            done_early = False

            while True:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = sentence_q.get(timeout=remaining)
                    if item is None:
                        done_early = True
                        break
                    initial_sentences.append(item)
                except _queue.Empty:
                    break

            # Speak the buffered content as a single chunk (clean, no artefacts)
            if initial_sentences:
                first_chunk = " ".join(initial_sentences)
                full_text = first_chunk
                speaker.speak_stream(first_chunk)
                logger.debug(
                    f"Reading initial chunk spoken ({len(initial_sentences)} "
                    f"sentences, {len(first_chunk)} chars)"
                )

            # Phase 2: stream and speak remaining sentences
            if not done_early:
                sentence_count = len(initial_sentences)
                while True:
                    try:
                        item = sentence_q.get(timeout=30)
                        if item is None:
                            break
                        full_text += (" " if full_text else "") + item
                        sentence_count += 1
                        speaker.speak_stream(item)
                        logger.debug(
                            f"Reading sentence {sentence_count} spoken: "
                            f"'{item[:60]}'"
                        )
                    except _queue.Empty:
                        logger.warning("Reading VLM stream timed out waiting for next sentence")
                        break

            gen_thread.join(timeout=60)

            if not full_text.strip():
                fallback = "I could not read any text from the image. Please try again."
                speaker.speak_stream(fallback)
                return fallback

        else:
            # No speaker — collect the full text
            sentence_count = 0
            try:
                stream = _make_stream()
                for sentence in _iter_sentences(stream):
                    full_text += (" " if full_text else "") + sentence
                    sentence_count += 1
            except Exception as e:
                logger.error(f"Groq Vision streaming failed: {e}")
                return "I could not read the text. Please try again."

            if not full_text.strip():
                return "I could not read any text from the image. Please try again."

        logger.info(
            f"Reading complete — {len(full_text)} chars"
        )
        return full_text.strip()
