# modules/reading/reading_module.py
#
# Changes from original:
#   1. _capture_frames(count=2) — reduced from 3. Two frames is enough for
#      sharpness selection. Removes one 0.2s inter-frame sleep. Saves ~0.4s.
#   2. warmup reduced from 1.0s → 0.4s in acquire(). Saves ~0.6s.
#   3. Groq Vision called with stream=True — tokens are buffered into sentences
#      and each sentence is spoken immediately via speaker.speak().
#      User hears first sentence in ~2-3s, rest plays while VLM generates.
#   4. run() accepts optional speaker= argument (same pattern as scene_module).

import base64
import time

import cv2
import numpy as np

from utils.logger import logger
from utils.image_utils import frame_to_base64, resize_frame
from utils.camera_manager import camera_manager
from config import GROQ_API_KEY, VLM_MODEL, VLM_MAX_TOKENS


# Sentence boundaries for the streaming buffer
_SENTENCE_ENDS = frozenset({".", "!", "?", "\n"})
_CLAUSE_MIN_LEN = 40   # only split on comma/semicolon after this many chars


def _apply_noir_correction(frame_rgb: np.ndarray) -> np.ndarray:
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab   = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    except Exception:
        return frame_rgb


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
        try:
            _noir = __import__("config").NOIR_CORRECTION
        except AttributeError:
            _noir = True

        frames = []
        # warmup=0.4 — reduced from 1.0. Reading mode uses still config which
        # stabilises quickly. 400ms is enough in practice on Pi 5.
        picam2 = camera_manager.acquire(mode="reading", warmup=0.4)
        try:
            for i in range(count):
                raw = picam2.capture_array("main")
                if _noir:
                    raw = _apply_noir_correction(raw)
                bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                bgr = resize_frame(bgr, max_width=1920)
                frames.append(frame_to_base64(bgr, quality=92))
                logger.debug(f"Reading frame {i+1}/{count} captured ✓")
                if i < count - 1:
                    time.sleep(0.15)   # reduced from 0.2s
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
            speaker: optional Speaker instance. When provided, each sentence
                     is spoken immediately as it streams from the VLM.

        Returns:
            Full reading text (used for logging / state).
            If speaker was provided, audio has already been played — caller
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

        full_text      = ""
        sentence_count = 0
        buffer         = ""

        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)

            stream = client.chat.completions.create(
                model=VLM_MODEL,
                max_tokens=2048,   # reading may need more tokens than scene
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

            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta is None:
                    continue

                buffer += delta

                # Drain complete sentences from the front of the buffer
                while True:
                    idx = -1
                    for i, ch in enumerate(buffer):
                        if ch in _SENTENCE_ENDS:
                            idx = i
                            break
                        # split on comma/semicolon only for long buffers
                        if ch in {",", ";"} and i >= _CLAUSE_MIN_LEN:
                            idx = i
                            break

                    if idx == -1:
                        break

                    sentence = buffer[:idx + 1].strip()
                    buffer   = buffer[idx + 1:].lstrip()

                    if sentence:
                        full_text += (" " if full_text else "") + sentence
                        sentence_count += 1
                        if speaker:
                            speaker.speak_stream(sentence)
                            logger.debug(
                                f"Reading sentence {sentence_count} spoken: "
                                f"'{sentence[:60]}'"
                            )

            # Flush remainder
            remainder = buffer.strip()
            if remainder:
                full_text += (" " if full_text else "") + remainder
                sentence_count += 1
                if speaker:
                    speaker.speak_stream(remainder)

        except Exception as e:
            logger.error(f"Groq Vision streaming failed: {e}")
            fallback = "I could not read the text. Please try again."
            if speaker:
                speaker.speak_stream(fallback)
            return fallback

        if not full_text.strip():
            fallback = "I could not read any text from the image. Please try again."
            if speaker:
                speaker.speak_stream(fallback)
            return fallback

        logger.info(
            f"Reading complete — {sentence_count} sentences, "
            f"{len(full_text)} chars"
        )
        return full_text.strip()
