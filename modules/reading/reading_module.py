# modules/reading/reading_module.py
# Uses shared CameraManager instead of creating its own Picamera2 instance.

import base64
import time
import cv2
import numpy as np

from utils.logger import logger
from utils.image_utils import frame_to_base64, resize_frame
from utils.camera_manager import camera_manager
from config import GROQ_API_KEY, VLM_MODEL


def _apply_noir_correction(frame_rgb: np.ndarray) -> np.ndarray:
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab   = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    except Exception:
        return frame_rgb


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

    def _capture_frames(self, count: int = 3) -> list:
        try:
            _noir = __import__("config").NOIR_CORRECTION
        except AttributeError:
            _noir = True

        frames = []
        picam2 = camera_manager.acquire(mode="reading", warmup=1.0)
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
                    time.sleep(0.2)
        finally:
            camera_manager.release()

        return frames

    def _pick_sharpest(self, frames: list) -> str:
        scores     = [self._sharpness_score(f) for f in frames]
        best_index = scores.index(max(scores))
        logger.debug(f"Sharpness scores: {[round(s,1) for s in scores]} — using frame {best_index}")
        return frames[best_index]

    def run(self) -> str:
        logger.info("ReadingModule.run() | capturing frames via camera_manager")

        try:
            frames = self._capture_frames(3)
        except Exception as e:
            logger.error(f"Camera error: {e}")
            return "I could not access the camera. Please check it is connected."

        if not frames:
            return "I could not capture any frames from the camera."

        best_frame = self._pick_sharpest(frames)
        logger.info("Best frame selected for reading ✓")

        reading_prompt = """
You are a reading assistant for visually impaired users.

Read ALL visible text in this image completely. Do not skip or truncate anything.

Instructions:
- Read every single word exactly as written
- Read from top to bottom, left to right
- For medicine labels: name, dosage, instructions, warnings
- For receipts: every item, price, and total
- For documents: full text top to bottom
- For screens/phones: read all visible text
- Add brief context first e.g. "This is a medicine label" or "This is a receipt"
- If no text visible: say "I could not find any text. Please hold the document closer."
- Do NOT summarize — read the COMPLETE text
"""
        logger.info("Sending frame to Groq Vision...")
        try:
            from groq import Groq
            client   = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model=VLM_MODEL,
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{best_frame}"}},
                        {"type": "text", "text": reading_prompt}
                    ]
                }]
            )
            result = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq Vision failed: {e}")
            return "I could not read the text. Please try again."

        if not result:
            return "I could not read any text from the image. Please try again."

        logger.info(f"Reading result: {result[:100]}...")
        return result.strip()
