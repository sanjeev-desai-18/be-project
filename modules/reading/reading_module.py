# modules/reading/reading_module.py
#
# Reads text from an image captured via picamera2 (RPi 5 + Camera 3 NoIR).
# Sends the sharpest frame to Groq Vision for full OCR-style text reading.
#
# Changes from original:
#   - _capture_frames() uses picamera2 instead of cv2.VideoCapture (CSI camera)
#   - NoIR CLAHE correction applied before encoding (respects NOIR_CORRECTION flag)
#   - Higher resolution kept (1920x1080) for text reading — sharpness matters more here
#   - Higher JPEG quality (quality=92) to preserve fine text details
#   - All other logic (sharpness scoring, prompt, Groq call) unchanged

import base64
import time
import cv2
import numpy as np

from utils.logger import logger
from utils.image_utils import frame_to_base64, resize_frame
from config import GROQ_API_KEY, VLM_MODEL


def _apply_noir_correction(frame_rgb: np.ndarray) -> np.ndarray:
    """CLAHE on L-channel in LAB space — normalises NoIR colour shift."""
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab   = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    except Exception:
        return frame_rgb


class ReadingModule:

    # ── Sharpness scorer — unchanged ──────────────────────────────────────────
    def _sharpness_score(self, b64_image: str) -> float:
        try:
            img_bytes = base64.b64decode(b64_image)
            np_arr    = np.frombuffer(img_bytes, np.uint8)
            img       = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            return cv2.Laplacian(img, cv2.CV_64F).var()
        except Exception as e:
            logger.warning(f"Sharpness calculation failed: {e}")
            return 0.0

    # ── Camera — picamera2, open once for all frames ──────────────────────────
    def _capture_frames(self, count: int = 3) -> list:
        """
        Open picamera2 ONCE and capture `count` frames.
        Uses higher resolution (1920x1080) and quality for text legibility.
        Returns list of base64 JPEG strings.
        """
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise RuntimeError(
                f"picamera2 not available: {e}. "
                "Install with: sudo apt install python3-picamera2"
            )

        try:
            _noir = __import__("config").NOIR_CORRECTION
        except AttributeError:
            _noir = True

        frames = []
        picam2 = Picamera2()

        try:
            # Use still configuration for maximum image quality
            config = picam2.create_still_configuration(
                main={"size": (1920, 1080), "format": "RGB888"}
            )
            picam2.configure(config)
            picam2.start()

            # Warmup — let AE/AWB settle (slightly longer for still config)
            time.sleep(1.0)
            for _ in range(3):
                picam2.capture_array("main")  # discard dark/unstable frames

            for i in range(count):
                raw = picam2.capture_array("main")  # RGB888 numpy array

                if _noir:
                    raw = _apply_noir_correction(raw)

                # Convert RGB → BGR for OpenCV encode pipeline
                bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                bgr = resize_frame(bgr, max_width=1920)       # preserve full res for text
                b64 = frame_to_base64(bgr, quality=92)        # high quality for fine text
                frames.append(b64)
                logger.debug(f"Reading frame {i + 1}/{count} captured ✓")

                if i < count - 1:
                    time.sleep(0.2)

        except Exception as e:
            raise RuntimeError(f"picamera2 capture failed: {e}") from e
        finally:
            try:
                picam2.stop()
                picam2.close()
            except Exception:
                pass

        return frames

    # ── Pick sharpest frame — unchanged ───────────────────────────────────────
    def _pick_sharpest(self, frames: list) -> str:
        scores     = [self._sharpness_score(f) for f in frames]
        best_index = scores.index(max(scores))
        logger.debug(
            f"Sharpness scores: {[round(s, 1) for s in scores]} — using frame {best_index}"
        )
        return frames[best_index]

    # ── Main entry — unchanged logic ──────────────────────────────────────────
    def run(self) -> str:
        logger.info("ReadingModule.run() | capturing 3 frames via picamera2")

        try:
            frames = self._capture_frames(3)
        except RuntimeError as e:
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

            client = Groq(api_key=GROQ_API_KEY)

            response = client.chat.completions.create(
                model=VLM_MODEL,
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{best_frame}"
                                }
                            },
                            {
                                "type": "text",
                                "text": reading_prompt
                            }
                        ]
                    }
                ]
            )

            result = response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Groq Vision failed: {e}")
            return "I could not read the text. Please try again."

        if not result:
            return "I could not read any text from the image. Please try again."

        logger.info(f"Reading result: {result[:100]}...")
        return result.strip()
