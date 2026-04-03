# modules/scene/camera.py — Frame capture using picamera2 (Raspberry Pi 5 + Camera 3 NoIR)
#
# Replaces the original OpenCV-based capture which used cv2.VideoCapture().
# On RPi 5 with the official camera module, picamera2 is the correct driver.
# cv2.VideoCapture() does NOT reliably access the CSI camera on RPi 5.
#
# NoIR correction is applied (CLAHE on L-channel in LAB space) so frames sent
# to the VLM resemble normal RGB images — the model was trained on RGB data.
# Disable by setting NOIR_CORRECTION = False in config.py.

import time
import cv2
import numpy as np

from utils.logger import logger
from utils.image_utils import frame_to_base64, resize_frame


def _apply_noir_correction(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Lightweight CLAHE colour correction for NoIR camera frames.
    Equalises the L-channel in LAB space. < 1 ms on RPi 5.
    Input/output: RGB numpy array.
    """
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab   = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    except Exception as e:
        logger.warning(f"NoIR correction failed — returning raw frame: {e}")
        return frame_rgb


def capture_frame_as_base64() -> str:
    """
    Capture a single frame from the Raspberry Pi camera via picamera2.

    Returns:
        base64 JPEG string ready for the Groq Vision API.

    Raises:
        RuntimeError: if camera cannot be opened or frame capture fails.
    """
    try:
        from picamera2 import Picamera2
    except ImportError as e:
        raise RuntimeError(
            f"picamera2 not available: {e}. "
            "Install with: sudo apt install python3-picamera2"
        )

    try:
        _noir_correction = __import__("config").NOIR_CORRECTION
    except AttributeError:
        _noir_correction = True  # default on

    logger.debug("Opening camera via picamera2 for scene capture...")

    picam2 = Picamera2()
    try:
        config = picam2.create_still_configuration(
            main={"size": (1024, 768), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()

        # Warm up — let auto-exposure settle
        time.sleep(0.8)

        # Discard a few frames
        for _ in range(3):
            picam2.capture_array("main")

        frame = picam2.capture_array("main")  # RGB888 numpy array

    except Exception as e:
        raise RuntimeError(f"picamera2 capture failed: {e}") from e
    finally:
        try:
            picam2.stop()
            picam2.close()
        except Exception:
            pass

    if frame is None:
        raise RuntimeError("Camera returned an empty frame.")

    if _noir_correction:
        logger.debug("Applying NoIR colour correction to scene frame")
        frame = _apply_noir_correction(frame)

    # Convert RGB → BGR for OpenCV resize/encode pipeline
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_bgr = resize_frame(frame_bgr, max_width=1024)
    b64       = frame_to_base64(frame_bgr, quality=85)

    logger.debug("Scene frame captured and encoded ✓")
    return b64
