# modules/scene/camera.py — Frame capture using picamera2 (Raspberry Pi 5 + Camera 3 NoIR)
#
# Replaces the original OpenCV-based capture which used cv2.VideoCapture().
# On RPi 5 with the official camera module, picamera2 is the correct driver.
# cv2.VideoCapture() does NOT reliably access the CSI camera on RPi 5.
#
# Picamera2's "RGB888" format returns BGR byte order (DRM convention),
# which is OpenCV's native format — no colour conversion needed.

import time
import cv2

from utils.logger import logger
from utils.image_utils import frame_to_base64, resize_frame


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

        frame = picam2.capture_array("main")

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

    # RGB888 gives BGR natively (DRM convention) — OpenCV native format
    # No colour conversion needed for imencode
    frame_bgr = resize_frame(frame, max_width=1024)
    b64       = frame_to_base64(frame_bgr, quality=85)

    logger.debug("Scene frame captured and encoded ✓")
    return b64
