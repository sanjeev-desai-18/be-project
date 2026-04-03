# utils/image_utils.py — Image encoding helpers for VLM API calls

import base64
import cv2
import numpy as np
from utils.logger import logger


def frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """
    Convert an OpenCV frame (numpy array) to base64 string.
    GPT-4o Vision API expects this format.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buffer = cv2.imencode('.jpg', frame, encode_params)
    if not success:
        raise RuntimeError("Failed to encode image frame to JPEG")
    b64 = base64.b64encode(buffer).decode('utf-8')
    logger.debug(f"Image encoded — size: {len(b64) // 1024} KB")
    return b64


def resize_frame(frame: np.ndarray, max_width: int = 1024) -> np.ndarray:
    """
    Resize frame if too large — reduces API cost and speeds up upload.
    Maintains original aspect ratio.
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    ratio = max_width / w
    new_w = max_width
    new_h = int(h * ratio)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.debug(f"Frame resized: {w}x{h} → {new_w}x{new_h}")
    return resized