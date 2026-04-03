"""
modules/currency/currency_detector.py
──────────────────────────────────────
Currency detection using picamera2.devices.Hailo — the correct integration
for RPi 5 + Hailo-8 Hat+.

Based on the working hailo-rpi5-examples/basic_pipelines/detection.py pattern.

Hailo NMS output format (baked-in NMS):
  - Returns a list of 7 arrays (one per class)
  - Each array shape: (N_detections, 5) where cols = [y1, x1, y2, x2, score]
  - Coordinates are NORMALISED 0.0–1.0 relative to input size

Labels (class index 0–6):
  0: 100_rupees
  1: 10_rupees
  2: 2000_rupees
  3: 200_rupees
  4: 20_rupees
  5: 500_rupees
  6: 50_rupees
"""

import cv2
import json
import numpy as np
import os
import threading
import time

from utils.logger import logger
from .currency_logic import process_predictions

# ── Paths ──────────────────────────────────────────────────────────────────────
_DIR        = os.path.dirname(__file__)
HEF_PATH    = os.path.join(_DIR, "yolov11s_currency.hef")
LABELS_PATH = os.path.join(_DIR, "labels.json")

# ── Labels — MUST match HEF compilation order exactly ─────────────────────────
# From PROJECT_CONTEXT.md — yolov11s class order:
CLASS_NAMES_FALLBACK = [
    "100_rupees",   # 0
    "10_rupees",    # 1
    "2000_rupees",  # 2
    "200_rupees",   # 3
    "20_rupees",    # 4
    "500_rupees",   # 5
    "50_rupees",    # 6
]

CONFIDENCE_THRESHOLD = 0.30
WINDOW_NAME          = "Currency Detection - Hailo 8"

_thread   = None
_stop_evt = threading.Event()


# ══════════════════════════════════════════════════════════════════════════════
# LABELS
# ══════════════════════════════════════════════════════════════════════════════
def _load_labels():
    if not os.path.exists(LABELS_PATH):
        logger.warning("labels.json not found — using hardcoded labels from context doc")
        return CLASS_NAMES_FALLBACK
    try:
        with open(LABELS_PATH) as f:
            data = json.load(f)
        if isinstance(data, dict) and "labels" in data:
            lbs = [str(x) for x in data["labels"]]
            # Strip background if present at index 0
            if lbs and lbs[0].lower() == "background":
                lbs = lbs[1:]
                logger.info("Stripped 'background' from labels (index 0)")
            logger.info(f"Labels ({len(lbs)}): {lbs}")
            return lbs
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception as e:
        logger.warning(f"labels.json error: {e} — using fallback")
    return CLASS_NAMES_FALLBACK


CLASS_NAMES = _load_labels()


# ══════════════════════════════════════════════════════════════════════════════
# PARSE HAILO NMS OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
def _parse_hailo_output(hailo_output, img_w, img_h):
    """
    Parse Hailo baked-in NMS output.

    hailo_output is a list of arrays, one per class:
        hailo_output[class_id] = array of shape (N, 5)
        Each row: [y1, x1, y2, x2, score]  — normalised 0.0 to 1.0

    Returns list of detection dicts with pixel coordinates.
    """
    detections = []

    if hailo_output is None:
        return detections

    # Log format on first call for debugging
    if not hasattr(_parse_hailo_output, "_logged"):
        _parse_hailo_output._logged = True
        logger.info(f"Hailo output type: {type(hailo_output)}")
        if isinstance(hailo_output, (list, np.ndarray)):
            logger.info(f"Hailo output length: {len(hailo_output)}")
            for i, cls_dets in enumerate(hailo_output):
                arr = np.asarray(cls_dets) if not isinstance(cls_dets, np.ndarray) else cls_dets
                logger.info(f"  class {i} ({CLASS_NAMES[i] if i < len(CLASS_NAMES) else '?'}): "
                            f"shape={arr.shape}")

    for class_id, class_dets in enumerate(hailo_output):
        if class_dets is None:
            continue

        arr = np.asarray(class_dets, dtype=np.float32)
        if arr.size == 0 or arr.ndim < 2:
            continue

        for det in arr:
            if len(det) < 5:
                continue

            score = float(det[4])
            if score < CONFIDENCE_THRESHOLD:
                continue

            # Hailo NMS returns [y1, x1, y2, x2] normalised
            y1 = float(det[0]) * img_h
            x1 = float(det[1]) * img_w
            y2 = float(det[2]) * img_h
            x2 = float(det[3]) * img_w

            # Clamp
            x1 = max(0.0, min(x1, img_w))
            y1 = max(0.0, min(y1, img_h))
            x2 = max(0.0, min(x2, img_w))
            y2 = max(0.0, min(y2, img_h))

            if x2 <= x1 or y2 <= y1:
                continue

            label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"cls{class_id}"

            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "x": (x1 + x2) / 2, "y": (y1 + y2) / 2,
                "width": x2 - x1, "height": y2 - y1,
                "confidence": score,
                "class_id": class_id,
                "class": label,
            })

    return detections


# ══════════════════════════════════════════════════════════════════════════════
# DRAW BOXES
# ══════════════════════════════════════════════════════════════════════════════
def _draw_boxes(frame, detections, fps=0.0):
    out = frame.copy()
    for d in detections:
        x1, y1 = int(d["x1"]), int(d["y1"])
        x2, y2 = int(d["x2"]), int(d["y2"])
        txt = f"{d['class'].replace('_', ' ')}  {d['confidence']:.0%}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(out, txt, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out,
                f"Hailo-8  |  {len(detections)} det  |  {fps:.1f} FPS  |  Q=stop",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DETECTION LOOP
# ══════════════════════════════════════════════════════════════════════════════
def _run(stop_evt):
    # ── Imports ───────────────────────────────────────────────────────────────
    try:
        from picamera2 import Picamera2
        from picamera2.devices import Hailo
    except ImportError as e:
        logger.error(f"picamera2 or picamera2.devices.Hailo not available: {e}")
        return

    if not os.path.exists(HEF_PATH):
        logger.error(f"HEF not found: {HEF_PATH}")
        logger.error("Place yolov11s_currency.hef in modules/currency/ or models/")
        return

    logger.info(f"Loading HEF: {HEF_PATH}")
    logger.info(f"Labels: {CLASS_NAMES}")

    # ── Initialise Hailo and Camera ───────────────────────────────────────────
    try:
        hailo  = Hailo(HEF_PATH)
        model_h, model_w, _ = hailo.get_input_shape()
        logger.info(f"Hailo model input shape: {model_h}x{model_w}")
    except Exception as e:
        logger.error(f"Hailo init failed: {e}", exc_info=True)
        return

    try:
        picam2 = Picamera2()
        # Main stream at model resolution for Hailo
        # Lores stream at same size for display (can be higher res if needed)
        main_size  = (model_w, model_h)
        lores_size = (model_w, model_h)

        config = picam2.create_preview_configuration(
            main  ={"size": main_size,  "format": "RGB888"},
            lores ={"size": lores_size, "format": "YUV420"},
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(1.0)   # camera warm-up
        logger.info(f"Camera started at {main_size}")
    except Exception as e:
        logger.error(f"Camera init failed: {e}", exc_info=True)
        try:
            hailo.close()
        except Exception:
            pass
        return

    # ── Window ────────────────────────────────────────────────────────────────
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)

    fps_t     = time.time()
    fps_count = 0
    fps       = 0.0

    logger.info("Currency detection running on Hailo-8 via picamera2.devices.Hailo")

    try:
        while not stop_evt.is_set():
            t0 = time.time()

            # ── Capture ───────────────────────────────────────────────────────
            frame = picam2.capture_array("main")   # RGB888 numpy array
            img_h, img_w = frame.shape[:2]

            # ── NoIR colour correction ────────────────────────────────────────
            # The NoIR camera captures near-infrared light that a standard RGB
            # camera blocks with its IR-cut filter.  The result looks magenta /
            # washed-out and confuses a model trained on normal RGB images.
            #
            # Strategy: equalise each channel independently so the overall
            # colour distribution resembles what the model was trained on.
            # This is a lightweight CLAHE pass that runs on the CPU in < 1 ms
            # and is applied BEFORE Hailo inference.
            #
            # If your Picamera2 tuning file already handles this (--tuning-file
            # imx708_noir.json) you can disable the block by setting
            # NOIR_CORRECTION = False in config.py (add it if you want).
            try:
                _noir = getattr(__import__("config"), "NOIR_CORRECTION", True)
            except Exception:
                _noir = True

            if _noir:
                import cv2 as _cv2
                _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                # Work in LAB — equalise only the L (lightness) channel
                _lab   = _cv2.cvtColor(frame, _cv2.COLOR_RGB2LAB)
                _l, _a, _b = _cv2.split(_lab)
                _l = _clahe.apply(_l)
                _lab = _cv2.merge((_l, _a, _b))
                frame = _cv2.cvtColor(_lab, _cv2.COLOR_LAB2RGB)

            # ── Hailo inference ───────────────────────────────────────────────
            # picamera2.devices.Hailo.run() handles:
            #   - resize to model input
            #   - UINT8 input formatting
            #   - async DMA transfer to Hailo-8
            #   - returns raw NMS output
            hailo_output = hailo.run(frame)

            # ── Parse detections ──────────────────────────────────────────────
            detections = _parse_hailo_output(hailo_output, img_w, img_h)

            if detections:
                logger.info(f"Detected: {[d['class'] for d in detections]}")

            # ── TTS ───────────────────────────────────────────────────────────
            process_predictions({"predictions": detections,
                                  "image": {"width": img_w, "height": img_h}})

            # ── FPS ───────────────────────────────────────────────────────────
            fps_count += 1
            if time.time() - fps_t >= 2.0:
                fps       = fps_count / (time.time() - fps_t)
                fps_count = 0
                fps_t     = time.time()

            # ── Display ───────────────────────────────────────────────────────
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(WINDOW_NAME, _draw_boxes(bgr, detections, fps))

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                logger.info("Q pressed — stopping currency detection")
                stop_evt.set()
                break

    except Exception as e:
        logger.error(f"Detection loop error: {type(e).__name__}: {e}", exc_info=True)
    finally:
        cv2.destroyWindow(WINDOW_NAME)
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            hailo.close()
        except Exception:
            pass
        logger.info("Currency detection stopped — resources released")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
def start_currency_detection():
    global _thread, _stop_evt
    if _thread is not None and _thread.is_alive():
        logger.warning("Currency detection already running")
        return
    _stop_evt.clear()
    _thread = threading.Thread(target=_run, args=(_stop_evt,), daemon=True)
    _thread.start()
    logger.info("Currency detection thread launched")


def stop_currency_detection():
    global _thread, _stop_evt
    if _thread is None or not _thread.is_alive():
        logger.warning("Not running")
        return
    _stop_evt.set()
    _thread.join(timeout=8)
    _thread = None
    logger.info("Currency detection thread stopped")
