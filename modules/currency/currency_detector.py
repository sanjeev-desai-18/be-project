"""
modules/currency/currency_detector.py
"""

import cv2
import json
import numpy as np
import os
import threading
import time
from collections import deque

from utils.logger import logger
from utils.camera_manager import camera_manager
from .currency_logic import process_predictions

_DIR        = os.path.dirname(__file__)
HEF_PATH    = os.path.join(_DIR, "yolov11s_currency.hef")
LABELS_PATH = os.path.join(_DIR, "labels.json")

CLASS_NAMES_FALLBACK = [
    "100_rupees", "10_rupees", "2000_rupees", "200_rupees",
    "20_rupees",  "500_rupees", "50_rupees",
]

# ── Thresholds ────────────────────────────────────────────────────────────────
# Raw inference threshold — kept low so _draw_boxes shows candidate boxes
# as visual feedback even while the temporal gate hasn't fired yet.
CONFIDENCE_THRESHOLD = 0.75

# Temporal consistency gate — a class must appear in at least CONFIRM_HITS
# of the last CONFIRM_WINDOW consecutive frames before TTS fires.
# At ~20–30 fps this means ~0.15–0.25 s of stable detection before speaking.
CONFIRM_WINDOW = 6   # rolling frame window
CONFIRM_HITS   = 4   # minimum hits inside that window to announce

WINDOW_NAME = "Currency Detection - Hailo 8"


# ── Shared display frame ──────────────────────────────────────────────────────
_frame_lock   = threading.Lock()
_latest_frame = [None]


def get_latest_frame():
    with _frame_lock:
        return _latest_frame[0]


def _set_latest_frame(frame):
    with _frame_lock:
        _latest_frame[0] = frame


# ── Hailo singleton ───────────────────────────────────────────────────────────
class _HailoManager:
    def __init__(self):
        self._hailo   = None
        self._model_h = 640
        self._model_w = 640
        self._lock    = threading.Lock()

    def get(self):
        with self._lock:
            if self._hailo is not None:
                logger.info("HailoManager: reusing existing Hailo instance")
                return self._hailo, self._model_h, self._model_w
            if not os.path.exists(HEF_PATH):
                raise RuntimeError(f"HEF not found: {HEF_PATH}")
            from picamera2.devices import Hailo
            logger.info("HailoManager: opening Hailo device (first use)...")
            hailo = Hailo(HEF_PATH)
            h, w, _ = hailo.get_input_shape()
            self._hailo, self._model_h, self._model_w = hailo, h, w
            logger.info(f"HailoManager: ready {w}x{h}")
            return self._hailo, self._model_h, self._model_w

    def shutdown(self):
        with self._lock:
            if self._hailo is not None:
                try:
                    self._hailo.close()
                except Exception:
                    pass
                self._hailo = None


hailo_manager = _HailoManager()


# ── Labels ────────────────────────────────────────────────────────────────────
def _load_labels():
    if not os.path.exists(LABELS_PATH):
        return CLASS_NAMES_FALLBACK
    try:
        with open(LABELS_PATH) as f:
            data = json.load(f)
        if isinstance(data, dict) and "labels" in data:
            lbs = [str(x) for x in data["labels"]]
            if lbs and lbs[0].lower() == "background":
                lbs = lbs[1:]
            return lbs
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        pass
    return CLASS_NAMES_FALLBACK

CLASS_NAMES = _load_labels()


# ── Temporal consistency tracker ──────────────────────────────────────────────
class _TemporalGate:
    """
    Per-class rolling hit counter.

    Every frame, call update(detected_classes) with the set of class labels
    seen above CONFIDENCE_THRESHOLD.

    confirmed() returns the subset of classes that have appeared in at least
    CONFIRM_HITS of the last CONFIRM_WINDOW frames — i.e. genuinely stable
    detections, not one-frame noise.
    """

    def __init__(self, window: int = CONFIRM_WINDOW, min_hits: int = CONFIRM_HITS):
        self._window   = window
        self._min_hits = min_hits
        # label -> deque of booleans, one per recent frame
        self._history: dict[str, deque] = {}

    def update(self, detected_classes: set) -> set:
        """
        Record this frame's detections and return the confirmed set.
        """
        # Ensure every known class has a history entry (so absences are counted)
        all_classes = set(self._history.keys()) | detected_classes
        for cls in all_classes:
            if cls not in self._history:
                self._history[cls] = deque(maxlen=self._window)
            self._history[cls].append(cls in detected_classes)

        # A class is confirmed when it has enough hits inside the window
        return {
            cls for cls, hist in self._history.items()
            if sum(hist) >= self._min_hits
        }

    def reset(self):
        self._history.clear()


_temporal_gate = _TemporalGate()


# ── Interruptible capture ─────────────────────────────────────────────────────
def _capture_with_timeout(picam2, stop_evt, timeout=1.0):
    result = [None]; exc = [None]; done = threading.Event()
    def _do():
        try: result[0] = picam2.capture_array("main")
        except Exception as e: exc[0] = e
        finally: done.set()
    threading.Thread(target=_do, daemon=True).start()
    got = done.wait(timeout=timeout)
    if not got:
        return (None, "stop") if stop_evt.is_set() else (None, "timeout")
    if exc[0] is not None:
        logger.warning(f"capture error: {exc[0]}")
        return (None, "stop") if stop_evt.is_set() else (None, "error")
    return result[0], None


# ── Parse hailo output ────────────────────────────────────────────────────────
def _parse_hailo_output(hailo_output, img_w, img_h):
    detections = []
    if hailo_output is None:
        return detections
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
            y1 = float(det[0]) * img_h; x1 = float(det[1]) * img_w
            y2 = float(det[2]) * img_h; x2 = float(det[3]) * img_w
            x1, y1 = max(0., min(x1, img_w)), max(0., min(y1, img_h))
            x2, y2 = max(0., min(x2, img_w)), max(0., min(y2, img_h))
            if x2 <= x1 or y2 <= y1:
                continue
            label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"cls{class_id}"
            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "x": (x1 + x2) / 2, "y": (y1 + y2) / 2,
                "width": x2 - x1, "height": y2 - y1,
                "confidence": score, "class_id": class_id, "class": label,
            })
    return detections


# ── Draw boxes ────────────────────────────────────────────────────────────────
def _draw_boxes(frame_bgr, detections, confirmed_classes: set, fps: float = 0.0):
    """
    Draw all candidate detections (above CONFIDENCE_THRESHOLD) in yellow.
    Detections that have also passed the temporal gate are drawn in green.
    This gives immediate visual feedback: yellow = seen but not confirmed yet,
    green = stable enough to announce.
    """
    out = frame_bgr.copy()
    for d in detections:
        x1, y1 = int(d["x1"]), int(d["y1"])
        x2, y2 = int(d["x2"]), int(d["y2"])
        confirmed = d["class"] in confirmed_classes
        colour = (0, 255, 0) if confirmed else (0, 215, 255)  # green / yellow
        txt = f"{d['class'].replace('_', ' ')}  {d['confidence']:.0%}"
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(out, txt, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out,
                f"Hailo-8 | {len(detections)} det | {fps:.1f} FPS | Q=stop",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
    return out


# ── Module state ──────────────────────────────────────────────────────────────
_thread          = None
_stop_evt        = threading.Event()
_camera_released = threading.Event()
_camera_released.set()


def reset():
    global _thread, _stop_evt, _camera_released
    _stop_evt        = threading.Event()
    _camera_released = threading.Event()
    _camera_released.set()
    _thread          = None
    _temporal_gate.reset()
    logger.info("currency_detector: state reset")


# ── Detection loop ────────────────────────────────────────────────────────────
def _run(stop_evt, camera_released_evt):
    logger.info("_run(): started")
    camera_released_evt.clear()

    logger.info("_run(): getting Hailo...")
    try:
        hailo, model_h, model_w = hailo_manager.get()
        logger.info(f"_run(): Hailo OK {model_w}x{model_h}")
    except Exception as e:
        logger.error(f"_run(): Hailo failed: {e}", exc_info=True)
        camera_released_evt.set()
        return

    if stop_evt.is_set():
        camera_released_evt.set()
        return

    logger.info("_run(): acquiring camera...")
    camera_acquired = False
    try:
        picam2 = camera_manager.acquire(
            mode="currency",
            model_size=(model_w, model_h),
            warmup=1.0
        )
        camera_acquired = True
        logger.info(f"_run(): camera acquired {model_w}x{model_h}")
    except Exception as e:
        logger.error(f"_run(): camera acquire failed: {e}", exc_info=True)
        camera_released_evt.set()
        return

    if stop_evt.is_set():
        try: camera_manager.release()
        except Exception: pass
        camera_released_evt.set()
        return

    logger.info("_run(): entering detection loop")

    fps_t = time.time(); fps_count = 0; fps = 0.0

    try:
        while not stop_evt.is_set():
            frame, status = _capture_with_timeout(picam2, stop_evt, timeout=1.0)
            if status == "stop":
                break
            if status in ("timeout", "error"):
                continue
            if frame is None:
                continue

            # RGB888 from picamera2 is DRM/BGR byte order — OpenCV native.
            # Strip alpha channel if present (should not happen at 640×640 RGB888
            # but guard anyway).
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]

            img_h, img_w = frame.shape[:2]

            # Hailo expects RGB; frame is BGR.
            frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hailo_output = hailo.run(frame_rgb)

            # All candidates above CONFIDENCE_THRESHOLD (used for display)
            detections = _parse_hailo_output(hailo_output, img_w, img_h)

            # ── Temporal gate ─────────────────────────────────────────────────
            # Update the rolling window with classes seen this frame and get
            # back only the classes that have been consistently detected.
            detected_classes  = {d["class"] for d in detections}
            confirmed_classes = _temporal_gate.update(detected_classes)

            # Build confirmed-only list for TTS (keeps original detection dicts
            # so currency_logic can read confidence, bbox, etc.)
            confirmed_detections = [d for d in detections
                                    if d["class"] in confirmed_classes]

            if confirmed_detections:
                logger.info(f"Confirmed: {[d['class'] for d in confirmed_detections]}")

            # TTS fires only on confirmed detections — never on raw candidates
            process_predictions({
                "predictions": confirmed_detections,
                "image": {"width": img_w, "height": img_h},
            })

            # FPS counter
            fps_count += 1
            now = time.time()
            if now - fps_t >= 2.0:
                fps = fps_count / (now - fps_t)
                fps_count = 0
                fps_t = now

            # ── Display: push every frame unconditionally ─────────────────────
            # The output window must be live from the very first captured frame,
            # regardless of whether any currency has been detected yet.
            # Candidate boxes are yellow; confirmed (announced) boxes are green.
            _set_latest_frame(_draw_boxes(frame, detections, confirmed_classes, fps))

    except Exception as e:
        logger.error(f"_run(): loop exception: {e}", exc_info=True)
    finally:
        logger.info("_run(): entering finally block")
        if camera_acquired:
            try:
                camera_manager.release()
                logger.info("_run(): camera released")
            except Exception as e:
                logger.warning(f"_run(): camera release error: {e}")
        _set_latest_frame(None)   # signal main thread: detection stopped
        camera_released_evt.set()
        logger.info("_run(): fully exited")


# ── Public API ────────────────────────────────────────────────────────────────
def start_currency_detection():
    global _thread, _stop_evt, _camera_released
    if _thread is not None and _thread.is_alive():
        logger.warning("Currency already running")
        return
    logger.info(f"start_currency_detection: _stop_evt.is_set()={_stop_evt.is_set()}")
    _stop_evt.clear()
    _camera_released.clear()
    _thread = threading.Thread(
        target=_run,
        args=(_stop_evt, _camera_released),
        daemon=True
    )
    _thread.start()
    logger.info("Currency detection thread launched")


def stop_currency_detection():
    global _thread
    if _thread is None or not _thread.is_alive():
        logger.info("stop_currency_detection: not running")
        _camera_released.set()
        return
    _stop_evt.set()
    logger.info("Currency stop signal sent")


def wait_for_camera_release(timeout=5.0):
    released = _camera_released.wait(timeout=timeout)
    if not released:
        logger.error(f"Camera not released after {timeout}s")
    return released


def shutdown():
    hailo_manager.shutdown()
