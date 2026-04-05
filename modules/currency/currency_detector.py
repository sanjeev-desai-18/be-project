"""
modules/currency/currency_detector.py
──────────────────────────────────────
Complete reset approach: on stop, the module resets ALL state so the
next start() call is exactly identical to the very first launch.
No events to get stuck, no thread references to check — clean slate.
"""

import cv2
import json
import numpy as np
import os
import threading
import time

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

CONFIDENCE_THRESHOLD = 0.30
WINDOW_NAME          = "Currency Detection - Hailo 8"


# ══════════════════════════════════════════════════════════════════════════════
# HAILO MANAGER — one instance, never closed between sessions
# ══════════════════════════════════════════════════════════════════════════════
class _HailoManager:
    def __init__(self):
        self._hailo   = None
        self._model_h = 640
        self._model_w = 640
        self._lock    = threading.Lock()

    def get(self):
        with self._lock:
            if self._hailo is not None:
                return self._hailo, self._model_h, self._model_w
            if not os.path.exists(HEF_PATH):
                raise RuntimeError(f"HEF not found: {HEF_PATH}")
            from picamera2.devices import Hailo
            logger.info("HailoManager: opening Hailo device...")
            hailo = Hailo(HEF_PATH)
            h, w, _ = hailo.get_input_shape()
            self._hailo, self._model_h, self._model_w = hailo, h, w
            logger.info(f"HailoManager: ready {w}x{h} ✓")
            return self._hailo, self._model_h, self._model_w

    def shutdown(self):
        with self._lock:
            if self._hailo is not None:
                try:
                    self._hailo.close()
                except Exception:
                    pass
                self._hailo = None
                logger.info("HailoManager: closed ✓")


hailo_manager = _HailoManager()


# ══════════════════════════════════════════════════════════════════════════════
# LABELS
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# PARSE / DRAW
# ══════════════════════════════════════════════════════════════════════════════
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
            x1,y1 = max(0.,min(x1,img_w)), max(0.,min(y1,img_h))
            x2,y2 = max(0.,min(x2,img_w)), max(0.,min(y2,img_h))
            if x2 <= x1 or y2 <= y1:
                continue
            label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"cls{class_id}"
            detections.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,
                "x":(x1+x2)/2,"y":(y1+y2)/2,"width":x2-x1,"height":y2-y1,
                "confidence":score,"class_id":class_id,"class":label})
    return detections

def _draw_boxes(frame, detections, fps=0.0):
    out = frame.copy()
    for d in detections:
        x1,y1,x2,y2 = int(d["x1"]),int(d["y1"]),int(d["x2"]),int(d["y2"])
        txt = f"{d['class'].replace('_',' ')}  {d['confidence']:.0%}"
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,0),2)
        (tw,th),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.65,2)
        cv2.rectangle(out,(x1,y1-th-8),(x1+tw+4,y1),(0,255,0),-1)
        cv2.putText(out,txt,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(out,f"Hailo-8 | {len(detections)} det | {fps:.1f} FPS | Q=quit",
                (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,220,255),2,cv2.LINE_AA)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MODULE STATE — reset() wipes everything back to startup condition
# ══════════════════════════════════════════════════════════════════════════════
_thread          = None
_stop_evt        = threading.Event()
_camera_released = threading.Event()
_camera_released.set()   # nothing running at start


def reset():
    """
    Wipe all module state back to exactly-as-if-just-imported.
    Called after every stop so next start() is a clean first launch.
    """
    global _thread, _stop_evt, _camera_released
    _stop_evt        = threading.Event()          # fresh, not set
    _camera_released = threading.Event()
    _camera_released.set()                        # camera is free
    _thread          = None
    logger.info("currency_detector: state fully reset ✓")


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION LOOP
# ══════════════════════════════════════════════════════════════════════════════
def _run(stop_evt, done_evt):
    """
    done_evt is set when _run() exits (camera released).
    Passed in at thread creation so it's always a fresh Event object.
    """
    try:
        hailo, model_h, model_w = hailo_manager.get()
    except Exception as e:
        logger.error(f"Hailo unavailable: {e}")
        done_evt.set()
        return

    try:
        _noir = getattr(__import__("config"), "NOIR_CORRECTION", True)
    except Exception:
        _noir = True
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if _noir else None

    camera_acquired = False
    try:
        picam2 = camera_manager.acquire(mode="currency",
                                        model_size=(model_w, model_h),
                                        warmup=1.0)
        camera_acquired = True
        logger.info(f"Currency: camera acquired {model_w}x{model_h} ✓")
    except Exception as e:
        logger.error(f"Currency: camera acquire failed: {e}", exc_info=True)
        done_evt.set()
        return

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 960, 540)
    except Exception:
        pass

    fps_t = time.time(); fps_count = 0; fps = 0.0
    logger.info("Currency detection running ✓")

    try:
        while not stop_evt.is_set():
            frame = picam2.capture_array("main")
            img_h, img_w = frame.shape[:2]

            if _noir and clahe is not None:
                lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                frame = cv2.cvtColor(cv2.merge((clahe.apply(l),a,b)), cv2.COLOR_LAB2RGB)

            hailo_output = hailo.run(frame)
            detections   = _parse_hailo_output(hailo_output, img_w, img_h)
            if detections:
                logger.info(f"Detected: {[d['class'] for d in detections]}")
            process_predictions({"predictions": detections,
                                 "image": {"width": img_w, "height": img_h}})

            fps_count += 1
            if time.time() - fps_t >= 2.0:
                fps = fps_count/(time.time()-fps_t); fps_count=0; fps_t=time.time()

            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            try:
                cv2.imshow(WINDOW_NAME, _draw_boxes(bgr, detections, fps))
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    stop_evt.set(); break
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Detection loop error: {e}", exc_info=True)
    finally:
        if camera_acquired:
            try:
                camera_manager.release()
                logger.info("Currency: camera released ✓")
            except Exception as e:
                logger.warning(f"Camera release error: {e}")
        try: cv2.destroyWindow(WINDOW_NAME)
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass
        done_evt.set()
        logger.info("Currency _run() exited ✓")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
def start_currency_detection():
    global _thread, _stop_evt, _camera_released

    if _thread is not None and _thread.is_alive():
        logger.warning("Currency already running")
        return

    # Always start from a clean reset — guarantees no stale state
    reset()

    done_evt = threading.Event()

    # Wire _camera_released to done_evt so wait_for_camera_release() works
    _camera_released = done_evt

    _thread = threading.Thread(target=_run, args=(_stop_evt, done_evt), daemon=True)
    _thread.start()
    logger.info("Currency detection thread launched ✓")


def stop_currency_detection():
    """Signal stop. Thread will exit at next loop iteration."""
    global _thread
    if _thread is None or not _thread.is_alive():
        logger.info("stop_currency_detection: not running")
        _camera_released.set()
        return
    _stop_evt.set()
    logger.info("Currency stop signal sent ✓")


def wait_for_camera_release(timeout: float = 5.0) -> bool:
    """Block until _run() fully exits and camera is released."""
    released = _camera_released.wait(timeout=timeout)
    if not released:
        logger.error(f"Camera not released after {timeout}s — thread may be stuck in capture_array()")
    return released


def shutdown():
    """Call only from main.py at process exit."""
    hailo_manager.shutdown()
