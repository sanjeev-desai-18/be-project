"""
modules/currency/currency_detector.py

KEY FIX — cv2 display must run on the main thread:
────────────────────────────────────────────────────
On Linux (Qt/GTK backend), ALL cv2 GUI calls — namedWindow, imshow,
waitKey, destroyAllWindows — must be called from the SAME thread that
first called them, and that thread must be the process main thread.

When called from a background daemon thread (_run), the first session
works by luck. On the second launch the Qt event loop is in a broken
state from the previous session's window that was destroyed from the
wrong thread. cv2.namedWindow() then blocks waiting for the Qt loop
to respond — forever, hanging the second session.

Fix: the detection thread never calls any cv2 GUI function. It puts
rendered BGR frames into _display_queue (maxsize=1, drop-oldest).
main.py calls pump_display() on every pipeline iteration from the
main thread. pump_display() drains the queue and calls imshow/waitKey.
The window is created once and reused across sessions — no destroy/
recreate cycle that could leave Qt in a broken state.
"""

import cv2
import json
import numpy as np
import os
import queue
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


# ── Display queue (detection thread -> main thread) ───────────────────────────
# maxsize=1: if main thread is slow, drop oldest frame and keep latest.
# Sentinel value None signals the main thread to blank the window.
_display_queue  = queue.Queue(maxsize=1)
_window_created = False


def pump_display():
    """
    CALL THIS FROM THE MAIN THREAD on every pipeline loop iteration.
    Drains the display queue and renders any pending frame via cv2.
    Returns True if 'q' was pressed (relay stop signal to caller).
    """
    global _window_created
    q_pressed = False
    try:
        item = _display_queue.get_nowait()
    except queue.Empty:
        # Nothing queued — pump Qt event loop so window stays responsive
        if _window_created:
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                q_pressed = True
        return q_pressed

    if item is None:
        # Sentinel: detection stopped — destroy window cleanly from main thread.
        # Double waitKey(1) gives Qt two event loop ticks to fully process the
        # destroy before namedWindow() can be called again next session.
        if _window_created:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                cv2.waitKey(1)
                logger.info("pump_display: cv2 window destroyed ✓")
            except Exception as e:
                logger.warning(f"pump_display: window destroy error: {e}")
            _window_created = False
        return q_pressed

    # Normal annotated frame
    if not _window_created:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 960, 540)
        _window_created = True
        logger.info("pump_display: cv2 window created on main thread ✓")

    try:
        cv2.imshow(WINDOW_NAME, item)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            q_pressed = True
    except Exception as e:
        logger.warning(f"pump_display: imshow error: {e}")

    return q_pressed


def _enqueue_frame(frame_bgr_or_none):
    """Push frame (or None sentinel) to display queue, drop oldest if full."""
    try:
        _display_queue.put_nowait(frame_bgr_or_none)
    except queue.Full:
        try:
            _display_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            _display_queue.put_nowait(frame_bgr_or_none)
        except queue.Full:
            pass


# ── Interruptible capture ─────────────────────────────────────────────────────
def _capture_with_timeout(picam2, stop_evt, timeout=1.0):
    result=[None]; exc=[None]; done=threading.Event()
    def _do():
        try: result[0] = picam2.capture_array("main")
        except Exception as e: exc[0] = e
        finally: done.set()
    threading.Thread(target=_do, daemon=True).start()
    got = done.wait(timeout=timeout)
    if not got:
        return (None,"stop") if stop_evt.is_set() else (None,"timeout")
    if exc[0] is not None:
        logger.warning(f"capture error: {exc[0]}")
        return (None,"stop") if stop_evt.is_set() else (None,"error")
    return result[0], None


# ── Parse / draw ──────────────────────────────────────────────────────────────
def _parse_hailo_output(hailo_output, img_w, img_h):
    detections = []
    if hailo_output is None:
        return detections
    for class_id, class_dets in enumerate(hailo_output):
        if class_dets is None: continue
        arr = np.asarray(class_dets, dtype=np.float32)
        if arr.size == 0 or arr.ndim < 2: continue
        for det in arr:
            if len(det) < 5: continue
            score = float(det[4])
            if score < CONFIDENCE_THRESHOLD: continue
            y1=float(det[0])*img_h; x1=float(det[1])*img_w
            y2=float(det[2])*img_h; x2=float(det[3])*img_w
            x1,y1=max(0.,min(x1,img_w)),max(0.,min(y1,img_h))
            x2,y2=max(0.,min(x2,img_w)),max(0.,min(y2,img_h))
            if x2<=x1 or y2<=y1: continue
            label = CLASS_NAMES[class_id] if class_id<len(CLASS_NAMES) else f"cls{class_id}"
            detections.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,
                "x":(x1+x2)/2,"y":(y1+y2)/2,"width":x2-x1,"height":y2-y1,
                "confidence":score,"class_id":class_id,"class":label})
    return detections


def _draw_boxes(frame_bgr, detections, fps=0.0):
    out = frame_bgr.copy()
    for d in detections:
        x1,y1=int(d["x1"]),int(d["y1"]); x2,y2=int(d["x2"]),int(d["y2"])
        txt=f"{d['class'].replace('_',' ')}  {d['confidence']:.0%}"
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,0),2)
        (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.65,2)
        cv2.rectangle(out,(x1,y1-th-8),(x1+tw+4,y1),(0,255,0),-1)
        cv2.putText(out,txt,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(out,f"Hailo-8 | {len(detections)} det | {fps:.1f} FPS | Q=stop",
                (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,220,255),2,cv2.LINE_AA)
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
    logger.info("currency_detector: state reset ✓")


# ── Detection loop (background thread — ZERO cv2 GUI calls) ──────────────────
def _run(stop_evt, camera_released_evt):
    logger.info("_run(): started")
    camera_released_evt.clear()

    # Step 1: Hailo
    logger.info("_run(): getting Hailo...")
    try:
        hailo, model_h, model_w = hailo_manager.get()
        logger.info(f"_run(): Hailo OK {model_w}x{model_h}")
    except Exception as e:
        logger.error(f"_run(): Hailo failed: {e}", exc_info=True)
        camera_released_evt.set()
        return

    # Step 2: Check stop before camera acquire
    if stop_evt.is_set():
        logger.warning("_run(): stop_evt already set before camera acquire — aborting")
        camera_released_evt.set()
        return

    # Step 3: Camera
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

    # Step 4: Check stop again after slow acquire
    if stop_evt.is_set():
        logger.warning("_run(): stop_evt set during camera acquire — releasing and aborting")
        try:
            camera_manager.release()
        except Exception:
            pass
        camera_released_evt.set()
        return

    # Step 5: CLAHE setup only — no cv2 GUI calls in this thread ever
    try:
        _noir = getattr(__import__("config"), "NOIR_CORRECTION", True)
    except Exception:
        _noir = True
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if _noir else None
    logger.info(f"_run(): NOIR_CORRECTION={_noir}, entering detection loop ✓")

    fps_t=time.time(); fps_count=0; fps=0.0

    try:
        while not stop_evt.is_set():
            frame, status = _capture_with_timeout(picam2, stop_evt, timeout=1.0)
            if status == "stop":
                logger.info("_run(): stop via capture timeout")
                break
            if status in ("timeout", "error"):
                continue
            if frame is None:
                continue

            img_h, img_w = frame.shape[:2]

            if _noir and clahe is not None:
                lab=cv2.cvtColor(frame,cv2.COLOR_RGB2LAB)
                l,a,b=cv2.split(lab)
                l=clahe.apply(l)
                frame=cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2RGB)

            hailo_output = hailo.run(frame)
            detections   = _parse_hailo_output(hailo_output, img_w, img_h)
            if detections:
                logger.info(f"Detected: {[d['class'] for d in detections]}")
            process_predictions({"predictions":detections,
                                 "image":{"width":img_w,"height":img_h}})

            fps_count+=1
            if time.time()-fps_t>=2.0:
                fps=fps_count/(time.time()-fps_t); fps_count=0; fps_t=time.time()

            # Render and push to queue — main thread calls pump_display()
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _enqueue_frame(_draw_boxes(bgr, detections, fps))

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
        _enqueue_frame(None)   # sentinel → main thread blanks window
        camera_released_evt.set()
        logger.info("_run(): fully exited ✓")


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
    logger.info("Currency detection thread launched ✓")


def stop_currency_detection():
    global _thread
    if _thread is None or not _thread.is_alive():
        logger.info("stop_currency_detection: not running")
        _camera_released.set()
        return
    _stop_evt.set()
    logger.info("Currency stop signal sent ✓")


def wait_for_camera_release(timeout=5.0):
    released = _camera_released.wait(timeout=timeout)
    if not released:
        logger.error(f"Camera not released after {timeout}s")
    return released


def shutdown():
    hailo_manager.shutdown()
