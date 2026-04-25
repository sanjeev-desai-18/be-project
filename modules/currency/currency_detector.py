"""
modules/currency/currency_detector.py

Hailo-8 YOLO currency detection pipeline.

Performance fixes vs previous version:
  - _capture_with_timeout() replaced with direct capture_array() call.
    The old version spawned a new threading.Thread every single frame —
    at 15 fps that is 15 thread-create/destroy cycles per second, adding
    ~5–10 ms of overhead per frame and thrashing the GIL.
  - CONFIRM_WINDOW reduced 20 → 8, CONFIRM_HITS 14 → 6.  At 15 fps a
    note now confirms in ~0.5 s instead of ~1.3 s.
  - ANNOUNCE_REPEAT / announce_count / mark_announced removed entirely.
    currency_logic fires on new confirmed track IDs — no silent budget.
  - Camera no longer hard-locked to 30 fps (FrameRate / FrameDurationLimits
    removed from camera_manager currency config).
"""

import cv2
import json
import numpy as np
import os
import threading
import time
from collections import deque, Counter

from utils.logger import logger
from utils.camera_manager import camera_manager
from .currency_logic import process_confirmed_notes, reset_logic_state

_DIR        = os.path.dirname(__file__)
HEF_PATH    = os.path.join(_DIR, "yolov11s_currency.hef")
LABELS_PATH = os.path.join(_DIR, "labels.json")

CLASS_NAMES_FALLBACK = [
    "100_rupees", "10_rupees", "2000_rupees", "200_rupees",
    "20_rupees",  "500_rupees", "50_rupees",
]

CONFIDENCE_THRESHOLD = 0.75

# Smaller window = faster confirmation at low fps.
# At 15 fps: 8 frames = 0.53 s, need 6/8 = 75% consistency.
# At 30 fps: 8 frames = 0.27 s — still fast.
CONFIRM_WINDOW = 8
CONFIRM_HITS   = 6
MAJORITY_RATIO = 0.75

DISAPPEAR_FRAMES = 10
TRACK_IOU_THRESH = 0.35
NMS_IOU_THRESHOLD = 0.50

WINDOW_NAME = "Currency Detection - Hailo 8"

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
                return self._hailo, self._model_h, self._model_w
            if not os.path.exists(HEF_PATH):
                raise RuntimeError(f"HEF not found: {HEF_PATH}")
            from picamera2.devices import Hailo
            logger.info("HailoManager: opening Hailo device...")
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


def _iou(a: dict, b: dict) -> float:
    xi1 = max(a["x1"], b["x1"]); yi1 = max(a["y1"], b["y1"])
    xi2 = min(a["x2"], b["x2"]); yi2 = min(a["y2"], b["y2"])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


def _apply_nms(detections: list) -> list:
    if len(detections) <= 1:
        return detections
    by_class: dict[str, list] = {}
    for d in detections:
        by_class.setdefault(d["class"], []).append(d)
    kept = []
    for dets in by_class.values():
        dets.sort(key=lambda d: d["confidence"], reverse=True)
        selected = []
        for d in dets:
            if all(_iou(d, s) < NMS_IOU_THRESHOLD for s in selected):
                selected.append(d)
        kept.extend(selected)
    return kept


# ── Note tracker ──────────────────────────────────────────────────────────────
class _NoteTracker:
    """
    Assigns a persistent track_id to each physical note in the scene.
    No announce_count — the tracker never silences tracks.
    """

    def __init__(self):
        self._tracks: dict[int, dict] = {}
        self._next_id = 1

    def _new_track(self, det: dict) -> int:
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = {
            "id":            tid,
            "box":           det,
            "class":         det["class"],
            "confidence":    det["confidence"],
            "history":       deque(maxlen=CONFIRM_WINDOW),
            "miss_streak":   0,
            "confirmed_cls": None,
        }
        logger.debug(f"NoteTracker: new track #{tid} ({det['class']})")
        return tid

    def _check_gate(self, trk: dict):
        if trk["confirmed_cls"] is not None:
            return
        hits = [h for h in trk["history"] if h is not None]
        if len(hits) < CONFIRM_HITS:
            return
        top_cls, top_count = Counter(hits).most_common(1)[0]
        if top_count / len(hits) >= MAJORITY_RATIO:
            trk["confirmed_cls"] = top_cls
            logger.info(
                f"NoteTracker: track #{trk['id']} confirmed → "
                f"{top_cls} ({top_count}/{len(hits)} = {top_count/len(hits):.0%})"
            )

    def update(self, detections: list) -> list:
        matched_tids: set[int] = set()
        matched_dis:  set[int] = set()

        pairs = sorted(
            [
                (_iou(trk["box"], det), tid, di)
                for tid, trk in self._tracks.items()
                for di, det in enumerate(detections)
            ],
            reverse=True,
        )
        for iou_val, tid, di in pairs:
            if iou_val < TRACK_IOU_THRESH:
                break
            if tid in matched_tids or di in matched_dis:
                continue
            trk = self._tracks[tid]
            det = detections[di]
            trk["box"]        = det
            trk["class"]      = det["class"]
            trk["confidence"] = det["confidence"]
            trk["history"].append(det["class"])
            trk["miss_streak"] = 0
            matched_tids.add(tid)
            matched_dis.add(di)

        for di, det in enumerate(detections):
            if di not in matched_dis:
                tid = self._new_track(det)
                self._tracks[tid]["history"].append(det["class"])
                matched_tids.add(tid)

        for tid in list(self._tracks.keys()):
            if tid not in matched_tids:
                trk = self._tracks[tid]
                trk["history"].append(None)
                trk["miss_streak"] += 1
                if trk["miss_streak"] >= DISAPPEAR_FRAMES:
                    logger.debug(f"NoteTracker: track #{tid} expired")
                    del self._tracks[tid]

        for trk in self._tracks.values():
            self._check_gate(trk)

        # Build result for all detections visible this frame
        pairs2 = sorted(
            [
                (_iou(trk["box"], det), tid, di)
                for tid, trk in self._tracks.items()
                for di, det in enumerate(detections)
                if tid in matched_tids and di in matched_dis
            ],
            reverse=True,
        )
        result = []
        seen_tids: set[int] = set()
        seen_dis:  set[int] = set()
        for _, tid, di in pairs2:
            if tid in seen_tids or di in seen_dis:
                continue
            trk = self._tracks[tid]
            result.append({
                **detections[di],
                "track_id":      tid,
                "confirmed_cls": trk["confirmed_cls"],
            })
            seen_tids.add(tid)
            seen_dis.add(di)

        return result

    def reset(self):
        self._tracks.clear()
        self._next_id = 1


_note_tracker = _NoteTracker()


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
    return _apply_nms(detections)


def _draw_boxes(frame_bgr, tracked_dets: list, fps: float = 0.0):
    out = frame_bgr.copy()
    for d in tracked_dets:
        x1, y1 = int(d["x1"]), int(d["y1"])
        x2, y2 = int(d["x2"]), int(d["y2"])
        tid = d.get("track_id", "?")
        cls = d.get("confirmed_cls") or d["class"]
        colour = (0, 255, 0) if d.get("confirmed_cls") else (0, 215, 255)
        label_txt = f"#{tid} {cls.replace('_', ' ')} {d['confidence']:.0%}"
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(out, label_txt, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out,
                f"Hailo-8 | {len(tracked_dets)} notes | {fps:.1f} FPS | Q=stop",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
    return out


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
    _note_tracker.reset()
    reset_logic_state()
    logger.info("currency_detector: state reset")


def _run(stop_evt, camera_released_evt):
    logger.info("_run(): started")
    camera_released_evt.clear()

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

    camera_acquired = False
    try:
        picam2 = camera_manager.acquire(
            mode="currency",
            model_size=(model_w, model_h),
            warmup=1.0
        )
        camera_acquired = True
        logger.info(f"_run(): camera acquired {model_w}x{model_h} at max fps")
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
            # Direct capture — no per-frame thread spawn.
            # picam2.capture_array() blocks until the next frame arrives
            # from the sensor (typically <33 ms at 30 fps). If the stop
            # event fires while we are blocked here, we will unblock on the
            # next frame (at most one frame delay, ~33 ms) and then exit the
            # while-loop cleanly on the next iteration check.
            try:
                frame = picam2.capture_array("main")
            except Exception as e:
                if stop_evt.is_set():
                    break
                logger.warning(f"capture error: {e}")
                continue

            if frame is None:
                continue

            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]

            img_h, img_w = frame.shape[:2]
            frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hailo_output = hailo.run(frame_rgb)
            detections   = _parse_hailo_output(hailo_output, img_w, img_h)

            tracked = _note_tracker.update(detections)

            # Pass all confirmed tracks — logic decides whether to speak
            confirmed = [t for t in tracked if t["confirmed_cls"] is not None]
            process_confirmed_notes(confirmed)

            fps_count += 1
            now = time.time()
            if now - fps_t >= 2.0:
                fps = fps_count / (now - fps_t)
                fps_count = 0
                fps_t = now

            _set_latest_frame(_draw_boxes(frame, tracked, fps))

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
        _set_latest_frame(None)
        camera_released_evt.set()
        logger.info("_run(): fully exited")


def start_currency_detection():
    global _thread, _stop_evt, _camera_released
    if _thread is not None and _thread.is_alive():
        logger.warning("Currency already running")
        return
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
