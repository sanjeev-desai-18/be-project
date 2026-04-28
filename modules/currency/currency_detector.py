"""
modules/currency/currency_detector.py

Hailo-8 YOLO currency detection pipeline.

Changes from previous version
------------------------------
1.  Shared VDevice  — replaced `picamera2.devices.Hailo` (which opened its
    own implicit VDevice) with a direct `hailo_platform` call that uses the
    process-wide shared VDevice from utils.hailo_device.  This eliminates
    the ~5 s stall when switching between currency and navigation modes.

2.  shutdown()  — no longer releases the VDevice.  The device lifetime is
    managed by utils.hailo_device.shutdown() registered in main.py via
    atexit.  shutdown() here only deactivates the configured network.

Performance notes (unchanged from previous version)
-----------------------------------------------------
  - Direct capture_array() call instead of per-frame thread spawn.
  - CONFIRM_WINDOW=4, CONFIRM_HITS=3 → confirms in ~0.25 s at 15 fps.
  - ANNOUNCE_REPEAT / announce_count logic removed; currency_logic fires
    on new confirmed track IDs.
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

# Faster confirmation: 4 frames window, 3 hits required → confirms in ~0.25s at 15fps
CONFIRM_WINDOW = 4
CONFIRM_HITS   = 3
MAJORITY_RATIO = 0.75

# Low disappear frames: tracker drops a note after 3 missed frames (~200ms at 15fps)
# currency_logic.py has its own GONE_FRAMES debounce on top of this so we
# do NOT need a large value here — keeping it small means the tracker stays clean.
DISAPPEAR_FRAMES  = 3
TRACK_IOU_THRESH  = 0.35
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


# ── Hailo network manager (shared VDevice) ────────────────────────────────────
class _HailoManager:
    """
    Loads the currency HEF onto the shared VDevice and manages the
    configured network lifetime.

    The VDevice itself is owned by utils.hailo_device — we never call
    vdevice.release() here.
    """

    def __init__(self):
        self._network    = None
        self._model_h    = 640
        self._model_w    = 640
        self._lock       = threading.Lock()

    def get(self):
        """Return (network, model_h, model_w), loading HEF on first call."""
        with self._lock:
            if self._network is not None:
                return self._network, self._model_h, self._model_w

            if not os.path.exists(HEF_PATH):
                raise RuntimeError(f"Currency HEF not found: {HEF_PATH}")

            try:
                from hailo_platform import (        # type: ignore
                    HEF,
                    HailoStreamInterface,
                    ConfigureParams,
                )
                from utils.hailo_device import get_vdevice

                logger.info("HailoManager (currency): loading HEF on shared VDevice…")
                hef        = HEF(HEF_PATH)
                vdevice    = get_vdevice()
                cfg_params = ConfigureParams.create_from_hef(
                    hef, interface=HailoStreamInterface.PCIe
                )
                network = vdevice.configure(hef, cfg_params)[0]
                network.activate()

                # Derive input shape from vstream info
                info = network.get_input_vstream_infos()[0]
                # shape is (H, W, C) or similar; fall back to defaults if unknown
                shape = getattr(info, "shape", None)
                if shape and len(shape) >= 2:
                    self._model_h, self._model_w = int(shape[0]), int(shape[1])

                self._network = network
                logger.info(
                    f"HailoManager (currency): ready "
                    f"{self._model_w}x{self._model_h} ✓"
                )
            except ImportError:
                # hailo_platform not available — fall back to picamera2.devices.Hailo
                # (developer machine or older image without the shared-device stack)
                logger.warning(
                    "hailo_platform not found — falling back to "
                    "picamera2.devices.Hailo for currency"
                )
                from picamera2.devices import Hailo   # type: ignore
                hailo = Hailo(HEF_PATH)
                h, w, _ = hailo.get_input_shape()
                # Wrap legacy object so callers get a consistent interface
                self._network    = _LegacyHailoWrapper(hailo)
                self._model_h, self._model_w = h, w
                logger.info(
                    f"HailoManager (currency): legacy Hailo ready "
                    f"{self._model_w}x{self._model_h}"
                )

            return self._network, self._model_h, self._model_w

    def shutdown(self):
        """Deactivate the configured network.  Does NOT release the VDevice."""
        with self._lock:
            if self._network is None:
                return
            try:
                if hasattr(self._network, "deactivate"):
                    self._network.deactivate()
                elif hasattr(self._network, "close"):
                    self._network.close()
                logger.info("HailoManager (currency): network deactivated")
            except Exception as exc:
                logger.warning(f"HailoManager (currency) shutdown error: {exc}")
            finally:
                self._network = None


class _LegacyHailoWrapper:
    """
    Thin wrapper that makes a `picamera2.devices.Hailo` object look like a
    hailo_platform network for the inference call in `_run()`.
    Only used as a fallback when hailo_platform is not installed.
    """
    def __init__(self, hailo):
        self._hailo = hailo

    def run(self, frame_rgb):
        return self._hailo.run(frame_rgb)

    def deactivate(self):
        try:
            self._hailo.close()
        except Exception:
            pass


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
        network, model_h, model_w = hailo_manager.get()
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

    # Resolve the inference callable once — works for both the shared-VDevice
    # path (no .run()) and the legacy picamera2 wrapper (has .run()).
    #
    # Output format contract for _parse_hailo_output():
    #   A list where index = class_id and value = array of shape (N, 5)
    #   with columns [y1_norm, x1_norm, y2_norm, x2_norm, score].
    #   This is what picamera2.devices.Hailo.run() already returns.
    #   InferVStreams returns a dict keyed by layer name — we unpack it here
    #   so _parse_hailo_output never has to deal with dict keys.

    if hasattr(network, "run"):
        # Legacy _LegacyHailoWrapper — already returns the per-class list.
        infer_fn = network.run
    else:
        # hailo_platform configured network — use InferVStreams, then convert
        # the raw output dict into the per-class list format.
        from hailo_platform import (        # type: ignore
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
            FormatType,
        )

        num_classes = len(CLASS_NAMES)

        def _unpack_nms_output(raw_dict: dict) -> list:
            """
            Convert InferVStreams output dict -> per-class detection list.

            Hailo YOLOv8/YOLO11 NMS postprocess output is RAGGED: each class
            slot holds a variable number of detections so the whole structure
            cannot be cast to a uniform numpy float32 array. That is exactly
            the inhomogeneous-shape error that was crashing inference.

            Observed layout from the HEF:
              {
                'yolov11s/yolov8_nms_postprocess': [   # len=1  (batch)
                  [                                    # len=num_classes
                    ndarray(N0, 5),  # class 0  [y1, x1, y2, x2, score]
                    ndarray(N1, 5),  # class 1  (N may differ per class)
                    ...
                  ]
                ]
              }

            Fix: never call np.array() on the whole ragged structure.
            Iterate class-by-class; each slot IS homogeneous (Ni x 5).
            """
            _empty = np.empty((0, 5), np.float32)

            raw = list(raw_dict.values())[0]   # unwrap dict

            # Strip the batch dimension (works for list or object ndarray)
            if isinstance(raw, np.ndarray):
                per_class = raw[0] if raw.ndim >= 1 else raw
            elif isinstance(raw, (list, tuple)) and len(raw) == 1:
                per_class = raw[0]
            else:
                per_class = raw

            result = []
            for cls_item in per_class:
                if cls_item is None:
                    result.append(_empty)
                    continue
                try:
                    # Safe: each per-class slot is homogeneous (Ni x 5)
                    cls_arr = np.asarray(cls_item, dtype=np.float32)
                except (ValueError, TypeError):
                    result.append(_empty)
                    continue

                if cls_arr.ndim == 0 or cls_arr.size == 0:
                    result.append(_empty)
                elif cls_arr.ndim == 1:
                    # Single detection returned as a flat 1-D row
                    if cls_arr.shape[0] >= 5 and float(cls_arr[4]) > 0:
                        result.append(cls_arr[:5].reshape(1, 5))
                    else:
                        result.append(_empty)
                else:
                    # Normal (Ni, 5): keep rows with positive score
                    if cls_arr.shape[1] >= 5:
                        valid = cls_arr[cls_arr[:, 4] > 0]
                        result.append(valid if len(valid) else _empty)
                    else:
                        result.append(_empty)

            if not result:
                logger.warning("_unpack_nms_output: empty per-class result")
            return result

        def infer_fn(frame_rgb):
            inp_name   = network.get_input_vstream_infos()[0].name
            input_data = {inp_name: frame_rgb[np.newaxis, ...]}
            in_p  = InputVStreamParams.make(network, format_type=FormatType.UINT8)
            out_p = OutputVStreamParams.make(network, format_type=FormatType.FLOAT32)
            with InferVStreams(network, in_p, out_p) as pipe:
                raw = pipe.infer(input_data)
            return _unpack_nms_output(raw)

    try:
        while not stop_evt.is_set():
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

            try:
                hailo_output = infer_fn(frame_rgb)
            except Exception as e:
                logger.warning(f"inference error: {e}")
                continue

            detections = _parse_hailo_output(hailo_output, img_w, img_h)
            tracked    = _note_tracker.update(detections)

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
    """Deactivate the currency network.  VDevice lifetime is in hailo_device."""
    hailo_manager.shutdown()
