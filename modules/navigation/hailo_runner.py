"""
modules/navigation/hailo_runner.py

Hailo inference runners for navigation: SCDepthV3 + YOLOv8m.
Uses persistent InferVStreams threads (matching the proven standalone pattern).

Both runners share the process-wide VDevice from utils.hailo_device.
"""

from __future__ import annotations

import os
import threading
import queue
import time
import numpy as np
import cv2
from utils.logger import logger

# ── Try importing Hailo SDK ──────────────────────────────────────────────────
try:
    from hailo_platform import (          # type: ignore
        HEF,
        HailoStreamInterface,
        InferVStreams,
        ConfigureParams,
        InputVStreamParams,
        OutputVStreamParams,
        FormatType,
    )
    _HAILO_AVAILABLE = True
    logger.info("hailo_platform SDK loaded ✓")
except ImportError:
    _HAILO_AVAILABLE = False
    logger.warning(
        "hailo_platform not found — navigation will use CPU fallback. "
        "Install hailort wheel for full performance."
    )


# ════════════════════════════════════════════════════════════════════════════
#  DEPTH RUNNER  (scdepthv3.hef or Midas_v2_small_model.hef)
# ════════════════════════════════════════════════════════════════════════════

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


class HailoDepthRunner:
    """
    Wraps a Hailo depth-estimation HEF for monocular depth.

    Supported models:
      • scdepthv3.hef            — 320×256 input (preferred)
      • Midas_v2_small_model.hef — 256×256 input (fallback)

    Uses a persistent InferVStreams thread so the pipeline stays open.
    """

    INPUT_W = 320
    INPUT_H = 256

    def __init__(self, hef_path: str):
        self.hef_path = hef_path
        self._ready   = False
        self._is_scdepth = "scdepth" in os.path.basename(hef_path).lower()
        # Hailo pipeline thread
        self._infer_q: queue.Queue = queue.Queue(maxsize=1)
        self._result_q: queue.Queue = queue.Queue(maxsize=2)
        self._thread: threading.Thread | None = None
        self._pipeline_ready = threading.Event()
        # CPU fallback
        self._torch_model    = None
        self._torch_transform = None

    # ── public API ──────────────────────────────────────────────────────────

    def load(self) -> bool:
        if _HAILO_AVAILABLE and os.path.exists(self.hef_path):
            try:
                self._start_hailo_thread()
                self._ready = True
                hef_name = os.path.basename(self.hef_path)
                logger.info(f"HailoDepthRunner: {hef_name} loaded ({self.INPUT_W}×{self.INPUT_H}) ✓")
                return True
            except Exception as e:
                logger.error(f"HailoDepthRunner: HEF load failed ({e}).")
                if os.path.exists(self.hef_path):
                    self._ready = True
                    return False
        return self._load_cpu_fallback()

    def estimate(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Return a uint8 depth map (H×W) same size as input frame."""
        if not self._ready:
            return self._gradient_fallback(bgr_frame)

        if self._thread is not None and self._thread.is_alive():
            return self._estimate_hailo(bgr_frame)
        elif self._torch_model is not None:
            return self._estimate_torch(bgr_frame)
        else:
            return self._gradient_fallback(bgr_frame)

    def to_meters(self, val: float) -> float:
        """Map a depth pixel value (0–255, higher = closer) to metres."""
        norm = (255 - val) / 255.0
        return round(0.2 + norm * 12.0, 1)   # 0.2 m … 12.2 m

    def close(self):
        """Stop the inference thread."""
        try:
            if self._thread is not None and self._thread.is_alive():
                self._infer_q.put(None)  # sentinel to stop thread
                self._thread.join(timeout=3)
            self._thread = None
            logger.info("HailoDepthRunner: closed")
        except Exception as e:
            logger.warning(f"HailoDepthRunner.close(): {e}")

    # ── Hailo persistent thread ─────────────────────────────────────────────

    def _start_hailo_thread(self):
        from utils.hailo_device import get_vdevice
        hef    = HEF(self.hef_path)
        target = get_vdevice()
        cfg    = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng     = target.configure(hef, cfg)[0]

        # Detect input dimensions from HEF
        in_info  = hef.get_input_vstream_infos()[0]
        out_info = hef.get_output_vstream_infos()[0]
        in_name  = in_info.name
        out_name = out_info.name
        shape    = in_info.shape
        if len(shape) >= 3:
            if shape[-1] <= 4:   # HxWxC
                self.INPUT_H, self.INPUT_W = shape[0], shape[1]
            else:                # CxHxW
                self.INPUT_H, self.INPUT_W = shape[1], shape[2]
        logger.info(f"HailoDepthRunner: detected input {self.INPUT_W}×{self.INPUT_H}")

        def _worker():
            ivp = InputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
            ovp = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
            with InferVStreams(ng, ivp, ovp) as pipeline:
                self._pipeline_ready.set()
                while True:
                    item = self._infer_q.get()
                    if item is None:
                        break
                    try:
                        raw = pipeline.infer({in_name: item})
                        self._result_q.put(raw[out_name])
                    except Exception as e:
                        logger.warning(f"HailoDepthRunner inference error: {e}")
                        self._result_q.put(None)

        self._thread = threading.Thread(target=_worker, daemon=True, name="DepthInfer")
        self._thread.start()
        self._pipeline_ready.wait(timeout=10)

    def _estimate_hailo(self, bgr_frame: np.ndarray) -> np.ndarray:
        orig_h, orig_w = bgr_frame.shape[:2]

        rgb = cv2.cvtColor(
            cv2.resize(bgr_frame, (self.INPUT_W, self.INPUT_H)),
            cv2.COLOR_BGR2RGB
        ).astype(np.float32)

        # SCDepthV3 expects [0, 255] float32; MiDaS expects [0, 1]
        if not self._is_scdepth:
            rgb = rgb / 255.0

        inp = rgb[np.newaxis, ...]

        # Send to inference thread
        try:
            self._infer_q.put_nowait(inp)
        except queue.Full:
            try: self._infer_q.get_nowait()
            except queue.Empty: pass
            self._infer_q.put_nowait(inp)

        try:
            raw = self._result_q.get(timeout=2.0)
        except queue.Empty:
            return self._gradient_fallback(bgr_frame)

        if raw is None:
            return self._gradient_fallback(bgr_frame)

        raw = np.squeeze(raw)

        # Postprocess depends on model
        if self._is_scdepth:
            # SCDepthV3: sigmoid → inverse depth
            depth = 1.0 / (_sigmoid(raw) * 10.0 + 0.009)
            lo, hi = np.percentile(depth, 2), np.percentile(depth, 98)
            if hi - lo < 1e-6:
                dm = np.zeros((self.INPUT_H, self.INPUT_W), dtype=np.uint8)
            else:
                norm = np.clip((depth - lo) / (hi - lo), 0, 1)
                dm = (norm * 255).astype(np.uint8)
                # Invert so close objects = high value (matches MiDaS convention
                # and INFERNO colormap: bright = close, dark = far)
                dm = 255 - dm
        else:
            # MiDaS: simple normalize
            dm = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return cv2.resize(dm, (orig_w, orig_h))

    # ── CPU fallback (PyTorch MiDaS small) ──────────────────────────────────

    def _load_cpu_fallback(self) -> bool:
        try:
            import torch
            logger.info("HailoDepthRunner: loading MiDaS Small on CPU…")
            self._torch_model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
            )
            self._torch_model.eval()
            transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )
            self._torch_transform = transforms.small_transform
            self._ready = True
            logger.info("HailoDepthRunner: MiDaS CPU fallback ready ✓")
            return True
        except Exception as e:
            logger.error(f"HailoDepthRunner: CPU fallback also failed: {e}")
            self._ready = True
            return False

    def _estimate_torch(self, bgr_frame: np.ndarray) -> np.ndarray:
        import torch
        orig_h, orig_w = bgr_frame.shape[:2]
        try:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            inp = self._torch_transform(rgb)
            with torch.no_grad():
                pred = self._torch_model(inp)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=(orig_h, orig_w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().numpy()
            dm = cv2.normalize(pred, None, 0, 255, cv2.NORM_MINMAX)
            return dm.astype(np.uint8)
        except Exception as e:
            logger.warning(f"torch depth estimation failed: {e}")
            return self._gradient_fallback(bgr_frame)

    @staticmethod
    def _gradient_fallback(bgr_frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        gx   = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        gy   = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        dm   = cv2.normalize(np.sqrt(gx**2 + gy**2), None, 0, 255, cv2.NORM_MINMAX)
        return (255 - dm).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════════
#  YOLO RUNNER  (yolov8m.hef)
# ════════════════════════════════════════════════════════════════════════════

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]

HIGH_PRIORITY = {
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "dog", "cat", "chair", "dining table",
}


class HailoYOLORunner:
    """
    Wraps the YOLOv8m HEF for real-time object detection.
    Uses a persistent InferVStreams thread so the pipeline stays open.

    Returns detections as a list of dicts:
      {name, conf, bbox:(x1,y1,x2,y2), zone_idx:0-4, high_priority:bool}
    """

    INPUT_W = 640
    INPUT_H = 640

    def __init__(self, hef_path: str, conf_threshold: float = 0.35):
        self.hef_path       = hef_path
        self.conf_threshold = conf_threshold
        self._ready         = False
        # Hailo pipeline thread
        self._infer_q: queue.Queue = queue.Queue(maxsize=1)
        self._result_q: queue.Queue = queue.Queue(maxsize=2)
        self._thread: threading.Thread | None = None
        self._pipeline_ready = threading.Event()
        self._ultra_model = None

    # ── public API ──────────────────────────────────────────────────────────

    def load(self) -> bool:
        if _HAILO_AVAILABLE and os.path.exists(self.hef_path):
            try:
                self._start_hailo_thread()
                self._ready = True
                logger.info("HailoYOLORunner: YOLOv8m HEF loaded ✓")
                return True
            except Exception as e:
                logger.error(f"HailoYOLORunner: HEF load failed ({e}).")
                if os.path.exists(self.hef_path):
                    self._ready = True
                    return False
        return self._load_cpu_fallback()

    def detect(self, bgr_frame: np.ndarray) -> list:
        """Return list of detection dicts, or [] on failure."""
        if not self._ready:
            return []
        if self._thread is not None and self._thread.is_alive():
            return self._detect_hailo(bgr_frame)
        elif self._ultra_model is not None:
            return self._detect_ultralytics(bgr_frame)
        return []

    def close(self):
        """Stop the inference thread."""
        try:
            if self._thread is not None and self._thread.is_alive():
                self._infer_q.put(None)
                self._thread.join(timeout=3)
            self._thread = None
            logger.info("HailoYOLORunner: closed")
        except Exception as e:
            logger.warning(f"HailoYOLORunner.close(): {e}")

    # ── Hailo persistent thread ─────────────────────────────────────────────

    def _start_hailo_thread(self):
        from utils.hailo_device import get_vdevice
        hef    = HEF(self.hef_path)
        target = get_vdevice()
        cfg    = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng     = target.configure(hef, cfg)[0]

        in_name = hef.get_input_vstream_infos()[0].name

        def _worker():
            ivp = InputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
            ovp = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
            with InferVStreams(ng, ivp, ovp) as pipeline:
                self._pipeline_ready.set()
                while True:
                    item = self._infer_q.get()
                    if item is None:
                        break
                    try:
                        raw = pipeline.infer({in_name: item})
                        self._result_q.put(raw)
                    except Exception as e:
                        logger.warning(f"HailoYOLORunner inference error: {e}")
                        self._result_q.put(None)

        self._thread = threading.Thread(target=_worker, daemon=True, name="YOLOInfer")
        self._thread.start()
        self._pipeline_ready.wait(timeout=10)

    def _detect_hailo(self, bgr_frame: np.ndarray) -> list:
        orig_h, orig_w = bgr_frame.shape[:2]

        rgb = cv2.cvtColor(
            cv2.resize(bgr_frame, (self.INPUT_W, self.INPUT_H)),
            cv2.COLOR_BGR2RGB
        ).astype(np.float32)
        inp = rgb[np.newaxis, ...]

        # Send to inference thread
        try:
            self._infer_q.put_nowait(inp)
        except queue.Full:
            try: self._infer_q.get_nowait()
            except queue.Empty: pass
            self._infer_q.put_nowait(inp)

        try:
            raw_out = self._result_q.get(timeout=2.0)
        except queue.Empty:
            return []

        if raw_out is None:
            return []

        return self._parse_hailo_output(raw_out, orig_w, orig_h)

    def _parse_hailo_output(self, raw_out: dict,
                            orig_w: int, orig_h: int) -> list:
        """
        Parse Hailo YOLOv8 NMS output.

        Output is per-class: tensor shape (1, 80) where each element
        is an ndarray of shape (K, 5): [ymin, xmin, ymax, xmax, score].
        Coordinates normalised to [0, 1].
        """
        dets = []
        try:
            tensor = next(iter(raw_out.values()))

            # Unwrap batch dimension
            if isinstance(tensor, (list, tuple)):
                per_class = tensor[0]
            elif hasattr(tensor, '__getitem__'):
                try:
                    per_class = tensor[0]
                except (IndexError, TypeError):
                    per_class = tensor
            else:
                per_class = tensor

            # Check if this is per-class NMS format (80 items)
            try:
                n_classes = len(per_class)
            except TypeError:
                return dets

            if n_classes == len(COCO_CLASSES):
                # Per-class NMS output
                for cls_idx, class_dets in enumerate(per_class):
                    class_dets = np.array(class_dets, dtype=np.float32)
                    if class_dets.ndim != 2 or class_dets.shape[0] == 0:
                        continue

                    for det_row in class_dets:
                        if len(det_row) < 5:
                            continue
                        score = float(det_row[4])
                        if score < self.conf_threshold:
                            continue

                        ymin, xmin, ymax, xmax = det_row[0], det_row[1], det_row[2], det_row[3]
                        x1 = max(0, int(xmin * orig_w))
                        y1 = max(0, int(ymin * orig_h))
                        x2 = min(orig_w, int(xmax * orig_w))
                        y2 = min(orig_h, int(ymax * orig_h))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        name = COCO_CLASSES[cls_idx] if cls_idx < len(COCO_CLASSES) else str(cls_idx)
                        zone_idx = min(int(((x1 + x2) / 2) / orig_w * 5), 4)

                        dets.append({
                            "name":          name,
                            "conf":          score,
                            "bbox":          (x1, y1, x2, y2),
                            "zone_idx":      zone_idx,
                            "high_priority": name in HIGH_PRIORITY,
                        })
            else:
                # Fallback: combined tensor format
                try:
                    arr = np.array(tensor, dtype=np.float32).squeeze()
                except (ValueError, TypeError):
                    return dets
                if arr.ndim == 2 and arr.shape[-1] >= 85:
                    for row in arr:
                        obj_conf = float(row[4])
                        if obj_conf < self.conf_threshold:
                            continue
                        cls_idx  = int(np.argmax(row[5:85]))
                        cls_conf = float(row[5 + cls_idx]) * obj_conf
                        if cls_conf < self.conf_threshold:
                            continue
                        cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                        x1 = max(0, int((cx - bw / 2) * orig_w))
                        y1 = max(0, int((cy - bh / 2) * orig_h))
                        x2 = min(orig_w, int((cx + bw / 2) * orig_w))
                        y2 = min(orig_h, int((cy + bh / 2) * orig_h))
                        name = COCO_CLASSES[cls_idx] if cls_idx < len(COCO_CLASSES) else str(cls_idx)
                        zone_idx = min(int(((x1 + x2) / 2) / orig_w * 5), 4)
                        dets.append({
                            "name":          name,
                            "conf":          cls_conf,
                            "bbox":          (x1, y1, x2, y2),
                            "zone_idx":      zone_idx,
                            "high_priority": name in HIGH_PRIORITY,
                        })

        except Exception as e:
            logger.warning(f"HailoYOLORunner output parse error: {e}")

        return dets

    # ── CPU fallback (ultralytics) ───────────────────────────────────────────

    def _load_cpu_fallback(self) -> bool:
        try:
            from ultralytics import YOLO   # type: ignore
            logger.info("HailoYOLORunner: loading YOLOv8n via ultralytics (CPU)…")
            self._ultra_model = YOLO("yolov8n.pt")
            self._ready = True
            logger.info("HailoYOLORunner: ultralytics CPU fallback ready ✓")
            return True
        except Exception as e:
            logger.error(f"HailoYOLORunner: CPU fallback failed: {e}")
            self._ready = True
            return False

    def _detect_ultralytics(self, bgr_frame: np.ndarray) -> list:
        try:
            results = self._ultra_model(bgr_frame, conf=self.conf_threshold,
                                        verbose=False)
            dets = []
            if results and results[0].boxes is not None:
                fw, fh = bgr_frame.shape[1], bgr_frame.shape[0]
                for box in results[0].boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    conf     = float(box.conf[0])
                    name     = self._ultra_model.names[int(box.cls[0])]
                    zone_idx = min(int(((x1 + x2) / 2) / fw * 5), 4)
                    dets.append({
                        "name":          name,
                        "conf":          conf,
                        "bbox":          (x1, y1, x2, y2),
                        "zone_idx":      zone_idx,
                        "high_priority": name in HIGH_PRIORITY,
                    })
            return dets
        except Exception as e:
            logger.warning(f"ultralytics detect error: {e}")
            return []
