"""
modules/navigation/hailo_runner.py

Thin wrappers around the Hailo Runtime SDK (hailo_platform) for the two
HEF models used by the navigation module:

    Midas_v2_small_model.hef  — monocular depth estimation
    yolov8m.hef               — object detection (COCO 80-class)

Changes from previous version
------------------------------
1.  Shared VDevice  — both runners now call utils.hailo_device.get_vdevice()
    instead of opening their own VDevice().  A single VDevice with
    ROUND_ROBIN scheduling is shared across navigation AND currency, which
    eliminates the ~5 s stall that occurred when switching between modes.

2.  close() no longer releases the VDevice  — it only deactivates the
    configured network.  The device lifetime is managed by
    utils.hailo_device.shutdown() at process exit.

3.  CPU fallback guard  — if the HEF file exists on disk but fails to load
    (e.g. a transient driver error), we do NOT try to download yolov8n.pt.
    That download is only attempted when there is genuinely no HEF present
    (i.e. a developer workstation without the AI HAT).
"""

from __future__ import annotations

import os
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
#  DEPTH RUNNER  (Midas_v2_small_model.hef)
# ════════════════════════════════════════════════════════════════════════════

class HailoDepthRunner:
    """
    Wraps the MiDaS v2 small HEF for monocular depth estimation.

    Usage
    -----
    runner = HailoDepthRunner("/path/to/Midas_v2_small_model.hef")
    runner.load()                       # opens device once
    depth_u8 = runner.estimate(frame)   # returns H×W uint8 depth map
    runner.close()
    """

    INPUT_W = 256
    INPUT_H = 256

    def __init__(self, hef_path: str):
        self.hef_path = hef_path
        self._ready   = False
        self._target   = None   # shared VDevice reference (not owned)
        self._network  = None
        self._hef      = None
        self._torch_model    = None
        self._torch_transform = None

    # ── public API ──────────────────────────────────────────────────────────

    def load(self) -> bool:
        if _HAILO_AVAILABLE and os.path.exists(self.hef_path):
            try:
                self._load_hailo()
                self._ready = True
                logger.info("HailoDepthRunner: MiDaS HEF loaded ✓")
                return True
            except Exception as e:
                logger.error(
                    f"HailoDepthRunner: HEF load failed ({e}). "
                    f"{'Skipping CPU fallback (HEF present — Pi deployment).' if os.path.exists(self.hef_path) else 'Trying CPU fallback.'}"
                )
                if os.path.exists(self.hef_path):
                    # HEF present but broken — gradient fallback only, no download
                    self._ready = True
                    return False

        # Only attempt CPU fallback on dev machines where no HEF is installed
        return self._load_cpu_fallback()

    def estimate(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Return a uint8 depth map (H×W) same size as input frame."""
        if not self._ready:
            return self._gradient_fallback(bgr_frame)

        if self._network is not None:
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
        """Deactivate the configured network.  Does NOT release the VDevice."""
        try:
            if self._network is not None:
                self._network.deactivate()
                self._network = None
            # self._target is the shared VDevice — do NOT call release() on it
            self._target = None
            logger.info("HailoDepthRunner: closed")
        except Exception as e:
            logger.warning(f"HailoDepthRunner.close(): {e}")

    # ── Hailo internals ─────────────────────────────────────────────────────

    def _load_hailo(self):
        from utils.hailo_device import get_vdevice
        self._hef    = HEF(self.hef_path)
        self._target = get_vdevice()            # ← shared VDevice
        cfg_params   = ConfigureParams.create_from_hef(
            self._hef, interface=HailoStreamInterface.PCIe
        )
        self._network = self._target.configure(self._hef, cfg_params)[0]
        self._network.activate()

    def _estimate_hailo(self, bgr_frame: np.ndarray) -> np.ndarray:
        orig_h, orig_w = bgr_frame.shape[:2]

        rgb = cv2.cvtColor(
            cv2.resize(bgr_frame, (self.INPUT_W, self.INPUT_H)),
            cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0

        input_data = {
            self._network.get_input_vstream_infos()[0].name:
                rgb[np.newaxis, ...]
        }

        try:
            in_params  = InputVStreamParams.make(self._network,
                                                 format_type=FormatType.FLOAT32)
            out_params = OutputVStreamParams.make(self._network,
                                                  format_type=FormatType.FLOAT32)
            with InferVStreams(self._network, in_params, out_params) as pipeline:
                out = pipeline.infer(input_data)

            raw = list(out.values())[0].squeeze()
        except Exception as e:
            logger.warning(f"HailoDepthRunner inference error: {e}")
            return self._gradient_fallback(bgr_frame)

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
            self._ready = True   # gradient fallback will be used
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

    Returns detections as a list of dicts:
      {name, conf, bbox:(x1,y1,x2,y2), zone_idx:0-4, high_priority:bool}
    """

    INPUT_W = 640
    INPUT_H = 640

    def __init__(self, hef_path: str, conf_threshold: float = 0.45):
        self.hef_path       = hef_path
        self.conf_threshold = conf_threshold
        self._ready         = False
        self._target        = None   # shared VDevice reference (not owned)
        self._network       = None
        self._hef           = None
        self._ultra_model   = None

    # ── public API ──────────────────────────────────────────────────────────

    def load(self) -> bool:
        if _HAILO_AVAILABLE and os.path.exists(self.hef_path):
            try:
                self._load_hailo()
                self._ready = True
                logger.info("HailoYOLORunner: YOLOv8m HEF loaded ✓")
                return True
            except Exception as e:
                logger.error(
                    f"HailoYOLORunner: HEF load failed ({e}). "
                    f"{'Skipping CPU fallback (HEF present — Pi deployment).' if os.path.exists(self.hef_path) else 'Trying CPU fallback.'}"
                )
                if os.path.exists(self.hef_path):
                    # HEF is present — we're on a Pi.  Do NOT attempt to
                    # download yolov8n.pt; return empty detections instead.
                    self._ready = True
                    return False

        return self._load_cpu_fallback()

    def detect(self, bgr_frame: np.ndarray) -> list:
        """Return list of detection dicts, or [] on failure."""
        if not self._ready:
            return []
        if self._network is not None:
            return self._detect_hailo(bgr_frame)
        elif self._ultra_model is not None:
            return self._detect_ultralytics(bgr_frame)
        return []

    def close(self):
        """Deactivate the configured network.  Does NOT release the VDevice."""
        try:
            if self._network is not None:
                self._network.deactivate()
                self._network = None
            # self._target is the shared VDevice — do NOT call release() on it
            self._target = None
            logger.info("HailoYOLORunner: closed")
        except Exception as e:
            logger.warning(f"HailoYOLORunner.close(): {e}")

    # ── Hailo internals ─────────────────────────────────────────────────────

    def _load_hailo(self):
        from utils.hailo_device import get_vdevice
        self._hef    = HEF(self.hef_path)
        self._target = get_vdevice()            # ← shared VDevice
        cfg_params   = ConfigureParams.create_from_hef(
            self._hef, interface=HailoStreamInterface.PCIe
        )
        self._network = self._target.configure(self._hef, cfg_params)[0]
        self._network.activate()

    def _detect_hailo(self, bgr_frame: np.ndarray) -> list:
        orig_h, orig_w = bgr_frame.shape[:2]

        resized = cv2.resize(bgr_frame, (self.INPUT_W, self.INPUT_H))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        inp     = rgb[np.newaxis, ...]

        input_name = self._network.get_input_vstream_infos()[0].name
        input_data = {input_name: inp.astype(np.uint8)}

        try:
            in_params  = InputVStreamParams.make(self._network,
                                                 format_type=FormatType.UINT8)
            out_params = OutputVStreamParams.make(self._network,
                                                  format_type=FormatType.FLOAT32)
            with InferVStreams(self._network, in_params, out_params) as pipeline:
                raw_out = pipeline.infer(input_data)
        except Exception as e:
            logger.warning(f"HailoYOLORunner inference error: {e}")
            return []

        return self._parse_hailo_output(raw_out, orig_w, orig_h)

    def _parse_hailo_output(self, raw_out: dict,
                            orig_w: int, orig_h: int) -> list:
        dets = []

        try:
            combined = None
            for v in raw_out.values():
                arr = np.array(v).squeeze()
                if arr.ndim == 2 and arr.shape[-1] >= 85:
                    combined = arr
                    break

            if combined is not None:
                for row in combined:
                    obj_conf = float(row[4])
                    if obj_conf < self.conf_threshold:
                        continue
                    cls_idx  = int(np.argmax(row[5:85]))
                    cls_conf = float(row[5 + cls_idx]) * obj_conf
                    if cls_conf < self.conf_threshold:
                        continue

                    cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                    x1 = int((cx - bw / 2) * orig_w)
                    y1 = int((cy - bh / 2) * orig_h)
                    x2 = int((cx + bw / 2) * orig_w)
                    y2 = int((cy + bh / 2) * orig_h)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(orig_w, x2), min(orig_h, y2)

                    name     = COCO_CLASSES[cls_idx] if cls_idx < len(COCO_CLASSES) else str(cls_idx)
                    zone_idx = min(int(((x1 + x2) / 2) / orig_w * 5), 4)

                    dets.append({
                        "name":          name,
                        "conf":          cls_conf,
                        "bbox":          (x1, y1, x2, y2),
                        "zone_idx":      zone_idx,
                        "high_priority": name in HIGH_PRIORITY,
                    })
            else:
                boxes_tensor  = None
                scores_tensor = None
                for k, v in raw_out.items():
                    arr = np.array(v).squeeze()
                    if "box" in k.lower() and arr.ndim == 2 and arr.shape[-1] == 4:
                        boxes_tensor = arr
                    elif "score" in k.lower() or "class" in k.lower():
                        scores_tensor = arr

                if boxes_tensor is not None and scores_tensor is not None:
                    for i, box in enumerate(boxes_tensor):
                        scores   = scores_tensor[i] if i < len(scores_tensor) else []
                        if len(scores) == 0:
                            continue
                        cls_idx  = int(np.argmax(scores))
                        cls_conf = float(scores[cls_idx])
                        if cls_conf < self.conf_threshold:
                            continue

                        if box[0] < box[2] and box[1] < box[3]:
                            x1 = int(box[0] * orig_w); y1 = int(box[1] * orig_h)
                            x2 = int(box[2] * orig_w); y2 = int(box[3] * orig_h)
                        else:
                            y1 = int(box[0] * orig_h); x1 = int(box[1] * orig_w)
                            y2 = int(box[2] * orig_h); x2 = int(box[3] * orig_w)

                        name     = COCO_CLASSES[cls_idx] if cls_idx < len(COCO_CLASSES) else str(cls_idx)
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
    # Only reached when the HEF file does NOT exist (developer workstation).

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
            self._ready = True   # return empty detections gracefully
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
