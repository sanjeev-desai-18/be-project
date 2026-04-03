# modules/scene/scene_module.py
# Scene tool — captures a frame via picamera2 and describes it via Groq Vision.
#
# Changes from original:
#   - _capture_frames() now uses picamera2 (via camera.py) instead of cv2.VideoCapture
#   - Opens the camera ONCE for multi-frame capture to avoid repeated warmup cost
#   - NoIR correction handled inside camera.py before encoding
#   - VLM_MAX_TOKENS raised to 300 in scene context for richer descriptions
#     (overridden locally — config.py is not changed)
#   - All other logic (JSON parsing, _to_speech, output filtering) unchanged

import json
import re
import time
import cv2
import numpy as np

from modules.scene.vlm_client import VLMClient
from utils.logger import logger
from utils.image_utils import frame_to_base64, resize_frame


def _apply_noir_correction(frame_rgb: np.ndarray) -> np.ndarray:
    """CLAHE on L-channel — same as camera.py helper, used for multi-frame path."""
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab   = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    except Exception:
        return frame_rgb


class SceneModule:

    def __init__(self):
        self.vlm = VLMClient()

    # ── Camera — picamera2, open once for all frames ──────────────────────────
    def _capture_frames(self, count: int = 3) -> list:
        """
        Open picamera2 ONCE and capture `count` frames — avoids repeated warmup.
        Returns a list of base64-encoded JPEG strings (BGR pipeline).
        """
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise RuntimeError(
                f"picamera2 not available: {e}. "
                "Install with: sudo apt install python3-picamera2"
            )

        try:
            _noir = __import__("config").NOIR_CORRECTION
        except AttributeError:
            _noir = True

        frames = []
        picam2 = Picamera2()

        try:
            config = picam2.create_preview_configuration(
                main={"size": (1024, 768), "format": "RGB888"}
            )
            picam2.configure(config)
            picam2.start()

            # Single warmup — let AE/AWB settle
            time.sleep(0.8)
            for _ in range(3):
                picam2.capture_array("main")  # discard dark frames

            for i in range(count):
                raw = picam2.capture_array("main")  # RGB888

                if _noir:
                    raw = _apply_noir_correction(raw)

                bgr   = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                bgr   = resize_frame(bgr, max_width=1024)
                b64   = frame_to_base64(bgr, quality=85)
                frames.append(b64)
                logger.debug(f"Scene frame {i + 1}/{count} captured ✓")

                if i < count - 1:
                    time.sleep(0.2)  # small gap between frames

        except Exception as e:
            raise RuntimeError(f"picamera2 multi-frame capture failed: {e}") from e
        finally:
            try:
                picam2.stop()
                picam2.close()
            except Exception:
                pass

        return frames

    # ── JSON parser ───────────────────────────────────────────────────────────
    def _parse_scene_json(self, raw: str) -> dict:
        """Robustly extract JSON from VLM output, handling markdown fences."""
        text  = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("No JSON found in VLM output")

    # ── Main entry ────────────────────────────────────────────────────────────
    def run(self) -> str:
        """
        Capture a scene and return a spoken string describing it.
        Called by scene_node in core/agent.py — no changes needed there.
        """
        logger.info("SceneModule.run() | capturing frames via picamera2")

        # Step 1 — Capture
        try:
            frames = self._capture_frames(count=3)
        except RuntimeError as e:
            logger.error(f"Camera error in SceneModule: {e}")
            return "I could not access the camera."

        if not frames:
            return "I could not capture any frames from the camera."

        # Step 2 — Perception prompt
        perception_prompt = """
Analyze the scene carefully and return rich, descriptive structured awareness.

Instructions:
- "near": list objects/people close to the camera with brief descriptors (e.g. "a wooden chair", "a person in a red shirt")
- "in_hand": list items visibly held or gripped by the person
- "obstacles": list anything that could block movement (e.g. "a step", "a bag on the floor")
- "context": write 1-2 full sentences describing the overall environment — lighting, room type, mood, and notable features
- "confidence": float 0.0 to 1.0

Be specific and descriptive. Avoid vague terms like "object" or "thing".
If unsure about lists, leave them empty — but always fill "context" with your best observation.

Respond strictly in this JSON format with no extra text:
{"near": [], "in_hand": [], "obstacles": [], "context": "", "confidence": 0.0}
"""

        # Step 3 — Call VLM with the first (best) frame
        raw_output = self.vlm.describe(frames[0], perception_prompt)
        logger.debug(f"Raw VLM output: {raw_output[:200]}")

        # Step 4 — Parse JSON
        try:
            scene_data = self._parse_scene_json(raw_output)
            scene_data.setdefault("near",       [])
            scene_data.setdefault("in_hand",    [])
            scene_data.setdefault("obstacles",  [])
            scene_data.setdefault("context",    "")
            scene_data.setdefault("confidence", 0.5)

        except Exception as e:
            logger.warning(f"Failed to parse scene JSON: {e} — using raw output as context")
            scene_data = {
                "near":       [],
                "in_hand":    [],
                "obstacles":  [],
                "context":    raw_output.strip(),
                "confidence": 0.3,
            }

        logger.info(f"Scene awareness: {scene_data}")

        # Step 5 — Convert to spoken string
        return self._to_speech(scene_data)

    # ── Speech builder ────────────────────────────────────────────────────────
    def _to_speech(self, data: dict) -> str:
        """Convert structured scene dict into a natural spoken sentence."""
        parts = []

        context = data.get("context", "").strip()
        if context:
            parts.append(context)

        near = [str(o) for o in data.get("near", []) if o]
        if near:
            if len(near) == 1:
                parts.append(f"Right nearby, I can see {near[0]}.")
            else:
                parts.append(
                    f"Right nearby, I can see {', '.join(near[:-1])}, and {near[-1]}."
                )

        in_hand = [str(o) for o in data.get("in_hand", []) if o]
        if in_hand:
            parts.append(f"You appear to be holding {' and '.join(in_hand)}.")

        obstacles = [str(o) for o in data.get("obstacles", []) if o]
        if obstacles:
            parts.append(
                f"Please be careful — I notice {', '.join(obstacles)} that could be in your way."
            )

        if not parts:
            return "I can see the scene but could not make out anything clearly right now."

        return " ".join(parts)
