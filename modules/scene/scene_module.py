# modules/scene/scene_module.py
# Uses shared CameraManager instead of creating its own Picamera2 instance.

import json
import re
import cv2
import numpy as np

from modules.scene.vlm_client import VLMClient
from utils.logger import logger
from utils.image_utils import frame_to_base64, resize_frame
from utils.camera_manager import camera_manager


def _apply_noir_correction(frame_rgb: np.ndarray) -> np.ndarray:
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

    def _capture_frames(self, count: int = 3) -> list:
        try:
            _noir = __import__("config").NOIR_CORRECTION
        except AttributeError:
            _noir = True

        frames = []
        picam2 = camera_manager.acquire(mode="scene", warmup=0.8)
        try:
            for i in range(count):
                raw = picam2.capture_array("main")
                if _noir:
                    raw = _apply_noir_correction(raw)
                bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                bgr = resize_frame(bgr, max_width=1024)
                frames.append(frame_to_base64(bgr, quality=85))
                logger.debug(f"Scene frame {i+1}/{count} captured ✓")
        finally:
            camera_manager.release()

        return frames

    def _parse_scene_json(self, raw: str) -> dict:
        text  = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("No JSON found in VLM output")

    def run(self) -> str:
        logger.info("SceneModule.run() | capturing frames via camera_manager")

        try:
            frames = self._capture_frames(count=3)
        except Exception as e:
            logger.error(f"Camera error in SceneModule: {e}")
            return "I could not access the camera."

        if not frames:
            return "I could not capture any frames from the camera."

        perception_prompt = """
Analyze the scene carefully and return rich, descriptive structured awareness.

Instructions:
- "near": list objects/people close to the camera with brief descriptors
- "in_hand": list items visibly held or gripped by the person
- "obstacles": list anything that could block movement
- "context": write 1-2 full sentences describing the overall environment
- "confidence": float 0.0 to 1.0

Respond strictly in this JSON format with no extra text:
{"near": [], "in_hand": [], "obstacles": [], "context": "", "confidence": 0.0}
"""
        raw_output = self.vlm.describe(frames[0], perception_prompt)
        logger.debug(f"Raw VLM output: {raw_output[:200]}")

        try:
            scene_data = self._parse_scene_json(raw_output)
            scene_data.setdefault("near",       [])
            scene_data.setdefault("in_hand",    [])
            scene_data.setdefault("obstacles",  [])
            scene_data.setdefault("context",    "")
            scene_data.setdefault("confidence", 0.5)
        except Exception as e:
            logger.warning(f"Failed to parse scene JSON: {e}")
            scene_data = {
                "near": [], "in_hand": [], "obstacles": [],
                "context": raw_output.strip(), "confidence": 0.3,
            }

        logger.info(f"Scene awareness: {scene_data}")
        return self._to_speech(scene_data)

    def _to_speech(self, data: dict) -> str:
        parts = []
        context = data.get("context", "").strip()
        if context:
            parts.append(context)
        near = [str(o) for o in data.get("near", []) if o]
        if near:
            if len(near) == 1:
                parts.append(f"Right nearby, I can see {near[0]}.")
            else:
                parts.append(f"Right nearby, I can see {', '.join(near[:-1])}, and {near[-1]}.")
        in_hand = [str(o) for o in data.get("in_hand", []) if o]
        if in_hand:
            parts.append(f"You appear to be holding {' and '.join(in_hand)}.")
        obstacles = [str(o) for o in data.get("obstacles", []) if o]
        if obstacles:
            parts.append(f"Please be careful — I notice {', '.join(obstacles)} that could be in your way.")
        if not parts:
            return "I can see the scene but could not make out anything clearly right now."
        return " ".join(parts)
