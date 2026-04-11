# modules/scene/scene_module.py
#
# Changes from previous version:
#
#   RESOLUTION (camera_manager.py was the real fix):
#     camera_manager now uses create_still_configuration((2304, 1296)) for
#     scene mode, selecting the sensor's native 2304x1296 mode. The previous
#     hardcoded (1024, 768) preview config was the reason frames were 1024x768
#     regardless of what resize_frame was asked to do here.
#
#   SPEECH — full collection, single gTTS call:
#     The remaining inter-sentence pause came from two separate speak_stream()
#     calls (first sentence + tail), which produced two MP3 files with an
#     audible gap at the boundary.
#
#     The agent.py scene_node already says "Looking at your surroundings."
#     the instant routing completes (~200ms), so perceived latency is handled.
#     We can afford to collect the full VLM response (2-3 short sentences,
#     arrives in ~2-3s) and send it as one string to speaker.speak() — one
#     MP3, zero inter-sentence gaps, completely natural playback.
#
#   DISPLAY:
#     Raw frame stored to _latest_scene_frame immediately after capture so
#     it appears on screen while the VLM is still generating.

import threading
import time as _time

import numpy as np

from modules.scene.vlm_client import VLMClient
from utils.logger import logger
from utils.image_utils import frame_to_base64, resize_frame
from utils.camera_manager import camera_manager


# ── Shared display frame ───────────────────────────────────────────────────────
_frame_lock:         threading.Lock    = threading.Lock()
_latest_scene_frame: np.ndarray | None = None


def get_latest_scene_frame() -> np.ndarray | None:
    """Return the most recently captured scene frame (BGR), or None."""
    with _frame_lock:
        return _latest_scene_frame


def _store_scene_frame(bgr: np.ndarray) -> None:
    global _latest_scene_frame
    with _frame_lock:
        # Keep a 1280-wide copy for display — avoids holding a 2304x1296
        # array in RAM for the lifetime of the program.
        _latest_scene_frame = resize_frame(bgr.copy(), max_width=1280)


# ── Prompt ─────────────────────────────────────────────────────────────────────
_SCENE_PROMPT = """\
Describe this scene for a blind person in 2 to 3 natural spoken sentences.
Mention what is nearest to the camera, any obstacles or hazards, and the \
general environment.
Be direct and conversational. Do not use lists, bullet points, or headings.
Start immediately with the description — no preamble like "In this image..."
"""

# How long to wait for the VLM to finish collecting all sentences.
# 2-3 sentences normally arrive within 3s; 10s is a safe ceiling.
_COLLECT_TIMEOUT = 10.0


class SceneModule:

    def __init__(self):
        self.vlm = VLMClient()

    def _capture_frame(self) -> tuple[str, np.ndarray] | tuple[None, None]:
        """
        Capture one frame at full native resolution (2304x1296).
        Stores a display copy, returns base64 string for VLM.
        """
        picam2 = camera_manager.acquire(mode="scene", warmup=0.3)
        try:
            raw = picam2.capture_array("main")
            # No resize for VLM — send at full 2304x1296 native resolution.
            b64 = frame_to_base64(raw, quality=85)
            logger.debug(
                f"Scene frame captured at {raw.shape[1]}x{raw.shape[0]} "
                f"({len(b64) // 1024} KB) ✓"
            )
            _store_scene_frame(raw)
            return b64, raw
        except Exception as e:
            logger.error(f"Camera capture error in SceneModule: {e}")
            return None, None
        finally:
            camera_manager.release()

    def run(self, speaker=None) -> str:
        """
        Capture at full resolution, collect the complete VLM response,
        then speak it as a single TTS call — one MP3, no gaps, natural audio.

        The agent already plays "Looking at your surroundings." before calling
        run(), so the user hears feedback within ~200ms. This function then
        collects and plays the full description (~2-3s later) as one seamless
        audio clip.

        Returns:
            Full description text for logging / state.
        """
        logger.info("SceneModule.run() | capturing frame via camera_manager")

        b64_frame, _ = self._capture_frame()

        if b64_frame is None:
            msg = "I could not access the camera."
            if speaker:
                speaker.speak(msg)
            return msg

        # ── Collect full VLM response in background thread ────────────────
        collected: list[str] = []
        error_box: list[Exception] = []

        def _generate():
            try:
                for sentence in self.vlm.describe_stream(b64_frame, _SCENE_PROMPT):
                    collected.append(sentence)
            except Exception as exc:
                error_box.append(exc)
                logger.error(f"VLM streaming error in SceneModule: {exc}")

        gen = threading.Thread(target=_generate, daemon=True, name="scene-vlm-gen")
        gen.start()
        gen.join(timeout=_COLLECT_TIMEOUT)

        if gen.is_alive():
            logger.warning("VLM did not finish within timeout — using partial response")

        full_text = " ".join(s.strip() for s in collected if s.strip())

        if not full_text:
            msg = "I can see the scene but could not make out anything clearly."
            if speaker:
                speaker.speak(msg)
            return msg

        logger.info(f"Scene complete — {len(collected)} sentences, {len(full_text)} chars")

        # Single speak() call — one MP3, zero inter-sentence gaps.
        if speaker:
            speaker.speak(full_text)

        return full_text
