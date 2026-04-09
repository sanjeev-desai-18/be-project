# modules/scene/scene_module.py
#
# Changes from original:
#   1. _capture_frames(count=1) — one frame is all that's needed for scene.
#      Original captured 3 but only used frames[0]. Saves ~0.4s.
#   2. warmup reduced from 0.8s → 0.3s in acquire(). Saves ~0.5s.
#   3. Prompt returns plain prose instead of JSON — no parsing delay,
#      and works naturally with streaming (JSON needs complete response).
#   4. run() accepts optional speaker= argument. When provided, each sentence
#      is spoken immediately as it streams from the VLM. The full text is
#      still returned for logging / state.
#   5. run() returns spoken=True signal so tts_node skips re-speaking.

import cv2

from modules.scene.vlm_client import VLMClient
from utils.logger import logger

from utils.image_utils import frame_to_base64, resize_frame
from utils.camera_manager import camera_manager





# Plain-prose prompt — works with streaming, faster to generate than JSON,
# no parsing step needed.
_SCENE_PROMPT = """\
Describe this scene for a blind person in 2 to 3 natural spoken sentences.
Mention what is nearest to the camera, any obstacles or hazards, and the \
general environment.
Be direct and conversational. Do not use lists, bullet points, or headings.
Start immediately with the description — no preamble like "In this image..."
"""


class SceneModule:

    def __init__(self):
        self.vlm = VLMClient()

    def _capture_frames(self, count: int = 1) -> list:

        frames = []
        # warmup=0.3 — reduced from 0.8. Camera is already initialised from
        # previous use; 300ms is enough for the pipeline to stabilise.
        picam2 = camera_manager.acquire(mode="scene", warmup=0.3)
        try:
            for i in range(count):
                raw = picam2.capture_array("main")
                # RGB888 gives BGR natively (DRM convention) — OpenCV native
                bgr = resize_frame(raw, max_width=1024)
                frames.append(frame_to_base64(bgr, quality=85))
                logger.debug(f"Scene frame {i+1}/{count} captured ✓")
        finally:
            camera_manager.release()

        return frames

    def run(self, speaker=None) -> str:
        """
        Capture one frame and describe the scene.

        Args:
            speaker: optional Speaker instance. When provided, each sentence
                     is spoken immediately as it streams from the VLM, so the
                     user hears the first sentence in ~2-3s instead of waiting
                     for the full response.

        Returns:
            Full description text (used for logging / state).
            If speaker was provided, the audio has already been played — the
            caller should set spoken=True in state to skip tts_node.
        """
        logger.info("SceneModule.run() | capturing frame via camera_manager")

        try:
            frames = self._capture_frames(count=1)
        except Exception as e:
            logger.error(f"Camera error in SceneModule: {e}")
            return "I could not access the camera."

        if not frames:
            return "I could not capture any frames from the camera."

        full_text = ""
        sentence_count = 0

        try:
            for sentence in self.vlm.describe_stream(frames[0], _SCENE_PROMPT):
                full_text += (" " if full_text else "") + sentence
                sentence_count += 1
                if speaker:
                    speaker.speak_stream(sentence)
                    logger.debug(
                        f"Scene sentence {sentence_count} spoken: '{sentence[:60]}'"
                    )
        except Exception as e:
            logger.error(f"VLM streaming error in SceneModule: {e}")
            fallback = "I can see the scene but had trouble describing it."
            if speaker:
                speaker.speak_stream(fallback)
            return fallback

        if not full_text.strip():
            fallback = "I can see the scene but could not make out anything clearly right now."
            if speaker:
                speaker.speak_stream(fallback)
            return fallback

        logger.info(f"Scene complete — {sentence_count} sentences, "
                    f"{len(full_text)} chars")
        return full_text.strip()
