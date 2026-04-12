# modules/scene/scene_module.py
#
# Speech strategy — ONE gTTS call, ONE MP3, zero inter-sentence gaps:
#
#   Every previous attempt that split the response across multiple speak()
#   or speak_stream() calls produced an audible pause at the split point
#   because each call is a separate MP3 file. gTTS encodes a small silence
#   tail into every file, and pygame's music.load() adds a brief buffer
#   flush on top — together ~300-500ms of dead air per boundary.
#
#   Fix: collect the complete VLM response (2-3 short sentences, ~3s),
#   join into one string, synthesise ONCE. The ACK "Looking at your
#   surroundings." already plays at ~200ms from agent.py so the user
#   hears immediate feedback. The full description then plays as one
#   uninterrupted audio clip with natural prosody end-to-end.

import io
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
    with _frame_lock:
        return _latest_scene_frame


def _store_scene_frame(bgr: np.ndarray) -> None:
    global _latest_scene_frame
    with _frame_lock:
        _latest_scene_frame = resize_frame(bgr.copy(), max_width=1280)


# ── Prompt ─────────────────────────────────────────────────────────────────────
_SCENE_PROMPT = """\
Describe this scene for a blind person in 2 to 3 natural spoken sentences.
Mention what is nearest to the camera, any obstacles or hazards, and the \
general environment.
Be direct and conversational. Do not use lists, bullet points, or headings.
Start immediately with the description — no preamble like "In this image..."
"""

_VLM_TIMEOUT = 10.0


class SceneModule:

    def __init__(self):
        self.vlm = VLMClient()

    def _capture_frame(self) -> tuple[str, np.ndarray] | tuple[None, None]:
        picam2 = camera_manager.acquire(mode="scene", warmup=0.3)
        try:
            raw = picam2.capture_array("main")
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
        Capture frame, collect full VLM response, speak as a single TTS call.

        One gTTS call = one MP3 = no pauses mid-description.
        The ACK from agent.py covers perceived latency.
        """
        logger.info("SceneModule.run() | capturing frame via camera_manager")

        b64_frame, _ = self._capture_frame()

        if b64_frame is None:
            msg = "I could not access the camera."
            if speaker:
                speaker.speak_stream(msg)
            return msg

        # ── Collect full VLM response ─────────────────────────────────────
        collected: list[str] = []
        error_box: list[str] = []

        def _generate():
            try:
                for sentence in self.vlm.describe_stream(b64_frame, _SCENE_PROMPT):
                    collected.append(sentence)
            except Exception as exc:
                error_box.append(str(exc))
                logger.error(f"VLM error in SceneModule: {exc}")

        gen = threading.Thread(target=_generate, daemon=True, name="scene-vlm-gen")
        gen.start()
        gen.join(timeout=_VLM_TIMEOUT)

        if gen.is_alive():
            logger.warning("VLM timed out — using partial response")

        full_text = " ".join(s.strip() for s in collected if s.strip())

        if not full_text:
            msg = "I can see the scene but could not make out anything clearly."
            if speaker:
                speaker.speak_stream(msg)
            return msg

        logger.info(
            f"Scene complete — {len(collected)} chunk(s), {len(full_text)} chars"
        )

        # ── Single gTTS call — one MP3, no gaps, no cut-off words ─────────
        if speaker:
            speaker.speak_stream(full_text)

        return full_text
