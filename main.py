"""
main.py — Blind Assistant entry point for Raspberry Pi 5 + Hailo 8 AI HAT
"""

import sys
import os
import atexit
import warnings
import time
import threading

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Path setup — must be FIRST ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# ── Local imports ──
from utils.logger import logger
from utils.audio_utils import check_microphone_available
from tts.speaker import Speaker
from modules.stt.listener import listen
from core.agent import agent
from core.state import AssistantState

# ── Register shared Hailo VDevice shutdown at process exit ──────────────────
# This must be registered BEFORE any module opens the VDevice so it is
# guaranteed to run last (atexit handlers execute in LIFO order).
# All runners (depth, YOLO, currency) only deactivate their networks on
# close(); the actual PCIe device handle is released here, once, cleanly.
try:
    from utils.hailo_device import shutdown as _hailo_shutdown
    atexit.register(_hailo_shutdown)
    logger.info("Hailo VDevice atexit handler registered ✓")
except ImportError:
    pass   # hailo_platform not present (dev machine) — nothing to register


speaker = Speaker()

WINDOW_NAME = "Blind Assistant - Hailo 8"


# ══════════════════════════════════════════════
# STATE BUILDER
# ══════════════════════════════════════════════
def build_state(transcript: str) -> AssistantState:
    return {
        "raw_transcript":         transcript.strip(),
        "cleaned_transcript":     "",
        "mode":                   "unknown",
        "confidence":             0.0,
        "extra_context":          "",
        "raw_output":             {},
        "final_output":           "",
        "needs_clarification":    False,
        "clarification_question": "",
        "error":                  None,
        "retry_count":            0,
        "spoken":                 False,
    }


# ══════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════
def run_pipeline(transcript: str) -> None:
    if not transcript.strip():
        logger.warning("Skipping empty transcript")
        return

    logger.info(f"─── New request: '{transcript}' ───")
    state = build_state(transcript)

    try:
        result_state = agent.invoke(state)

        mode       = result_state.get("mode", "unknown")
        confidence = result_state.get("confidence", 0.0)
        output     = result_state.get("final_output", "")

        if not output:
            output = (result_state.get("response") or
                      result_state.get("output") or
                      result_state.get("answer") or
                      result_state.get("text") or "")

        logger.debug(f"mode={mode} | confidence={confidence:.2f} | output={output[:80]}")
        logger.info("─── Request complete ───")

    except KeyError as e:
        logger.warning(f"Pipeline state key error: {e}")
        speaker.speak("Sorry, I had trouble understanding that.")
    except ValueError as e:
        logger.warning(f"Pipeline value error: {e}")
        speaker.speak("Sorry, I had trouble understanding that.")
    except Exception as e:
        logger.warning(f"Pipeline error — {type(e).__name__}: {e}")
        speaker.speak("Sorry, I had trouble understanding that.")


# ══════════════════════════════════════════════
# MIC LOOP — runs in background thread
# ══════════════════════════════════════════════
def mic_loop():
    if not check_microphone_available():
        logger.error("No microphone found — mic loop cannot start")
        speaker.speak("No microphone detected. Please connect a USB microphone.")
        return

    logger.info("Microphone loop started — listening for commands")
    speaker.speak(
        "Hii, your personal visionmate assistant is ready. "
        "I can help you with scene description, currency check, "
        "navigate your surrounding, read documents and improve knowledge."
    )

    while True:
        try:
            transcript = listen()
            if transcript:
                logger.info(f"Heard: \"{transcript}\"")
                run_pipeline(transcript)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.warning(f"Mic loop error: {type(e).__name__}: {e}")
            time.sleep(1)


# ══════════════════════════════════════════════
# DISPLAY LOOP — runs on main thread
# Priority order for frame display:
#   1. Navigation (real-time, two-panel)
#   2. Currency   (real-time, detection overlay)
#   3. Scene      (static captured image)
#   4. Waiting    (blank placeholder)
# ══════════════════════════════════════════════
def display_loop():
    import cv2
    import numpy as np
    from modules.currency.currency_detector import get_latest_frame as get_currency_frame
    from modules.scene.scene_module import get_latest_scene_frame
    from modules.navigation.navigation_module import get_latest_nav_frame

    window_open = False
    _last_scene_id: list = [None]

    # ── Placeholder frames ────────────────────────────────────────────────
    waiting = np.zeros((380, 960, 3), dtype=np.uint8)
    cv2.putText(waiting, "Waiting for a command…",
                (280, 190), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (80, 80, 80), 2, cv2.LINE_AA)
    cv2.putText(waiting,
                "Say: 'activate navigation' | 'currency check' | 'describe surroundings'",
                (60, 230), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (60, 60, 60), 1, cv2.LINE_AA)

    stopped = np.zeros((380, 960, 3), dtype=np.uint8)
    cv2.putText(stopped, "Module stopped",
                (360, 190), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (80, 80, 80), 2, cv2.LINE_AA)

    logger.info("Display loop started on main thread")

    while True:
        nav_frame      = get_latest_nav_frame()
        currency_frame = get_currency_frame()
        scene_frame    = get_latest_scene_frame()

        # ── 1. Navigation (highest priority — real-time) ─────────────────
        if nav_frame is not None:
            if not window_open:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, 960, 380)
                window_open = True
                logger.info("Display: window opened for navigation")
            cv2.imshow(WINDOW_NAME, nav_frame)

        # ── 2. Currency (real-time detection overlay) ────────────────────
        elif currency_frame is not None:
            if not window_open:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, 960, 540)
                window_open = True
                logger.info("Display: window opened for currency")
            cv2.imshow(WINDOW_NAME, currency_frame)

        # ── 3. Scene (static captured image) ────────────────────────────
        elif scene_frame is not None:
            scene_id = id(scene_frame)
            if not window_open:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, 960, 540)
                window_open = True
                logger.info("Display: window opened for scene frame")

            if scene_id != _last_scene_id[0]:
                display = cv2.resize(scene_frame, (960, 540))
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (960, 44), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)
                cv2.putText(display, "Scene captured",
                            (14, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.85, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, display)
                _last_scene_id[0] = scene_id

        # ── 4. Nothing running ───────────────────────────────────────────
        elif window_open:
            cv2.imshow(WINDOW_NAME, stopped)

        # ── waitKey keeps Qt event loop alive ────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and window_open:
            logger.info("'q' pressed — stopping active modules")
            try:
                import modules.navigation.navigation_module as _nav
                if _nav.navigation_active:
                    _nav.stop_navigation_mode()
            except Exception as e:
                logger.warning(f"q-press nav stop error: {e}")
            try:
                from modules.currency.currency_module import stop_currency_mode, currency_active
                if currency_active:
                    stop_currency_mode()
            except Exception as e:
                logger.warning(f"q-press currency stop error: {e}")

        time.sleep(0.033)   # ~30 Hz display update


# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == "__main__":
    logger.info("══════════════════════════════════════")
    logger.info("    Blind Assistant — Raspberry Pi 5  ")
    logger.info("    Hailo 8 AI HAT + RPi Camera 3     ")
    logger.info("    Press Ctrl+C to stop              ")
    logger.info("══════════════════════════════════════")

    mic_thread = threading.Thread(target=mic_loop, daemon=True, name="MicLoop")
    mic_thread.start()
    logger.info("Mic loop thread started")

    try:
        display_loop()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down")
        speaker.speak("Goodbye!")

        # ── Stop all active modules cleanly ──────────────────────────────
        try:
            import modules.navigation.navigation_module as _nav
            if _nav.navigation_active:
                _nav.stop_navigation_mode()
        except Exception:
            pass

        try:
            from modules.currency.currency_module import stop_currency_mode, currency_active
            if currency_active:
                stop_currency_mode()
        except Exception:
            pass

        # NOTE: hailo_device.shutdown() is called automatically via atexit —
        # no manual call needed here.  camera_manager and cv2 still shut down
        # explicitly since they are not covered by atexit.

        try:
            from utils.camera_manager import camera_manager
            camera_manager.shutdown()
        except Exception:
            pass

        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

        os._exit(0)
