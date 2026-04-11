"""
main.py — Blind Assistant entry point for Raspberry Pi 5 + Hailo 8 AI HAT
"""

import sys
import os
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

speaker = Speaker()

WINDOW_NAME = "Currency Detection - Hailo 8"


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
    speaker.speak("Blind assistant ready. Say currency check to detect notes.")

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
# cv2 imshow/waitKey must be called from the
# same thread (main) on Linux Qt/GTK backends.
# _run() writes frames to shared memory,
# this loop reads and renders them at ~30fps.
# ══════════════════════════════════════════════
def display_loop():
    import cv2
    import numpy as np
    from modules.currency.currency_detector import get_latest_frame
    from modules.scene.scene_module import get_latest_scene_frame

    window_open = False

    # Track which scene frame we last showed so we only re-render on change
    _last_scene_id: list = [None]

    # Blank "waiting" frame shown when nothing is running
    waiting = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.putText(waiting, "Waiting for currency detection...",
                (160, 270), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (80, 80, 80), 2, cv2.LINE_AA)

    stopped = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.putText(stopped, "Currency detection stopped",
                (210, 270), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (80, 80, 80), 2, cv2.LINE_AA)

    logger.info("Display loop started on main thread")

    while True:
        currency_frame = get_latest_frame()
        scene_frame    = get_latest_scene_frame()

        if currency_frame is not None:
            # Live currency detection — always takes priority
            if not window_open:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, 960, 540)
                window_open = True
                logger.info("Display: window opened")
            cv2.imshow(WINDOW_NAME, currency_frame)

        elif scene_frame is not None:
            # Show the latest captured scene frame with a label overlay
            scene_id = id(scene_frame)
            if not window_open:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, 960, 540)
                window_open = True
                logger.info("Display: window opened for scene frame")

            if scene_id != _last_scene_id[0]:
                # New frame arrived — render label overlay once
                display = cv2.resize(scene_frame, (960, 540))

                # Semi-transparent dark bar at the top for the label
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (960, 44), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

                cv2.putText(display, "Scene captured",
                            (14, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.85, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow(WINDOW_NAME, display)
                _last_scene_id[0] = scene_id

        elif window_open:
            # Something was running but has now stopped
            cv2.imshow(WINDOW_NAME, stopped)

        # waitKey must be called continuously to keep Qt event loop alive
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and window_open:
            logger.info("'q' pressed — stopping currency detection")
            try:
                from modules.currency.currency_module import stop_currency_mode, currency_active
                if currency_active:
                    stop_currency_mode()
            except Exception as e:
                logger.warning(f"q-press stop error: {e}")

        time.sleep(0.033)   # ~30 Hz is plenty for display


# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == "__main__":
    logger.info("══════════════════════════════════════")
    logger.info("    Blind Assistant — Raspberry Pi 5  ")
    logger.info("    Hailo 8 AI HAT + RPi Camera 3     ")
    logger.info("    Press Ctrl+C to stop              ")
    logger.info("══════════════════════════════════════")

    # Mic loop in background thread — blocks on listen() so must not be main
    mic_thread = threading.Thread(target=mic_loop, daemon=True, name="MicLoop")
    mic_thread.start()
    logger.info("Mic loop thread started")

    try:
        # Main thread owns all cv2 GUI calls
        display_loop()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down")
        speaker.speak("Goodbye!")

        try:
            from modules.currency.currency_module import stop_currency_mode, currency_active
            if currency_active:
                stop_currency_mode()
        except Exception:
            pass

        try:
            from modules.currency.currency_detector import shutdown as hailo_shutdown
            hailo_shutdown()
        except Exception:
            pass

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
