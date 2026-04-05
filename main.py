"""
main.py — Blind Assistant entry point for Raspberry Pi 5 + Hailo 8 AI HAT
- Pure terminal / mic loop (no web server, no browser, no FastAPI)
- Listens on USB mic via Groq Whisper STT
- Routes intent via LangGraph agent
- Currency detection runs on Hailo 8 using .hef model
- TTS output to Bluetooth earbuds via gTTS / ElevenLabs
- CV2 window shows live bounding boxes during currency mode
- Mic loop keeps running so user can switch modes at any time

DISPLAY ARCHITECTURE:
  cv2 GUI calls (namedWindow, imshow, waitKey) MUST run on the main thread
  on Linux Qt/GTK backends. mic_loop() blocks for seconds inside listen(),
  which starves the Qt event loop and causes the window to freeze/hang on
  second launch. Fix: mic_loop() runs in a daemon thread. The main thread
  does nothing except call pump_display() in a tight loop, keeping Qt alive.
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
from modules.currency.currency_detector import pump_display

speaker = Speaker()


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
        logger.warning(f"Pipeline state key error — missing key: {e}")
        speaker.speak("Sorry, I had trouble understanding that.")
    except ValueError as e:
        logger.warning(f"Pipeline value error: {e}")
        speaker.speak("Sorry, I had trouble understanding that.")
    except Exception as e:
        logger.warning(f"Pipeline error — {type(e).__name__}: {e}")
        speaker.speak("Sorry, I had trouble understanding that.")


# ══════════════════════════════════════════════
# MIC LOOP — runs in a background thread
# Main thread must stay free for cv2 GUI calls
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
# ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == "__main__":
    logger.info("══════════════════════════════════════")
    logger.info("    Blind Assistant — Raspberry Pi 5  ")
    logger.info("    Hailo 8 AI HAT + RPi Camera 3     ")
    logger.info("    Press Ctrl+C to stop              ")
    logger.info("══════════════════════════════════════")

    # Start mic loop in a daemon thread so the main thread stays free
    # for cv2 GUI calls (Qt/GTK requires all imshow/waitKey on main thread)
    mic_thread = threading.Thread(target=mic_loop, daemon=True, name="MicLoop")
    mic_thread.start()
    logger.info("Mic loop thread started ✓")

    try:
        # Main thread: pump cv2 display queue at ~30Hz.
        # pump_display() is a no-op when currency is not running.
        # If user presses 'q' in the cv2 window, treat it as stop signal.
        while True:
            try:
                q_pressed = pump_display()
                if q_pressed:
                    logger.info("'q' pressed in cv2 window — stopping currency")
                    try:
                        from modules.currency.currency_module import stop_currency_mode, currency_active
                        if currency_active:
                            stop_currency_mode()
                    except Exception as e:
                        logger.warning(f"q-press stop error: {e}")
            except Exception as e:
                logger.warning(f"pump_display error: {e}")
            time.sleep(0.033)   # ~30Hz — fast enough for smooth display

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down")
        speaker.speak("Goodbye!")

        # Stop currency detection if running
        try:
            from modules.currency.currency_module import stop_currency_mode, currency_active
            if currency_active:
                stop_currency_mode()
        except Exception:
            pass

        # Shutdown Hailo device (only called once, here at exit)
        try:
            from modules.currency.currency_detector import shutdown as hailo_shutdown
            hailo_shutdown()
        except Exception:
            pass

        # Shutdown shared camera manager
        try:
            from utils.camera_manager import camera_manager
            camera_manager.shutdown()
        except Exception:
            pass

        # Close cv2 window from main thread (correct thread to call this)
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

        os._exit(0)
