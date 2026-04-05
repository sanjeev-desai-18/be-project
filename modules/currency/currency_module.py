"""
modules/currency/currency_module.py
"""
import threading
import time
from utils.logger import logger

currency_active = False
_lock           = threading.Lock()


def reset_currency_state():
    global currency_active
    with _lock:
        currency_active = False


def start_currency_mode():
    global currency_active
    with _lock:
        import modules.currency.currency_detector as _det
        if _det._thread is not None and _det._thread.is_alive():
            logger.warning("Currency already active")
            currency_active = True
            return
        currency_active = True
        _det.start_currency_detection()
        logger.info("Currency mode started ✓")


def stop_currency_mode():
    global currency_active
    from utils.camera_manager import camera_manager
    import modules.currency.currency_detector as _det

    with _lock:
        if not currency_active:
            logger.info("Currency not active")
            return
        _det.stop_currency_detection()
        currency_active = False

    # Wait outside lock for clean thread exit
    released = _det.wait_for_camera_release(timeout=2.0)

    if not released:
        # Thread is stuck in capture_array(). Fix:
        # 1. Stop the camera from outside — this raises an exception
        #    inside capture_array(), letting the thread exit its loop
        # 2. Force-release the camera lock so next acquire() can proceed
        logger.warning("Thread stuck in capture_array() — calling force_release()")
        camera_manager.force_release()

        # Give thread a moment to process the exception and exit
        time.sleep(0.5)

    # Always reset module state so next start() is a clean launch
    _det.reset()
    logger.info("Currency stopped and reset ✓")
