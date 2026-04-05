"""
modules/currency/currency_module.py
─────────────────────────────────────
Thread manager for currency detection.
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
        logger.debug("currency_module: state reset to inactive")


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
            logger.info("Currency not active — nothing to stop")
            return
        _det.stop_currency_detection()
        currency_active = False
        logger.info("Currency stop signal sent ✓")

    # Wait outside lock — allows start to proceed if called immediately after
    released = _det.wait_for_camera_release(timeout=3.0)

    if not released:
        # Thread stuck — _capture_with_timeout should have prevented this,
        # but as last resort: stop camera from outside to unblock it
        logger.warning("Thread still stuck after 3s — calling force_release()")
        camera_manager.force_release()

    # Join the thread so we know it's fully dead before reset
    t = _det._thread
    if t is not None and t.is_alive():
        logger.info("Joining currency thread...")
        t.join(timeout=3.0)
        if t.is_alive():
            logger.error("Thread still alive after join — resetting anyway")

    # Full state reset — next start() gets a clean slate
    _det.reset()
    logger.info("Currency stopped and reset ✓")
