"""
modules/currency/currency_module.py
─────────────────────────────────────
Thread manager for currency detection.
Exposes start_currency_mode() and stop_currency_mode() to the LangGraph agent.

The actual inference (Hailo 8) lives in currency_detector.py.
"""

import threading
from utils.logger import logger
from .currency_detector import start_currency_detection, stop_currency_detection

currency_active = False
currency_thread = None
_lock           = threading.Lock()


def start_currency_mode():
    global currency_active, currency_thread

    with _lock:
        if currency_active and currency_thread and currency_thread.is_alive():
            logger.warning("Currency mode already active — ignoring duplicate start")
            return

        currency_active = True
        currency_thread = threading.Thread(
            target=start_currency_detection,
            daemon=True
        )
        currency_thread.start()
        logger.info("Currency mode started ✓")


def stop_currency_mode():
    global currency_active, currency_thread

    with _lock:
        if not currency_active:
            logger.warning("Currency mode not active — nothing to stop")
            return

        stop_currency_detection()
        currency_active = False
        currency_thread = None
        logger.info("Currency mode stopped ✓")
