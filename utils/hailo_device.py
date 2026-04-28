"""
utils/hailo_device.py

Shared Hailo VDevice singleton — ROUND_ROBIN scheduler.

Root cause of the 5-second switch lag
--------------------------------------
The old code let each module open its own VDevice:

    currency_detector  →  picamera2.devices.Hailo  →  VDevice()   (implicit)
    HailoDepthRunner   →  hailo_platform.VDevice()
    HailoYOLORunner    →  hailo_platform.VDevice()

The Hailo PCIe driver serialises VDevice open() calls.  When a second
module tried to open the device while another was running, the driver
stalled for ~5 s waiting for the first device handle to be released.

The fix
-------
One VDevice is opened at process start with HailoSchedulingAlgorithm.ROUND_ROBIN.
Every runner (depth, YOLO, currency) configures its HEF on *this* device
and calls network.activate() / network.deactivate() rather than
opening/closing the device itself.  ROUND_ROBIN lets the firmware time-
slice the hardware between multiple active networks with no driver stall.

Usage
-----
    from utils.hailo_device import get_vdevice
    vdevice = get_vdevice()          # same object on every call
    network = vdevice.configure(hef, cfg_params)[0]
    network.activate()
    # … inference …
    network.deactivate()             # do NOT call vdevice.release()

Call shutdown() only at clean process exit (e.g. main.py atexit).
"""

from __future__ import annotations

import threading
from utils.logger import logger

_lock: threading.Lock = threading.Lock()
_vdevice = None          # hailo_platform.VDevice | None
_available: bool | None = None   # cached import probe result


def hailo_available() -> bool:
    """Return True if hailo_platform can be imported (cached after first call)."""
    global _available
    if _available is not None:
        return _available
    try:
        import hailo_platform   # type: ignore  # noqa: F401
        _available = True
    except ImportError:
        _available = False
    return _available


def get_vdevice():
    """
    Return the process-wide shared VDevice, creating it on first call.

    Thread-safe.  Raises RuntimeError if hailo_platform is unavailable
    or the device cannot be opened.
    """
    global _vdevice
    if _vdevice is not None:
        return _vdevice
    with _lock:
        if _vdevice is not None:   # double-checked locking
            return _vdevice
        if not hailo_available():
            raise RuntimeError(
                "hailo_platform is not installed — cannot open shared VDevice"
            )
        try:
            from hailo_platform import VDevice, HailoSchedulingAlgorithm  # type: ignore
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            _vdevice = VDevice(params)
            logger.info("HailoSharedDevice: VDevice opened (ROUND_ROBIN) ✓")
        except Exception as exc:
            logger.error(f"HailoSharedDevice: VDevice open failed — {exc}")
            raise RuntimeError(f"Cannot open shared Hailo VDevice: {exc}") from exc
    return _vdevice


def shutdown() -> None:
    """
    Release the shared VDevice.  Call once at process exit *after* all
    networks have been deactivated.  Safe to call even if get_vdevice()
    was never called or hailo_platform is absent.
    """
    global _vdevice
    with _lock:
        if _vdevice is None:
            return
        try:
            _vdevice.release()
            logger.info("HailoSharedDevice: VDevice released ✓")
        except Exception as exc:
            logger.warning(f"HailoSharedDevice shutdown error: {exc}")
        finally:
            _vdevice = None
