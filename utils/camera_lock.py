# utils/camera_lock.py
#
# With the shared CameraManager, camera_exclusive() is no longer needed —
# camera_manager.acquire() blocks until the camera is free via its internal
# threading.Lock. This file is kept for import compatibility but is now a
# simple passthrough context manager.

from contextlib import contextmanager
from utils.logger import logger


@contextmanager
def camera_exclusive():
    """
    No-op context manager — camera exclusivity is now handled by
    camera_manager.acquire() / camera_manager.release() internally.
    Kept for import compatibility with scene_module and reading_module.
    """
    logger.debug("camera_exclusive: passthrough (camera_manager handles locking)")
    yield
