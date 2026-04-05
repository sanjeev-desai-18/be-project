# utils/camera_manager.py

import threading
import time
from utils.logger import logger


class CameraManager:

    def __init__(self):
        self._lock    = threading.Lock()
        self._picam2  = None
        self._mode    = None
        self._started = False

    def _get_picam2(self):
        if self._picam2 is None:
            from picamera2 import Picamera2
            self._picam2 = Picamera2()
            logger.info("CameraManager: Picamera2 instance created ✓")
        return self._picam2

    def _configure(self, mode: str, model_size: tuple = (640, 640)):
        cam = self._get_picam2()
        if self._started:
            try:
                cam.stop()
            except Exception:
                pass
            self._started = False

        if mode == "currency":
            w, h = model_size
            config = cam.create_preview_configuration(
                main={"size": (w, h), "format": "RGB888"})
        elif mode == "reading":
            config = cam.create_still_configuration(
                main={"size": (1920, 1080), "format": "RGB888"})
        else:
            config = cam.create_preview_configuration(
                main={"size": (1024, 768), "format": "RGB888"})

        cam.configure(config)
        self._mode = mode
        logger.info(f"CameraManager: configured for mode='{mode}'")

    def acquire(self, mode: str, model_size: tuple = (640, 640),
                warmup: float = 0.8) -> "Picamera2":
        self._lock.acquire()
        try:
            self._configure(mode, model_size)
            cam = self._get_picam2()
            cam.start()
            self._started = True
            logger.info(f"CameraManager: started mode='{mode}', warming up {warmup}s")
            time.sleep(warmup)
            for _ in range(3):
                try:
                    cam.capture_array("main")
                except Exception:
                    pass
            return cam
        except Exception as e:
            self._lock.release()
            logger.error(f"CameraManager.acquire() failed: {e}")
            raise RuntimeError(f"Camera acquire failed: {e}") from e

    def release(self):
        try:
            if self._picam2 is not None and self._started:
                self._picam2.stop()
                self._started = False
                logger.info("CameraManager: stopped streaming")
        except Exception as e:
            logger.warning(f"CameraManager.release() stop error: {e}")
        finally:
            try:
                self._lock.release()
            except RuntimeError:
                pass

    def force_release(self):
        """
        Call this when the thread holding the camera is stuck in capture_array()
        and will never call release() itself.

        Strategy:
          1. Call picam2.stop() — this terminates the libcamera pipeline and
             causes any blocked capture_array() to raise an exception, which
             lets the stuck thread exit its loop and reach its finally block.
          2. Force-release the lock regardless of who holds it, so the next
             acquire() can proceed.

        This is safe because:
          - picam2.stop() is idempotent (calling it twice is harmless)
          - The stuck thread's finally block calls release() which calls
            stop() again (no-op) and tries lock.release() (RuntimeError
            caught and ignored)
          - The Picamera2 instance stays alive — only streaming stops
        """
        logger.warning("CameraManager: force_release() called — stopping camera to unblock stuck thread")

        # Step 1: stop the camera — this unblocks capture_array() in the stuck thread
        try:
            if self._picam2 is not None:
                self._picam2.stop()
                self._started = False
                logger.info("CameraManager: force-stopped streaming ✓")
        except Exception as e:
            logger.warning(f"CameraManager: force stop error: {e}")

        # Step 2: give the stuck thread a moment to receive the exception
        # and exit its loop before we release the lock
        time.sleep(0.3)

        # Step 3: force-release the lock
        # threading.Lock has no "force release" — we release it only if
        # we can determine it's locked. We do this by trying to acquire
        # with timeout=0. If it succeeds, the lock was FREE (release it
        # back). If it fails, lock is HELD — release it.
        acquired = self._lock.acquire(blocking=False)
        if acquired:
            # Lock was free — we just grabbed it, release it back
            self._lock.release()
            logger.info("CameraManager: lock was already free")
        else:
            # Lock is held by stuck thread — force release
            # threading.Lock allows release from any thread in Python
            try:
                self._lock.release()
                logger.info("CameraManager: lock force-released ✓")
            except RuntimeError as e:
                logger.warning(f"CameraManager: lock release error: {e}")

    def shutdown(self):
        try:
            if self._picam2 is not None:
                if self._started:
                    self._picam2.stop()
                self._picam2.close()
                self._picam2  = None
                self._started = False
                logger.info("CameraManager: shutdown complete")
        except Exception as e:
            logger.warning(f"CameraManager.shutdown() error: {e}")


camera_manager = CameraManager()
