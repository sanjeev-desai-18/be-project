# utils/camera_manager.py
#
# Singleton Picamera2.
#
# FORMAT NOTE:
# Picamera2's "RGB888" format returns BGR byte order (DRM convention).
# This is OpenCV's native format — we can use frames directly with
# cv2.imshow / cv2.imencode without colour conversion.
# For Hailo inference (expects RGB), convert BGR→RGB before hailo.run().
#
# WHY _verify_frames_flowing() WAS REMOVED:
# ──────────────────────────────────────────
# _verify_frames_flowing() called capture_array() in a daemon thread to
# confirm the pipeline was live. The problem: cam.stop() (called in release())
# does NOT interrupt a thread already blocked inside capture_array(). So on
# second launch the ghost verification thread from session 1 was still alive
# and blocked inside capture_array() when session 2 called cam.configure()
# and cam.start() underneath it. Session 2 then spawned its own verification
# thread, resulting in two concurrent threads hitting picamera2's internal
# request queue simultaneously — corrupting it. Every capture in the second
# session's detection loop then returned immediately with an error or None,
# making the loop spin at 100% CPU and produce zero detections.
#
# The fix: replace pre-verification with a longer sleep warmup. The detection
# loop's _capture_with_timeout() already handles stalls gracefully — if the
# first few frames are bad, it logs and continues. No pre-verification needed.
#
# HOW SECOND-LAUNCH STALLS ARE NOW HANDLED:
# ──────────────────────────────────────────
# _configure() always does stop() + 0.3s sleep before reconfigure, which
# clears picamera2's internal request queue. acquire() then sleeps warmup
# seconds (1.0s for currency) after start(). By the time _run()'s first
# _capture_with_timeout() fires, the pipeline has had 1.3s to stabilise.
# If a frame still doesn't arrive within 1s, _capture_with_timeout() returns
# ("timeout") and the loop retries — no hang, no ghost threads.

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

        # Always do a full stop before reconfiguring — clears request queue.
        if self._started:
            try:
                cam.stop()
                logger.info("CameraManager: stopped before reconfigure")
            except Exception:
                pass
            self._started = False

        # Brief sleep after stop lets picamera2 flush any stale requests
        # that were queued before stop() was called.
        time.sleep(0.3)

        if mode == "currency":
            w, h = model_size
            config = cam.create_preview_configuration(
                main={"size": (w, h), "format": "RGB888"})
        elif mode == "reading":
            config = cam.create_still_configuration(
                main={"size": (1920, 1080), "format": "RGB888"})
        elif mode == "scene":
            # Full native resolution — imx708_wide_noir selects 2304x1296
            # sensor mode when asked for this size (score 1000 = exact match).
            config = cam.create_still_configuration(
                main={"size": (2304, 1296), "format": "RGB888"})
        else:
            config = cam.create_preview_configuration(
                main={"size": (1024, 768), "format": "RGB888"})

        cam.configure(config)
        self._mode = mode
        logger.info(f"CameraManager: configured for mode='{mode}'")

    def acquire(self, mode: str, model_size: tuple = (640, 640),
                warmup: float = 0.8, lock_timeout: float = 10.0) -> "Picamera2":
        # Use a timeout so callers (scene, reading) never freeze indefinitely
        # if the currency thread is slow to release. 10s is generous — currency
        # stop + camera release normally completes in under 4s.
        if not self._lock.acquire(timeout=lock_timeout):
            raise RuntimeError(
                f"CameraManager: could not acquire lock in {lock_timeout}s — "
                "another module may still be holding the camera"
            )
        try:
            self._configure(mode, model_size)
            cam = self._get_picam2()

            cam.start()
            self._started = True
            logger.info(
                f"CameraManager: started mode='{mode}', warming up {warmup}s"
            )
            # Sleep-only warmup. Do NOT call capture_array() here.
            # Any capture_array() call in a thread at this point can outlive
            # release() and corrupt the next session's request queue.
            # The detection loop's _capture_with_timeout() handles stalls.
            time.sleep(warmup)

            logger.info("CameraManager: acquire complete ✓")
            return cam

        except Exception as e:
            try:
                self._lock.release()
            except RuntimeError:
                pass
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
        logger.warning("CameraManager: force_release() called")
        try:
            if self._picam2 is not None:
                self._picam2.stop()
                self._started = False
                logger.info("CameraManager: force-stopped ✓")
        except Exception as e:
            logger.warning(f"CameraManager: force stop error: {e}")
        time.sleep(0.3)
        acquired = self._lock.acquire(blocking=False)
        if acquired:
            self._lock.release()
        else:
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
