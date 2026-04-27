"""
modules/navigation/navigation_module.py

BlindNav — proximity navigation for visually impaired users.
Integrated into the Blind Assistant framework (RPi 5 + Hailo 8 AI HAT).

Architecture
────────────
  start_navigation_mode()   → called by agent's navigation_node
  stop_navigation_mode()    → called by stop_node or when another mode starts
  get_latest_nav_frame()    → called by main.py's display_loop (main thread)

The module runs two background threads:
  • _vision_thread  — captures frames, runs Hailo depth + YOLO, analyses zones
  • (speech is dispatched from _vision_thread via the project Speaker)

The main thread calls get_latest_nav_frame() at ~30 Hz and imshow()s the
result — exactly the same pattern used by currency_module.

Camera access goes through the shared CameraManager singleton so navigation
cannot conflict with currency or reading modes.
"""

from __future__ import annotations

import threading
import time
import queue
import numpy as np
import cv2
import os

from utils.logger import logger
from utils.camera_manager import camera_manager

# ── Lazy imports of heavy runners (only loaded when mode starts) ─────────────
# This keeps startup fast and avoids loading Hailo for unused modes.
_depth_runner = None
_yolo_runner  = None

# ── HEF paths — stored in modules/navigation/models/ ────────────────────────
# Resolve relative to THIS file so no dependency on config.BASE_DIR
_NAV_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_MIDAS_HEF_PATH = os.path.join(_NAV_DIR, "Midas_v2_small_model.hef")
_YOLO_HEF_PATH  = os.path.join(_NAV_DIR, "yolov8m.hef")

# ── Module state ─────────────────────────────────────────────────────────────
navigation_active: bool          = False
_nav_thread: threading.Thread | None = None
_stop_event: threading.Event     = threading.Event()
_camera_released: threading.Event = threading.Event()

# ── Shared frame (vision thread writes, display loop reads) ──────────────────
_latest_frame_lock = threading.Lock()
_latest_nav_frame: np.ndarray | None = None


def get_latest_nav_frame() -> np.ndarray | None:
    """Called by main.py's display_loop on the main thread."""
    with _latest_frame_lock:
        return _latest_nav_frame


def _set_latest_nav_frame(frame: np.ndarray | None):
    with _latest_frame_lock:
        global _latest_nav_frame
        _latest_nav_frame = frame


# ════════════════════════════════════════════════════════════════════════════
#  ZONE DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════

ZONE_DANGER  = 0.5    # metres — stop immediately
ZONE_WARN    = 1.5    # metres — caution
ZONE_NOTICE  = 3.5    # metres — be aware

ZONES = [
    ("far left",  0.00, 0.20),
    ("left",      0.20, 0.40),
    ("center",    0.40, 0.60),
    ("right",     0.60, 0.80),
    ("far right", 0.80, 1.00),
]

ZONE_COL = {
    "danger": (0,   0,   255),
    "warn":   (0,   110, 255),
    "notice": (0,   210, 255),
    "clear":  (0,   210, 80),
}

# Speech cooldowns (seconds) — prevent audio spam
_SPEECH_CD = {"danger": 4.0, "warn": 6.0, "notice": 10.0, "clear": 12.0}


# ════════════════════════════════════════════════════════════════════════════
#  ZONE ANALYSER
# ════════════════════════════════════════════════════════════════════════════

class ZoneAnalyser:
    def analyse(self, depth_map: np.ndarray, depth_runner) -> list:
        h, w = depth_map.shape
        roi  = depth_map[int(h * 0.25):, :]   # ignore sky / ceiling

        zones = []
        for name, x0f, x1f in ZONES:
            x0     = int(x0f * w)
            x1     = int(x1f * w)
            col    = roi[:, x0:x1]
            val    = float(np.percentile(col, 85))
            dist_m = depth_runner.to_meters(val)
            level  = (
                "danger" if dist_m <= ZONE_DANGER else
                "warn"   if dist_m <= ZONE_WARN   else
                "notice" if dist_m <= ZONE_NOTICE  else
                "clear"
            )
            zones.append({
                "name": name, "distance_m": dist_m,
                "level": level, "x0": x0, "x1": x1,
            })
        return zones

    def safe_direction(self, zones: list) -> str:
        weights    = [0.6, 0.8, 1.0, 0.8, 0.6]
        best_score = -1
        best_name  = "center"
        for i, z in enumerate(zones):
            score = z["distance_m"] * weights[i]
            if score > best_score:
                best_score = score
                best_name  = z["name"]
        return best_name


# ════════════════════════════════════════════════════════════════════════════
#  SPEECH SCHEDULER  (uses project Speaker, non-blocking)
# ════════════════════════════════════════════════════════════════════════════

class NavSpeechScheduler:
    """
    Decides WHAT to say and WHEN — delegates to the project's TTS Speaker.
    All calls are non-blocking (Speaker uses its own thread internally).
    """

    def __init__(self):
        self._last:       dict = {}
        self._prev_lvl:   dict = {}
        self._last_obj:   dict = {}
        self._last_guide: float = 0.0

    def update(self, zones: list, detections: list, safe_dir: str,
               speaker) -> None:
        now = time.time()

        # ── 1. Immediate danger ──────────────────────────────────────────────
        danger_zones = [z for z in zones if z["level"] == "danger"]
        if danger_zones:
            dz = min(danger_zones, key=lambda x: x["distance_m"])
            if now - self._last.get("danger_alert", 0) >= _SPEECH_CD["danger"]:
                escape = self._escape_direction(zones, dz["name"])
                if escape:
                    msg = (f"Stop! Obstacle {dz['distance_m']} metres "
                           f"{dz['name']}. Step {escape}.")
                else:
                    msg = f"Stop! Very close obstacle {dz['name']}. Do not move."
                speaker.speak(msg)
                self._last["danger_alert"] = now
                return

        # ── 2. High-priority YOLO objects ────────────────────────────────────
        from modules.navigation.hailo_runner import HIGH_PRIORITY
        for det in detections:
            if det["high_priority"]:
                name = det["name"]
                if now - self._last_obj.get(name, 0) >= 4.0:
                    zone_name = ZONES[det["zone_idx"]][0]
                    avoid = self._escape_direction(zones, zone_name)
                    msg = (f"{name} on your {zone_name}. Move {avoid}."
                           if avoid else f"{name} ahead. Slow down.")
                    speaker.speak(msg)
                    self._last_obj[name] = now
                    return

        # ── 3. Navigation guidance every 5 s ────────────────────────────────
        if now - self._last_guide >= 5.0:
            guide = self._navigation_instruction(zones, safe_dir)
            speaker.speak(guide)
            self._last_guide = now
            return

        # ── 4. Zone-cleared reassurance ──────────────────────────────────────
        for z in zones:
            prev = self._prev_lvl.get(z["name"], "clear")
            if prev in ("danger", "warn") and z["level"] == "clear":
                key = z["name"] + "_clear"
                if now - self._last.get(key, 0) >= 5.0:
                    speaker.speak(f"{z['name']} is now clear.")
                    self._last[key] = now

        for z in zones:
            self._prev_lvl[z["name"]] = z["level"]

    # ── helpers ─────────────────────────────────────────────────────────────

    def _navigation_instruction(self, zones: list, safe_dir: str) -> str:
        zm         = {z["name"]: z for z in zones}
        center     = zm.get("center", {})
        left       = zm.get("left",   {})
        right      = zm.get("right",  {})
        far_left   = zm.get("far left",  {})
        far_right  = zm.get("far right", {})

        if all(z["level"] == "clear" for z in zones):
            return f"All clear. Walk straight, {self._steps(center.get('distance_m', 5))} steps ahead."

        if center.get("level") == "clear" and \
           center.get("distance_m", 0) >= left.get("distance_m", 0) and \
           center.get("distance_m", 0) >= right.get("distance_m", 0):
            return f"Walk straight. {self._steps(center.get('distance_m', 5))} steps ahead."

        if center.get("level") in ("danger", "warn"):
            if left.get("distance_m", 0) > right.get("distance_m", 0) and \
               left.get("level") in ("clear", "notice"):
                return (f"Obstacle ahead. Turn left. "
                        f"{self._steps(left.get('distance_m', 3))} steps clear on left.")
            elif right.get("distance_m", 0) > left.get("distance_m", 0) and \
                 right.get("level") in ("clear", "notice"):
                return (f"Obstacle ahead. Turn right. "
                        f"{self._steps(right.get('distance_m', 3))} steps clear on right.")
            elif far_left.get("level") == "clear":
                return "Move to your far left. Path is clear there."
            elif far_right.get("level") == "clear":
                return "Move to your far right. Path is clear there."
            else:
                return "Path blocked ahead. Stop and wait."

        ld, rd = left.get("distance_m", 0), right.get("distance_m", 0)
        if ld > rd + 1.0:
            return f"Slight left. {self._steps(ld)} steps of space on left."
        elif rd > ld + 1.0:
            return f"Slight right. {self._steps(rd)} steps of space on right."

        sd   = zm.get(safe_dir, {})
        dist = sd.get("distance_m", 5)
        if safe_dir == "center":
            return f"Walk straight. {self._steps(dist)} steps ahead."
        return f"Head {safe_dir}. {self._steps(dist)} steps clear."

    @staticmethod
    def _steps(metres: float) -> int:
        return max(1, round(metres / 0.75))

    @staticmethod
    def _escape_direction(zones: list, blocked: str) -> str:
        order       = ["far left", "left", "center", "right", "far right"]
        zm          = {z["name"]: z for z in zones}
        blocked_idx = next((i for i, n in enumerate(order) if n == blocked), 2)
        candidates  = sorted(
            range(len(order)),
            key=lambda i: (abs(i - blocked_idx),
                           -zm.get(order[i], {}).get("distance_m", 0))
        )
        for i in candidates:
            z = zm.get(order[i])
            if z and z["level"] in ("clear", "notice") and order[i] != blocked:
                return order[i]
        return ""


# ════════════════════════════════════════════════════════════════════════════
#  VISUALISER  (two-panel: camera zones | depth heatmap)
# ════════════════════════════════════════════════════════════════════════════

class NavVisualiser:

    def build(self, frame: np.ndarray, depth_map: np.ndarray,
              zones: list, detections: list,
              safe_dir: str, fps: float) -> np.ndarray:
        left  = self._cam_panel(frame, zones, detections, safe_dir, fps)
        right = self._depth_panel(frame, depth_map, zones, safe_dir)
        lp = cv2.resize(left,  (480, 380))
        rp = cv2.resize(right, (480, 380))
        return cv2.hconcat([lp, rp])

    # ── Camera panel ──────────────────────────────────────────────────────

    def _cam_panel(self, frame, zones, detections, safe_dir, fps):
        p    = frame.copy()
        h, w = p.shape[:2]

        # Zone overlays
        for z in zones:
            col   = ZONE_COL[z["level"]]
            alpha = 0.18 if z["level"] == "clear" else 0.38
            x0, x1 = z["x0"], z["x1"]
            ov = p.copy()
            cv2.rectangle(ov, (x0, 0), (x1, h), col, -1)
            cv2.addWeighted(ov, alpha, p, 1 - alpha, 0, p)
            cv2.rectangle(p, (x0, 0), (x1, h), col, 2)

            bx = x0 + 4
            cv2.putText(p, z["name"],
                        (bx, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 220, 220), 1)
            cv2.putText(p, f"{z['distance_m']}m",
                        (bx, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2)
            if z["level"] == "danger":
                cv2.rectangle(p, (x0 + 3, 3), (x1 - 3, h - 3), col, 5)

        # YOLO bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            col = (0, 0, 255) if det["high_priority"] else (0, 200, 200)
            cv2.rectangle(p, (x1, y1), (x2, y2), col, 2)
            lbl = f"{det['name']} {det['conf']:.0%}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(p, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
            cv2.putText(p, lbl, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # Safe path arrow
        self._draw_arrow(p, zones, safe_dir)

        # Title bar
        cv2.rectangle(p, (0, 0), (w, 34), (12, 12, 12), -1)
        cv2.putText(p, f"BLINDNAV  FPS:{fps:.1f}  [Q=stop]",
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 230), 1)
        return p

    def _draw_arrow(self, panel, zones, safe_dir):
        h, w = panel.shape[:2]
        cx   = w // 2
        for z in zones:
            if z["name"] == safe_dir:
                cx = (z["x0"] + z["x1"]) // 2
                break
        src = (w // 2, h - 60)
        dst = (cx,     h // 2 + 20)
        col = (0, 255, 120)
        cv2.arrowedLine(panel, src, dst, col, 4, tipLength=0.3)
        cv2.putText(panel, "SAFE", (cx - 18, h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    # ── Depth panel ───────────────────────────────────────────────────────

    def _depth_panel(self, frame, depth_map, zones, safe_dir):
        h, w = frame.shape[:2]
        dm   = cv2.resize(depth_map, (w, h))
        p    = cv2.applyColorMap(dm, cv2.COLORMAP_INFERNO)

        for z in zones:
            col = ZONE_COL[z["level"]]
            cv2.rectangle(p, (z["x0"], 0), (z["x1"], h), col, 2)
            cv2.putText(p, f"{z['distance_m']}m",
                        (z["x0"] + 4, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            if z["name"] == safe_dir:
                cv2.rectangle(p, (z["x0"] + 4, 38),
                              (z["x1"] - 4, h - 4), (0, 255, 120), 3)

        cv2.rectangle(p, (0, 0), (w, 34), (12, 12, 12), -1)
        cv2.putText(p, f"DEPTH MAP  |  Safe: {safe_dir.upper()}",
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return p


# ════════════════════════════════════════════════════════════════════════════
#  VISION THREAD
# ════════════════════════════════════════════════════════════════════════════

def _vision_loop(speaker):
    """
    Background thread — acquires camera, runs Hailo inference, updates frame.
    Releases camera cleanly when stopped.
    """
    global navigation_active

    _camera_released.clear()
    analyser  = ZoneAnalyser()
    scheduler = NavSpeechScheduler()
    vis       = NavVisualiser()

    frame_count = 0
    start_time  = time.time()

    # ── Load Hailo runners (lazy, once per activation) ───────────────────────
    global _depth_runner, _yolo_runner
    if _depth_runner is None:
        from modules.navigation.hailo_runner import HailoDepthRunner
        _depth_runner = HailoDepthRunner(_MIDAS_HEF_PATH)
        _depth_runner.load()

    if _yolo_runner is None:
        from modules.navigation.hailo_runner import HailoYOLORunner
        _yolo_runner = HailoYOLORunner(_YOLO_HEF_PATH)
        _yolo_runner.load()

    # ── Acquire camera via shared CameraManager ──────────────────────────────
    try:
        cam = camera_manager.acquire(
            mode="navigation",
            model_size=(640, 480),
            warmup=1.0,
        )
        logger.info("Navigation: camera acquired ✓")
    except Exception as e:
        logger.error(f"Navigation: camera acquire failed: {e}")
        speaker.speak("Could not access camera for navigation.")
        navigation_active = False
        _camera_released.set()
        return

    try:
        while not _stop_event.is_set():
            # ── Capture frame ───────────────────────────────────────────────
            try:
                bgr = cam.capture_array()
            except Exception as e:
                logger.warning(f"Navigation: capture error: {e}")
                time.sleep(0.05)
                continue

            if bgr is None:
                time.sleep(0.05)
                continue

            # camera_manager returns BGR (RGB888 in libcamera → OpenCV native)
            frame = bgr

            # ── Depth estimation ────────────────────────────────────────────
            depth_map = _depth_runner.estimate(frame)

            # ── Zone analysis ───────────────────────────────────────────────
            zones    = analyser.analyse(depth_map, _depth_runner)
            safe_dir = analyser.safe_direction(zones)

            # ── Object detection ────────────────────────────────────────────
            detections = _yolo_runner.detect(frame)

            # ── Speech scheduler ────────────────────────────────────────────
            try:
                scheduler.update(zones, detections, safe_dir, speaker)
            except Exception as e:
                logger.warning(f"Navigation speech scheduler error: {e}")

            # ── Build display frame ─────────────────────────────────────────
            frame_count += 1
            fps = frame_count / max(time.time() - start_time, 0.001)
            try:
                display = vis.build(frame, depth_map, zones,
                                    detections, safe_dir, fps)
                _set_latest_nav_frame(display)
            except Exception as e:
                logger.warning(f"Navigation visualiser error: {e}")
                _set_latest_nav_frame(frame)   # fallback: raw frame

    finally:
        # ── Cleanup ─────────────────────────────────────────────────────────
        _set_latest_nav_frame(None)
        try:
            camera_manager.release()
            logger.info("Navigation: camera released ✓")
        except Exception as e:
            logger.warning(f"Navigation camera release error: {e}")
        navigation_active = False
        _camera_released.set()
        logger.info("Navigation vision thread stopped ✓")


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def start_navigation_mode(speaker=None) -> None:
    """Start navigation mode. Called from agent's navigation_node."""
    global navigation_active, _nav_thread

    if navigation_active:
        logger.info("Navigation: already active — ignoring duplicate start")
        return

    if speaker is None:
        from tts.speaker import Speaker
        speaker = Speaker()

    _stop_event.clear()
    navigation_active = True

    _nav_thread = threading.Thread(
        target=_vision_loop,
        args=(speaker,),
        daemon=True,
        name="NavVisionLoop",
    )
    _nav_thread.start()
    logger.info("Navigation mode started ✓")


def stop_navigation_mode() -> None:
    """Stop navigation mode. Called from stop_node or before another mode starts."""
    global navigation_active

    if not navigation_active:
        logger.info("Navigation: not active — nothing to stop")
        return

    logger.info("Navigation: stopping…")
    _stop_event.set()

    # Wait up to 3 s for the camera to be released
    released = _camera_released.wait(timeout=3.0)
    if not released:
        logger.warning("Navigation: camera not released within timeout")

    navigation_active = False
    _set_latest_nav_frame(None)
    logger.info("Navigation mode stopped ✓")
