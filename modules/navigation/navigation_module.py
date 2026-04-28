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
_NAV_DIR           = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_SCDEPTH_HEF_PATH  = os.path.join(_NAV_DIR, "scdepthv3.hef")
_MIDAS_HEF_PATH    = os.path.join(_NAV_DIR, "Midas_v2_small_model.hef")
_YOLO_HEF_PATH     = os.path.join(_NAV_DIR, "yolov8m.hef")

# Prefer SCDepthV3 if available, fall back to MiDaS
_DEPTH_HEF_PATH = _SCDEPTH_HEF_PATH if os.path.exists(_SCDEPTH_HEF_PATH) else _MIDAS_HEF_PATH

# ── Module state ─────────────────────────────────────────────────────────────
navigation_active: bool          = False
_nav_thread: threading.Thread | None = None
_stop_event: threading.Event     = threading.Event()
_camera_released: threading.Event = threading.Event()

# ── Speech pause flag — set by mic_loop while user is speaking ───────────────
# This lets the voice listener temporarily silence nav speech so the user's
# voice command is not drowned out by navigation announcements.
nav_speech_paused: threading.Event = threading.Event()   # SET = paused

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
_SPEECH_CD = {"danger": 5.0, "warn": 8.0, "notice": 12.0, "clear": 15.0}


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
    Respects nav_speech_paused flag so mic can capture voice commands.
    Caches last message to avoid repeating identical guidance.
    """

    def __init__(self):
        self._last:       dict = {}
        self._prev_lvl:   dict = {}
        self._last_obj:   dict = {}
        self._last_guide: float = 0.0
        self._last_msg:   str  = ""   # cache to skip duplicate messages

    def _speak_cached(self, speaker, msg: str):
        """Only speak if the message differs from the last one spoken."""
        if msg != self._last_msg:
            speaker.speak(msg)
            self._last_msg = msg

    def update(self, zones: list, detections: list, safe_dir: str,
               speaker) -> None:
        now = time.time()

        # ── Respect speech-pause flag (mic listening) ────────────────────────
        paused = nav_speech_paused.is_set()

        # ── 1. Danger — only when center + at least one adjacent are blocked ─
        danger_zones = [z for z in zones if z["level"] == "danger"]
        if danger_zones:
            zm = {z["name"]: z for z in zones}
            center_bad = zm.get("center", {}).get("level") in ("danger", "warn")
            left_bad   = zm.get("left", {}).get("level") in ("danger", "warn")
            right_bad  = zm.get("right", {}).get("level") in ("danger", "warn")
            # Only say "stop" when center AND at least one adjacent are blocked
            if center_bad and (left_bad or right_bad):
                dz = min(danger_zones, key=lambda x: x["distance_m"])
                if now - self._last.get("danger_alert", 0) >= _SPEECH_CD["danger"]:
                    escape = self._escape_direction(zones, dz["name"])
                    if escape:
                        msg = f"Stop! Move {escape}."
                    else:
                        msg = "Stop! Do not move forward."
                    self._speak_cached(speaker, msg)
                    self._last["danger_alert"] = now
                    return

        # If paused for mic listening, skip non-critical speech
        if paused:
            for z in zones:
                self._prev_lvl[z["name"]] = z["level"]
            return

        # ── 2. YOLO object announcements ─────────────────────────────────────
        from modules.navigation.hailo_runner import HIGH_PRIORITY
        for det in detections:
            if det["high_priority"]:
                name = det["name"]
                if now - self._last_obj.get(name, 0) >= 6.0:
                    zone_name = ZONES[det["zone_idx"]][0]
                    zone_info = next((z for z in zones if z["name"] == zone_name), None)
                    dist_m = zone_info["distance_m"] if zone_info else 3.0
                    steps = self._steps(dist_m)
                    avoid = self._escape_direction(zones, zone_name)
                    if zone_name == "center":
                        pos = "ahead"
                    elif zone_name in ("left", "far left"):
                        pos = "to your left"
                    else:
                        pos = "to your right"
                    msg = f"{name} {steps} step{'s' if steps != 1 else ''} {pos}."
                    if avoid:
                        msg += f" Move {avoid}."
                    self._speak_cached(speaker, msg)
                    self._last_obj[name] = now
                    return

        # ── 3. Navigation guidance every 6 s ─────────────────────────────────
        if now - self._last_guide >= 6.0:
            guide = self._navigation_instruction(zones, detections, safe_dir)
            self._speak_cached(speaker, guide)
            self._last_guide = now
            return

        # ── 4. Zone-cleared reassurance ──────────────────────────────────────
        for z in zones:
            prev = self._prev_lvl.get(z["name"], "clear")
            if prev in ("danger", "warn") and z["level"] == "clear":
                key = z["name"] + "_clear"
                if now - self._last.get(key, 0) >= 8.0:
                    self._speak_cached(speaker, f"{z['name']} is now clear.")
                    self._last[key] = now

        for z in zones:
            self._prev_lvl[z["name"]] = z["level"]

    # ── helpers ─────────────────────────────────────────────────────────────

    def _navigation_instruction(self, zones: list, detections: list, safe_dir: str) -> str:
        zm         = {z["name"]: z for z in zones}
        center     = zm.get("center", {})
        left       = zm.get("left",   {})
        right      = zm.get("right",  {})
        far_left   = zm.get("far left",  {})
        far_right  = zm.get("far right", {})

        # YOLO object suffix (max 1 object to keep speech short)
        obj_suffix = ""
        for det in detections:
            if det.get("high_priority"):
                zn = ZONES[det["zone_idx"]][0]
                zi = next((z for z in zones if z["name"] == zn), None)
                d  = zi["distance_m"] if zi else 3.0
                s  = self._steps(d)
                pos = "ahead" if zn == "center" else ("to your left" if zn in ("left", "far left") else "to your right")
                obj_suffix = f". {det['name']} {s} step{'s' if s != 1 else ''} {pos}"
                break

        # All clear
        if all(z["level"] == "clear" for z in zones):
            return f"Forward path is clear{obj_suffix}"

        # Center clear and best
        if center.get("level") == "clear" and \
           center.get("distance_m", 0) >= left.get("distance_m", 0) and \
           center.get("distance_m", 0) >= right.get("distance_m", 0):
            return f"Walk straight{obj_suffix}"

        # Center blocked
        if center.get("level") in ("danger", "warn"):
            if left.get("distance_m", 0) > right.get("distance_m", 0) and \
               left.get("level") in ("clear", "notice"):
                return f"Move one step left{obj_suffix}"
            elif right.get("distance_m", 0) > left.get("distance_m", 0) and \
                 right.get("level") in ("clear", "notice"):
                return f"Move one step right{obj_suffix}"
            elif far_left.get("level") == "clear":
                return f"Move to your far left{obj_suffix}"
            elif far_right.get("level") == "clear":
                return f"Move to your far right{obj_suffix}"
            else:
                return f"Path blocked. Stop{obj_suffix}"

        # Sides imbalanced
        ld, rd = left.get("distance_m", 0), right.get("distance_m", 0)
        if ld > rd + 0.8:
            return f"Move slightly left{obj_suffix}"
        elif rd > ld + 0.8:
            return f"Move slightly right{obj_suffix}"

        # Fallback
        if safe_dir == "center":
            return f"Walk straight{obj_suffix}"
        return f"Move slightly {safe_dir}{obj_suffix}"

    @staticmethod
    def _steps(metres: float) -> int:
        return max(1, round(metres / 1.5))

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
        _depth_runner = HailoDepthRunner(_DEPTH_HEF_PATH)
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

    # Cached results for frame-skipping (depth every 2nd, yolo every 3rd)
    cached_depth_map  = None
    cached_zones      = []
    cached_safe_dir   = "center"
    cached_detections = []

    try:
        while not _stop_event.is_set():
            # ── Capture frame ───────────────────────────────────────────────
            try:
                bgr = cam.capture_array()
            except Exception as e:
                logger.warning(f"Navigation: capture error: {e}")
                time.sleep(0.02)
                continue

            if bgr is None:
                time.sleep(0.02)
                continue

            frame = bgr
            frame_count += 1

            # ── Depth estimation (every 2nd frame) ──────────────────────────
            if frame_count % 2 == 0 or cached_depth_map is None:
                depth_map = _depth_runner.estimate(frame)
                if depth_map is not None:
                    cached_depth_map = depth_map
                    cached_zones     = analyser.analyse(depth_map, _depth_runner)
                    cached_safe_dir  = analyser.safe_direction(cached_zones)

            # ── Object detection (every 3rd frame) ──────────────────────────
            if frame_count % 3 == 0 or not cached_detections:
                dets = _yolo_runner.detect(frame)
                if dets is not None:
                    cached_detections = dets

            # Use cached results for this frame
            depth_map  = cached_depth_map
            zones      = cached_zones
            safe_dir   = cached_safe_dir
            detections = cached_detections

            if depth_map is None:
                continue

            # ── Speech scheduler ────────────────────────────────────────────
            try:
                scheduler.update(zones, detections, safe_dir, speaker)
            except Exception as e:
                logger.warning(f"Navigation speech scheduler error: {e}")

            # ── Build display frame ─────────────────────────────────────────
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
