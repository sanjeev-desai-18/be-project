#!/usr/bin/env python3
"""
diag_camera.py — Camera format diagnostic for blind_assistant
Run on the Raspberry Pi to verify what capture_array() actually returns.

Saves test images for BOTH the default format (XBGR8888) and explicit
RGB888, so you can compare which one gives correct colours.

Usage:
    python3 diag_camera.py

Output files:
    diag_default_raw.jpg     — default format, saved directly (no conversion)
    diag_default_rgb2bgr.jpg — default format, RGB→BGR conversion then save
    diag_rgb888_raw.jpg      — RGB888 format, saved directly (no conversion)
    diag_rgb888_rgb2bgr.jpg  — RGB888 format, RGB→BGR conversion then save

Check which images have CORRECT colours. That tells us the true channel
order of capture_array() on your system.
"""

import time
import cv2
import numpy as np

print("=" * 60)
print("Camera Format Diagnostic")
print("=" * 60)

try:
    from picamera2 import Picamera2
except ImportError:
    print("ERROR: picamera2 not installed")
    exit(1)


def test_format(label, config_fn):
    """Capture with a given config and save diagnostic images."""
    print(f"\n--- Testing: {label} ---")
    picam2 = Picamera2()
    try:
        config = config_fn(picam2)
        picam2.configure(config)
        picam2.start()
        time.sleep(2.0)  # generous warmup for AWB

        # discard frames
        for _ in range(5):
            picam2.capture_array("main")

        frame = picam2.capture_array("main")
        print(f"  Shape: {frame.shape}, dtype: {frame.dtype}")
        print(f"  Channels: {frame.shape[2] if frame.ndim == 3 else 1}")
        print(f"  Pixel[0,0] sample: {frame[100, 100]}")

        # Strip alpha if 4-channel
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame3 = frame[:, :, :3]
            print(f"  Stripped to 3 channels: {frame3.shape}")
        else:
            frame3 = frame

        tag = label.replace(" ", "_").lower()

        # Save RAW (no colour conversion) — imencode treats as BGR
        # If colours look correct here, capture_array gives BGR natively
        cv2.imwrite(f"diag_{tag}_raw.jpg", frame3)
        print(f"  Saved: diag_{tag}_raw.jpg  (no conversion)")

        # Save with RGB→BGR conversion
        # If colours look correct here, capture_array gives RGB natively
        converted = cv2.cvtColor(frame3, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"diag_{tag}_rgb2bgr.jpg", converted)
        print(f"  Saved: diag_{tag}_rgb2bgr.jpg  (RGB→BGR)")

    finally:
        picam2.stop()
        picam2.close()


# Test 1: Default format (no format specified — matches standalone script)
test_format("default", lambda cam: cam.create_preview_configuration(
    main={"size": (640, 640)}
))

# Brief pause between tests
time.sleep(1)

# Test 2: Explicit RGB888 (what the project was using)
test_format("rgb888", lambda cam: cam.create_preview_configuration(
    main={"size": (640, 640), "format": "RGB888"}
))

print("\n" + "=" * 60)
print("DONE! Check the 4 saved images:")
print("  - If *_raw.jpg has correct colours → capture gives BGR")
print("  - If *_rgb2bgr.jpg has correct colours → capture gives RGB")
print("  - Compare default vs rgb888 to see if format matters")
print("=" * 60)
