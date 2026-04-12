"""
config.py — Blind Assistant configuration for Raspberry Pi 5 + Hailo 8
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "Rachel")

# ── STT (Groq Whisper) ────────────────────────────────────────────────────────
SAMPLE_RATE         = 44100      # USB mic native rate; resampled to 16k for Whisper
SILENCE_THRESHOLD   = 2.0        # seconds of silence before recording ends

# ── Agent (Groq LLM for routing) ──────────────────────────────────────────────
AGENT_MODEL         = "llama-3.1-8b-instant"
AGENT_TEMPERATURE   = 0.1

# ── Confidence Thresholds ─────────────────────────────────────────────────────
CONFIDENCE_HIGH     = 0.75
CONFIDENCE_MEDIUM   = 0.50

# ── Vision / VLM (Groq vision model) ─────────────────────────────────────────
VLM_MODEL           = "meta-llama/llama-4-scout-17b-16e-instruct"
VLM_MAX_TOKENS      = 300        # slightly higher to allow complete sentences
CAMERA_INDEX        = 0
CAMERA_WARMUP_MS    = 500

# ── TTS ───────────────────────────────────────────────────────────────────────
# Options:
#   "pyttsx3"    — offline, zero-latency (espeak-ng on Pi). Recommended.
#                  Install: pip install pyttsx3 && sudo apt install espeak-ng
#   "gtts"       — Google TTS, online, saves to disk first (~1-3s extra)
#   "elevenlabs" — highest quality, requires API key
# Options:
#   "piper"      — RECOMMENDED. Offline neural TTS, <200ms latency, natural voice.
#                  pip install piper-tts
#                  Download voice: en_US-lessac-medium.onnx + .onnx.json
#                  https://github.com/rhasspy/piper/blob/master/VOICES.md
#   "gtts"       — Google Translate TTS, online, ~1-2s network delay per call
#   "pyttsx3"    — espeak-ng, offline, robotic
#   "elevenlabs" — highest quality, requires API key
TTS_ENGINE          = "gtts"     # "gtts" now uses edge-tts (Microsoft neural) under the hood
TTS_LANGUAGE        = "en"
TTS_SLOW            = False

# edge-tts settings (used when TTS_ENGINE = "gtts" — edge-tts replaces gTTS)
# Voice options (natural, free, no API key):
#   "en-US-JennyNeural"   — warm, clear American female (recommended)
#   "en-US-GuyNeural"     — natural American male
#   "en-GB-SoniaNeural"   — British female
#   "en-IN-NeerjaNeural"  — Indian English female
# Rate: "+0%" normal, "+10%" faster, "-10%" slower
EDGE_TTS_VOICE      = "en-US-JennyNeural"
EDGE_TTS_RATE       = "+0%"

# Piper settings (only used when TTS_ENGINE = "piper")
# Path to the .onnx model file. Download from:
# https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium
PIPER_MODEL         = "/home/pi/piper-voices/en_US-lessac-medium.onnx"
PIPER_LENGTH_SCALE  = 1.15       # slightly slower — more natural cadence        # speech speed: >1.0 = slower, <1.0 = faster

# pyttsx3 settings (only used when TTS_ENGINE = "pyttsx3")
PYTTSX3_RATE        = 165        # words per minute — lower = slower/clearer
PYTTSX3_VOLUME      = 1.0        # 0.0 to 1.0

# ── Currency / Hailo 8 ────────────────────────────────────────────────────────
HAILO_HEF_PATH      = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "modules", "currency", "yolov11s_currency.hef"
)
CURRENCY_CONFIDENCE = 0.8
CURRENCY_MAX_FPS    = 30
