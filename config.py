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
VLM_MAX_TOKENS      = 200
CAMERA_INDEX        = 0
CAMERA_WARMUP_MS    = 500

# ── TTS ───────────────────────────────────────────────────────────────────────
# Use "gtts" for offline / free, "elevenlabs" for higher quality
TTS_ENGINE          = "gtts"
TTS_LANGUAGE        = "en"
TTS_SLOW            = False

# ── Currency / Hailo 8 ────────────────────────────────────────────────────────
HAILO_HEF_PATH      = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "modules", "currency", "yolov11s_currency.hef"
)
CURRENCY_CONFIDENCE = 0.50
CURRENCY_MAX_FPS    = 30
