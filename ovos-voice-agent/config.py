"""Centralized configuration for OVOS voice agent.

Read environment variables and expose defaults to be used across the codebase.
This keeps URLs, ports and secret names in one place and avoids hard-coded values.
"""
from pathlib import Path
import os

# Server network
VOICE_AGENT_HOST = os.getenv("VOICE_AGENT_HOST", "localhost")
VOICE_AGENT_PORT = int(os.getenv("VOICE_AGENT_PORT", os.getenv("PORT", "60200")))
VOICE_AGENT_BASE = os.getenv("VOICE_AGENT_BASE", f"http://{VOICE_AGENT_HOST}:{VOICE_AGENT_PORT}")
VOICE_AGENT_WS_BASE = os.getenv("VOICE_AGENT_WS_BASE", f"ws://{VOICE_AGENT_HOST}:{VOICE_AGENT_PORT}")

# LLM / API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss-20b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# TTS / Kokoro
TTS_ENGINE = os.getenv("TTS_ENGINE", "kokoro")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "am_onyx")
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", os.getenv("KOKORO_SPEED", "1.1")))
KOKORO_MODEL_DIR = Path(os.getenv("KOKORO_MODEL_DIR", str(Path.cwd() / "cache" / "kokoro"))).expanduser()
KOKORO_MODEL_FILE = os.getenv("KOKORO_MODEL_FILE", "kokoro-v1.0.onnx")
KOKORO_VOICES_FILE = os.getenv("KOKORO_VOICES_FILE", "voices-v1.0.bin")
KOKORO_MODEL_URL = os.getenv("KOKORO_MODEL_URL", "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx")
KOKORO_VOICES_URL = os.getenv("KOKORO_VOICES_URL", "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin")

# Runtime defaults
DEFAULT_HOST = VOICE_AGENT_HOST
DEFAULT_PORT = VOICE_AGENT_PORT

def voice_agent_base():
    return VOICE_AGENT_BASE

def voice_agent_ws_base():
    return VOICE_AGENT_WS_BASE
