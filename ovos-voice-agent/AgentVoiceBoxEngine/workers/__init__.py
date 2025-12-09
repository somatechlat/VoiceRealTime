"""AgentVoiceBox Worker Services.

This package contains standalone worker services that consume from Redis Streams
and perform CPU/GPU-intensive tasks:

- STT Worker: Speech-to-text transcription using Faster-Whisper
- TTS Worker: Text-to-speech synthesis using Kokoro ONNX
- LLM Worker: Language model inference with multi-provider support
"""

__all__ = []
