"""TTS provider abstraction.

The real‑time WebSocket handler (`realtime_ws.py`) needs a clean way to
select an engine (Kokoro, Piper or Espeak) and stream audio chunks.  This
module defines a tiny plug‑in system:

* ``TTSProvider`` – abstract base class with ``synthesize`` that yields
  base‑64‑encoded ``audio/wav`` chunks.
* ``KokoroProvider`` – streams sentence‑by‑sentence using the Kokoro ONNX
  model.  It respects ``voice`` and ``speed`` arguments and checks the
  ``_cancel_current`` flag on the ``RealtimeWS`` instance.
* ``PiperProvider`` – falls back to the ``piper`` binary.
* ``EspeakProvider`` – final fallback using ``espeak-ng``.

Each provider is registered in ``PROVIDERS`` and selected via the
``TTS_ENGINE`` environment variable.  The ``get_provider`` helper returns
an instantiated provider ready for use.
"""

from __future__ import annotations

import base64
import io
import os
import subprocess
import tempfile

# Standard library imports – sorted alphabetically per PEP8
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, Dict


class TTSProvider(ABC):
    """Abstract TTS provider.

    Implementations must yield *base64* strings that represent a WAV file.
    The caller is responsible for wrapping the string in a ``response.audio.delta``
    event.
    """

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        cancel_flag: Any = None,
    ) -> AsyncGenerator[str, None]:
        """Yield base64‑encoded WAV chunks.

        ``cancel_flag`` is a mutable object (e.g., ``RealtimeWS``) that
        provides a ``_cancel_current`` attribute.  The provider should break
        the iteration as soon as the flag becomes ``True``.
        """
        ...


class KokoroProvider(TTSProvider):
    def __init__(self) -> None:
        # Lazy import – will raise ImportError if kokoro is not installed.
        import kokoro_onnx as K  # type: ignore

        self._K = K
        model_dir = os.getenv("KOKORO_MODEL_DIR", "/app/cache/kokoro")
        model_file = os.getenv("KOKORO_MODEL_FILE", "kokoro-v1.0.onnx")
        voices_file = os.getenv("KOKORO_VOICES_FILE", "voices-v1.0.bin")
        self._model_path = os.path.join(model_dir, model_file)
        self._voices_path = os.path.join(model_dir, voices_file)
        self._engine = None  # type: ignore
        self._load_engine()

    def _load_engine(self) -> None:
        if os.path.isfile(self._model_path) and os.path.isfile(self._voices_path):
            try:
                self._engine = self._K.Kokoro(
                    model_path=self._model_path,
                    voices_path=self._voices_path,
                )
            except Exception:
                self._engine = None
        else:
            self._engine = None

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        cancel_flag: Any = None,
    ) -> AsyncGenerator[str, None]:
        import soundfile as sf  # type: ignore

        if self._engine is None:
            return

        voice_name = voice or os.getenv("KOKORO_VOICE", "am_onyx")
        speed_val = float(speed or os.getenv("KOKORO_SPEED", "1.1"))

        try:
            voice_list = self._engine.get_voices()
            if voice_name not in voice_list:
                voice_name = voice_list[0] if voice_list else voice_name
        except Exception:
            pass

        async for audio_arr, sr in self._engine.create_stream(
            text=text,
            voice=voice_name,
            speed=speed_val,
        ):
            if getattr(cancel_flag, "_cancel_current", False):
                break
            buf = io.BytesIO()
            sf.write(buf, audio_arr, sr, format="WAV")
            yield base64.b64encode(buf.getvalue()).decode("utf-8")


class PiperProvider(TTSProvider):
    def __init__(self) -> None:
        self._piper_bin = os.getenv("PIPER_BIN", "/usr/local/bin/piper")
        self._default_voice = os.getenv("PIPER_VOICE", "/app/voices/en_US-amy-medium.onnx")

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        cancel_flag: Any = None,
    ) -> AsyncGenerator[str, None]:
        if not os.path.isfile(self._piper_bin) or not os.access(self._piper_bin, os.X_OK):
            return
        voice_path = voice or self._default_voice
        if not os.path.isfile(voice_path):
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_path = tf.name
        try:
            subprocess.run(
                [self._piper_bin, "-m", voice_path, "-f", tmp_path],
                input=text.encode("utf-8"),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open(tmp_path, "rb") as f:
                data = f.read()
            yield base64.b64encode(data).decode("utf-8")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


class EspeakProvider(TTSProvider):
    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        cancel_flag: Any = None,
    ) -> AsyncGenerator[str, None]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_path = tf.name
        try:
            cmd = [
                "espeak-ng",
                "-w",
                tmp_path,
                "-s",
                str(int((speed or 1.0) * 180)),
                "-v",
                "en-us",
                text,
            ]
            subprocess.run(cmd, check=True)
            with open(tmp_path, "rb") as f:
                data = f.read()
            yield base64.b64encode(data).decode("utf-8")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


PROVIDERS: Dict[str, type[TTSProvider]] = {
    "kokoro": KokoroProvider,
    "piper": PiperProvider,
    "espeak": EspeakProvider,
}


def get_provider() -> TTSProvider:
    """Factory that returns a concrete ``TTSProvider`` based on ``TTS_ENGINE``.

    If the requested engine cannot be instantiated (e.g., missing binary), the
    function falls back to the next engine in the priority list ``kokoro →
    piper → espeak``.
    """
    engine = os.getenv("TTS_ENGINE", "kokoro").lower()
    for candidate in [engine, "kokoro", "piper", "espeak"]:
        provider_cls = PROVIDERS.get(candidate)
        if provider_cls is None:
            continue
        try:
            return provider_cls()
        except Exception:
            continue

    class NullProvider(TTSProvider):
        async def synthesize(self, *_, **__) -> AsyncGenerator[str, None]:
            if False:
                yield ""

    return NullProvider()
