from __future__ import annotations

import os
from flask import Blueprint, jsonify

tts_blueprint = Blueprint("tts", __name__)


@tts_blueprint.get("/tts/voices")
def list_voices():
    """List available TTS voices.

    Prefers Kokoro voices (if kokoro_onnx is available), otherwise returns Piper default.
    """
    voices = []
    engine = os.getenv("TTS_ENGINE", "kokoro").lower()
    if engine in {"kokoro", "piper"}:  # try kokoro first
        try:
            import kokoro_onnx as K  # type: ignore
            model_dir = os.getenv("KOKORO_MODEL_DIR", "/app/cache/kokoro")
            model_file = os.getenv("KOKORO_MODEL_FILE", "kokoro-v1.0.onnx")
            voices_file = os.getenv("KOKORO_VOICES_FILE", "voices-v1.0.bin")
            mp = os.path.join(model_dir, model_file)
            vp = os.path.join(model_dir, voices_file)
            if os.path.isfile(mp) and os.path.isfile(vp):
                engine_inst = K.Kokoro(model_path=mp, voices_path=vp)
                for name in engine_inst.get_voices():
                    voices.append({"id": name, "styles": []})
        except Exception:
            pass
    if not voices:
        # Fallback: Piper baked voice
        voices = [{"id": os.getenv("PIPER_VOICE", "/app/voices/default.onnx"), "styles": []}]
    return jsonify({"voices": voices})
