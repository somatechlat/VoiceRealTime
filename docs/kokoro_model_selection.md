# Kokoro Model Selection

The Voice Engine now uses the **medium-sized** Kokoro model (`Kokoro-82M.onnx`) by default. You can switch to a different model (e.g., a larger model) without changing the code.

## Environment Variable

- **`KOKORO_MODEL_FILE`** – Name of the ONNX model file to load from the cache directory (`/app/cache/kokoro`).
  - Default: `Kokoro-82M.onnx` (the 82‑M parameter medium model).
  - Example to use a larger model:
    ```bash
    export KOKORO_MODEL_FILE=Kokoro-200M.onnx
    ```

The variable is read in three places:

1. **Dockerfile** – during the image build the model is copied to the target name defined by `KOKORO_MODEL_FILE`.
2. **Realtime WebSocket transport** – loads the model file using the same variable.
3. **TTS route** – resolves the model path for the HTTP endpoint.

## How to Provide a New Model
1. Place the ONNX model file (e.g., `Kokoro-200M.onnx`) in a location accessible during the Docker build, such as the local cache or a volume mount.
2. Set `KOKORO_MODEL_FILE` in your `.env` (or export it in the shell) before rebuilding the containers.
3. Rebuild and restart the compose stack.

## Rebuilding the Stack
```bash
# From the repository root
cd ovos-docker/compose
# Rebuild images with the new configuration
docker compose -f voice-agent-compose.yml build --no-cache
# Start the services in detached mode
docker compose -f voice-agent-compose.yml up -d
```

The service will expose the health endpoint at `http://localhost:<VOICE_AGENT_PORT>/health` (default `60200`).

## Verifying the Model
You can check the logs of the `voice-agent` container to see which model file was loaded:
```bash
docker logs ovos_voice_agent 2>&1 | grep Kokoro-.*\.onnx
```
If the correct model name appears, the configuration is successful.
