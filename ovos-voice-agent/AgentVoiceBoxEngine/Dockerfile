# syntax=docker/dockerfile:1.7

# -----------------------------
# Builder stage installs Python dependencies
# -----------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install build dependencies
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Leverage cached wheels when requirements do not change
COPY enterprise/requirements-enterprise.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip wheel --wheel-dir /tmp/wheels --no-cache-dir -r requirements.txt

# -----------------------------
# Runtime image with non-root user
# -----------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    PORT=8000 \
    KOKORO_MODEL_DIR=/app/cache/kokoro \
    KOKORO_MODEL_FILE=kokoro-v1.0.onnx \
    KOKORO_VOICES_FILE=voices-v1.0.bin \
    KOKORO_MODEL_URL=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx \
    KOKORO_VOICES_URL=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

WORKDIR ${APP_HOME}

# System deps for runtime (e.g., certificates)
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl espeak-ng libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --system ovos && useradd --system --create-home --gid ovos ovos

# Copy Python wheels from builder and install
COPY --from=builder /tmp/wheels /tmp/wheels
COPY enterprise/requirements-enterprise.txt ./requirements.txt
RUN pip install --no-cache-dir --find-links=/tmp/wheels -r requirements.txt \
    && rm -rf /tmp/wheels ./requirements.txt
# Explicitly install gunicorn and gevent for WebSocket support
RUN pip install --no-cache-dir gunicorn gevent \
    && pip install --no-cache-dir onnxruntime \
    && pip install --no-cache-dir kokoro-tts kokoro-onnx soundfile numpy

# Copy application source
COPY enterprise /app
USER root
# Prefetch Kokoro model and voices into a writable cache so runtime can use kokoro_onnx without falling back
USER root
# Prefetch Kokoro model and voices into a writable cache so runtime can use kokoro_onnx without falling back
RUN set -e \
    && mkdir -p "${KOKORO_MODEL_DIR}" \
    && curl -fL --retry 5 --retry-delay 5 "$KOKORO_MODEL_URL" -o "${KOKORO_MODEL_DIR}/${KOKORO_MODEL_FILE}" \
    && curl -fL --retry 5 --retry-delay 5 "$KOKORO_VOICES_URL" -o "${KOKORO_MODEL_DIR}/${KOKORO_VOICES_FILE}" \
    && ls -lh "${KOKORO_MODEL_DIR}"

# Also include top-level LLM integration module so it can be imported at runtime
COPY llm_integration.py /app/llm_integration.py
