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
    PORT=8000

WORKDIR ${APP_HOME}

# System deps for runtime (e.g., certificates)
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --system ovos && useradd --system --create-home --gid ovos ovos

# Copy Python wheels from builder and install
COPY --from=builder /tmp/wheels /tmp/wheels
COPY enterprise/requirements-enterprise.txt ./requirements.txt
RUN pip install --no-cache-dir --find-links=/tmp/wheels -r requirements.txt \
    && rm -rf /tmp/wheels ./requirements.txt
# Explicitly install gunicorn and gevent for WebSocket support
RUN pip install gunicorn gevent

# Copy application source
COPY enterprise /app

# Ensure runtime directories exist and permissions set
RUN mkdir -p /app/logs /app/data \
    && chown -R ovos:ovos /app

USER ovos

EXPOSE ${PORT}

ENV FLASK_ENV=production

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--worker-class", "gevent", "--workers", "1", "--timeout", "300", "app:create_app()"]
