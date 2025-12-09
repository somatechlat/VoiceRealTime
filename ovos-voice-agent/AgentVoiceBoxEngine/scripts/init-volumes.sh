#!/bin/bash
# Initialize persistent volume directories for Docker Compose
# Run this before first `docker compose up`
#
# Creates data directories with proper permissions for:
# - Redis (AOF/RDB persistence)
# - PostgreSQL (data + WAL)
# - Whisper model cache
# - Kokoro TTS models
# - TTS synthesis cache

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_DIR}/data"

echo "Initializing volume directories in ${DATA_DIR}..."

# Create directories
mkdir -p "${DATA_DIR}/redis"
mkdir -p "${DATA_DIR}/postgres"
mkdir -p "${DATA_DIR}/whisper"
mkdir -p "${DATA_DIR}/kokoro"
mkdir -p "${DATA_DIR}/tts-cache"

# Set permissions (Redis and PostgreSQL need specific UIDs)
# Redis runs as redis user (UID 999 in alpine)
# PostgreSQL runs as postgres user (UID 70 in alpine)
chmod 755 "${DATA_DIR}/redis"
chmod 755 "${DATA_DIR}/postgres"
chmod 755 "${DATA_DIR}/whisper"
chmod 755 "${DATA_DIR}/kokoro"
chmod 755 "${DATA_DIR}/tts-cache"

echo "Volume directories initialized:"
ls -la "${DATA_DIR}"

echo ""
echo "Ready to start services with:"
echo "  docker compose -f docker-compose.test.yml up -d"
