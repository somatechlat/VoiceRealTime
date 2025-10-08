#!/bin/bash
# Quick start script for OVOS Realtime WebSocket Server

echo "🎤 Starting OVOS Realtime Voice Server..."
echo ""

cd "$(dirname "$0")/sprint4-websocket"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Install sprint2 speech dependencies
echo "Installing speech processing dependencies..."
pip install -q -r ../sprint2-speech/requirements.txt

echo ""
echo "✅ Starting server on http://localhost:8765"
echo "✅ WebSocket endpoint: ws://localhost:8765/v1/realtime"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run the server
python3 realtime_server.py
