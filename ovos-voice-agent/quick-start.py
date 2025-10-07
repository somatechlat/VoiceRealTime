#!/usr/bin/env python3
"""
Quick Start Script for OVOS Voice Agent
Starts both servers needed for the simple chat interface
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import websockets
        print("✅ Dependencies look good!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\n📦 Please install dependencies:")
        print("cd sprint4-websocket && pip install -r requirements.txt")
        return False

def start_websocket_server():
    """Start the WebSocket realtime server"""
    print("🚀 Starting WebSocket server on port 8001...")
    
    # Change to sprint4 directory
    sprint4_dir = Path(__file__).parent / "sprint4-websocket"
    
    if not sprint4_dir.exists():
        print(f"❌ Directory not found: {sprint4_dir}")
        return None
        
    # Start the server
    try:
        process = subprocess.Popen([
            sys.executable, "realtime_server.py"
        ], cwd=sprint4_dir)
        return process
    except Exception as e:
        print(f"❌ Failed to start WebSocket server: {e}")
        return None

def main():
    print("🎤 OVOS Voice Agent - Quick Start")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("sprint4-websocket").exists():
        print("❌ Please run this script from the ovos-voice-agent directory")
        print("   Current directory should contain: sprint4-websocket/")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start WebSocket server
    ws_process = start_websocket_server()
    if not ws_process:
        sys.exit(1)
    
    print("✅ WebSocket server started!")
    print("\n🌐 Ready to test!")
    print("=" * 40)
    print("1. Open 'simple-chat.html' in your browser")
    print("2. Click 'Connect to OVOS'") 
    print("3. Click the 🎤 button and start talking!")
    print("4. Click ⏹️ to stop and get a response")
    print("\n🔗 Direct link: file://" + str(Path(__file__).parent / "simple-chat.html"))
    print("\n⏹️  Press Ctrl+C to stop servers")
    
    try:
        # Wait for interrupt
        ws_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        ws_process.terminate()
        print("✅ Servers stopped!")

if __name__ == "__main__":
    main()