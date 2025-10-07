#!/usr/bin/env python3
"""
OVOS Voice Agent - Sprint 1: Real-time Streaming Server
FastAPI + WebSocket server for real-time audio processing
"""

import asyncio
import json
import logging
from typing import Dict, Set
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import redis.asyncio as redis
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OVOS Voice Agent Server",
    description="Real-time speech-to-speech voice agent server",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_data: Dict[str, dict] = {}
        
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_data[session_id] = {
            "connected_at": datetime.now().isoformat(),
            "audio_buffer": [],
            "state": "idle"
        }
        logger.info(f"Client {session_id} connected")
        
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_data:
            del self.session_data[session_id]
        logger.info(f"Client {session_id} disconnected")
        
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            await websocket.send_text(json.dumps(message))
            
    async def broadcast(self, message: dict):
        for session_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except:
                logger.error(f"Failed to send to {session_id}")

manager = ConnectionManager()

# Redis connection for session persistence
async def get_redis():
    return await redis.from_url("redis://localhost:6379", decode_responses=True)

# Audio buffer management
class AudioBuffer:
    def __init__(self, max_size: int = 1024 * 1024):  # 1MB buffer
        self.buffer = bytearray()
        self.max_size = max_size
        
    def append(self, data: bytes):
        self.buffer.extend(data)
        if len(self.buffer) > self.max_size:
            # Keep only the last max_size bytes
            self.buffer = self.buffer[-self.max_size:]
            
    def get_and_clear(self) -> bytes:
        data = bytes(self.buffer)
        self.buffer.clear()
        return data
        
    def get_size(self) -> int:
        return len(self.buffer)

# Session management
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
        
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
            "audio_buffer": AudioBuffer(),
            "conversation_state": "idle",
            "context": []
        }
        return session_id
        
    def get_session(self, session_id: str) -> dict:
        return self.sessions.get(session_id)
        
    def update_session_state(self, session_id: str, state: str):
        if session_id in self.sessions:
            self.sessions[session_id]["conversation_state"] = state
            
    def cleanup_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

session_manager = SessionManager()

# API Models
class SessionCreate(BaseModel):
    user_id: str = None
    config: dict = {}

class AudioMessage(BaseModel):
    type: str
    session_id: str
    audio_data: str = None
    metadata: dict = {}

# REST API Endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(manager.active_connections)
    }

@app.post("/sessions")
async def create_session(session_data: SessionCreate):
    session_id = session_manager.create_session()
    return {
        "session_id": session_id,
        "status": "created",
        "websocket_url": f"/ws/{session_id}"
    }

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Don't return the audio buffer in the response
    response_session = {k: v for k, v in session.items() if k != "audio_buffer"}
    response_session["buffer_size"] = session["audio_buffer"].get_size()
    return response_session

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    session_manager.cleanup_session(session_id)
    manager.disconnect(session_id)
    return {"status": "deleted"}

# WebSocket endpoint for real-time audio
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    # Verify session exists
    session = session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=1008, reason="Session not found")
        return
        
    await manager.connect(websocket, session_id)
    
    try:
        # Send connection confirmation
        await manager.send_message(session_id, {
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            await handle_websocket_message(session_id, message)
            
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        manager.disconnect(session_id)

async def handle_websocket_message(session_id: str, message: dict):
    """Handle incoming WebSocket messages"""
    message_type = message.get("type")
    session = session_manager.get_session(session_id)
    
    if not session:
        return
        
    try:
        if message_type == "audio_chunk":
            # Handle incoming audio data
            audio_data = message.get("audio_data", "")
            if audio_data:
                # Decode base64 audio data
                import base64
                audio_bytes = base64.b64decode(audio_data)
                session["audio_buffer"].append(audio_bytes)
                
                # Update session state
                session_manager.update_session_state(session_id, "receiving_audio")
                
                # Echo back for now (will be replaced with STT processing)
                await manager.send_message(session_id, {
                    "type": "audio_received",
                    "buffer_size": session["audio_buffer"].get_size(),
                    "timestamp": datetime.now().isoformat()
                })
                
        elif message_type == "start_conversation":
            session_manager.update_session_state(session_id, "conversation_active")
            await manager.send_message(session_id, {
                "type": "conversation_started",
                "message": "Ready to receive audio",
                "timestamp": datetime.now().isoformat()
            })
            
        elif message_type == "end_conversation":
            session_manager.update_session_state(session_id, "idle")
            # Clear audio buffer
            session["audio_buffer"].get_and_clear()
            await manager.send_message(session_id, {
                "type": "conversation_ended",
                "message": "Conversation terminated",
                "timestamp": datetime.now().isoformat()
            })
            
        elif message_type == "ping":
            await manager.send_message(session_id, {
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            })
            
        else:
            logger.warning(f"Unknown message type: {message_type}")
            
    except Exception as e:
        logger.error(f"Error handling message {message_type}: {e}")
        await manager.send_message(session_id, {
            "type": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
