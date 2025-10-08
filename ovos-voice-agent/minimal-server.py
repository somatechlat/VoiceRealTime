#!/usr/bin/env python3
"""
OVOS Voice Agent - Minimal Test Server
Simplified version for quick testing without heavy dependencies
"""

import asyncio
import json
import logging
import base64
import uuid
from typing import Dict, Set
from datetime import datetime
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OVOS Realtime Voice API - Minimal",
    description="Simplified OpenAI-compatible WebSocket voice processing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import LLM integration
import sys
sys.path.append('../')
from llm_integration import generate_ai_response

class MinimalConnectionManager:
    """Simplified connection manager"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_data: Dict[str, dict] = {}
        
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_data[session_id] = {
            "id": session_id,
            "created_at": int(time.time()),
            "status": "connected"
        }
        
        logger.info(f"Session {session_id} connected")
        
        # Send session.created event
        await self.send_event(session_id, {
            "type": "session.created",
            "session": {
                "id": session_id,
                "object": "realtime.session",
                "model": "ovos-voice-1",
                "created_at": int(time.time())
            }
        })
    
    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)
        self.session_data.pop(session_id, None)
        logger.info(f"Session {session_id} disconnected")
    
    async def send_event(self, session_id: str, event: dict):
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_text(json.dumps(event))
            except Exception as e:
                logger.error(f"Failed to send event to {session_id}: {e}")

# Global connection manager
connection_manager = MinimalConnectionManager()

class MinimalEventProcessor:
    """Simplified event processor"""
    
    def __init__(self, manager):
        self.manager = manager
        self.conversations: Dict[str, list] = {}
    
    async def process_event(self, session_id: str, event: dict):
        event_type = event.get("type")
        logger.info(f"Processing event: {event_type}")
        
        try:
            if event_type == "session.update":
                await self.handle_session_update(session_id, event)
            elif event_type == "input_audio_buffer.append":
                await self.handle_audio_append(session_id, event)
            elif event_type == "input_audio_buffer.commit":
                await self.handle_audio_commit(session_id, event)
            elif event_type == "conversation.item.create":
                await self.handle_conversation_item(session_id, event)
            elif event_type == "response.create":
                await self.handle_response_create(session_id, event)
            else:
                logger.warning(f"Unknown event type: {event_type}")
                
        except Exception as e:
            logger.error(f"Error processing {event_type}: {e}")
            await self.send_error(session_id, "processing_error", str(e))
    
    async def handle_session_update(self, session_id: str, event: dict):
        await self.manager.send_event(session_id, {
            "type": "session.updated",
            "session": self.manager.session_data.get(session_id, {})
        })
    
    async def handle_audio_append(self, session_id: str, event: dict):
        # Simulate audio processing
        await self.manager.send_event(session_id, {
            "type": "input_audio_buffer.speech_started",
            "audio_start_ms": int(time.time() * 1000),
            "item_id": f"item_{uuid.uuid4().hex[:16]}"
        })
    
    async def handle_audio_commit(self, session_id: str, event: dict):
        # Simulate transcription
        item_id = f"item_{uuid.uuid4().hex[:16]}"
        
        # Fake transcription for demo
        fake_transcription = "Hello, this is a test transcription from the audio you just sent."
        
        # Create conversation item
        conversation_item = {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "status": "completed",
            "role": "user",
            "content": [{
                "type": "input_audio",
                "transcript": fake_transcription
            }]
        }
        
        # Store conversation
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        self.conversations[session_id].append(conversation_item)
        
        await self.manager.send_event(session_id, {
            "type": "conversation.item.created",
            "item": conversation_item
        })
        
        await self.manager.send_event(session_id, {
            "type": "input_audio_buffer.speech_stopped",
            "audio_end_ms": int(time.time() * 1000),
            "item_id": item_id
        })
    
    async def handle_conversation_item(self, session_id: str, event: dict):
        item = event.get("item", {})
        
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        self.conversations[session_id].append(item)
        
        await self.manager.send_event(session_id, {
            "type": "conversation.item.created",
            "item": item
        })
    
    async def handle_response_create(self, session_id: str, event: dict):
        # Generate AI response
        await self.generate_response(session_id)
    
    async def generate_response(self, session_id: str):
        try:
            response_id = f"resp_{uuid.uuid4().hex[:16]}"
            item_id = f"item_{uuid.uuid4().hex[:16]}"
            
            # Get last user message
            conversation = self.conversations.get(session_id, [])
            user_text = "Hello"  # Default
            
            for item in reversed(conversation):
                if item.get("role") == "user":
                    content = item.get("content", [])
                    if content:
                        user_text = content[0].get("transcript") or content[0].get("text", "Hello")
                    break
            
            # Send response.created
            await self.manager.send_event(session_id, {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "realtime.response",
                    "status": "in_progress"
                }
            })
            
            # Generate AI response using Groq
            try:
                ai_response = await generate_ai_response(session_id, user_text)
            except Exception as e:
                logger.error(f"AI generation failed: {e}")
                ai_response = f"I heard you say: {user_text}. How can I help you with that?"
            
            # Create response item
            response_item = {
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "audio",
                    "transcript": ai_response  # Browser looks for 'transcript' field
                }]
            }
            
            # Send response events
            await self.manager.send_event(session_id, {
                "type": "response.output_item.added",
                "response_id": response_id,
                "output_index": 0,
                "item": response_item
            })

            # Generate fake audio response (simple beep pattern)
            # In a real implementation, this would use TTS
            import base64
            
            # Create a simple audio buffer (silence/beep pattern)
            fake_audio_data = b'\x00' * 1024  # 1KB of silence
            audio_b64 = base64.b64encode(fake_audio_data).decode('utf-8')
            
            # Send audio response
            await self.manager.send_event(session_id, {
                "type": "response.audio.delta",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": audio_b64
            })
            
            # Send audio done
            await self.manager.send_event(session_id, {
                "type": "response.audio.done",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0
            })

            await self.manager.send_event(session_id, {
                "type": "response.done",
                "response": {
                    "id": response_id,
                    "object": "realtime.response",
                    "status": "completed",
                    "output": [response_item]
                }
            })            # Store response in conversation
            self.conversations[session_id].append(response_item)
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            await self.send_error(session_id, "response_error", str(e))
    
    async def send_error(self, session_id: str, error_type: str, message: str):
        await self.manager.send_event(session_id, {
            "type": "error",
            "error": {
                "type": error_type,
                "message": message
            }
        })

# Global event processor
event_processor = MinimalEventProcessor(connection_manager)

@app.websocket("/v1/realtime")
async def websocket_realtime(websocket: WebSocket):
    """Simplified realtime WebSocket endpoint"""
    session_id = f"sess_{uuid.uuid4().hex[:24]}"
    
    try:
        await connection_manager.connect(websocket, session_id)
        
        while True:
            try:
                data = await websocket.receive_text()
                event = json.loads(data)
                await event_processor.process_event(session_id, event)
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
            except Exception as e:
                logger.error(f"Processing error: {e}")
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connection_manager.disconnect(session_id)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_connections": len(connection_manager.active_connections),
        "timestamp": datetime.now().isoformat(),
        "version": "minimal-1.0"
    }

if __name__ == "__main__":
    print("🎤 OVOS Minimal Voice Agent - Starting...")
    print("🌐 WebSocket: ws://localhost:65000/v1/realtime")
    print("📋 Health: http://localhost:65000/health")
    print("✅ Ready for voice chat testing!")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=65000,
        log_level="info"
    )