#!/usr/bin/env python3
"""
OVOS Voice Agent - Sprint 4: WebSocket Realtime Protocol
OpenAI-compatible WebSocket server for real-time voice processing
"""

import asyncio
import json
import logging
import base64
import uuid
from typing import Dict, Any, Optional, Set
from datetime import datetime, timezone
import time
import wave
import io

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import enhanced components
import sys
sys.path.append('../sprint2-speech')
sys.path.append('../sprint3-api')
sys.path.append('../')
from speech_pipeline import create_realtime_pipeline, SpeechPipeline
from llm_integration import generate_ai_response, clear_session_memory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OVOS Realtime Voice API",
    description="OpenAI-compatible WebSocket realtime voice processing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global connection management
class RealtimeConnectionManager:
    """Manages WebSocket connections and sessions"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_data: Dict[str, dict] = {}
        self.speech_pipelines: Dict[str, SpeechPipeline] = {}
        self.conversation_states: Dict[str, dict] = {}
        
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept WebSocket connection and initialize session"""
        await websocket.accept()
        
        # Store connection
        self.active_connections[session_id] = websocket
        
        # Initialize session data
        self.session_data[session_id] = {
            "id": session_id,
            "object": "realtime.session",
            "model": "ovos-voice-1",
            "created_at": int(time.time()),
            "status": "connected",
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500
            },
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "voice": "default",
            "temperature": 0.8,
            "max_response_output_tokens": "inf"
        }
        
        # Initialize conversation state
        self.conversation_states[session_id] = {
            "items": [],
            "response": None,
            "input_audio_buffer": b"",
            "output_audio_buffer": b"",
            "conversation_id": f"conv_{uuid.uuid4().hex[:16]}"
        }
        
        # Create speech pipeline
        pipeline = create_realtime_pipeline(session_id)
        self.speech_pipelines[session_id] = pipeline
        
        logger.info(f"WebSocket connected for session {session_id}")
        
        # Send session.created event
        await self.send_event(session_id, {
            "type": "session.created",
            "session": self.session_data[session_id]
        })
    
    def disconnect(self, session_id: str):
        """Cleanup connection and session data"""
        # Cleanup WebSocket connection
        self.active_connections.pop(session_id, None)
        
        # Cleanup session data
        self.session_data.pop(session_id, None)
        self.conversation_states.pop(session_id, None)
        
        # Cleanup speech pipeline
        pipeline = self.speech_pipelines.pop(session_id, None)
        if pipeline:
            pipeline.reset_session()
            
        logger.info(f"WebSocket disconnected for session {session_id}")
    
    async def send_event(self, session_id: str, event: dict):
        """Send event to specific session"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_text(json.dumps(event))
            except Exception as e:
                logger.error(f"Failed to send event to {session_id}: {e}")
    
    async def broadcast_event(self, event: dict, exclude: Set[str] = None):
        """Broadcast event to all connected sessions"""
        exclude = exclude or set()
        
        for session_id, websocket in self.active_connections.items():
            if session_id not in exclude:
                try:
                    await websocket.send_text(json.dumps(event))
                except Exception as e:
                    logger.error(f"Failed to broadcast to {session_id}: {e}")

# Global connection manager
connection_manager = RealtimeConnectionManager()

class RealtimeEventProcessor:
    """Processes OpenAI Realtime API events"""
    
    def __init__(self, manager: RealtimeConnectionManager):
        self.manager = manager
    
    async def process_event(self, session_id: str, event: dict):
        """Process incoming WebSocket event"""
        event_type = event.get("type")
        
        if not event_type:
            await self.send_error(session_id, "missing_event_type", "Event type is required")
            return
        
        try:
            # Route to appropriate handler
            if event_type == "session.update":
                await self.handle_session_update(session_id, event)
            elif event_type == "input_audio_buffer.append":
                await self.handle_audio_buffer_append(session_id, event)
            elif event_type == "input_audio_buffer.commit":
                await self.handle_audio_buffer_commit(session_id, event)
            elif event_type == "input_audio_buffer.clear":
                await self.handle_audio_buffer_clear(session_id, event)
            elif event_type == "conversation.item.create":
                await self.handle_conversation_item_create(session_id, event)
            elif event_type == "conversation.item.truncate":
                await self.handle_conversation_item_truncate(session_id, event)
            elif event_type == "conversation.item.delete":
                await self.handle_conversation_item_delete(session_id, event)
            elif event_type == "response.create":
                await self.handle_response_create(session_id, event)
            elif event_type == "response.cancel":
                await self.handle_response_cancel(session_id, event)
            else:
                await self.send_error(session_id, "unknown_event", f"Unknown event type: {event_type}")
                
        except Exception as e:
            logger.error(f"Error processing event {event_type} for session {session_id}: {e}")
            await self.send_error(session_id, "processing_error", str(e))
    
    async def handle_session_update(self, session_id: str, event: dict):
        """Handle session.update event"""
        session_data = self.manager.session_data.get(session_id)
        if not session_data:
            return
        
        # Update session configuration
        session_update = event.get("session", {})
        
        for key, value in session_update.items():
            if key in session_data:
                session_data[key] = value
        
        # Apply updates to speech pipeline
        pipeline = self.manager.speech_pipelines.get(session_id)
        if pipeline:
            if "voice" in session_update:
                pipeline.set_voice(session_update["voice"])
        
        # Send session.updated event
        await self.manager.send_event(session_id, {
            "type": "session.updated",
            "session": session_data
        })
    
    async def handle_audio_buffer_append(self, session_id: str, event: dict):
        """Handle input_audio_buffer.append event"""
        try:
            # Decode base64 audio data
            audio_b64 = event.get("audio")
            if not audio_b64:
                await self.send_error(session_id, "missing_audio", "Audio data is required")
                return
            
            audio_data = base64.b64decode(audio_b64)
            
            # Add to conversation buffer
            conversation_state = self.manager.conversation_states.get(session_id, {})
            conversation_state["input_audio_buffer"] += audio_data
            
            # Process through speech pipeline
            pipeline = self.manager.speech_pipelines.get(session_id)
            if pipeline:
                result = await pipeline.process_audio_chunk(audio_data)
                
                # Handle speech detection
                if result.get('audio_info', {}).get('speech_detected'):
                    await self.manager.send_event(session_id, {
                        "type": "input_audio_buffer.speech_started",
                        "audio_start_ms": int(time.time() * 1000),
                        "item_id": f"item_{uuid.uuid4().hex[:16]}"
                    })
                
                # Handle turn detection
                if result.get('turn_detected'):
                    await self.manager.send_event(session_id, {
                        "type": "conversation.item.input_audio_transcription.started",
                        "item_id": f"item_{uuid.uuid4().hex[:16]}"
                    })
                
                # Handle completed transcription
                if result.get('transcription'):
                    await self.handle_transcription_completed(session_id, result['transcription'])
        
        except Exception as e:
            logger.error(f"Audio buffer append error: {e}")
            await self.send_error(session_id, "audio_processing_error", str(e))
    
    async def handle_audio_buffer_commit(self, session_id: str, event: dict):
        """Handle input_audio_buffer.commit event"""
        try:
            conversation_state = self.manager.conversation_states.get(session_id, {})
            audio_buffer = conversation_state.get("input_audio_buffer", b"")
            
            if len(audio_buffer) > 0:
                # Process accumulated audio
                pipeline = self.manager.speech_pipelines.get(session_id)
                if pipeline:
                    # Force processing of current buffer
                    transcription_result = await pipeline.stt_engine.transcribe(audio_buffer)
                    
                    if transcription_result.get('text'):
                        await self.handle_transcription_completed(session_id, transcription_result)
                
                # Clear buffer after commit
                conversation_state["input_audio_buffer"] = b""
            
            await self.manager.send_event(session_id, {
                "type": "input_audio_buffer.committed"
            })
            
        except Exception as e:
            logger.error(f"Audio buffer commit error: {e}")
            await self.send_error(session_id, "commit_error", str(e))
    
    async def handle_audio_buffer_clear(self, session_id: str, event: dict):
        """Handle input_audio_buffer.clear event"""
        conversation_state = self.manager.conversation_states.get(session_id, {})
        conversation_state["input_audio_buffer"] = b""
        
        # Reset pipeline state
        pipeline = self.manager.speech_pipelines.get(session_id)
        if pipeline:
            pipeline.reset_session()
        
        await self.manager.send_event(session_id, {
            "type": "input_audio_buffer.cleared"
        })
    
    async def handle_transcription_completed(self, session_id: str, transcription: dict):
        """Handle completed transcription"""
        item_id = f"item_{uuid.uuid4().hex[:16]}"
        
        # Create conversation item
        conversation_item = {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "status": "completed",
            "role": "user",
            "content": [{
                "type": "input_audio",
                "transcript": transcription.get('text', ''),
                "audio": None  # Audio data not included in response
            }]
        }
        
        # Add to conversation state
        conversation_state = self.manager.conversation_states.get(session_id, {})
        conversation_state["items"].append(conversation_item)
        
        # Send events
        await self.manager.send_event(session_id, {
            "type": "conversation.item.created", 
            "item": conversation_item
        })
        
        await self.manager.send_event(session_id, {
            "type": "input_audio_buffer.speech_stopped",
            "audio_end_ms": int(time.time() * 1000),
            "item_id": item_id
        })
        
        # Trigger response generation
        await self.generate_response(session_id, transcription.get('text', ''))
    
    async def handle_conversation_item_create(self, session_id: str, event: dict):
        """Handle conversation.item.create event"""
        item = event.get("item", {})
        
        # Add item to conversation
        conversation_state = self.manager.conversation_states.get(session_id, {})
        conversation_state["items"].append(item)
        
        await self.manager.send_event(session_id, {
            "type": "conversation.item.created",
            "item": item
        })
    
    async def handle_response_create(self, session_id: str, event: dict):
        """Handle response.create event"""
        response_config = event.get("response", {})
        
        # Extract the last user message for response generation
        conversation_state = self.manager.conversation_states.get(session_id, {})
        items = conversation_state.get("items", [])
        
        # Find the last user message
        last_user_message = None
        for item in reversed(items):
            if item.get("role") == "user":
                content = item.get("content", [])
                if content and content[0].get("type") in ["input_audio", "input_text"]:
                    last_user_message = content[0].get("transcript") or content[0].get("text")
                    break
        
        if last_user_message:
            await self.generate_response(session_id, last_user_message)
        else:
            await self.send_error(session_id, "no_user_message", "No user message found for response generation")
    
    async def generate_response(self, session_id: str, user_input: str):
        """Generate and stream response"""
        try:
            response_id = f"resp_{uuid.uuid4().hex[:16]}"
            item_id = f"item_{uuid.uuid4().hex[:16]}"
            
            # Send response.created event
            await self.manager.send_event(session_id, {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "realtime.response",
                    "status": "in_progress",
                    "status_details": None,
                    "output": [],
                    "usage": None
                }
            })
            
            # Generate response text (placeholder - integrate with OVOS persona/LLM)
            response_text = await self.generate_response_text(session_id, user_input)
            
            # Create response item
            response_item = {
                "id": item_id,
                "object": "realtime.item", 
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [{
                    "type": "audio",
                    "transcript": response_text
                }]
            }
            
            # Send response.output_item.added
            await self.manager.send_event(session_id, {
                "type": "response.output_item.added",
                "response_id": response_id,
                "output_index": 0,
                "item": response_item
            })
            
            # Generate and stream audio
            await self.generate_and_stream_audio(session_id, response_id, item_id, response_text)
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            await self.send_error(session_id, "response_error", str(e))
    
    async def generate_response_text(self, session_id: str, user_input: str) -> str:
        """Generate response text using Groq LLM integration"""
        try:
            # Use Groq API for intelligent responses
            response = await generate_ai_response(session_id, user_input)
            return response
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            
            # Fallback to simple responses if LLM fails
            if "hello" in user_input.lower():
                return "Hello! How can I help you today?"
            elif "how are you" in user_input.lower():
                return "I'm doing well, thank you for asking! How are you?"
            elif "what time" in user_input.lower():
                return f"The current time is {datetime.now().strftime('%I:%M %p')}."
            else:
                return f"I heard you say: {user_input}. How can I help you with that?"
    
    async def generate_and_stream_audio(self, session_id: str, response_id: str, item_id: str, text: str):
        """Generate TTS audio and stream it"""
        try:
            # Get pipeline for TTS synthesis
            pipeline = self.manager.speech_pipelines.get(session_id)
            if not pipeline:
                return
            
            # Get session voice settings
            session_data = self.manager.session_data.get(session_id, {})
            voice = session_data.get("voice", "default")
            
            # Synthesize audio
            tts_result = await pipeline.synthesize_response(text, voice=voice, streaming=True)
            
            if tts_result.get('audio'):
                # Convert audio to base64 for transmission
                audio_b64 = base64.b64encode(tts_result['audio']).decode('utf-8')
                
                # Send audio delta event
                await self.manager.send_event(session_id, {
                    "type": "response.audio.delta",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": audio_b64
                })
                
                # Send transcript delta
                await self.manager.send_event(session_id, {
                    "type": "response.audio_transcript.delta",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text
                })
            
            # Mark item as completed
            await self.manager.send_event(session_id, {
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": 0,
                "item": {
                    "id": item_id,
                    "object": "realtime.item",
                    "type": "message", 
                    "status": "completed",
                    "role": "assistant",
                    "content": [{
                        "type": "audio",
                        "transcript": text
                    }]
                }
            })
            
            # Mark response as completed
            await self.manager.send_event(session_id, {
                "type": "response.done",
                "response": {
                    "id": response_id,
                    "object": "realtime.response",
                    "status": "completed",
                    "status_details": None,
                    "output": [{
                        "id": item_id,
                        "object": "realtime.item",
                        "type": "message",
                        "status": "completed", 
                        "role": "assistant",
                        "content": [{
                            "type": "audio",
                            "transcript": text
                        }]
                    }],
                    "usage": {
                        "total_tokens": len(text.split()),
                        "input_tokens": 0,
                        "output_tokens": len(text.split())
                    }
                }
            })
            
        except Exception as e:
            logger.error(f"Audio generation error: {e}")
            await self.send_error(session_id, "audio_generation_error", str(e))
    
    async def handle_response_cancel(self, session_id: str, event: dict):
        """Handle response.cancel event"""
        # Stop any ongoing response generation
        conversation_state = self.manager.conversation_states.get(session_id, {})
        current_response = conversation_state.get("response")
        
        if current_response:
            await self.manager.send_event(session_id, {
                "type": "response.cancelled",
                "response": current_response
            })
            
            conversation_state["response"] = None
    
    async def send_error(self, session_id: str, error_type: str, message: str):
        """Send error event to client"""
        await self.manager.send_event(session_id, {
            "type": "error",
            "error": {
                "type": error_type,
                "code": error_type,
                "message": message,
                "param": None,
                "event_id": None
            }
        })

# Global event processor
event_processor = RealtimeEventProcessor(connection_manager)

# ==================== WebSocket Endpoint ====================

@app.websocket("/v1/realtime")
async def websocket_realtime(websocket: WebSocket):
    """OpenAI-compatible realtime WebSocket endpoint"""
    session_id = f"sess_{uuid.uuid4().hex[:24]}"
    
    try:
        # Connect and initialize session
        await connection_manager.connect(websocket, session_id)
        
        # Main message loop
        while True:
            # Receive message from client
            try:
                data = await websocket.receive_text()
                event = json.loads(data)
                
                # Process event through event processor
                await event_processor.process_event(session_id, event)
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from session {session_id}: {e}")
                await event_processor.send_error(session_id, "invalid_json", "Invalid JSON format")
            except Exception as e:
                logger.error(f"Error processing message from session {session_id}: {e}")
                await event_processor.send_error(session_id, "processing_error", str(e))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        # Cleanup connection
        connection_manager.disconnect(session_id)

# ==================== HTTP Endpoints ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_connections": len(connection_manager.active_connections),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "OpenAI Realtime API v1"
    }

@app.get("/v1/realtime/sessions")
async def list_realtime_sessions():
    """List active realtime sessions"""
    sessions = []
    for session_id, session_data in connection_manager.session_data.items():
        sessions.append({
            **session_data,
            "connection_status": "connected" if session_id in connection_manager.active_connections else "disconnected"
        })
    
    return {
        "object": "list",
        "data": sessions
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,  # Different port from REST API
        log_level="info",
        reload=True
    )