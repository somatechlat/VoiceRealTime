#!/usr/bin/env python3
"""
OVOS Voice Agent - Sprint 3: OpenAI API Compatibility Layer
FastAPI server providing OpenAI-compatible endpoints for voice processing
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union
import tempfile
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Import enhanced speech pipeline
import sys
sys.path.append('../sprint2-speech')
from speech_pipeline import create_realtime_pipeline, SpeechPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with OpenAI-compatible metadata
app = FastAPI(
    title="OVOS Voice Agent API",
    description="OpenAI-compatible voice processing API using OVOS ecosystem",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for browser compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Global state management
active_sessions: Dict[str, dict] = {}
speech_pipelines: Dict[str, SpeechPipeline] = {}

# ==================== OpenAI-Compatible Models ====================

class RealtimeSession(BaseModel):
    """OpenAI-compatible realtime session model"""
    id: str = Field(..., description="Session ID")
    object: str = Field(default="realtime.session", description="Object type")
    model: str = Field(default="ovos-voice-1", description="Model identifier")
    expires_at: Optional[int] = Field(None, description="Session expiration timestamp")
    created_at: int = Field(..., description="Session creation timestamp")
    
    # OVOS-specific extensions
    voice: str = Field(default="default", description="TTS voice")
    language: str = Field(default="en-US", description="Processing language")
    turn_detection: bool = Field(default=True, description="Enable turn detection")
    
    class Config:
        json_encoders = {
            datetime: lambda v: int(v.timestamp())
        }

class SessionCreateRequest(BaseModel):
    """Request model for creating realtime sessions"""
    model: str = Field(default="ovos-voice-1", description="Model to use")
    voice: Optional[str] = Field(default="default", description="TTS voice")
    language: Optional[str] = Field(default="en-US", description="Processing language")
    turn_detection: Optional[bool] = Field(default=True, description="Enable turn detection")
    session_config: Optional[dict] = Field(default_factory=dict, description="Additional session config")

class SessionUpdateRequest(BaseModel):
    """Request model for updating realtime sessions"""
    voice: Optional[str] = Field(None, description="TTS voice")
    language: Optional[str] = Field(None, description="Processing language")
    turn_detection: Optional[bool] = Field(None, description="Enable turn detection")

class AudioSpeechRequest(BaseModel):
    """OpenAI-compatible TTS request"""
    model: str = Field(default="ovos-tts-1", description="TTS model")
    input: str = Field(..., description="Text to synthesize")
    voice: Optional[str] = Field(default="default", description="Voice to use")
    response_format: Optional[str] = Field(default="mp3", description="Audio format")
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0, description="Playback speed")

class AudioTranscriptionRequest(BaseModel):
    """OpenAI-compatible STT request model"""
    model: str = Field(default="ovos-whisper-1", description="STT model")
    language: Optional[str] = Field(None, description="Input language")
    prompt: Optional[str] = Field(None, description="Optional prompt")
    response_format: Optional[str] = Field(default="json", description="Response format")
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")

class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response"""
    text: str = Field(..., description="Transcribed text")
    
    # Extended format fields
    language: Optional[str] = Field(None, description="Detected language")
    duration: Optional[float] = Field(None, description="Audio duration")
    confidence: Optional[float] = Field(None, description="Transcription confidence")

class ErrorResponse(BaseModel):
    """OpenAI-compatible error response"""
    error: dict = Field(..., description="Error details")

# ==================== Authentication ====================

async def get_current_session(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Extract session ID from authorization header (optional)"""
    if credentials and credentials.scheme == "Bearer":
        return credentials.credentials
    return None

def create_error_response(error_type: str, message: str, code: str = None) -> dict:
    """Create OpenAI-compatible error response"""
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
            "param": None
        }
    }

# ==================== Session Management Endpoints ====================

@app.post("/v1/realtime/sessions", response_model=RealtimeSession)
async def create_realtime_session(request: SessionCreateRequest) -> RealtimeSession:
    """Create a new realtime session - OpenAI compatible"""
    try:
        session_id = f"sess_{uuid.uuid4().hex[:24]}"
        created_at = int(datetime.now(timezone.utc).timestamp())
        
        # Create session metadata
        session_data = {
            "id": session_id,
            "object": "realtime.session",
            "model": request.model,
            "created_at": created_at,
            "expires_at": None,  # Sessions don't expire by default
            "voice": request.voice,
            "language": request.language,
            "turn_detection": request.turn_detection,
            "status": "created"
        }
        
        # Initialize speech pipeline for this session
        pipeline_config = {
            "sample_rate": 24000,
            "turn_detection_enabled": request.turn_detection,
            **request.session_config
        }
        
        pipeline = create_realtime_pipeline(session_id, **pipeline_config)
        if request.voice:
            pipeline.set_voice(request.voice)
        if request.language:
            pipeline.set_language(request.language)
        
        # Store session data
        active_sessions[session_id] = session_data
        speech_pipelines[session_id] = pipeline
        
        logger.info(f"Created realtime session {session_id} with model {request.model}")
        
        return RealtimeSession(**session_data)
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response("internal_error", f"Failed to create session: {str(e)}")
        )

@app.get("/v1/realtime/sessions/{session_id}", response_model=RealtimeSession)
async def get_realtime_session(session_id: str) -> RealtimeSession:
    """Get realtime session details - OpenAI compatible"""
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=create_error_response("not_found", f"Session {session_id} not found")
        )
    
    session_data = active_sessions[session_id]
    return RealtimeSession(**session_data)

@app.patch("/v1/realtime/sessions/{session_id}", response_model=RealtimeSession)
async def update_realtime_session(session_id: str, request: SessionUpdateRequest) -> RealtimeSession:
    """Update realtime session configuration - OpenAI compatible"""
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=create_error_response("not_found", f"Session {session_id} not found")
        )
    
    session_data = active_sessions[session_id]
    pipeline = speech_pipelines.get(session_id)
    
    # Update session configuration
    if request.voice is not None:
        session_data["voice"] = request.voice
        if pipeline:
            pipeline.set_voice(request.voice)
    
    if request.language is not None:
        session_data["language"] = request.language
        if pipeline:
            pipeline.set_language(request.language)
    
    if request.turn_detection is not None:
        session_data["turn_detection"] = request.turn_detection
        # Update pipeline config if needed
    
    logger.info(f"Updated session {session_id}")
    return RealtimeSession(**session_data)

@app.delete("/v1/realtime/sessions/{session_id}")
async def delete_realtime_session(session_id: str) -> dict:
    """Delete realtime session - OpenAI compatible"""
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=create_error_response("not_found", f"Session {session_id} not found")
        )
    
    # Cleanup session resources
    active_sessions.pop(session_id, None)
    pipeline = speech_pipelines.pop(session_id, None)
    
    if pipeline:
        pipeline.reset_session()
    
    logger.info(f"Deleted session {session_id}")
    return {"deleted": True}

# ==================== Audio Processing Endpoints ====================

@app.post("/v1/audio/speech")
async def create_speech(request: AudioSpeechRequest) -> Response:
    """Create speech from text - OpenAI compatible TTS endpoint"""
    try:
        # Create temporary pipeline for TTS
        pipeline = create_realtime_pipeline()
        
        # Set voice if specified
        if request.voice:
            pipeline.set_voice(request.voice)
        
        # Synthesize speech
        result = await pipeline.synthesize_response(
            text=request.input,
            voice=request.voice,
            streaming=False
        )
        
        if not result.get('audio'):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=create_error_response("synthesis_error", "Failed to synthesize speech")
            )
        
        # Convert audio format if needed (placeholder)
        audio_data = result['audio']
        
        # Determine content type based on format
        content_type = "audio/wav"  # Default
        if request.response_format == "mp3":
            content_type = "audio/mpeg"
        elif request.response_format == "opus":
            content_type = "audio/opus"
        elif request.response_format == "flac":
            content_type = "audio/flac"
        
        return Response(
            content=audio_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
            }
        )
        
    except Exception as e:
        logger.error(f"Speech synthesis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response("synthesis_error", str(e))
        )

@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(default="ovos-whisper-1"),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0)
) -> Union[TranscriptionResponse, dict]:
    """Create transcription from audio - OpenAI compatible STT endpoint"""
    try:
        # Read uploaded audio file
        audio_content = await file.read()
        
        if len(audio_content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=create_error_response("invalid_request", "Empty audio file")
            )
        
        # Create temporary pipeline for transcription
        pipeline = create_realtime_pipeline()
        
        # Process audio through pipeline
        transcription_result = await pipeline.stt_engine.transcribe(audio_content, language)
        
        if not transcription_result.get('text'):
            # Return empty transcription if no speech detected
            response_data = {
                "text": "",
                "language": language,
                "duration": 0.0,
                "confidence": 0.0
            }
        else:
            response_data = {
                "text": transcription_result['text'],
                "language": transcription_result.get('language', language),
                "duration": len(audio_content) / (16000 * 2),  # Approximate duration
                "confidence": transcription_result.get('confidence', 0.0)
            }
        
        # Handle different response formats
        if response_format == "json":
            return TranscriptionResponse(**response_data)
        elif response_format == "text":
            return Response(content=response_data["text"], media_type="text/plain")
        elif response_format == "verbose_json":
            # Extended format with additional metadata
            return {
                **response_data,
                "segments": [],  # Could include segment-level data
                "model": model
            }
        else:
            return TranscriptionResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response("transcription_error", str(e))
        )

# ==================== Health and Info Endpoints ====================

@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_sessions": len(active_sessions),
        "version": "1.0.0",
        "api_compatibility": "OpenAI v1"
    }

@app.get("/v1/models")
async def list_models() -> dict:
    """List available models - OpenAI compatible"""
    models = [
        {
            "id": "ovos-voice-1",
            "object": "model",
            "created": 1699401600,  # Static timestamp
            "owned_by": "ovos",
            "permission": [],
            "root": "ovos-voice-1",
            "parent": None,
            "description": "OVOS real-time voice processing model"
        },
        {
            "id": "ovos-whisper-1", 
            "object": "model",
            "created": 1699401600,
            "owned_by": "ovos",
            "permission": [],
            "root": "ovos-whisper-1",
            "parent": None,
            "description": "OVOS Whisper-based STT model"
        },
        {
            "id": "ovos-tts-1",
            "object": "model", 
            "created": 1699401600,
            "owned_by": "ovos",
            "permission": [],
            "root": "ovos-tts-1",
            "parent": None,
            "description": "OVOS phoonnx-based TTS model"
        }
    ]
    
    return {
        "object": "list",
        "data": models
    }

@app.get("/v1/dashboard")
async def get_dashboard() -> dict:
    """OVOS-specific dashboard endpoint"""
    pipeline_statuses = {}
    
    for session_id, pipeline in speech_pipelines.items():
        pipeline_statuses[session_id] = pipeline.get_comprehensive_status()
    
    return {
        "active_sessions": len(active_sessions),
        "sessions": list(active_sessions.keys()),
        "pipeline_statuses": pipeline_statuses,
        "system_info": {
            "version": "1.0.0",
            "api_type": "OpenAI Compatible",
            "engine": "OVOS"
        }
    }

# ==================== Background Tasks ====================

async def cleanup_expired_sessions():
    """Background task to cleanup expired sessions"""
    while True:
        try:
            current_time = datetime.now(timezone.utc).timestamp()
            expired_sessions = []
            
            for session_id, session_data in active_sessions.items():
                expires_at = session_data.get("expires_at")
                if expires_at and current_time > expires_at:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                logger.info(f"Cleaning up expired session: {session_id}")
                await delete_realtime_session(session_id)
            
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")
            await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    """Initialize background tasks"""
    asyncio.create_task(cleanup_expired_sessions())
    logger.info("OVOS Voice Agent API started - OpenAI compatible endpoints active")

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown"""
    # Cleanup all active sessions
    for session_id in list(active_sessions.keys()):
        try:
            await delete_realtime_session(session_id)
        except:
            pass
    
    logger.info("OVOS Voice Agent API shutdown complete")

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=True
    )