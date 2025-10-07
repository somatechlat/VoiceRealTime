#!/usr/bin/env python3
"""
OVOS Voice Agent - Sprint 2: Speech Processing Pipeline
Real-time STT, TTS, and audio processing components
"""

import asyncio
import logging
import numpy as np
import torch
import threading
from typing import Optional, Callable, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
import time
import io
import wave

# OVOS components
try:
    from ovos_plugin_manager.stt import OVOSSTTFactory
    from ovos_plugin_manager.tts import OVOSTTSFactory
    OVOS_AVAILABLE = True
except ImportError:
    OVOS_AVAILABLE = False
    logging.warning("OVOS plugins not available, using fallback implementations")

# Faster Whisper for STT
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    
# WebRTC VAD
try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 30  # 30ms chunks for VAD
    frame_duration_ms: int = 30
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive
    silence_threshold_ms: int = 1000  # 1 second of silence to trigger processing
    
class AudioProcessor:
    """Handles audio preprocessing and VAD"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.vad = None
        
        if WEBRTC_VAD_AVAILABLE:
            self.vad = webrtcvad.Vad(config.vad_aggressiveness)
            logger.info("WebRTC VAD initialized")
        else:
            logger.warning("WebRTC VAD not available, using basic energy-based VAD")
    
    def preprocess_audio(self, audio_data: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array and preprocess"""
        try:
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Resample if necessary (implement resampling if needed)
            # For now, assume input is already at correct sample rate
            
            return audio_float
            
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            return np.array([], dtype=np.float32)
    
    def detect_speech(self, audio_data: bytes) -> bool:
        """Detect if audio contains speech using VAD"""
        if self.vad and len(audio_data) > 0:
            try:
                # WebRTC VAD requires specific frame sizes
                frame_size = int(self.config.sample_rate * self.config.frame_duration_ms / 1000)
                
                if len(audio_data) >= frame_size * 2:  # 2 bytes per sample for 16-bit
                    # Take the first complete frame
                    frame_bytes = audio_data[:frame_size * 2]
                    return self.vad.is_speech(frame_bytes, self.config.sample_rate)
                    
            except Exception as e:
                logger.error(f"VAD error: {e}")
        
        # Fallback: simple energy-based detection
        return self._energy_based_vad(audio_data)
    
    def _energy_based_vad(self, audio_data: bytes) -> bool:
        """Simple energy-based voice activity detection"""
        if len(audio_data) == 0:
            return False
            
        # Convert to numpy and calculate RMS energy
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        if len(audio_array) == 0:
            return False
            
        rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
        threshold = 500  # Adjust based on testing
        
        return rms > threshold

class STTEngine:
    """Speech-to-Text engine with multiple backends"""
    
    def __init__(self, model_name: str = "base"):
        self.model = None
        self.model_name = model_name
        self.is_loaded = False
        
        # Try to load Faster Whisper first
        if FASTER_WHISPER_AVAILABLE:
            self._load_faster_whisper()
        elif OVOS_AVAILABLE:
            self._load_ovos_stt()
        else:
            logger.error("No STT engine available")
    
    def _load_faster_whisper(self):
        """Load Faster Whisper model"""
        try:
            # Use CPU for now, can be changed to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = WhisperModel(
                self.model_name,
                device=device,
                compute_type="float16" if device == "cuda" else "int8"
            )
            self.is_loaded = True
            logger.info(f"Faster Whisper model '{self.model_name}' loaded on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load Faster Whisper: {e}")
    
    def _load_ovos_stt(self):
        """Load OVOS STT plugin"""
        try:
            self.model = OVOSSTTFactory.create()
            self.is_loaded = True
            logger.info("OVOS STT plugin loaded")
        except Exception as e:
            logger.error(f"Failed to load OVOS STT: {e}")
    
    async def transcribe(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio to text"""
        if not self.is_loaded:
            return None
            
        try:
            # Convert bytes to proper format for STT
            if isinstance(self.model, WhisperModel):
                return await self._transcribe_whisper(audio_data)
            else:
                return await self._transcribe_ovos(audio_data)
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    async def _transcribe_whisper(self, audio_data: bytes) -> Optional[str]:
        """Transcribe using Faster Whisper"""
        try:
            # Convert audio bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None,
                self.model.transcribe,
                audio_float
            )
            
            # Combine all segments
            text = " ".join([segment.text for segment in segments]).strip()
            
            if text:
                logger.info(f"Transcribed: '{text}' (language: {info.language}, probability: {info.language_probability:.2f})")
                return text
                
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            
        return None
    
    async def _transcribe_ovos(self, audio_data: bytes) -> Optional[str]:
        """Transcribe using OVOS STT"""
        try:
            # Convert to WAV format for OVOS
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_data)
            
            wav_buffer.seek(0)
            
            # Run OVOS STT in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.model.execute,
                wav_buffer.read()
            )
            
            if result:
                logger.info(f"OVOS STT result: {result}")
                return result
                
        except Exception as e:
            logger.error(f"OVOS STT error: {e}")
            
        return None

class TTSEngine:
    """Text-to-Speech engine with multiple backends"""
    
    def __init__(self, voice: str = "default"):
        self.engine = None
        self.voice = voice
        self.is_loaded = False
        
        # Try to load OVOS TTS first
        if OVOS_AVAILABLE:
            self._load_ovos_tts()
        else:
            logger.error("No TTS engine available")
    
    def _load_ovos_tts(self):
        """Load OVOS TTS plugin"""
        try:
            self.engine = OVOSTTSFactory.create()
            self.is_loaded = True
            logger.info("OVOS TTS plugin loaded")
        except Exception as e:
            logger.error(f"Failed to load OVOS TTS: {e}")
    
    async def synthesize(self, text: str) -> Optional[bytes]:
        """Convert text to speech audio"""
        if not self.is_loaded or not text.strip():
            return None
            
        try:
            # Run TTS in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            if OVOS_AVAILABLE and hasattr(self.engine, 'get_tts'):
                audio_file, phonemes = await loop.run_in_executor(
                    None,
                    self.engine.get_tts,
                    text,
                    None  # audio file path will be returned
                )
                
                # Read the generated audio file
                if audio_file and Path(audio_file).exists():
                    with open(audio_file, 'rb') as f:
                        audio_data = f.read()
                    
                    # Clean up temporary file
                    Path(audio_file).unlink(missing_ok=True)
                    
                    logger.info(f"TTS synthesized: '{text}' -> {len(audio_data)} bytes")
                    return audio_data
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            
        return None

class SpeechPipeline:
    """Complete speech processing pipeline"""
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.audio_processor = AudioProcessor(self.config)
        self.stt_engine = STTEngine()
        self.tts_engine = TTSEngine()
        
        # Audio buffers
        self.audio_buffer = bytearray()
        self.silence_start = None
        self.is_processing = False
        
        logger.info("Speech pipeline initialized")
    
    async def process_audio_chunk(self, audio_data: bytes) -> Optional[str]:
        """Process incoming audio chunk and return transcription if complete"""
        if not audio_data:
            return None
            
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        
        # Check for speech activity
        has_speech = self.audio_processor.detect_speech(audio_data)
        
        current_time = time.time()
        
        if has_speech:
            # Reset silence timer
            self.silence_start = None
        else:
            # Start or continue silence timer
            if self.silence_start is None:
                self.silence_start = current_time
                
        # Check if we should process the buffer
        if (self.silence_start and 
            (current_time - self.silence_start) * 1000 > self.config.silence_threshold_ms and
            len(self.audio_buffer) > 0 and
            not self.is_processing):
            
            # Process the accumulated audio
            return await self._process_buffer()
            
        return None
    
    async def _process_buffer(self) -> Optional[str]:
        """Process the current audio buffer"""
        if self.is_processing or len(self.audio_buffer) == 0:
            return None
            
        self.is_processing = True
        
        try:
            # Copy buffer and clear it
            audio_to_process = bytes(self.audio_buffer)
            self.audio_buffer.clear()
            self.silence_start = None
            
            logger.info(f"Processing {len(audio_to_process)} bytes of audio")
            
            # Transcribe
            transcription = await self.stt_engine.transcribe(audio_to_process)
            
            return transcription
            
        except Exception as e:
            logger.error(f"Buffer processing error: {e}")
            return None
        finally:
            self.is_processing = False
    
    async def synthesize_response(self, text: str) -> Optional[bytes]:
        """Generate speech from text response"""
        return await self.tts_engine.synthesize(text)
    
    def get_buffer_info(self) -> dict:
        """Get current buffer status"""
        return {
            "buffer_size": len(self.audio_buffer),
            "is_processing": self.is_processing,
            "silence_duration": (time.time() - self.silence_start) * 1000 if self.silence_start else 0
        }

# Factory function for easy initialization
def create_speech_pipeline(config: dict = None) -> SpeechPipeline:
    """Create a speech pipeline with given configuration"""
    audio_config = AudioConfig()
    
    if config:
        for key, value in config.items():
            if hasattr(audio_config, key):
                setattr(audio_config, key, value)
    
    return SpeechPipeline(audio_config)

if __name__ == "__main__":
    # Test the speech pipeline
    import asyncio
    
    async def test_pipeline():
        pipeline = create_speech_pipeline()
        
        # Test with some dummy audio data
        dummy_audio = b'\x00' * 32000  # 1 second of silence at 16kHz
        
        result = await pipeline.process_audio_chunk(dummy_audio)
        print(f"Test result: {result}")
        
        # Test TTS
        tts_result = await pipeline.synthesize_response("Hello, this is a test.")
        print(f"TTS result: {len(tts_result) if tts_result else 0} bytes")
    
    asyncio.run(test_pipeline())
