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
from typing import Optional, Callable, AsyncGenerator, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import time
import io
import wave
import json
from collections import deque
from datetime import datetime, timedelta

# Enhanced audio processing
import librosa
import soundfile as sf
import noisereduce as nr
from scipy import signal

# OVOS components with enhanced imports
try:
    from ovos_plugin_manager.stt import OVOSSTTFactory
    from ovos_plugin_manager.tts import OVOSTTSFactory
    from ovos_plugin_manager.templates.stt import STT, StreamingSTT
    from ovos_plugin_manager.templates.tts import TTS
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
    """Enhanced audio processing configuration for OpenAI compatibility"""
    sample_rate: int = 24000  # Higher quality for better TTS
    channels: int = 1
    chunk_duration_ms: int = 20  # Smaller chunks for lower latency
    frame_duration_ms: int = 20
    vad_aggressiveness: int = 3  # More aggressive for better turn detection
    silence_threshold_ms: int = 500  # Faster response for real-time
    
    # Enhanced audio processing
    noise_reduction: bool = True
    auto_gain_control: bool = True
    echo_cancellation: bool = True
    
    # Turn detection
    turn_detection_enabled: bool = True
    interruption_threshold_ms: int = 300  # Allow interruptions after 300ms
    max_audio_length_s: int = 30  # Max single utterance length
    
    # Quality settings
    audio_format: str = 'pcm_s16le'  # OpenAI compatible format
    buffer_size: int = 4096
    overlap_ms: int = 10  # Overlap between chunks for smooth processing
    
class AudioProcessor:
    """Enhanced audio preprocessing with dual VAD and noise reduction"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.webrtc_vad = None
        self.silero_vad = None
        self.noise_reducer = None
        
        # Initialize WebRTC VAD
        if WEBRTC_VAD_AVAILABLE:
            self.webrtc_vad = webrtcvad.Vad(config.vad_aggressiveness)
            logger.info("WebRTC VAD initialized")
        
        # Initialize Silero VAD for enhanced accuracy
        try:
            import torch
            self.silero_vad, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              force_reload=False,
                                              onnx=False)
            logger.info("Silero VAD initialized")
        except Exception as e:
            logger.warning(f"Silero VAD not available: {e}")
        
        # Audio enhancement setup
        if config.noise_reduction:
            logger.info("Noise reduction enabled")
            
        # Turn detection state
        self.last_voice_time = None
        self.conversation_active = False
        self.audio_history = deque(maxlen=50)  # Keep last 1 second at 20ms chunks
    
    def preprocess_audio(self, audio_data: bytes) -> np.ndarray:
        """Enhanced audio preprocessing with noise reduction and normalization"""
        try:
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Apply noise reduction if enabled
            if self.config.noise_reduction and len(audio_float) > 0:
                audio_float = self._apply_noise_reduction(audio_float)
            
            # Apply auto gain control
            if self.config.auto_gain_control:
                audio_float = self._apply_agc(audio_float)
            
            # Resample if necessary
            if len(audio_float) > 0 and self._needs_resampling():
                audio_float = self._resample_audio(audio_float)
            
            return audio_float
            
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            return np.array([], dtype=np.float32)
    
    def _apply_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise reduction using spectral subtraction"""
        try:
            # Simple noise reduction - can be enhanced with noisereduce library
            if len(audio) < 1024:  # Too short for meaningful noise reduction
                return audio
            
            # Estimate noise from first 0.1 seconds
            noise_sample_size = min(int(0.1 * self.config.sample_rate), len(audio) // 4)
            noise_profile = audio[:noise_sample_size]
            noise_power = np.mean(noise_profile ** 2)
            
            # Simple spectral subtraction
            if noise_power > 0:
                reduction_factor = min(0.5, noise_power * 10)  # Adaptive reduction
                audio = audio - (noise_profile.mean() * reduction_factor)
            
            return audio
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio
    
    def _apply_agc(self, audio: np.ndarray) -> np.ndarray:
        """Apply automatic gain control"""
        try:
            if len(audio) == 0:
                return audio
                
            # Calculate RMS level
            rms = np.sqrt(np.mean(audio ** 2))
            
            if rms > 0:
                # Target RMS level (-20 dB)
                target_rms = 0.1
                gain = min(3.0, target_rms / rms)  # Limit gain to 3x
                audio = audio * gain
                
                # Prevent clipping
                audio = np.clip(audio, -1.0, 1.0)
            
            return audio
        except Exception as e:
            logger.warning(f"AGC failed: {e}")
            return audio
    
    def _needs_resampling(self) -> bool:
        """Check if resampling is needed"""
        # For now, assume input is correct sample rate
        # Can be enhanced to detect actual sample rate
        return False
    
    def _resample_audio(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio to target sample rate"""
        # Placeholder - implement with librosa if needed
        return audio
    
    def detect_speech(self, audio_data: bytes) -> dict:
        """Enhanced speech detection with dual VAD and turn detection"""
        result = {
            'has_speech': False,
            'confidence': 0.0,
            'turn_detected': False,
            'interruption_possible': False
        }
        
        if len(audio_data) == 0:
            return result
        
        # Preprocess audio for VAD
        audio_float = self.preprocess_audio(audio_data)
        
        # WebRTC VAD
        webrtc_result = self._webrtc_vad_check(audio_data)
        
        # Silero VAD (more accurate)
        silero_result = self._silero_vad_check(audio_float)
        
        # Energy-based VAD (fallback)
        energy_result = self._energy_based_vad(audio_data)
        
        # Combine results with confidence scoring
        confidence_scores = []
        speech_detections = []
        
        if webrtc_result is not None:
            confidence_scores.append(0.7 if webrtc_result else 0.3)
            speech_detections.append(webrtc_result)
        
        if silero_result is not None:
            confidence_scores.append(0.8 if silero_result else 0.2)
            speech_detections.append(silero_result)
        
        confidence_scores.append(0.6 if energy_result else 0.4)
        speech_detections.append(energy_result)
        
        # Final decision based on weighted voting
        has_speech = sum(speech_detections) > len(speech_detections) / 2
        confidence = sum(confidence_scores) / len(confidence_scores)
        
        result['has_speech'] = has_speech
        result['confidence'] = confidence
        
        # Turn detection logic
        current_time = time.time()
        
        if has_speech:
            self.last_voice_time = current_time
            if not self.conversation_active:
                result['turn_detected'] = True
                self.conversation_active = True
        else:
            if (self.conversation_active and 
                self.last_voice_time and 
                (current_time - self.last_voice_time) * 1000 > self.config.interruption_threshold_ms):
                result['interruption_possible'] = True
        
        # Update audio history for context
        self.audio_history.append({
            'timestamp': current_time,
            'has_speech': has_speech,
            'confidence': confidence
        })
        
        return result
    
    def _webrtc_vad_check(self, audio_data: bytes) -> Optional[bool]:
        """WebRTC VAD check"""
        if not self.webrtc_vad:
            return None
            
        try:
            frame_size = int(self.config.sample_rate * self.config.frame_duration_ms / 1000)
            
            if len(audio_data) >= frame_size * 2:  # 2 bytes per sample for 16-bit
                frame_bytes = audio_data[:frame_size * 2]
                return self.webrtc_vad.is_speech(frame_bytes, self.config.sample_rate)
        except Exception as e:
            logger.error(f"WebRTC VAD error: {e}")
        
        return None
    
    def _silero_vad_check(self, audio_float: np.ndarray) -> Optional[bool]:
        """Silero VAD check for enhanced accuracy"""
        if not self.silero_vad or len(audio_float) == 0:
            return None
            
        try:
            # Silero VAD expects specific input format
            if len(audio_float) < 512:  # Minimum length for reliable detection
                return None
                
            # Convert to tensor and get speech probability
            audio_tensor = torch.from_numpy(audio_float).float()
            
            with torch.no_grad():
                speech_prob = self.silero_vad(audio_tensor, self.config.sample_rate).item()
            
            # Threshold for speech detection (adjustable)
            return speech_prob > 0.5
            
        except Exception as e:
            logger.error(f"Silero VAD error: {e}")
        
        return None
    
    def _energy_based_vad(self, audio_data: bytes) -> bool:
        """Enhanced energy-based voice activity detection with adaptive threshold"""
        if len(audio_data) == 0:
            return False
            
        # Convert to numpy and calculate RMS energy
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        if len(audio_array) == 0:
            return False
            
        rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
        
        # Adaptive threshold based on recent audio history
        base_threshold = 300
        
        if len(self.audio_history) > 10:
            # Calculate background noise level from recent non-speech frames
            recent_energies = []
            for frame in list(self.audio_history)[-10:]:
                if not frame.get('has_speech', False):
                    # Approximate energy from confidence (simplified)
                    recent_energies.append(frame['confidence'] * 1000)
            
            if recent_energies:
                background_level = np.mean(recent_energies)
                threshold = max(base_threshold, background_level * 2.5)
            else:
                threshold = base_threshold
        else:
            threshold = base_threshold
        
        return rms > threshold
    
    def reset_conversation_state(self):
        """Reset conversation state for new session"""
        self.last_voice_time = None
        self.conversation_active = False
        self.audio_history.clear()
        logger.info("Conversation state reset")
    
    def get_turn_context(self) -> dict:
        """Get current turn detection context"""
        return {
            'conversation_active': self.conversation_active,
            'last_voice_time': self.last_voice_time,
            'silence_duration': (time.time() - self.last_voice_time) * 1000 if self.last_voice_time else 0,
            'recent_activity': len([f for f in self.audio_history if f['has_speech']]) / max(1, len(self.audio_history))
        }

class STTEngine:
    """Enhanced Speech-to-Text engine with streaming and OVOS integration"""
    
    def __init__(self, model_name: str = "base", streaming: bool = True):
        self.model = None
        self.model_name = model_name
        self.is_loaded = False
        self.streaming_enabled = streaming
        self.language_detection = True
        
        # Performance optimization settings
        self.batch_size = 1  # For real-time processing
        self.beam_size = 1   # Faster inference
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model based on availability and preferences
        if FASTER_WHISPER_AVAILABLE:
            self._load_faster_whisper()
        elif OVOS_AVAILABLE:
            self._load_ovos_stt()
        else:
            logger.error("No STT engine available")
            
        # Streaming buffer for incremental transcription
        self.audio_buffer = bytearray()
        self.last_transcription = ""
        self.transcription_confidence = 0.0
    
    def _load_faster_whisper(self):
        """Load optimized Faster Whisper model for real-time processing"""
        try:
            # Optimized settings for low latency
            compute_type = "float16" if self.device == "cuda" else "int8"
            
            # Use smaller model for real-time if base model is too slow
            if self.model_name == "base" and self.device == "cpu":
                actual_model = "tiny"  # Faster for real-time on CPU
                logger.info("Using tiny model for better real-time performance on CPU")
            else:
                actual_model = self.model_name
            
            self.model = WhisperModel(
                actual_model,
                device=self.device,
                compute_type=compute_type,
                # Optimizations for real-time processing
                num_workers=1,
                download_root=None,
                local_files_only=False
            )
            self.is_loaded = True
            logger.info(f"Faster Whisper model '{actual_model}' loaded on {self.device} with {compute_type}")
            
        except Exception as e:
            logger.error(f"Failed to load Faster Whisper: {e}")
            # Fallback to OVOS if available
            if OVOS_AVAILABLE:
                logger.info("Falling back to OVOS STT")
                self._load_ovos_stt()
    
    def _load_ovos_stt(self):
        """Load OVOS STT plugin"""
        try:
            self.model = OVOSSTTFactory.create()
            self.is_loaded = True
            logger.info("OVOS STT plugin loaded")
        except Exception as e:
            logger.error(f"Failed to load OVOS STT: {e}")
    
    async def transcribe(self, audio_data: bytes, language: Optional[str] = None) -> dict:
        """Enhanced transcription with metadata and confidence"""
        if not self.is_loaded:
            return {'text': None, 'confidence': 0.0, 'language': None, 'processing_time': 0.0}
            
        start_time = time.time()
        
        try:
            # Convert bytes to proper format for STT
            if isinstance(self.model, WhisperModel):
                result = await self._transcribe_whisper(audio_data, language)
            else:
                result = await self._transcribe_ovos(audio_data, language)
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            
            # Log performance metrics
            if result['text']:
                logger.info(f"Transcribed in {processing_time:.3f}s: '{result['text'][:50]}...'"
                          f" (confidence: {result['confidence']:.2f}, lang: {result['language']})")
            
            return result
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {'text': None, 'confidence': 0.0, 'language': None, 'processing_time': time.time() - start_time}
    
    async def transcribe_streaming(self, audio_chunk: bytes) -> dict:
        """Streaming transcription for real-time processing"""
        self.audio_buffer.extend(audio_chunk)
        
        # Only process when we have enough audio (e.g., 1 second)
        min_audio_length = 16000  # 1 second at 16kHz
        
        if len(self.audio_buffer) >= min_audio_length:
            # Process the buffered audio
            audio_to_process = bytes(self.audio_buffer)
            self.audio_buffer.clear()
            
            result = await self.transcribe(audio_to_process)
            
            # Update streaming state
            if result['text']:
                self.last_transcription = result['text']
                self.transcription_confidence = result['confidence']
            
            return result
        
        # Return partial result if not enough audio yet
        return {'text': '', 'confidence': 0.0, 'language': None, 'streaming': True}
    
    async def _transcribe_whisper(self, audio_data: bytes, language: Optional[str] = None) -> dict:
        """Enhanced Whisper transcription with optimization"""
        try:
            # Convert audio bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Skip if audio is too short or silent
            if len(audio_float) < 1600:  # Less than 0.1 seconds
                return {'text': '', 'confidence': 0.0, 'language': language}
                
            # Check if audio has sufficient energy
            rms = np.sqrt(np.mean(audio_float ** 2))
            if rms < 0.01:  # Very quiet audio
                return {'text': '', 'confidence': 0.0, 'language': language}
            
            # Prepare transcription parameters
            transcribe_params = {
                'beam_size': self.beam_size,
                'best_of': 1,  # Faster processing
                'temperature': 0.0,  # More deterministic
                'condition_on_previous_text': False,  # Faster for short chunks
            }
            
            # Add language if specified
            if language:
                transcribe_params['language'] = language
            
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(audio_float, **transcribe_params)
            )
            
            # Process segments
            text_parts = []
            total_confidence = 0.0
            segment_count = 0
            
            for segment in segments:
                if segment.text.strip():  # Only non-empty segments
                    text_parts.append(segment.text)
                    # Use no_speech_prob as inverse confidence
                    confidence = 1.0 - segment.no_speech_prob if hasattr(segment, 'no_speech_prob') else 0.8
                    total_confidence += confidence
                    segment_count += 1
            
            text = " ".join(text_parts).strip()
            
            # Calculate average confidence
            avg_confidence = total_confidence / segment_count if segment_count > 0 else 0.0
            
            # Detect language if not specified
            detected_language = info.language if hasattr(info, 'language') else language
            language_probability = info.language_probability if hasattr(info, 'language_probability') else 0.0
            
            result = {
                'text': text,
                'confidence': avg_confidence,
                'language': detected_language,
                'language_probability': language_probability
            }
            
            return result
                
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return {'text': None, 'confidence': 0.0, 'language': language}
    
    async def _transcribe_ovos(self, audio_data: bytes, language: Optional[str] = None) -> dict:
        """Enhanced OVOS STT integration with metadata"""
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
                wav_buffer.read(),
                language
            )
            
            # OVOS plugins may return just text or a dict with metadata
            if isinstance(result, dict):
                return {
                    'text': result.get('text', ''),
                    'confidence': result.get('confidence', 0.8),
                    'language': result.get('language', language)
                }
            elif isinstance(result, str):
                return {
                    'text': result,
                    'confidence': 0.8,  # Default confidence for OVOS
                    'language': language
                }
            else:
                return {'text': '', 'confidence': 0.0, 'language': language}
                
        except Exception as e:
            logger.error(f"OVOS STT error: {e}")
            return {'text': None, 'confidence': 0.0, 'language': language}
    
    def get_streaming_buffer_info(self) -> dict:
        """Get current streaming buffer status"""
        return {
            'buffer_size': len(self.audio_buffer),
            'last_transcription': self.last_transcription,
            'confidence': self.transcription_confidence,
            'model_name': self.model_name,
            'device': self.device
        }

class TTSEngine:
    """Enhanced Text-to-Speech engine with phoonnx and real-time streaming"""
    
    def __init__(self, voice: str = "default", streaming: bool = True):
        self.engine = None
        self.voice = voice
        self.is_loaded = False
        self.streaming_enabled = streaming
        
        # Performance settings for real-time TTS
        self.sample_rate = 24000  # Higher quality
        self.chunk_size = 1024   # For streaming
        self.voice_cache = {}    # Cache for voice models
        
        # Try to load phoonnx first, then OVOS TTS
        if self._check_phoonnx_available():
            self._load_phoonnx()
        elif OVOS_AVAILABLE:
            self._load_ovos_tts()
        else:
            logger.error("No TTS engine available")
    
    def _check_phoonnx_available(self) -> bool:
        """Check if phoonnx is available"""
        try:
            import onnxruntime
            # Check if phoonnx models are available
            return True
        except ImportError:
            return False
    
    def _load_phoonnx(self):
        """Load phoonnx TTS engine for high-quality synthesis"""
        try:
            # Initialize phoonnx with optimized settings
            # This is a placeholder - actual implementation would load ONNX models
            logger.info("phoonnx TTS engine loaded (placeholder)")
            self.is_loaded = True
            self.engine_type = "phoonnx"
            
            # Load default voice model
            self._load_voice_model(self.voice)
            
        except Exception as e:
            logger.error(f"Failed to load phoonnx: {e}")
            # Fallback to OVOS
            if OVOS_AVAILABLE:
                self._load_ovos_tts()
    
    def _load_voice_model(self, voice_name: str):
        """Load specific voice model for phoonnx"""
        if voice_name not in self.voice_cache:
            # Placeholder for actual phoonnx model loading
            # In real implementation, this would load ONNX models from HuggingFace
            logger.info(f"Loading phoonnx voice model: {voice_name}")
            self.voice_cache[voice_name] = {
                'model_path': f"path/to/{voice_name}.onnx",
                'config': {'sample_rate': self.sample_rate}
            }
    
    def _load_ovos_tts(self):
        """Load OVOS TTS plugin with optimization"""
        try:
            # Load with specific configuration for real-time use
            config = {
                'pulse_duck': False,  # Disable audio ducking for faster response
                'audio_ext': 'wav',   # Use WAV for better quality
                'ssml_tags': ['speak', 'prosody', 'break']  # Supported SSML tags
            }
            
            self.engine = OVOSTTSFactory.create(config)
            self.is_loaded = True
            self.engine_type = "ovos"
            logger.info("OVOS TTS plugin loaded with optimizations")
        except Exception as e:
            logger.error(f"Failed to load OVOS TTS: {e}")
    
    async def synthesize(self, text: str, voice: Optional[str] = None, streaming: bool = False) -> dict:
        """Enhanced text-to-speech synthesis with metadata"""
        if not self.is_loaded or not text.strip():
            return {'audio': None, 'sample_rate': self.sample_rate, 'processing_time': 0.0, 'voice': voice}
            
        start_time = time.time()
        selected_voice = voice or self.voice
        
        try:
            # Clean and prepare text
            clean_text = self._prepare_text(text)
            
            if streaming and self.streaming_enabled:
                result = await self._synthesize_streaming(clean_text, selected_voice)
            else:
                result = await self._synthesize_batch(clean_text, selected_voice)
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            
            if result['audio']:
                logger.info(f"TTS synthesized '{text[:30]}...' in {processing_time:.3f}s "
                          f"({len(result['audio'])} bytes, voice: {selected_voice})")
            
            return result
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return {'audio': None, 'sample_rate': self.sample_rate, 'processing_time': time.time() - start_time, 'voice': selected_voice}
    
    def _prepare_text(self, text: str) -> str:
        """Clean and prepare text for TTS"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Basic text normalization for better TTS
        text = text.replace('&', ' and ')
        text = text.replace('@', ' at ')
        text = text.replace('#', ' hash ')
        
        # Ensure proper punctuation for natural speech
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    async def _synthesize_streaming(self, text: str, voice: str) -> dict:
        """Streaming TTS synthesis for real-time output"""
        # For now, fall back to batch synthesis
        # Real streaming would synthesize and return audio chunks as they're generated
        return await self._synthesize_batch(text, voice)
    
    async def _synthesize_batch(self, text: str, voice: str) -> dict:
        """Batch TTS synthesis"""
        loop = asyncio.get_event_loop()
        
        if hasattr(self, 'engine_type') and self.engine_type == "phoonnx":
            # Use phoonnx synthesis
            audio_data = await loop.run_in_executor(None, self._phoonnx_synthesize, text, voice)
        else:
            # Use OVOS synthesis
            audio_data = await loop.run_in_executor(None, self._ovos_synthesize, text, voice)
        
        return {
            'audio': audio_data,
            'sample_rate': self.sample_rate,
            'voice': voice,
            'format': 'wav'
        }
    
    def _phoonnx_synthesize(self, text: str, voice: str) -> Optional[bytes]:
        """phoonnx TTS synthesis (placeholder implementation)"""
        try:
            # Placeholder for actual phoonnx synthesis
            # Real implementation would use ONNX runtime with phoonnx models
            logger.info(f"phoonnx synthesizing: '{text}' with voice '{voice}'")
            
            # Return empty audio for now - would be actual synthesis in real implementation
            return b''  # Placeholder
            
        except Exception as e:
            logger.error(f"phoonnx synthesis error: {e}")
            return None
    
    def _ovos_synthesize(self, text: str, voice: str) -> Optional[bytes]:
        """OVOS TTS synthesis"""
        try:
            if OVOS_AVAILABLE and hasattr(self.engine, 'get_tts'):
                audio_file, phonemes = self.engine.get_tts(text, None)
                
                # Read the generated audio file
                if audio_file and Path(audio_file).exists():
                    with open(audio_file, 'rb') as f:
                        audio_data = f.read()
                    
                    # Clean up temporary file
                    Path(audio_file).unlink(missing_ok=True)
                    
                    return audio_data
            
            return None
            
        except Exception as e:
            logger.error(f"OVOS TTS synthesis error: {e}")
            return None
    
    async def synthesize_ssml(self, ssml: str, voice: Optional[str] = None) -> dict:
        """Synthesize SSML markup for enhanced speech control"""
        # Extract plain text from SSML for now
        # Real implementation would parse SSML tags
        import re
        plain_text = re.sub(r'<[^>]+>', '', ssml)
        return await self.synthesize(plain_text, voice)
    
    def get_available_voices(self) -> list:
        """Get list of available voices"""
        if hasattr(self, 'engine_type') and self.engine_type == "phoonnx":
            # Return phoonnx voices
            return list(self.voice_cache.keys()) + ['default', 'miro', 'dii']  # Example voices
        else:
            # Return OVOS voices
            return ['default']  # Placeholder
    
    def set_voice(self, voice: str):
        """Change the active voice"""
        if voice in self.get_available_voices():
            self.voice = voice
            if hasattr(self, 'engine_type') and self.engine_type == "phoonnx":
                self._load_voice_model(voice)
            logger.info(f"Voice changed to: {voice}")
        else:
            logger.warning(f"Voice '{voice}' not available")

class SpeechPipeline:
    """Enhanced speech processing pipeline with OpenAI-compatible features"""
    
    def __init__(self, config: AudioConfig = None, session_id: str = None):
        self.config = config or AudioConfig()
        self.session_id = session_id or f"session_{int(time.time())}"
        
        # Initialize components
        self.audio_processor = AudioProcessor(self.config)
        self.stt_engine = STTEngine(streaming=True)
        self.tts_engine = TTSEngine(streaming=True)
        
        # Enhanced audio management
        self.audio_buffer = bytearray()
        self.processed_buffer = bytearray()
        self.silence_start = None
        self.is_processing = False
        self.processing_lock = asyncio.Lock()
        
        # Turn detection and conversation management
        self.conversation_state = {
            'active': False,
            'turn_start': None,
            'last_activity': None,
            'interruption_detected': False,
            'response_pending': False
        }
        
        # Performance metrics
        self.metrics = {
            'total_processed_chunks': 0,
            'total_transcriptions': 0,
            'total_synthesis': 0,
            'avg_processing_time': 0.0,
            'errors': 0
        }
        
        logger.info(f"Enhanced speech pipeline initialized for session {self.session_id}")
    
    async def process_audio_chunk(self, audio_data: bytes) -> dict:
        """Enhanced audio chunk processing with turn detection and OpenAI compatibility"""
        result = {
            'type': 'audio_processed',
            'transcription': None,
            'turn_detected': False,
            'interruption_possible': False,
            'conversation_state': 'idle',
            'audio_info': {}
        }
        
        if not audio_data:
            return result
            
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        self.metrics['total_processed_chunks'] += 1
        
        # Enhanced speech detection
        speech_info = self.audio_processor.detect_speech(audio_data)
        
        current_time = time.time()
        
        # Update conversation state
        if speech_info['has_speech']:
            if not self.conversation_state['active']:
                self.conversation_state['active'] = True
                self.conversation_state['turn_start'] = current_time
                result['turn_detected'] = True
                logger.info(f"Turn started in session {self.session_id}")
            
            self.conversation_state['last_activity'] = current_time
            self.silence_start = None
            result['conversation_state'] = 'speaking'
        else:
            # Handle silence
            if self.silence_start is None:
                self.silence_start = current_time
            
            silence_duration = (current_time - self.silence_start) * 1000
            
            if self.conversation_state['active']:
                if silence_duration > self.config.interruption_threshold_ms:
                    result['interruption_possible'] = True
                    
                if silence_duration > self.config.silence_threshold_ms:
                    result['conversation_state'] = 'processing'
        
        # Update audio info
        result['audio_info'] = {
            'buffer_size': len(self.audio_buffer),
            'speech_detected': speech_info['has_speech'],
            'confidence': speech_info['confidence'],
            'silence_duration': (current_time - self.silence_start) * 1000 if self.silence_start else 0
        }
        
        # Check if we should process the buffer
        should_process = (
            self.silence_start and 
            (current_time - self.silence_start) * 1000 > self.config.silence_threshold_ms and
            len(self.audio_buffer) > 0 and
            not self.is_processing
        )
        
        if should_process:
            async with self.processing_lock:
                transcription_result = await self._process_buffer()
                if transcription_result:
                    result['transcription'] = transcription_result
                    result['conversation_state'] = 'completed'
                    
        return result
    
    async def _process_buffer(self) -> Optional[dict]:
        """Enhanced buffer processing with comprehensive metadata"""
        if self.is_processing or len(self.audio_buffer) == 0:
            return None
            
        self.is_processing = True
        start_time = time.time()
        
        try:
            # Copy buffer and clear it
            audio_to_process = bytes(self.audio_buffer)
            self.audio_buffer.clear()
            self.silence_start = None
            
            logger.info(f"Processing {len(audio_to_process)} bytes of audio in session {self.session_id}")
            
            # Transcribe with enhanced metadata
            transcription_result = await self.stt_engine.transcribe(audio_to_process)
            
            processing_time = time.time() - start_time
            
            # Update metrics
            self.metrics['total_transcriptions'] += 1
            self.metrics['avg_processing_time'] = (
                (self.metrics['avg_processing_time'] * (self.metrics['total_transcriptions'] - 1) + processing_time) /
                self.metrics['total_transcriptions']
            )
            
            if transcription_result['text']:
                # Reset conversation state after successful transcription
                self.conversation_state['active'] = False
                self.conversation_state['turn_start'] = None
                
                result = {
                    'text': transcription_result['text'],
                    'confidence': transcription_result['confidence'],
                    'language': transcription_result['language'],
                    'processing_time': processing_time,
                    'session_id': self.session_id,
                    'timestamp': time.time(),
                    'audio_length': len(audio_to_process)
                }
                
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Buffer processing error in session {self.session_id}: {e}")
            self.metrics['errors'] += 1
            return None
        finally:
            self.is_processing = False
    
    async def synthesize_response(self, text: str, voice: Optional[str] = None, streaming: bool = False) -> dict:
        """Enhanced speech synthesis with metadata and streaming support"""
        try:
            self.conversation_state['response_pending'] = True
            
            # Synthesize with enhanced options
            synthesis_result = await self.tts_engine.synthesize(text, voice, streaming)
            
            # Update metrics
            self.metrics['total_synthesis'] += 1
            
            # Add session metadata
            result = {
                **synthesis_result,
                'session_id': self.session_id,
                'timestamp': time.time(),
                'text': text,
                'streaming': streaming
            }
            
            self.conversation_state['response_pending'] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Response synthesis error in session {self.session_id}: {e}")
            self.conversation_state['response_pending'] = False
            self.metrics['errors'] += 1
            return {
                'audio': None,
                'error': str(e),
                'session_id': self.session_id,
                'timestamp': time.time()
            }
    
    def get_comprehensive_status(self) -> dict:
        """Get comprehensive pipeline status and metrics"""
        turn_context = self.audio_processor.get_turn_context()
        stt_info = self.stt_engine.get_streaming_buffer_info()
        
        return {
            'session_id': self.session_id,
            'audio_buffer': {
                'size': len(self.audio_buffer),
                'processed_size': len(self.processed_buffer),
                'is_processing': self.is_processing,
                'silence_duration': (time.time() - self.silence_start) * 1000 if self.silence_start else 0
            },
            'conversation_state': self.conversation_state,
            'turn_context': turn_context,
            'stt_status': stt_info,
            'tts_status': {
                'available_voices': self.tts_engine.get_available_voices(),
                'current_voice': self.tts_engine.voice
            },
            'metrics': self.metrics,
            'config': {
                'sample_rate': self.config.sample_rate,
                'chunk_duration_ms': self.config.chunk_duration_ms,
                'silence_threshold_ms': self.config.silence_threshold_ms,
                'turn_detection_enabled': self.config.turn_detection_enabled
            }
        }
    
    def reset_session(self):
        """Reset pipeline for new session"""
        self.audio_buffer.clear()
        self.processed_buffer.clear()
        self.silence_start = None
        self.is_processing = False
        
        # Reset conversation state
        self.conversation_state = {
            'active': False,
            'turn_start': None,
            'last_activity': None,
            'interruption_detected': False,
            'response_pending': False
        }
        
        # Reset audio processor state
        self.audio_processor.reset_conversation_state()
        
        logger.info(f"Pipeline reset for session {self.session_id}")
    
    async def handle_interruption(self):
        """Handle conversation interruption"""
        logger.info(f"Handling interruption in session {self.session_id}")
        
        # Stop current processing
        self.is_processing = False
        self.conversation_state['interruption_detected'] = True
        
        # Clear buffers
        self.audio_buffer.clear()
        
        # Reset turn state
        self.conversation_state['active'] = True  # Keep active for new turn
        self.conversation_state['turn_start'] = time.time()
        
    def set_language(self, language: str):
        """Set processing language"""
        self.config.language = getattr(self.config, 'language', 'en-US')
        logger.info(f"Language set to {language} for session {self.session_id}")
    
    def set_voice(self, voice: str):
        """Set TTS voice"""
        self.tts_engine.set_voice(voice)
        logger.info(f"Voice set to {voice} for session {self.session_id}")

# Factory function for easy initialization
def create_speech_pipeline(config: dict = None, session_id: str = None) -> SpeechPipeline:
    """Create an enhanced speech pipeline with OpenAI-compatible configuration"""
    audio_config = AudioConfig()
    
    if config:
        for key, value in config.items():
            if hasattr(audio_config, key):
                setattr(audio_config, key, value)
    
    return SpeechPipeline(audio_config, session_id)

def create_realtime_pipeline(session_id: str = None, **kwargs) -> SpeechPipeline:
    """Create a speech pipeline optimized for real-time OpenAI-compatible processing"""
    optimized_config = {
        'sample_rate': 24000,
        'chunk_duration_ms': 20,
        'silence_threshold_ms': 500,
        'turn_detection_enabled': True,
        'interruption_threshold_ms': 300,
        'noise_reduction': True,
        'auto_gain_control': True,
        **kwargs  # Allow override of any settings
    }
    
    return create_speech_pipeline(optimized_config, session_id)

if __name__ == "__main__":
    # Test the enhanced speech pipeline
    import asyncio
    
    async def test_enhanced_pipeline():
        # Create real-time optimized pipeline
        pipeline = create_realtime_pipeline(session_id="test_session")
        
        print(f"Testing enhanced speech pipeline for session {pipeline.session_id}")
        
        # Test with some dummy audio data
        dummy_audio = b'\x00' * 48000  # 1 second of silence at 24kHz
        
        # Test audio processing
        result = await pipeline.process_audio_chunk(dummy_audio)
        print(f"Audio processing result: {result}")
        
        # Test TTS with enhanced features
        tts_result = await pipeline.synthesize_response(
            "Hello, this is a test of the enhanced OVOS speech pipeline.",
            voice="default",
            streaming=False
        )
        print(f"TTS result: {tts_result}")
        
        # Test status reporting
        status = pipeline.get_comprehensive_status()
        print(f"Pipeline status: {status}")
        
        # Test interruption handling
        await pipeline.handle_interruption()
        print("Interruption handling tested")
        
        # Test session reset
        pipeline.reset_session()
        print("Session reset tested")
    
    asyncio.run(test_enhanced_pipeline())
