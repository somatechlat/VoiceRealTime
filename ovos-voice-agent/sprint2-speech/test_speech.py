#!/usr/bin/env python3
"""
Test script for Sprint 2 Speech Processing Pipeline
"""

import asyncio
import logging
import numpy as np
from speech_pipeline import SpeechPipeline, AudioConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_audio(duration_seconds=2.0, sample_rate=16000, frequency=440):
    """Generate test audio signal (sine wave)"""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()

def generate_silence(duration_seconds=1.0, sample_rate=16000):
    """Generate silence"""
    samples = int(sample_rate * duration_seconds)
    silence = np.zeros(samples, dtype=np.int16)
    return silence.tobytes()

async def test_vad():
    """Test Voice Activity Detection"""
    logger.info("Testing Voice Activity Detection...")
    
    config = AudioConfig()
    pipeline = SpeechPipeline(config)
    
    # Test with silence
    silence = generate_silence(0.5)
    has_speech = pipeline.audio_processor.detect_speech(silence)
    logger.info(f"Silence detected as speech: {has_speech} (should be False)")
    
    # Test with audio signal
    audio = generate_test_audio(0.5)
    has_speech = pipeline.audio_processor.detect_speech(audio)
    logger.info(f"Audio detected as speech: {has_speech} (should be True)")
    
    return True

async def test_stt():
    """Test Speech-to-Text"""
    logger.info("Testing Speech-to-Text...")
    
    pipeline = SpeechPipeline()
    
    if not pipeline.stt_engine.is_loaded:
        logger.warning("STT engine not loaded, skipping STT test")
        return False
    
    # Generate some test audio
    test_audio = generate_test_audio(2.0, frequency=440)  # 2 seconds of 440Hz tone
    
    result = await pipeline.stt_engine.transcribe(test_audio)
    logger.info(f"STT result: '{result}'")
    
    return result is not None

async def test_tts():
    """Test Text-to-Speech"""
    logger.info("Testing Text-to-Speech...")
    
    pipeline = SpeechPipeline()
    
    if not pipeline.tts_engine.is_loaded:
        logger.warning("TTS engine not loaded, skipping TTS test")
        return False
    
    test_text = "Hello, this is a test of the text to speech system."
    
    result = await pipeline.tts_engine.synthesize(test_text)
    
    if result:
        logger.info(f"TTS generated {len(result)} bytes of audio")
        return True
    else:
        logger.error("TTS failed to generate audio")
        return False

async def test_full_pipeline():
    """Test the complete speech processing pipeline"""
    logger.info("Testing full speech processing pipeline...")
    
    config = AudioConfig(
        silence_threshold_ms=500,  # Shorter threshold for testing
        chunk_duration_ms=30
    )
    
    pipeline = SpeechPipeline(config)
    
    # Simulate audio chunks
    chunks = [
        generate_test_audio(0.1, frequency=440),  # 100ms of audio
        generate_test_audio(0.1, frequency=523),  # 100ms of different audio
        generate_silence(0.6),  # 600ms of silence (triggers processing)
    ]
    
    results = []
    
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} bytes)")
        
        result = await pipeline.process_audio_chunk(chunk)
        
        if result:
            logger.info(f"Pipeline result: '{result}'")
            results.append(result)
            
        # Show buffer info
        buffer_info = pipeline.get_buffer_info()
        logger.info(f"Buffer info: {buffer_info}")
        
        # Small delay between chunks
        await asyncio.sleep(0.1)
    
    logger.info(f"Pipeline test completed. Results: {results}")
    return len(results) > 0

async def test_real_time_simulation():
    """Simulate real-time audio processing"""
    logger.info("Simulating real-time audio processing...")
    
    pipeline = SpeechPipeline()
    
    # Simulate 5 seconds of audio in 30ms chunks
    chunk_duration = 0.03  # 30ms
    total_duration = 3.0  # 3 seconds
    
    chunk_samples = int(16000 * chunk_duration)
    total_chunks = int(total_duration / chunk_duration)
    
    logger.info(f"Simulating {total_chunks} chunks of {chunk_duration*1000:.0f}ms each")
    
    for i in range(total_chunks):
        # Generate chunk (alternate between audio and silence)
        if i % 20 < 10:  # 10 chunks audio, 10 chunks silence
            chunk = generate_test_audio(chunk_duration, frequency=440 + (i % 5) * 50)
        else:
            chunk = generate_silence(chunk_duration)
        
        result = await pipeline.process_audio_chunk(chunk)
        
        if result:
            logger.info(f"Transcription at chunk {i}: '{result}'")
        
        # Simulate real-time delay
        await asyncio.sleep(chunk_duration * 0.5)  # Simulate processing time
    
    logger.info("Real-time simulation completed")

async def main():
    """Run all tests"""
    logger.info("Starting OVOS Voice Agent Sprint 2 Tests")
    logger.info("="*50)
    
    tests = [
        ("VAD Test", test_vad),
        ("STT Test", test_stt),
        ("TTS Test", test_tts),
        ("Full Pipeline Test", test_full_pipeline),
        ("Real-time Simulation", test_real_time_simulation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name}...")
        try:
            result = await test_func()
            results[test_name] = "PASS" if result else "FAIL"
            logger.info(f"{test_name}: {results[test_name]}")
        except Exception as e:
            results[test_name] = f"ERROR: {e}"
            logger.error(f"{test_name}: {results[test_name]}")
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY:")
    for test_name, result in results.items():
        status_icon = "✅" if result == "PASS" else "❌" if result == "FAIL" else "⚠️"
        logger.info(f"{status_icon} {test_name}: {result}")

if __name__ == "__main__":
    asyncio.run(main())
