#!/usr/bin/env python3
"""
Audio Codecs - Sprint F
G.711 μ-law and A-law codec support for OpenAI Realtime API
"""

import audioop
import struct
from typing import Optional


class AudioCodec:
    """Audio codec handler for multiple formats"""
    
    @staticmethod
    def encode_pcm16(data: bytes, sample_rate: int = 24000) -> bytes:
        """PCM16 passthrough"""
        return data
    
    @staticmethod
    def decode_pcm16(data: bytes, sample_rate: int = 24000) -> bytes:
        """PCM16 passthrough"""
        return data
    
    @staticmethod
    def encode_g711_ulaw(pcm_data: bytes) -> bytes:
        """Encode PCM16 to G.711 μ-law"""
        return audioop.lin2ulaw(pcm_data, 2)
    
    @staticmethod
    def decode_g711_ulaw(ulaw_data: bytes) -> bytes:
        """Decode G.711 μ-law to PCM16"""
        return audioop.ulaw2lin(ulaw_data, 2)
    
    @staticmethod
    def encode_g711_alaw(pcm_data: bytes) -> bytes:
        """Encode PCM16 to G.711 A-law"""
        return audioop.lin2alaw(pcm_data, 2)
    
    @staticmethod
    def decode_g711_alaw(alaw_data: bytes) -> bytes:
        """Decode G.711 A-law to PCM16"""
        return audioop.alaw2lin(alaw_data, 2)
    
    @staticmethod
    def resample(data: bytes, from_rate: int, to_rate: int, channels: int = 1) -> bytes:
        """Resample audio data"""
        if from_rate == to_rate:
            return data
        
        state = None
        resampled, state = audioop.ratecv(data, 2, channels, from_rate, to_rate, state)
        return resampled
    
    @staticmethod
    def get_codec(format_type: str):
        """Get encoder/decoder for format"""
        codecs = {
            "pcm16": (AudioCodec.encode_pcm16, AudioCodec.decode_pcm16),
            "g711_ulaw": (AudioCodec.encode_g711_ulaw, AudioCodec.decode_g711_ulaw),
            "g711_alaw": (AudioCodec.encode_g711_alaw, AudioCodec.decode_g711_alaw),
        }
        return codecs.get(format_type, (AudioCodec.encode_pcm16, AudioCodec.decode_pcm16))


class AudioFormatConverter:
    """Convert between audio formats"""
    
    def __init__(self):
        self.codec = AudioCodec()
    
    def convert(self, data: bytes, from_format: str, to_format: str, 
                from_rate: int = 24000, to_rate: int = 24000) -> bytes:
        """Convert audio from one format to another"""
        
        # Decode to PCM16
        _, decoder = self.codec.get_codec(from_format)
        pcm_data = decoder(data, from_rate)
        
        # Resample if needed
        if from_rate != to_rate:
            pcm_data = self.codec.resample(pcm_data, from_rate, to_rate)
        
        # Encode to target format
        encoder, _ = self.codec.get_codec(to_format)
        return encoder(pcm_data, to_rate)
