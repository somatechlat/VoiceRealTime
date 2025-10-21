#!/usr/bin/env python3
"""
Rate Limiter - Sprint E
Token-based rate limiting for OpenAI Realtime API compatibility
"""

import time
from typing import Dict, Tuple


class RateLimiter:
    """Rate limiter with token and request counting"""
    
    def __init__(self):
        self.limits: Dict[str, dict] = {}
        self.usage: Dict[str, dict] = {}
    
    def set_limits(self, session_id: str, requests_per_minute: int = 100, tokens_per_minute: int = 10000):
        """Set rate limits for a session"""
        self.limits[session_id] = {
            "requests": requests_per_minute,
            "tokens": tokens_per_minute,
            "window_start": time.time()
        }
        self.usage[session_id] = {
            "requests": 0,
            "tokens": 0
        }
    
    def _reset_if_needed(self, session_id: str):
        """Reset counters if window expired"""
        if session_id not in self.limits:
            return
        
        window_start = self.limits[session_id]["window_start"]
        if time.time() - window_start >= 60:
            self.limits[session_id]["window_start"] = time.time()
            self.usage[session_id] = {"requests": 0, "tokens": 0}
    
    def check_limit(self, session_id: str, tokens: int = 0) -> Tuple[bool, dict]:
        """Check if request is within limits"""
        if session_id not in self.limits:
            self.set_limits(session_id)
        
        self._reset_if_needed(session_id)
        
        usage = self.usage[session_id]
        limits = self.limits[session_id]
        
        requests_remaining = limits["requests"] - usage["requests"]
        tokens_remaining = limits["tokens"] - usage["tokens"]
        
        window_elapsed = time.time() - limits["window_start"]
        reset_seconds = max(0, 60 - window_elapsed)
        
        allowed = (usage["requests"] < limits["requests"] and 
                  usage["tokens"] + tokens <= limits["tokens"])
        
        return allowed, {
            "requests_limit": limits["requests"],
            "requests_remaining": max(0, requests_remaining),
            "tokens_limit": limits["tokens"],
            "tokens_remaining": max(0, tokens_remaining - tokens),
            "reset_seconds": reset_seconds
        }
    
    def consume(self, session_id: str, tokens: int = 0):
        """Consume rate limit quota"""
        if session_id not in self.usage:
            self.set_limits(session_id)
        
        self.usage[session_id]["requests"] += 1
        self.usage[session_id]["tokens"] += tokens
    
    def get_limits(self, session_id: str) -> dict:
        """Get current limits and usage"""
        if session_id not in self.limits:
            self.set_limits(session_id)
        
        self._reset_if_needed(session_id)
        
        usage = self.usage[session_id]
        limits = self.limits[session_id]
        
        window_elapsed = time.time() - limits["window_start"]
        reset_seconds = max(0, 60 - window_elapsed)
        
        return {
            "requests_limit": limits["requests"],
            "requests_remaining": max(0, limits["requests"] - usage["requests"]),
            "tokens_limit": limits["tokens"],
            "tokens_remaining": max(0, limits["tokens"] - usage["tokens"]),
            "reset_seconds": reset_seconds
        }


def count_tokens(text: str) -> int:
    """Approximate token count (1 token ≈ 4 characters)"""
    return max(1, len(text) // 4)
