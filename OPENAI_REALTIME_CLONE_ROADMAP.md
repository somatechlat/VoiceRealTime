# OpenAI Realtime API - Complete Implementation Roadmap

**Goal:** 100% compatible clone of OpenAI Realtime API  
**Current Status:** 45% complete  
**Target:** Full feature parity with OpenAI's Realtime API  
**Approach:** Parallel sprint execution

---

## Current State Assessment

### ✅ Working (45%)
- Basic WebSocket protocol
- Audio streaming (PCM16 24kHz)
- Session management
- Basic conversation flow
- TTS/STT pipeline
- Response cancellation
- Turn detection (VAD)

### ❌ Missing (55%)
- Function calling (0%)
- Conversation item truncate/delete (broken - code crashes)
- Transcription events (incomplete)
- Rate limiting (fake)
- Config field usage (instructions, temperature, modalities ignored)
- Audio format negotiation
- Error code taxonomy

---

## Sprint Structure

**Duration:** 6 weeks total  
**Sprints:** 6 parallel tracks  
**Team Capacity:** Unlimited parallel execution  

---

## SPRINT A: Critical Fixes (Week 1)

**Priority:** CRITICAL - Code currently crashes  
**Files:** `sprint4-websocket/realtime_server.py`, `enterprise/app/transports/realtime_ws.py`

### Tasks

#### A1: Implement conversation.item.truncate
```python
async def handle_conversation_item_truncate(self, session_id: str, event: dict):
    item_id = event.get("item_id")
    content_index = event.get("content_index")
    audio_end_ms = event.get("audio_end_ms")
    
    conversation_state = self.manager.conversation_states.get(session_id, {})
    items = conversation_state.get("items", [])
    
    for item in items:
        if item["id"] == item_id:
            if content_index is not None:
                item["content"] = item["content"][:content_index + 1]
            
            await self.manager.send_event(session_id, {
                "type": "conversation.item.truncated",
                "item_id": item_id,
                "content_index": content_index,
                "audio_end_ms": audio_end_ms
            })
            break
```

#### A2: Implement conversation.item.delete
```python
async def handle_conversation_item_delete(self, session_id: str, event: dict):
    item_id = event.get("item_id")
    
    conversation_state = self.manager.conversation_states.get(session_id, {})
    items = conversation_state.get("items", [])
    
    items[:] = [item for item in items if item["id"] != item_id]
    
    await self.manager.send_event(session_id, {
        "type": "conversation.item.deleted",
        "item_id": item_id
    })
```

#### A3: Add missing server events
- `conversation.item.truncated`
- `conversation.item.deleted`

**Deliverable:** Code doesn't crash on truncate/delete events

---

## SPRINT B: Configuration Fields (Week 1-2)

**Priority:** HIGH - Config fields currently ignored  
**Files:** `llm_integration.py`, `sprint4-websocket/realtime_server.py`

### Tasks

#### B1: Use instructions field
```python
# In llm_integration.py
async def generate_ai_response(session_id: str, user_input: str, instructions: str = None):
    system_prompt = instructions or "You are a helpful assistant."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    response = await groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.8,
        max_tokens=1024
    )
    return response.choices[0].message.content
```

#### B2: Use temperature field
```python
# Pass temperature from session config to LLM
session_data = self.manager.session_data.get(session_id)
temperature = session_data.get("temperature", 0.8)
```

#### B3: Use max_output_tokens field
```python
# Pass max_output_tokens from session config to LLM
max_tokens = session_data.get("max_output_tokens", "inf")
if max_tokens != "inf":
    max_tokens = int(max_tokens)
else:
    max_tokens = 4096  # reasonable default
```

#### B4: Respect output_modalities
```python
async def generate_and_stream_audio(self, session_id: str, response_id: str, item_id: str, text: str):
    session_data = self.manager.session_data.get(session_id, {})
    modalities = session_data.get("output_modalities", ["audio"])
    
    # Send transcript if text modality enabled
    if "text" in modalities:
        await self.manager.send_event(session_id, {
            "type": "response.audio_transcript.delta",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": text
        })
    
    # Generate audio if audio modality enabled
    if "audio" in modalities:
        # ... TTS synthesis
```

**Deliverable:** All session config fields are used

---

## SPRINT C: Function Calling (Week 1-3)

**Priority:** CRITICAL - Core feature (30% of OpenAI value)  
**Files:** New file `function_calling.py`, `sprint4-websocket/realtime_server.py`

### Tasks

#### C1: Create function calling engine
```python
# function_calling.py
class FunctionCallingEngine:
    def __init__(self):
        self.functions = {}
    
    def register_function(self, name: str, schema: dict, handler: callable):
        self.functions[name] = {
            "schema": schema,
            "handler": handler
        }
    
    async def detect_function_call(self, text: str, tools: list) -> dict:
        # Use LLM to detect if text contains function call
        # Return function name and arguments
        pass
    
    async def execute_function(self, function_name: str, arguments: dict) -> dict:
        if function_name not in self.functions:
            return {"error": "Function not found"}
        
        handler = self.functions[function_name]["handler"]
        result = await handler(**arguments)
        return result
```

#### C2: Integrate function detection in conversation flow
```python
async def handle_transcription_completed(self, session_id: str, transcription: dict):
    text = transcription.get('text', '')
    
    # Check for function calls
    session_data = self.manager.session_data.get(session_id)
    tools = session_data.get("tools", [])
    
    if tools:
        function_call = await self.detect_function_call(text, tools)
        if function_call:
            await self.handle_function_call(session_id, function_call)
            return
    
    # Normal conversation flow
    # ...
```

#### C3: Add function call events
```python
async def handle_function_call(self, session_id: str, function_call: dict):
    response_id = f"resp_{uuid.uuid4().hex[:16]}"
    item_id = f"item_{uuid.uuid4().hex[:16]}"
    
    # Send function_call_arguments.delta
    await self.manager.send_event(session_id, {
        "type": "response.function_call_arguments.delta",
        "response_id": response_id,
        "item_id": item_id,
        "delta": json.dumps(function_call["arguments"])
    })
    
    # Send function_call_arguments.done
    await self.manager.send_event(session_id, {
        "type": "response.function_call_arguments.done",
        "response_id": response_id,
        "item_id": item_id,
        "arguments": json.dumps(function_call["arguments"])
    })
    
    # Execute function
    result = await self.function_engine.execute_function(
        function_call["name"],
        function_call["arguments"]
    )
    
    # Create function_call_output item
    await self.create_function_output_item(session_id, item_id, result)
```

#### C4: Support function_call and function_call_output item types
```python
# Add to conversation item creation
if item_type == "function_call":
    # Handle function call item
    pass
elif item_type == "function_call_output":
    # Handle function output item
    pass
```

**Deliverable:** Complete function calling system

---

## SPRINT D: Transcription Events (Week 2)

**Priority:** MEDIUM - Improves UX  
**Files:** `sprint4-websocket/realtime_server.py`, `sprint2-speech/speech_pipeline.py`

### Tasks

#### D1: Add transcription.started event
```python
async def handle_audio_buffer_commit(self, session_id: str, event: dict):
    item_id = f"item_{uuid.uuid4().hex[:16]}"
    
    # Send started event
    await self.manager.send_event(session_id, {
        "type": "conversation.item.input_audio_transcription.started",
        "item_id": item_id,
        "content_index": 0
    })
    
    # Process transcription
    # ...
```

#### D2: Add transcription.completed event
```python
async def handle_transcription_completed(self, session_id: str, transcription: dict):
    item_id = transcription.get("item_id")
    
    await self.manager.send_event(session_id, {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": item_id,
        "content_index": 0,
        "transcript": transcription.get("text", "")
    })
```

#### D3: Add transcription.failed event
```python
# In speech_pipeline.py - handle transcription errors
try:
    result = await self.stt_engine.transcribe(audio_data)
except Exception as e:
    await self.send_transcription_failed_event(session_id, item_id, str(e))
```

**Deliverable:** All transcription events implemented

---

## SPRINT E: Rate Limiting (Week 2-3)

**Priority:** MEDIUM - Production requirement  
**Files:** New file `rate_limiter.py`, `sprint4-websocket/realtime_server.py`

### Tasks

#### E1: Create rate limiter
```python
# rate_limiter.py
class RateLimiter:
    def __init__(self):
        self.limits = {}
        self.usage = {}
    
    def set_limits(self, session_id: str, requests_per_minute: int, tokens_per_minute: int):
        self.limits[session_id] = {
            "requests": requests_per_minute,
            "tokens": tokens_per_minute,
            "window_start": time.time()
        }
        self.usage[session_id] = {
            "requests": 0,
            "tokens": 0
        }
    
    def check_limit(self, session_id: str, tokens: int = 0) -> tuple[bool, dict]:
        # Check if within limits
        # Return (allowed, remaining_limits)
        pass
    
    def consume(self, session_id: str, tokens: int):
        # Increment usage counters
        pass
```

#### E2: Integrate rate limiting
```python
async def process_event(self, session_id: str, event: dict):
    # Check rate limits
    allowed, limits = self.rate_limiter.check_limit(session_id)
    
    if not allowed:
        await self.send_error(session_id, "rate_limit_error", "Rate limit exceeded")
        return
    
    # Process event
    # ...
    
    # Consume rate limit
    self.rate_limiter.consume(session_id, tokens_used)
```

#### E3: Send rate_limits.updated events
```python
async def send_rate_limit_update(self, session_id: str):
    limits = self.rate_limiter.get_limits(session_id)
    
    await self.manager.send_event(session_id, {
        "type": "rate_limits.updated",
        "rate_limits": {
            "requests": {
                "limit": limits["requests_limit"],
                "remaining": limits["requests_remaining"],
                "reset_seconds": limits["reset_seconds"]
            },
            "tokens": {
                "limit": limits["tokens_limit"],
                "remaining": limits["tokens_remaining"],
                "reset_seconds": limits["reset_seconds"]
            }
        }
    })
```

#### E4: Token counting
```python
def count_tokens(text: str) -> int:
    # Simple approximation: 1 token ≈ 4 characters
    return len(text) // 4
```

**Deliverable:** Working rate limiting with token counting

---

## SPRINT F: Audio Formats & Advanced Features (Week 3-4)

**Priority:** LOW - Nice to have  
**Files:** `sprint2-speech/speech_pipeline.py`, `sprint4-websocket/realtime_server.py`

### Tasks

#### F1: Add g711_ulaw codec support
```python
# In AudioProcessor
def encode_g711_ulaw(self, pcm_data: bytes) -> bytes:
    # Implement μ-law encoding
    pass

def decode_g711_ulaw(self, ulaw_data: bytes) -> bytes:
    # Implement μ-law decoding
    pass
```

#### F2: Add g711_alaw codec support
```python
def encode_g711_alaw(self, pcm_data: bytes) -> bytes:
    # Implement A-law encoding
    pass

def decode_g711_alaw(self, alaw_data: bytes) -> bytes:
    # Implement A-law decoding
    pass
```

#### F3: Audio format negotiation
```python
async def handle_session_update(self, session_id: str, event: dict):
    session_update = event.get("session", {})
    audio_config = session_update.get("audio", {})
    
    input_format = audio_config.get("input", {}).get("format", {})
    output_format = audio_config.get("output", {}).get("format", {})
    
    # Update audio processing pipeline with new formats
    if input_format:
        self.update_input_format(session_id, input_format)
    if output_format:
        self.update_output_format(session_id, output_format)
```

#### F4: Turn detection configuration
```python
async def handle_session_update(self, session_id: str, event: dict):
    turn_detection = event.get("session", {}).get("audio", {}).get("input", {}).get("turn_detection")
    
    if turn_detection:
        pipeline = self.manager.speech_pipelines.get(session_id)
        if pipeline:
            if turn_detection.get("type") == "none":
                pipeline.config.turn_detection_enabled = False
            else:
                pipeline.config.turn_detection_enabled = True
                pipeline.config.vad_threshold = turn_detection.get("threshold", 0.5)
                pipeline.config.silence_threshold_ms = turn_detection.get("silence_duration_ms", 500)
```

**Deliverable:** Multiple audio formats supported

---

## SPRINT G: Error Handling (Week 4)

**Priority:** MEDIUM - Production quality  
**Files:** `sprint4-websocket/realtime_server.py`, `enterprise/app/transports/realtime_ws.py`

### Tasks

#### G1: Define error code taxonomy
```python
ERROR_CODES = {
    "invalid_request_error": "Request validation failed",
    "authentication_error": "Authentication failed",
    "permission_error": "Insufficient permissions",
    "not_found_error": "Resource not found",
    "rate_limit_error": "Rate limit exceeded",
    "api_error": "Internal server error",
    "overloaded_error": "Server overloaded"
}
```

#### G2: Implement error helpers
```python
async def send_error(self, session_id: str, error_type: str, message: str, param: str = None):
    await self.manager.send_event(session_id, {
        "type": "error",
        "error": {
            "type": error_type,
            "code": error_type,
            "message": message,
            "param": param,
            "event_id": f"event_{uuid.uuid4().hex[:16]}"
        }
    })
```

#### G3: Add error handling throughout
```python
# Validation errors
if not event.get("item"):
    await self.send_error(session_id, "invalid_request_error", "Missing item field", "item")
    return

# Not found errors
if session_id not in self.manager.session_data:
    await self.send_error(session_id, "not_found_error", f"Session {session_id} not found")
    return

# Rate limit errors
if not allowed:
    await self.send_error(session_id, "rate_limit_error", "Rate limit exceeded")
    return
```

**Deliverable:** Complete error code taxonomy

---

## SPRINT H: Testing & Validation (Week 5-6)

**Priority:** CRITICAL - Verify compatibility  
**Files:** New directory `tests/openai_compatibility/`

### Tasks

#### H1: Create test suite
```python
# tests/openai_compatibility/test_events.py
import pytest

class TestOpenAICompatibility:
    async def test_session_lifecycle(self):
        # Test session.created, session.updated
        pass
    
    async def test_audio_streaming(self):
        # Test input_audio_buffer.* events
        pass
    
    async def test_conversation_management(self):
        # Test conversation.item.* events
        pass
    
    async def test_function_calling(self):
        # Test function call detection and execution
        pass
    
    async def test_rate_limiting(self):
        # Test rate limit enforcement
        pass
```

#### H2: Create compatibility checker
```python
# tests/openai_compatibility/compatibility_checker.py
class CompatibilityChecker:
    def check_event_format(self, event: dict, expected_schema: dict) -> bool:
        # Validate event matches OpenAI schema
        pass
    
    def check_all_events_supported(self) -> list:
        # Return list of missing events
        pass
```

#### H3: Integration tests with OpenAI SDK
```python
# Test that OpenAI's official SDK works with our server
from openai import OpenAI

client = OpenAI(base_url="http://localhost:60200/v1")
# Run OpenAI SDK tests against our server
```

#### H4: Load testing
```python
# Test concurrent sessions
async def test_concurrent_sessions():
    sessions = []
    for i in range(100):
        session = await create_session()
        sessions.append(session)
    
    # Verify all sessions work correctly
```

**Deliverable:** Comprehensive test suite proving 100% compatibility

---

## Implementation Order

### Week 1 (Parallel)
- Sprint A: Critical fixes (MUST DO FIRST)
- Sprint B: Configuration fields
- Sprint C: Function calling (start)

### Week 2 (Parallel)
- Sprint C: Function calling (continue)
- Sprint D: Transcription events
- Sprint E: Rate limiting (start)

### Week 3 (Parallel)
- Sprint C: Function calling (finish)
- Sprint E: Rate limiting (finish)
- Sprint F: Audio formats (start)

### Week 4 (Parallel)
- Sprint F: Audio formats (finish)
- Sprint G: Error handling

### Week 5-6 (Parallel)
- Sprint H: Testing & validation
- Bug fixes and polish

---

## Success Criteria

### Must Have (100% Required)
- ✅ All client → server events handled
- ✅ All server → client events sent
- ✅ Function calling works
- ✅ All config fields used
- ✅ Rate limiting enforced
- ✅ Error codes match OpenAI

### Should Have (95% Required)
- ✅ Multiple audio formats
- ✅ Turn detection configurable
- ✅ Transcription events complete

### Nice to Have (Optional)
- ✅ OpenAI SDK compatibility
- ✅ Load tested to 1000 concurrent sessions

---

## Verification Checklist

After completion, verify:

1. [ ] Code doesn't crash on any OpenAI event
2. [ ] All session config fields are used
3. [ ] Function calling works end-to-end
4. [ ] Rate limiting enforces limits
5. [ ] All transcription events sent
6. [ ] Error codes match OpenAI taxonomy
7. [ ] Multiple audio formats supported
8. [ ] Turn detection is configurable
9. [ ] OpenAI SDK works with our server
10. [ ] 100+ concurrent sessions work

---

## Files to Modify

### Core Implementation
- `sprint4-websocket/realtime_server.py` (main file)
- `enterprise/app/transports/realtime_ws.py` (enterprise version)
- `llm_integration.py` (LLM calls)
- `sprint2-speech/speech_pipeline.py` (audio processing)

### New Files to Create
- `function_calling.py` (function calling engine)
- `rate_limiter.py` (rate limiting)
- `audio_codecs.py` (g711 codecs)
- `tests/openai_compatibility/` (test suite)

### Documentation
- Update `VOICE_AGENT_ROADMAP.md` with completion status
- Create `OPENAI_COMPATIBILITY.md` with feature matrix

---

## Current vs Target

| Feature | Current | Target | Sprint |
|---------|---------|--------|--------|
| Basic events | 85% | 100% | A |
| Config fields | 20% | 100% | B |
| Function calling | 0% | 100% | C |
| Transcription | 40% | 100% | D |
| Rate limiting | 10% | 100% | E |
| Audio formats | 25% | 100% | F |
| Error handling | 50% | 100% | G |
| Testing | 0% | 100% | H |

**Overall: 45% → 100%**

---

**Last Updated:** 2025-01-20  
**Status:** Ready for implementation  
**Estimated Completion:** 6 weeks with parallel execution
