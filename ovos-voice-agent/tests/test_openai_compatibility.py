#!/usr/bin/env python3
"""
OpenAI Compatibility Tests - Sprint H
Comprehensive test suite for OpenAI Realtime API compatibility
"""

import pytest
import asyncio
import json
import websockets
from typing import Dict, List


class TestOpenAICompatibility:
    """Test suite for OpenAI Realtime API compatibility"""
    
    BASE_URL = "ws://localhost:60200/v1/realtime"
    
    async def connect_websocket(self):
        """Connect to WebSocket server"""
        ws = await websockets.connect(f"{self.BASE_URL}?access_token=test_token")
        return ws
    
    async def receive_event(self, ws, event_type: str, timeout: float = 5.0):
        """Receive specific event type"""
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                event = json.loads(msg)
                if event.get("type") == event_type:
                    return event
            except asyncio.TimeoutError:
                continue
        raise TimeoutError(f"Event {event_type} not received within {timeout}s")
    
    @pytest.mark.asyncio
    async def test_session_lifecycle(self):
        """Test session.created and session.updated events"""
        ws = await self.connect_websocket()
        
        # Should receive session.created
        event = await self.receive_event(ws, "session.created")
        assert event["type"] == "session.created"
        assert "session" in event
        assert event["session"]["object"] == "realtime.session"
        
        # Update session
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "voice": "am_onyx",
                "temperature": 0.9
            }
        }))
        
        # Should receive session.updated
        event = await self.receive_event(ws, "session.updated")
        assert event["type"] == "session.updated"
        assert event["session"]["voice"] == "am_onyx"
        
        await ws.close()
    
    @pytest.mark.asyncio
    async def test_audio_buffer_events(self):
        """Test input_audio_buffer.* events"""
        ws = await self.connect_websocket()
        await self.receive_event(ws, "session.created")
        
        # Append audio
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": "AAAA"  # dummy base64
        }))
        
        # Commit audio
        await ws.send(json.dumps({
            "type": "input_audio_buffer.commit"
        }))
        
        event = await self.receive_event(ws, "input_audio_buffer.committed")
        assert event["type"] == "input_audio_buffer.committed"
        
        await ws.close()
    
    @pytest.mark.asyncio
    async def test_conversation_item_create(self):
        """Test conversation.item.create event"""
        ws = await self.connect_websocket()
        await self.receive_event(ws, "session.created")
        
        # Create conversation item
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "id": "item_test123",
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "Hello"
                }]
            }
        }))
        
        event = await self.receive_event(ws, "conversation.item.created")
        assert event["type"] == "conversation.item.created"
        assert event["item"]["id"] == "item_test123"
        
        await ws.close()
    
    @pytest.mark.asyncio
    async def test_conversation_item_truncate(self):
        """Test conversation.item.truncate event"""
        ws = await self.connect_websocket()
        await self.receive_event(ws, "session.created")
        
        # Create item first
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "id": "item_truncate_test",
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Part 1"},
                    {"type": "input_text", "text": "Part 2"}
                ]
            }
        }))
        await self.receive_event(ws, "conversation.item.created")
        
        # Truncate item
        await ws.send(json.dumps({
            "type": "conversation.item.truncate",
            "item_id": "item_truncate_test",
            "content_index": 0
        }))
        
        event = await self.receive_event(ws, "conversation.item.truncated")
        assert event["type"] == "conversation.item.truncated"
        assert event["item_id"] == "item_truncate_test"
        
        await ws.close()
    
    @pytest.mark.asyncio
    async def test_conversation_item_delete(self):
        """Test conversation.item.delete event"""
        ws = await self.connect_websocket()
        await self.receive_event(ws, "session.created")
        
        # Create item first
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "id": "item_delete_test",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Delete me"}]
            }
        }))
        await self.receive_event(ws, "conversation.item.created")
        
        # Delete item
        await ws.send(json.dumps({
            "type": "conversation.item.delete",
            "item_id": "item_delete_test"
        }))
        
        event = await self.receive_event(ws, "conversation.item.deleted")
        assert event["type"] == "conversation.item.deleted"
        assert event["item_id"] == "item_delete_test"
        
        await ws.close()
    
    @pytest.mark.asyncio
    async def test_response_create(self):
        """Test response.create event"""
        ws = await self.connect_websocket()
        await self.receive_event(ws, "session.created")
        
        # Create user message
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}]
            }
        }))
        await self.receive_event(ws, "conversation.item.created")
        
        # Request response
        await ws.send(json.dumps({
            "type": "response.create"
        }))
        
        event = await self.receive_event(ws, "response.created")
        assert event["type"] == "response.created"
        assert "response" in event
        
        await ws.close()
    
    @pytest.mark.asyncio
    async def test_response_cancel(self):
        """Test response.cancel event"""
        ws = await self.connect_websocket()
        await self.receive_event(ws, "session.created")
        
        # Create user message and start response
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Tell me a long story"}]
            }
        }))
        await self.receive_event(ws, "conversation.item.created")
        
        await ws.send(json.dumps({"type": "response.create"}))
        await self.receive_event(ws, "response.created")
        
        # Cancel response
        await ws.send(json.dumps({"type": "response.cancel"}))
        
        event = await self.receive_event(ws, "response.cancelled")
        assert event["type"] == "response.cancelled"
        
        await ws.close()
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limit enforcement"""
        ws = await self.connect_websocket()
        await self.receive_event(ws, "session.created")
        
        # Send many requests rapidly
        for i in range(150):
            await ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"Message {i}"}]
                }
            }))
        
        # Should eventually receive rate limit error
        error_received = False
        for _ in range(10):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                event = json.loads(msg)
                if event.get("type") == "error" and event.get("error", {}).get("type") == "rate_limit_error":
                    error_received = True
                    break
            except asyncio.TimeoutError:
                continue
        
        assert error_received, "Rate limit error not received"
        
        await ws.close()
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error event format"""
        ws = await self.connect_websocket()
        await self.receive_event(ws, "session.created")
        
        # Send invalid event
        await ws.send(json.dumps({
            "type": "invalid_event_type"
        }))
        
        event = await self.receive_event(ws, "error")
        assert event["type"] == "error"
        assert "error" in event
        assert "type" in event["error"]
        assert "message" in event["error"]
        
        await ws.close()
    
    @pytest.mark.asyncio
    async def test_function_calling(self):
        """Test function calling flow"""
        ws = await self.connect_websocket()
        await self.receive_event(ws, "session.created")
        
        # Update session with tools
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "tools": [{
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }]
            }
        }))
        await self.receive_event(ws, "session.updated")
        
        # Create message that should trigger function
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "What's the weather in London?"}]
            }
        }))
        await self.receive_event(ws, "conversation.item.created")
        
        await ws.send(json.dumps({"type": "response.create"}))
        
        # Should receive function call events
        event = await self.receive_event(ws, "response.function_call_arguments.delta", timeout=10.0)
        assert event["type"] == "response.function_call_arguments.delta"
        
        await ws.close()


class CompatibilityChecker:
    """Check OpenAI API compatibility"""
    
    REQUIRED_CLIENT_EVENTS = [
        "session.update",
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "input_audio_buffer.clear",
        "conversation.item.create",
        "conversation.item.truncate",
        "conversation.item.delete",
        "response.create",
        "response.cancel"
    ]
    
    REQUIRED_SERVER_EVENTS = [
        "error",
        "session.created",
        "session.updated",
        "conversation.created",
        "input_audio_buffer.committed",
        "input_audio_buffer.cleared",
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "conversation.item.created",
        "conversation.item.input_audio_transcription.completed",
        "conversation.item.truncated",
        "conversation.item.deleted",
        "response.created",
        "response.done",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.text.delta",
        "response.text.done",
        "response.audio_transcript.delta",
        "response.audio_transcript.done",
        "response.audio.delta",
        "response.audio.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "rate_limits.updated"
    ]
    
    def check_event_coverage(self, implemented_events: List[str]) -> Dict[str, List[str]]:
        """Check which events are implemented"""
        missing_client = [e for e in self.REQUIRED_CLIENT_EVENTS if e not in implemented_events]
        missing_server = [e for e in self.REQUIRED_SERVER_EVENTS if e not in implemented_events]
        
        return {
            "missing_client_events": missing_client,
            "missing_server_events": missing_server,
            "coverage_percent": (
                (len(self.REQUIRED_CLIENT_EVENTS) + len(self.REQUIRED_SERVER_EVENTS) - 
                 len(missing_client) - len(missing_server)) / 
                (len(self.REQUIRED_CLIENT_EVENTS) + len(self.REQUIRED_SERVER_EVENTS)) * 100
            )
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
