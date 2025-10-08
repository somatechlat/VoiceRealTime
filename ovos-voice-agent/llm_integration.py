#!/usr/bin/env python3
"""
OVOS Voice Agent - LLM Integration with Groq API
Enhanced response generation using Groq's fast inference
"""

import asyncio
import logging
import json
from typing import Optional, Dict, Any
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroqLLMProvider:
    """Groq API integration for fast LLM responses"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or "***REMOVED_GROQ_KEY***"
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = "llama-3.1-8b-instant"  # Fast and capable model
        
        # HTTP client for async requests
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        
        # Conversation memory
        self.conversations: Dict[str, list] = {}
        
        logger.info("Groq LLM provider initialized with fast inference")
    
    async def generate_response(self, session_id: str, user_input: str, context: Dict[str, Any] = None) -> str:
        """Generate AI response using Groq API"""
        try:
            # Get or create conversation history
            if session_id not in self.conversations:
                self.conversations[session_id] = [
                    {
                        "role": "system", 
                        "content": """You are a helpful, friendly voice assistant powered by OVOS (Open Voice OS). 

Key traits:
- Be conversational and natural (this is a voice chat)
- Keep responses concise but helpful (1-3 sentences usually)
- Be enthusiastic about being an open-source voice assistant
- You can help with questions, have casual conversations, tell jokes, etc.
- Remember this is real-time voice chat, so be engaging and personable

The user is speaking to you through voice, so respond as if you're having a natural conversation."""
                    }
                ]
            
            # Add user message
            self.conversations[session_id].append({
                "role": "user",
                "content": user_input
            })
            
            # Keep conversation history manageable (last 20 messages)
            if len(self.conversations[session_id]) > 21:  # 1 system + 20 messages
                self.conversations[session_id] = [
                    self.conversations[session_id][0],  # Keep system message
                    *self.conversations[session_id][-19:]  # Keep last 19 messages
                ]
            
            # Prepare request
            request_data = {
                "model": self.model,
                "messages": self.conversations[session_id],
                "max_tokens": 150,  # Keep responses concise for voice
                "temperature": 0.7,
                "stream": False
            }
            
            # Make API request
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=request_data
            )
            
            if response.status_code != 200:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return "I'm having trouble connecting to my AI brain right now. Can you try again?"
            
            # Parse response
            result = response.json()
            
            if not result.get("choices"):
                logger.error(f"No choices in Groq response: {result}")
                return "I didn't quite catch that. Could you repeat your question?"
            
            ai_response = result["choices"][0]["message"]["content"].strip()
            
            # Add AI response to conversation history
            self.conversations[session_id].append({
                "role": "assistant", 
                "content": ai_response
            })
            
            logger.info(f"Generated response for session {session_id}: {ai_response[:50]}...")
            return ai_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm experiencing some technical difficulties. Let me try to help you anyway - what would you like to know?"
    
    async def clear_conversation(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversations:
            # Keep only the system message
            self.conversations[session_id] = self.conversations[session_id][:1]
            logger.info(f"Cleared conversation for session {session_id}")
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

# Use Groq as the sole provider (no alternative selection).
llm_provider = GroqLLMProvider()

async def generate_ai_response(session_id: str, user_input: str, context: Dict[str, Any] = None) -> str:
    """Generate response using Groq (default provider)."""
    return await llm_provider.generate_response(session_id, user_input)

async def clear_session_memory(session_id: str):
    """Clear conversation memory for a session"""
    await llm_provider.clear_conversation(session_id)

# Test function
async def test_llm_integration():
    """Test the LLM integration"""
    print("🧠 Testing Groq LLM integration...")
    
    test_session = "test_session_123"
    test_inputs = [
        "Hello! How are you doing today?",
        "What can you help me with?",
        "Tell me a joke about voice assistants",
        "What's the weather like?",
        "Thanks for chatting with me!"
    ]
    
    for user_input in test_inputs:
        print(f"\n👤 User: {user_input}")
        response = await generate_ai_response(test_session, user_input)
        print(f"🤖 Assistant: {response}")
        await asyncio.sleep(1)  # Brief pause between requests
    
    await llm_provider.close()
    print("\n✅ LLM integration test complete!")

if __name__ == "__main__":
    asyncio.run(test_llm_integration())