#!/usr/bin/env python3
"""LLM integration for the voice agent.

This module provides a thin wrapper around an OpenAI‑compatible API (e.g. the
OpenAI hosted ``gpt-oss-20b`` model). The implementation is asynchronous and
keeps the same public functions that the rest of the codebase expects:

* ``generate_ai_response(session_id, user_input, context=None)`` – returns the
    assistant's reply.
* ``clear_session_memory(session_id)`` – clears any cached conversation state.

Only a minimal in‑memory history is kept to provide a ``system`` prompt and to
preserve recent turns. The default model is ``gpt-oss-20b``; it can be overridden
with the ``LLM_MODEL`` environment variable. The OpenAI API key is read from
``OPENAI_API_KEY``.

The code is deliberately simple because the user requested a *single* model for
testing and no UI for model selection.
"""

import asyncio
import os
from typing import Any, Dict, List
import httpx
from ovos_voice_agent import config

"""Simplified LLM integration stub.

The original implementation performed remote inference via Groq and also provided a local
fallback. The user has requested that *all* LLM inference be performed as external calls
and that the heavy local logic be removed. To satisfy this, the module now contains a very
lightweight stub that simply returns a deterministic echo of the user input. This keeps the
public API (`generate_ai_response` and `clear_session_memory`) unchanged for the rest of the
codebase while eliminating any heavy dependencies, network calls, or local state.

If a real external service is desired later, the `generate_ai_response` function can be
updated to perform an HTTP request to the appropriate endpoint.
"""

# Imports are already placed at the top of the file.

# No heavy imports – the stub does not perform any network I/O.

class OpenAIProvider:
    """Simple wrapper for an OpenAI‑compatible chat endpoint.

    The provider maintains a per‑session message list so that the model receives a
    conversational context. Only the last 20 user/assistant turns are kept to avoid
    payload bloat.
    """

    def __init__(self) -> None:
        self.api_key: str = config.OPENAI_API_KEY
        self.base_url: str = config.OPENAI_API_BASE
        self.model: str = os.getenv("LLM_MODEL", config.LLM_MODEL)
        self.client: httpx.AsyncClient | None = None
        if self.api_key:
            self.client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        else:
            # No key – the provider will fall back to a deterministic response.
            self.client = None
        # In‑memory conversation history per session.
        self.conversations: Dict[str, List[Dict[str, str]]] = {}

    async def _ensure_history(self, session_id: str) -> List[Dict[str, str]]:
        """Create a default history with a system prompt if missing."""
        if session_id not in self.conversations:
            self.conversations[session_id] = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful, friendly voice assistant powered by OVOS. "
                        "Keep responses concise (1‑3 sentences) and conversational."
                    ),
                }
            ]
        return self.conversations[session_id]

    async def generate_response(
        self, 
        session_id: str, 
        user_input: str, 
        context: Dict[str, Any] | None = None,
        instructions: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | str = 200
    ) -> str:
        """Call the remote model or fall back to a deterministic reply.

        Args:
            session_id: Session identifier
            user_input: User's message
            context: Additional context (unused)
            instructions: System prompt override
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate (or "inf")
        """
        history = await self._ensure_history(session_id)
        
        # Override system prompt if instructions provided
        if instructions:
            history[0] = {"role": "system", "content": instructions}
        
        # Append the new user message.
        history.append({"role": "user", "content": user_input})

        # Trim history to last 20 turns (plus the system prompt).
        if len(history) > 41:  # 1 system + 20 pairs = 41 entries max
            history[:] = [history[0]] + history[-40:]

        if not self.client:
            # Deterministic fallback when no API key is configured.
            response_text = (
                f"You said: {user_input}" if user_input else "I am ready to assist you."
            )
        else:
            # Handle max_tokens
            if max_tokens == "inf":
                max_tokens = 4096
            else:
                max_tokens = int(max_tokens)
            
            payload = {
                "model": self.model,
                "messages": history,
                "max_tokens": max_tokens,
                "temperature": float(temperature),
            }
            try:
                resp = await self.client.post(
                    f"{self.base_url}/chat/completions", json=payload
                )
                resp.raise_for_status()
                data = resp.json()
                response_text = data["choices"][0]["message"]["content"].strip()
            except Exception as exc:  # pragma: no cover – network failures are rare in tests
                # Log and fall back to deterministic response.
                print(f"LLM request failed: {exc}")
                response_text = (
                    f"You said: {user_input}" if user_input else "I am ready to assist you."
                )

        # Store assistant reply.
        history.append({"role": "assistant", "content": response_text})
        return response_text

    async def clear_conversation(self, session_id: str) -> None:
        """Reset a session's history, keeping only the system prompt."""
        if session_id in self.conversations:
            self.conversations[session_id] = self.conversations[session_id][:1]

    async def close(self) -> None:
        if self.client:
            await self.client.aclose()


# Singleton provider used by the public helper functions.
_provider = OpenAIProvider()


async def generate_ai_response(
    session_id: str, 
    user_input: str, 
    context: Dict[str, Any] | None = None,
    instructions: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | str = 200
) -> str:
    """Public wrapper that forwards to the ``OpenAIProvider`` instance."""
    return await _provider.generate_response(
        session_id, user_input, context, instructions, temperature, max_tokens
    )

async def clear_session_memory(session_id: str) -> None:
    """Clear cached conversation history for a session."""
    await _provider.clear_conversation(session_id)

# Optional test harness retained for developer convenience.
async def test_llm_integration():
    """Run a quick end‑to‑end check of the OpenAI provider.

    This function is useful during development; it does not require any UI.
    """
    print("🧠 Testing OpenAI LLM integration (model: gpt-oss-20b)...")
    test_session = "test_session_123"
    test_inputs = [
        "Hello! How are you doing today?",
        "What can you help me with?",
        "Tell me a joke about voice assistants",
        "What's the weather like?",
        "Thanks for chatting with me!",
    ]
    for user_input in test_inputs:
        resp = await generate_ai_response(test_session, user_input)
        print(f"User: {user_input}\nAssistant: {resp}\n")
        await asyncio.sleep(0.2)

    # Clean up the HTTP client.
    await _provider.close()

if __name__ == "__main__":
    asyncio.run(test_llm_integration())