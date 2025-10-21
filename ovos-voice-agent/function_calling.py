#!/usr/bin/env python3
"""Function calling engine for OpenAI Realtime API compatibility.

This module provides function detection, validation, and execution capabilities
for voice-based function calling during conversations.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional
import asyncio

logger = logging.getLogger(__name__)


class FunctionCallingEngine:
    """Manages function registration, detection, and execution."""
    
    def __init__(self):
        self.functions: Dict[str, Dict[str, Any]] = {}
    
    def register_function(self, name: str, schema: dict, handler: Callable) -> None:
        """Register a function with its schema and handler.
        
        Args:
            name: Function name
            schema: OpenAI function schema
            handler: Async callable that executes the function
        """
        self.functions[name] = {
            "schema": schema,
            "handler": handler
        }
        logger.info(f"Registered function: {name}")
    
    async def detect_function_call(self, text: str, tools: List[dict]) -> Optional[dict]:
        """Detect if text contains a function call using LLM.
        
        Args:
            text: User's transcribed text
            tools: List of available tools from session config
            
        Returns:
            Dict with function name and arguments, or None
        """
        if not tools:
            return None
        
        # Simple keyword-based detection (can be enhanced with LLM)
        text_lower = text.lower()
        
        for tool in tools:
            if tool.get("type") != "function":
                continue
            
            function = tool.get("function", {})
            function_name = function.get("name", "")
            description = function.get("description", "").lower()
            
            # Check if function name or keywords in description match
            if function_name.lower() in text_lower:
                # Extract parameters (simplified)
                parameters = function.get("parameters", {})
                properties = parameters.get("properties", {})
                
                # Build arguments dict
                arguments = {}
                for param_name, param_schema in properties.items():
                    # Simple extraction - can be enhanced
                    if param_name.lower() in text_lower:
                        arguments[param_name] = text  # Simplified
                
                return {
                    "name": function_name,
                    "arguments": arguments,
                    "call_id": f"call_{id(text)}"
                }
        
        return None
    
    async def execute_function(self, function_name: str, arguments: dict) -> dict:
        """Execute a registered function.
        
        Args:
            function_name: Name of function to execute
            arguments: Function arguments
            
        Returns:
            Function result or error
        """
        if function_name not in self.functions:
            return {
                "error": f"Function '{function_name}' not found",
                "type": "function_not_found"
            }
        
        try:
            handler = self.functions[function_name]["handler"]
            
            # Execute handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**arguments)
            else:
                result = handler(**arguments)
            
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            logger.error(f"Function execution error: {e}")
            return {
                "error": str(e),
                "type": "execution_error"
            }
    
    def validate_arguments(self, function_name: str, arguments: dict) -> tuple[bool, Optional[str]]:
        """Validate function arguments against schema.
        
        Args:
            function_name: Function name
            arguments: Arguments to validate
            
        Returns:
            (is_valid, error_message)
        """
        if function_name not in self.functions:
            return False, f"Function '{function_name}' not found"
        
        schema = self.functions[function_name]["schema"]
        parameters = schema.get("parameters", {})
        required = parameters.get("required", [])
        properties = parameters.get("properties", {})
        
        # Check required parameters
        for param in required:
            if param not in arguments:
                return False, f"Missing required parameter: {param}"
        
        # Check parameter types (simplified)
        for param, value in arguments.items():
            if param not in properties:
                return False, f"Unknown parameter: {param}"
        
        return True, None


# Example function handlers
async def example_get_weather(location: str, unit: str = "celsius") -> dict:
    """Example weather function."""
    return {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "condition": "sunny"
    }


async def example_set_timer(duration: int, label: str = "") -> dict:
    """Example timer function."""
    return {
        "duration": duration,
        "label": label,
        "status": "timer_set"
    }


# Global engine instance
_engine = FunctionCallingEngine()


def get_function_engine() -> FunctionCallingEngine:
    """Get the global function calling engine."""
    return _engine


def register_default_functions():
    """Register default example functions."""
    _engine.register_function(
        "get_weather",
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        },
        example_get_weather
    )
    
    _engine.register_function(
        "set_timer",
        {
            "name": "set_timer",
            "description": "Set a timer for a specified duration",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "integer",
                        "description": "Duration in seconds"
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional timer label"
                    }
                },
                "required": ["duration"]
            }
        },
        example_set_timer
    )


# Register default functions on module load
register_default_functions()
