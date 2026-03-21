"""LLM service with OpenRouter and Ollama support"""

import json
import time
import httpx
from typing import Optional, Callable, Awaitable
from ..config import settings
from .cost_tracker import cost_tracker


# Available OpenRouter models
OPENROUTER_MODELS = [
    # Default
    "anthropic/claude-sonnet-4-5",
    # Best for natural dialog
    "x-ai/grok-4",
    "x-ai/grok-4-fast",
    "minimax/minimax-m2-her",
    "mistralai/mistral-small-creative",
    "deepseek/deepseek-v3.2",
    # Other
    "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-flash",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    # New dialog models
    "deepseek/deepseek-chat-v3-0324",
    "moonshotai/kimi-k2",
    "mistralai/mistral-medium-3",
    "meta-llama/llama-4-maverick",
    "qwen/qwen3-235b-a22b",
    "google/gemini-2.5-pro",
    # Legacy
    "anthropic/claude-3-haiku",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.1-8b-instruct",
]

# Fast models to try as fallbacks (cheap, fast, good enough for conversation)
FALLBACK_MODELS = [
    "mistralai/mistral-small-creative",
    "google/gemini-2.5-flash",
    "openai/gpt-4o-mini",
]


class LLMService:
    """Abstraction layer for LLM providers"""

    def __init__(self):
        self.provider = settings.llm_provider
        self.openrouter_model = settings.openrouter_model
        self.ollama_model = settings.ollama_model
        self.ollama_host = settings.ollama_host
        self.tts_provider = settings.tts_provider
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    def update_settings(
        self,
        provider: Optional[str] = None,
        openrouter_model: Optional[str] = None,
        ollama_model: Optional[str] = None,
        ollama_host: Optional[str] = None,
        tts_provider: Optional[str] = None,
        category_models: Optional[dict] = None
    ):
        """Update LLM settings"""
        if provider:
            self.provider = provider
        if openrouter_model:
            self.openrouter_model = openrouter_model
        if ollama_model:
            self.ollama_model = ollama_model
        if ollama_host:
            self.ollama_host = ollama_host
        if tts_provider:
            self.tts_provider = tts_provider
            settings.tts_provider = tts_provider
        if category_models:
            settings.category_models.update(category_models)

    async def get_ollama_models(self) -> list[str]:
        """Fetch available models from Ollama"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.ollama_host}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            print(f"Failed to fetch Ollama models: {e}")
            return []

    def get_settings(self) -> dict:
        """Get current settings (sync version without Ollama models)"""
        return {
            "provider": self.provider,
            "openrouter_model": self.openrouter_model,
            "ollama_model": self.ollama_model,
            "ollama_host": self.ollama_host,
            "tts_provider": self.tts_provider,
            "category_models": settings.category_models,
            "available_openrouter_models": OPENROUTER_MODELS,
            "available_ollama_models": []
        }

    async def get_settings_async(self) -> dict:
        """Get current settings with Ollama models"""
        ollama_models = await self.get_ollama_models()
        return {
            "provider": self.provider,
            "openrouter_model": self.openrouter_model,
            "ollama_model": self.ollama_model,
            "ollama_host": self.ollama_host,
            "tts_provider": self.tts_provider,
            "category_models": settings.category_models,
            "available_openrouter_models": OPENROUTER_MODELS,
            "available_ollama_models": ollama_models
        }

    async def generate(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
        category: str = "unknown",
        caller_name: str = "",
        model_override: Optional[str] = None,
    ) -> str:
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        if self.provider == "openrouter":
            return await self._call_openrouter_with_fallback(messages, max_tokens=max_tokens, response_format=response_format, category=category, caller_name=caller_name, model_override=model_override)
        else:
            return await self._call_ollama(messages, max_tokens=max_tokens)

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_executor: Callable[[str, dict], Awaitable[str]],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 500,
        max_tool_rounds: int = 3,
        category: str = "unknown",
        caller_name: str = "",
    ) -> tuple[str, list[dict]]:
        """Generate a response with OpenRouter function calling.

        Args:
            messages: Conversation messages
            tools: Tool definitions in OpenAI function-calling format
            tool_executor: async function(tool_name, arguments) -> result string
            system_prompt: Optional system prompt
            model: Model to use (defaults to primary openrouter_model)
            max_tokens: Max tokens for response
            max_tool_rounds: Max tool call rounds to prevent loops

        Returns:
            (final_text, tool_calls_made) where tool_calls_made is a list of
            {"name": str, "arguments": dict, "result": str} dicts
        """
        model = model or self._get_model_for_category(category)
        msgs = list(messages)
        if system_prompt:
            msgs = [{"role": "system", "content": system_prompt}] + msgs

        all_tool_calls = []

        for round_num in range(max_tool_rounds + 1):
            payload = {
                "model": model,
                "messages": msgs,
                "max_tokens": max_tokens,
                "temperature": 0.65,
                "tools": tools,
                "tool_choice": "auto",
            }

            start_time = time.time()
            try:
                response = await self.client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.openrouter_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=15.0,
                )
                response.raise_for_status()
                data = response.json()
            except httpx.TimeoutException:
                print(f"[LLM-Tools] {model} timed out (round {round_num})")
                break
            except Exception as e:
                print(f"[LLM-Tools] {model} error (round {round_num}): {e}")
                break

            latency_ms = (time.time() - start_time) * 1000
            usage = data.get("usage", {})
            if usage:
                cost_tracker.record_llm_call(
                    category=category,
                    model=model,
                    usage_data=usage,
                    max_tokens=max_tokens,
                    latency_ms=latency_ms,
                    caller_name=caller_name,
                )

            choice = data["choices"][0]
            msg = choice["message"]

            # Check for tool calls
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                # No tool calls — LLM returned a final text response
                content = msg.get("content", "")
                return content or "", all_tool_calls

            # Append assistant message with tool calls to conversation
            msgs.append(msg)

            # Execute each tool call
            for tc in tool_calls:
                func = tc["function"]
                tool_name = func["name"]
                try:
                    arguments = json.loads(func["arguments"])
                except (json.JSONDecodeError, TypeError):
                    arguments = {}

                print(f"[LLM-Tools] Round {round_num}: calling {tool_name}({arguments})")

                try:
                    result = await tool_executor(tool_name, arguments)
                except Exception as e:
                    result = f"Tool unavailable — could not complete {tool_name} right now."
                    print(f"[LLM-Tools] Tool {tool_name} failed: {e}")

                all_tool_calls.append({
                    "name": tool_name,
                    "arguments": arguments,
                    "result": result[:500],
                })

                # Append tool result to conversation
                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        # Exhausted tool rounds or hit an error — do one final call without tools
        print(f"[LLM-Tools] Finishing after {len(all_tool_calls)} tool calls")
        start_time = time.time()
        try:
            final_payload = {
                "model": model,
                "messages": msgs,
                "max_tokens": max_tokens,
                "temperature": 0.65,
            }
            response = await self.client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json=final_payload,
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()
            latency_ms = (time.time() - start_time) * 1000
            usage = data.get("usage", {})
            if usage:
                cost_tracker.record_llm_call(
                    category=category,
                    model=model,
                    usage_data=usage,
                    max_tokens=max_tokens,
                    latency_ms=latency_ms,
                    caller_name=caller_name,
                )
            content = data["choices"][0]["message"].get("content", "")
            return content or "", all_tool_calls
        except Exception as e:
            print(f"[LLM-Tools] Final call failed: {e}")
            return "", all_tool_calls

    def _get_model_for_category(self, category: str) -> str:
        """Get the best model for a given category based on config routing."""
        return settings.category_models.get(category, self.openrouter_model)

    async def _call_openrouter_with_fallback(self, messages: list[dict], max_tokens: Optional[int] = None, response_format: Optional[dict] = None, category: str = "unknown", caller_name: str = "", model_override: Optional[str] = None) -> str:
        """Try category-specific model, then fallback models. Always returns a response."""

        # Use explicit override if provided, else category routing, else primary
        model = model_override or self._get_model_for_category(category)
        result = await self._call_openrouter_once(messages, model, max_tokens=max_tokens, response_format=response_format, category=category, caller_name=caller_name)
        if result is not None:
            return result

        # Try fallback models (drop response_format for fallbacks — not all models support it)
        for model in FALLBACK_MODELS:
            if model == self.openrouter_model:
                continue  # Already tried
            print(f"[LLM] Falling back to {model}...")
            result = await self._call_openrouter_once(messages, model, timeout=8.0, max_tokens=max_tokens, category=category, caller_name=caller_name)
            if result is not None:
                return result

        # Everything failed — return an in-character line so the show continues
        print("[LLM] All models failed, using canned response")
        return "Sorry, I totally blanked out for a second. What were you saying?"

    async def _call_openrouter_once(self, messages: list[dict], model: str, timeout: float = 10.0, max_tokens: Optional[int] = None, response_format: Optional[dict] = None, category: str = "unknown", caller_name: str = "") -> str | None:
        """Single attempt to call OpenRouter. Returns None on failure (not a fallback string)."""
        start_time = time.time()
        try:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens or 500,
                "temperature": 0.65,
                "top_p": 0.9,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.15,
            }
            if response_format:
                payload["response_format"] = response_format
            response = await self.client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            latency_ms = (time.time() - start_time) * 1000
            usage = data.get("usage", {})
            if usage:
                cost_tracker.record_llm_call(
                    category=category,
                    model=model,
                    usage_data=usage,
                    max_tokens=max_tokens or 500,
                    latency_ms=latency_ms,
                    caller_name=caller_name,
                )
            content = data["choices"][0]["message"]["content"]
            if content and content.strip():
                return content
            print(f"[LLM] {model} returned empty response")
            return None
        except httpx.TimeoutException:
            print(f"[LLM] {model} timed out ({timeout}s)")
            return None
        except Exception as e:
            print(f"[LLM] {model} error: {e}")
            return None

    async def _call_ollama(self, messages: list[dict], max_tokens: Optional[int] = None) -> str:
        """Call Ollama API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_host}/api/chat",
                    json={
                        "model": self.ollama_model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens or 100,
                            "temperature": 0.8,
                            "top_p": 0.9,
                            "repeat_penalty": 1.3,
                            "top_k": 50,
                        },
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]
        except httpx.TimeoutException:
            print("Ollama timeout")
            return "Sorry, I totally blanked out for a second. What were you saying?"
        except Exception as e:
            print(f"Ollama error: {e}")
            return "Sorry, I totally blanked out for a second. What were you saying?"


# Global instance
llm_service = LLMService()
