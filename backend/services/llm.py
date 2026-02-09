"""LLM service with OpenRouter and Ollama support"""

import httpx
from typing import Optional
from ..config import settings


# Available OpenRouter models
OPENROUTER_MODELS = [
    # Best for natural dialog (ranked)
    "minimax/minimax-m2-her",
    "mistralai/mistral-small-creative",
    "x-ai/grok-4-fast",
    "deepseek/deepseek-v3.2",
    # Updated standard models
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4-5",
    "google/gemini-2.5-flash",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
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
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    def update_settings(
        self,
        provider: Optional[str] = None,
        openrouter_model: Optional[str] = None,
        ollama_model: Optional[str] = None,
        ollama_host: Optional[str] = None,
        tts_provider: Optional[str] = None
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
            "available_openrouter_models": OPENROUTER_MODELS,
            "available_ollama_models": ollama_models
        }

    async def generate(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        if self.provider == "openrouter":
            return await self._call_openrouter_with_fallback(messages, max_tokens=max_tokens)
        else:
            return await self._call_ollama(messages, max_tokens=max_tokens)

    async def _call_openrouter_with_fallback(self, messages: list[dict], max_tokens: Optional[int] = None) -> str:
        """Try primary model, then fallback models. Always returns a response."""

        # Try primary model first
        result = await self._call_openrouter_once(messages, self.openrouter_model, max_tokens=max_tokens)
        if result is not None:
            return result

        # Try fallback models
        for model in FALLBACK_MODELS:
            if model == self.openrouter_model:
                continue  # Already tried
            print(f"[LLM] Falling back to {model}...")
            result = await self._call_openrouter_once(messages, model, timeout=10.0, max_tokens=max_tokens)
            if result is not None:
                return result

        # Everything failed — return an in-character line so the show continues
        print("[LLM] All models failed, using canned response")
        return "Sorry, I totally blanked out for a second. What were you saying?"

    async def _call_openrouter_once(self, messages: list[dict], model: str, timeout: float = 15.0, max_tokens: Optional[int] = None) -> str | None:
        """Single attempt to call OpenRouter. Returns None on failure (not a fallback string)."""
        try:
            response = await self.client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens or 150,
                    "temperature": 0.8,
                    "top_p": 0.92,
                    "frequency_penalty": 0.5,
                    "presence_penalty": 0.3,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
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
