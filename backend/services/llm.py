"""LLM service with OpenRouter and Ollama support"""

import httpx
from typing import Optional
from ..config import settings


# Available OpenRouter models
OPENROUTER_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-3-haiku",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-flash-1.5",
    "google/gemini-pro-1.5",
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct",
]


class LLMService:
    """Abstraction layer for LLM providers"""

    def __init__(self):
        self.provider = settings.llm_provider
        self.openrouter_model = settings.openrouter_model
        self.ollama_model = settings.ollama_model
        self.ollama_host = settings.ollama_host
        self.tts_provider = settings.tts_provider

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
            # Also update the global settings so TTS service picks it up
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
            "available_ollama_models": []  # Fetched separately
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
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt to prepend

        Returns:
            Generated text response
        """
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        if self.provider == "openrouter":
            return await self._call_openrouter(messages)
        else:
            return await self._call_ollama(messages)

    async def _call_openrouter(self, messages: list[dict]) -> str:
        """Call OpenRouter API with retry"""
        for attempt in range(2):  # Try twice
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {settings.openrouter_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.openrouter_model,
                            "messages": messages,
                            "max_tokens": 150,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    if not content or not content.strip():
                        print(f"OpenRouter returned empty response")
                        return ""
                    return content
            except (httpx.TimeoutException, httpx.ReadTimeout):
                print(f"OpenRouter timeout (attempt {attempt + 1})")
                if attempt == 0:
                    continue  # Retry once
                return "Uh, sorry, I lost you there for a second. What was that?"
            except Exception as e:
                print(f"OpenRouter error: {e}")
                return "Yeah... I don't know, man."
        return "Uh, hold on a sec..."

    async def _call_ollama(self, messages: list[dict]) -> str:
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
                            "num_predict": 100,     # Allow complete thoughts
                            "temperature": 0.8,     # Balanced creativity/coherence
                            "top_p": 0.9,           # Focused word choices
                            "repeat_penalty": 1.3,  # Avoid repetition
                            "top_k": 50,            # Reasonable token variety
                        },
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]
        except httpx.TimeoutException:
            print("Ollama timeout")
            return "Uh, sorry, I lost you there for a second. What was that?"
        except Exception as e:
            print(f"Ollama error: {e}")
            return "Yeah... I don't know, man."


# Global instance
llm_service = LLMService()
