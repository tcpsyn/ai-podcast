"""Configuration settings for the AI Radio Show backend"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env from parent directory
load_dotenv(Path(__file__).parent.parent / ".env")


class Settings(BaseSettings):
    # API Keys
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    inworld_api_key: str = os.getenv("INWORLD_API_KEY", "")

    # Twilio Settings
    twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    twilio_phone_number: str = os.getenv("TWILIO_PHONE_NUMBER", "")
    twilio_webhook_base_url: str = os.getenv("TWILIO_WEBHOOK_BASE_URL", "")

    # LLM Settings
    llm_provider: str = "openrouter"  # "openrouter" or "ollama"
    openrouter_model: str = "anthropic/claude-3-haiku"
    ollama_model: str = "llama3.2"
    ollama_host: str = "http://localhost:11434"

    # TTS Settings
    tts_provider: str = "kokoro"  # "kokoro", "elevenlabs", "vits", or "bark"

    # Audio Settings
    sample_rate: int = 24000

    # Paths
    base_dir: Path = Path(__file__).parent.parent
    sounds_dir: Path = base_dir / "sounds"
    music_dir: Path = base_dir / "music"
    sessions_dir: Path = base_dir / "sessions"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
