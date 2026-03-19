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

    # SignalWire
    signalwire_project_id: str = os.getenv("SIGNALWIRE_PROJECT_ID", "")
    signalwire_space: str = os.getenv("SIGNALWIRE_SPACE", "")
    signalwire_token: str = os.getenv("SIGNALWIRE_TOKEN", "")
    signalwire_phone: str = os.getenv("SIGNALWIRE_PHONE", "")
    signalwire_stream_url: str = os.getenv("SIGNALWIRE_STREAM_URL", "")

    # Email (IMAP)
    submissions_imap_host: str = os.getenv("SUBMISSIONS_IMAP_HOST", "")
    submissions_imap_user: str = os.getenv("SUBMISSIONS_IMAP_USER", "")
    submissions_imap_pass: str = os.getenv("SUBMISSIONS_IMAP_PASS", "")

    # LLM Settings
    llm_provider: str = "openrouter"  # "openrouter" or "ollama"
    openrouter_model: str = "anthropic/claude-sonnet-4-5"  # primary/default model
    ollama_model: str = "llama3.2"
    ollama_host: str = "http://localhost:11434"

    # Per-category model routing — cheaper models for non-critical tasks
    # Categories: caller_dialog, devon_monitor, devon_ask, background_gen,
    #             call_summary, news_summary, topic_gen, unknown
    category_models: dict = {
        "caller_dialog": "x-ai/grok-4",                         # full Grok 4 — edgier dialog, latency OK (gaps cut in post)
        "devon_ask": "x-ai/grok-4",                             # Devon should match the show's edgy energy
        "devon_monitor": "google/gemini-2.5-flash",           # Devon polling — just decisions, keep cheap
        "background_gen": "x-ai/grok-4",                      # wilder, more specific caller backgrounds
        "call_summary": "google/gemini-2.5-flash",            # post-call summaries
        "news_summary": "google/gemini-2.5-flash",            # news digests
        "topic_gen": "google/gemini-2.5-flash",               # topic generation
    }

    # TTS Settings
    tts_provider: str = "inworld"  # "kokoro", "elevenlabs", "inworld", "vits", or "bark"

    # Audio Settings
    sample_rate: int = 24000

    # Paths
    base_dir: Path = Path(__file__).parent.parent
    sounds_dir: Path = base_dir / "sounds"
    music_dir: Path = base_dir / "music"
    ads_dir: Path = base_dir / "ads"
    idents_dir: Path = base_dir / "idents"
    sessions_dir: Path = base_dir / "sessions"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
