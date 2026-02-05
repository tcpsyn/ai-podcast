"""Piper TTS service using sherpa-onnx for fast local voice synthesis"""

import asyncio
import numpy as np
from pathlib import Path
from typing import Optional

# Models directory
MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "sherpa"

# Try to import sherpa-onnx
try:
    import sherpa_onnx
    SHERPA_AVAILABLE = True
except ImportError:
    SHERPA_AVAILABLE = False
    sherpa_onnx = None


# Available sherpa-onnx Piper models
PIPER_MODELS = {
    "amy": {
        "dir": "vits-piper-en_US-amy-low",
        "model": "en_US-amy-low.onnx",
        "name": "Amy (US Female)",
        "sample_rate": 16000,
    },
    "joe": {
        "dir": "vits-piper-en_US-joe-medium",
        "model": "en_US-joe-medium.onnx",
        "name": "Joe (US Male)",
        "sample_rate": 22050,
    },
    "lessac": {
        "dir": "vits-piper-en_US-lessac-medium",
        "model": "en_US-lessac-medium.onnx",
        "name": "Lessac (US Female)",
        "sample_rate": 22050,
    },
    "alan": {
        "dir": "vits-piper-en_GB-alan-medium",
        "model": "en_GB-alan-medium.onnx",
        "name": "Alan (UK Male)",
        "sample_rate": 22050,
    },
}


class PiperTTSService:
    """Fast local TTS using sherpa-onnx with Piper models"""

    def __init__(self):
        self.output_sample_rate = 24000  # Our standard output rate
        self._tts_engines: dict[str, any] = {}

    def is_available(self) -> bool:
        """Check if sherpa-onnx is available"""
        return SHERPA_AVAILABLE

    def _get_engine(self, model_key: str):
        """Get or create a TTS engine for the given model"""
        if model_key in self._tts_engines:
            return self._tts_engines[model_key], PIPER_MODELS[model_key]["sample_rate"]

        if model_key not in PIPER_MODELS:
            raise ValueError(f"Unknown model: {model_key}")

        model_info = PIPER_MODELS[model_key]
        model_dir = MODELS_DIR / model_info["dir"]

        if not model_dir.exists():
            raise RuntimeError(f"Model not found: {model_dir}")

        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model_dir / model_info["model"]),
                    tokens=str(model_dir / "tokens.txt"),
                    data_dir=str(model_dir / "espeak-ng-data"),
                ),
                num_threads=2,
            ),
        )
        tts = sherpa_onnx.OfflineTts(config)
        self._tts_engines[model_key] = tts
        return tts, model_info["sample_rate"]

    async def generate_speech(self, text: str, model_key: str = "amy") -> bytes:
        """Generate speech from text using sherpa-onnx

        Args:
            text: Text to synthesize
            model_key: Model key (amy, joe, lessac, alan)

        Returns:
            Raw PCM audio bytes (16-bit signed int, 24kHz mono)
        """
        if not SHERPA_AVAILABLE:
            raise RuntimeError("sherpa-onnx not installed. Run: pip install sherpa-onnx")

        loop = asyncio.get_event_loop()

        def run_tts():
            tts, model_sample_rate = self._get_engine(model_key)
            audio = tts.generate(text)
            samples = np.array(audio.samples, dtype=np.float32)

            # Resample to 24kHz if needed
            if model_sample_rate != self.output_sample_rate:
                ratio = self.output_sample_rate / model_sample_rate
                new_length = int(len(samples) * ratio)
                samples = np.interp(
                    np.linspace(0, len(samples) - 1, new_length),
                    np.arange(len(samples)),
                    samples
                ).astype(np.float32)

            # Convert to int16
            audio_int16 = (samples * 32767).astype(np.int16)
            return audio_int16.tobytes()

        return await loop.run_in_executor(None, run_tts)

    def list_available_models(self) -> list[dict]:
        """List available models"""
        available = []
        for key, info in PIPER_MODELS.items():
            model_dir = MODELS_DIR / info["dir"]
            if model_dir.exists():
                available.append({
                    "id": key,
                    "name": info["name"],
                    "sample_rate": info["sample_rate"],
                })
        return available


# Global instance
piper_service = PiperTTSService()


def is_piper_available() -> bool:
    """Check if Piper (sherpa-onnx) is available"""
    return piper_service.is_available()
