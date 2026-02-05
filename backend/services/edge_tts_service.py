"""Edge TTS service - free Microsoft TTS API"""

import asyncio
import io
import numpy as np
from typing import Optional

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


class EdgeTTSService:
    """TTS using Microsoft Edge's free API"""

    def __init__(self):
        self.sample_rate = 24000  # Edge TTS outputs 24kHz

    def is_available(self) -> bool:
        return EDGE_TTS_AVAILABLE

    async def generate_speech(self, text: str, voice: str = "en-US-JennyNeural") -> bytes:
        """Generate speech from text using Edge TTS

        Args:
            text: Text to synthesize
            voice: Edge TTS voice name (e.g., "en-US-JennyNeural")

        Returns:
            Raw PCM audio bytes (16-bit signed int, 24kHz mono)
        """
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")

        communicate = edge_tts.Communicate(text, voice)

        # Collect MP3 audio data
        mp3_data = b''
        async for chunk in communicate.stream():
            if chunk['type'] == 'audio':
                mp3_data += chunk['data']

        if not mp3_data:
            raise RuntimeError("No audio generated")

        # Convert MP3 to PCM
        pcm_data = await self._mp3_to_pcm(mp3_data)
        return pcm_data

    async def _mp3_to_pcm(self, mp3_data: bytes) -> bytes:
        """Convert MP3 to raw PCM using ffmpeg or pydub"""
        loop = asyncio.get_event_loop()

        def convert():
            try:
                # Try pydub first (more reliable)
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(io.BytesIO(mp3_data))
                # Convert to 24kHz mono 16-bit
                audio = audio.set_frame_rate(24000).set_channels(1).set_sample_width(2)
                return audio.raw_data
            except ImportError:
                pass

            # Fallback to ffmpeg subprocess
            import subprocess
            process = subprocess.Popen(
                [
                    'ffmpeg', '-i', 'pipe:0',
                    '-f', 's16le',
                    '-acodec', 'pcm_s16le',
                    '-ar', '24000',
                    '-ac', '1',
                    'pipe:1'
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            pcm_data, stderr = process.communicate(input=mp3_data)
            if process.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")
            return pcm_data

        return await loop.run_in_executor(None, convert)

    async def list_voices(self) -> list[dict]:
        """List available Edge TTS voices"""
        if not EDGE_TTS_AVAILABLE:
            return []

        voices = await edge_tts.list_voices()
        return [
            {
                "id": v["ShortName"],
                "name": v["ShortName"].replace("Neural", ""),
                "gender": v["Gender"],
                "locale": v["Locale"],
            }
            for v in voices
            if v["Locale"].startswith("en-")
        ]


# Global instance
edge_tts_service = EdgeTTSService()


def is_edge_tts_available() -> bool:
    return edge_tts_service.is_available()
