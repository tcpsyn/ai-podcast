"""Whisper transcription service"""

import tempfile
import numpy as np
from faster_whisper import WhisperModel
import librosa

# Global model instance (loaded once)
_whisper_model = None


def get_whisper_model() -> WhisperModel:
    """Get or create Whisper model instance"""
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper tiny model for fast transcription...")
        # Use tiny model for speed - about 3-4x faster than base
        # beam_size=1 and best_of=1 for fastest inference
        _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("Whisper model loaded")
    return _whisper_model


def decode_audio(audio_data: bytes, source_sample_rate: int = None) -> tuple[np.ndarray, int]:
    """
    Decode audio from various formats to numpy array.

    Args:
        audio_data: Raw audio bytes
        source_sample_rate: If provided, treat as raw PCM at this sample rate

    Returns:
        Tuple of (audio array as float32, sample rate)
    """
    # If sample rate is provided, assume raw PCM (from server-side recording)
    if source_sample_rate is not None:
        print(f"Decoding raw PCM at {source_sample_rate}Hz, {len(audio_data)} bytes")
        if len(audio_data) % 2 != 0:
            audio_data = audio_data + b'\x00'
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, source_sample_rate

    print(f"First 20 bytes: {audio_data[:20].hex()}")

    # Try to decode with librosa first (handles webm, ogg, wav, mp3, etc via ffmpeg)
    try:
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        audio, sample_rate = librosa.load(temp_path, sr=None, mono=True)
        print(f"Decoded with librosa: {len(audio)} samples at {sample_rate}Hz")

        import os
        os.unlink(temp_path)

        return audio.astype(np.float32), sample_rate

    except Exception as e:
        print(f"librosa decode failed: {e}, trying raw PCM at 16kHz...")

        # Fall back to raw PCM (16-bit signed int, 16kHz mono - Whisper's rate)
        if len(audio_data) % 2 != 0:
            audio_data = audio_data + b'\x00'

        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, 16000


async def transcribe_audio(audio_data: bytes, source_sample_rate: int = None) -> str:
    """
    Transcribe audio data to text using Whisper.

    Args:
        audio_data: Audio bytes (webm, ogg, wav, or raw PCM)
        source_sample_rate: If provided, treat audio_data as raw PCM at this rate

    Returns:
        Transcribed text
    """
    model = get_whisper_model()

    print(f"Transcribing audio: {len(audio_data)} bytes")

    # Decode audio from whatever format
    audio, detected_sample_rate = decode_audio(audio_data, source_sample_rate)

    print(f"Audio samples: {len(audio)}, duration: {len(audio)/detected_sample_rate:.2f}s")
    print(f"Audio range: min={audio.min():.4f}, max={audio.max():.4f}")

    # Check if audio is too quiet
    if np.abs(audio).max() < 0.01:
        print("Warning: Audio appears to be silent or very quiet")
        return ""

    # Resample to 16kHz for Whisper
    if detected_sample_rate != 16000:
        audio_16k = librosa.resample(audio, orig_sr=detected_sample_rate, target_sr=16000)
        print(f"Resampled to {len(audio_16k)} samples at 16kHz")
    else:
        audio_16k = audio

    # Transcribe with speed optimizations
    segments, info = model.transcribe(
        audio_16k,
        beam_size=1,  # Faster, slightly less accurate
        best_of=1,
        language="en",  # Skip language detection
        vad_filter=True,  # Skip silence
    )
    segments_list = list(segments)
    text = " ".join([s.text for s in segments_list]).strip()

    print(f"Transcription result: '{text}' (language: {info.language}, prob: {info.language_probability:.2f})")

    return text
