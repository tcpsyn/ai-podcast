"""TTS service with ElevenLabs, F5-TTS, MLX Kokoro, StyleTTS2, VITS, and Bark support"""

import os
import numpy as np
from scipy.signal import butter, filtfilt
from pathlib import Path
import tempfile
import torch

from ..config import settings

# Patch torch.load for compatibility with PyTorch 2.6+
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Global clients
_elevenlabs_client = None
_vits_tts = None
_bark_loaded = False
_kokoro_model = None
_styletts2_model = None
_f5tts_model = None
_chattts_model = None
_chattts_speakers = {}  # Cache for speaker embeddings

# Kokoro voice mapping - using highest-graded voices
# Grades from https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
KOKORO_VOICES = {
    # Male voices (best available are C+ grade)
    "VR6AewLTigWG4xSOukaG": "am_fenrir",  # Tony - deep/powerful (C+)
    "TxGEqnHWrfWFTfGW9XjX": "am_michael", # Rick - solid male voice (C+)
    "pNInz6obpgDQGcFmaJgB": "am_puck",    # Dennis - anxious dad (C+)
    "ODq5zmih8GrVes37Dizd": "bm_george",  # Earl - older/distinguished British (C)
    "IKne3meq5aSn9XLyUdCD": "bm_fable",   # Marcus - young British (C)
    # Female voices (much better quality available)
    "jBpfuIE2acCO8z3wKNLl": "af_heart",   # Jasmine - best quality (A)
    "EXAVITQu4vr4xnSDxMaL": "af_bella",   # Megan - warm/friendly (A-)
    "21m00Tcm4TlvDq8ikWAM": "bf_emma",    # Tanya - professional British (B-)
    "XB0fDUnXU5powFXDhCwa": "af_nicole",  # Carla - Jersey mom (B-)
    "pFZP5JQG7iQjIQuC4Bku": "af_sarah",   # Brenda - overthinker (C+)
}

# Speed adjustments per voice (1.0 = normal, lower = slower/more natural)
# Slower speeds (0.85-0.95) generally sound more natural
KOKORO_SPEEDS = {
    # Male voices - slower speeds help with C+ grade voices
    "VR6AewLTigWG4xSOukaG": 0.9,   # Tony (am_fenrir) - deep voice, slower
    "TxGEqnHWrfWFTfGW9XjX": 0.92,  # Rick (am_michael) - solid pace
    "pNInz6obpgDQGcFmaJgB": 0.95,  # Dennis (am_puck) - anxious but not rushed
    "ODq5zmih8GrVes37Dizd": 0.85,  # Earl (bm_george) - older, slower British
    "IKne3meq5aSn9XLyUdCD": 0.95,  # Marcus (bm_fable) - young, natural
    # Female voices - A-grade voices can handle faster speeds
    "jBpfuIE2acCO8z3wKNLl": 0.95,  # Jasmine (af_heart) - best voice, natural pace
    "EXAVITQu4vr4xnSDxMaL": 0.95,  # Megan (af_bella) - warm
    "21m00Tcm4TlvDq8ikWAM": 0.9,   # Tanya (bf_emma) - professional British
    "XB0fDUnXU5powFXDhCwa": 0.95,  # Carla (af_nicole) - animated but clear
    "pFZP5JQG7iQjIQuC4Bku": 0.92,  # Brenda (af_sarah) - overthinker, measured
}

DEFAULT_KOKORO_VOICE = "af_heart"
DEFAULT_KOKORO_SPEED = 0.95

# VCTK speaker mapping - different voices for different callers
VITS_SPEAKERS = {
    # Male voices
    "VR6AewLTigWG4xSOukaG": "p226",  # Tony
    "TxGEqnHWrfWFTfGW9XjX": "p251",  # Rick
    "pNInz6obpgDQGcFmaJgB": "p245",  # Dennis
    "ODq5zmih8GrVes37Dizd": "p232",  # Earl
    "IKne3meq5aSn9XLyUdCD": "p252",  # Marcus
    # Female voices
    "jBpfuIE2acCO8z3wKNLl": "p225",  # Jasmine
    "EXAVITQu4vr4xnSDxMaL": "p228",  # Megan
    "21m00Tcm4TlvDq8ikWAM": "p229",  # Tanya
    "XB0fDUnXU5powFXDhCwa": "p231",  # Carla
    "pFZP5JQG7iQjIQuC4Bku": "p233",  # Brenda
}

DEFAULT_VITS_SPEAKER = "p225"

# Inworld voice mapping - maps ElevenLabs voice IDs to Inworld voices
# Full voice list from API (English): Abby, Alex, Amina, Anjali, Arjun, Ashley,
# Blake, Brian, Callum, Carter, Celeste, Chloe, Claire, Clive, Craig, Darlene,
# Deborah, Dennis, Derek, Dominus, Edward, Elizabeth, Elliot, Ethan, Evan, Evelyn,
# Gareth, Graham, Grant, Hades, Hamish, Hana, Hank, Jake, James, Jason, Jessica,
# Julia, Kayla, Kelsey, Lauren, Liam, Loretta, Luna, Malcolm, Mark, Marlene,
# Miranda, Mortimer, Nate, Oliver, Olivia, Pippa, Pixie, Priya, Ronald, Rupert,
# Saanvi, Sarah, Sebastian, Serena, Shaun, Simon, Snik, Tessa, Theodore, Timothy,
# Tyler, Veronica, Victor, Victoria, Vinny, Wendy
INWORLD_VOICES = {
    # Original voice IDs
    "VR6AewLTigWG4xSOukaG": "Edward",    # Tony - fast-talking, emphatic, streetwise
    "TxGEqnHWrfWFTfGW9XjX": "Shaun",     # Rick - friendly, dynamic, conversational
    "pNInz6obpgDQGcFmaJgB": "Alex",      # Dennis - energetic, expressive, mildly nasal
    "ODq5zmih8GrVes37Dizd": "Craig",     # Earl - older British, refined, articulate
    "IKne3meq5aSn9XLyUdCD": "Timothy",   # Marcus/Jerome - lively, upbeat American
    "jBpfuIE2acCO8z3wKNLl": "Hana",      # Jasmine - bright, expressive young female
    "EXAVITQu4vr4xnSDxMaL": "Ashley",    # Megan - warm, natural female
    "21m00Tcm4TlvDq8ikWAM": "Wendy",     # Tanya - posh, middle-aged British
    "XB0fDUnXU5powFXDhCwa": "Sarah",     # Carla - fast-talking, questioning tone
    "pFZP5JQG7iQjIQuC4Bku": "Deborah",   # Brenda (original) - gentle, elegant
    # Regular caller voice IDs (backfilled)
    "onwK4e9ZLuTAKqWW03F9": "Ronald",    # Bobby - repo man
    "FGY2WhTYpPnrIDTdsKH5": "Julia",     # Carla (regular) - Jersey mom
    "CwhRBWXzGAHq8TQ4Fs17": "Mark",      # Leon - male caller
    "SOYHLrjzK2X1ezoPC6cr": "Carter",    # Carl - male caller
    "N2lVS1w4EtoT3dr4eOWO": "Clive",     # Reggie - male caller
    "hpp4J3VqNfWAUOO0d1Us": "Olivia",    # Brenda (regular) - ambulance driver
    "nPczCjzI2devNBz1zQrb": "Theodore",  # Keith - male caller
    "JBFqnCBsd6RMkjVDRZzb": "Blake",     # Andre - male caller
    "TX3LPaxmHKxFdv7VOQHJ": "Dennis",    # Rick (regular) - male caller
    "cgSgspJ2msm6clMCkdW9": "Priya",     # Megan (regular) - female caller
}
DEFAULT_INWORLD_VOICE = "Dennis"

# Inworld voices that speak too slowly at default rate — bump them up
# Range is 0.5 to 1.5, where 1.0 is the voice's native speed
INWORLD_SPEED_OVERRIDES = {
    "Wendy": 1.15,
    "Craig": 1.15,
    "Deborah": 1.15,
    "Sarah": 1.1,
    "Hana": 1.1,
    "Theodore": 1.15,
    "Blake": 1.1,
    "Priya": 1.1,
}
DEFAULT_INWORLD_SPEED = 1.1  # Slight bump for all voices


def preprocess_text_for_kokoro(text: str) -> str:
    """
    Preprocess text to improve Kokoro prosody and naturalness.

    - Adds slight pauses via punctuation
    - Handles contractions and abbreviations
    - Normalizes spacing
    """
    import re

    # Normalize whitespace
    text = ' '.join(text.split())

    # Add comma pauses after common transition words (if no punctuation follows)
    transitions = [
        r'\b(Well)\s+(?=[A-Za-z])',
        r'\b(So)\s+(?=[A-Za-z])',
        r'\b(Now)\s+(?=[A-Za-z])',
        r'\b(Look)\s+(?=[A-Za-z])',
        r'\b(See)\s+(?=[A-Za-z])',
        r'\b(Anyway)\s+(?=[A-Za-z])',
        r'\b(Actually)\s+(?=[A-Za-z])',
        r'\b(Honestly)\s+(?=[A-Za-z])',
        r'\b(Basically)\s+(?=[A-Za-z])',
    ]
    for pattern in transitions:
        text = re.sub(pattern, r'\1, ', text)

    # Add pause after "I mean" at start of sentence
    text = re.sub(r'^(I mean)\s+', r'\1, ', text)
    text = re.sub(r'\.\s+(I mean)\s+', r'. \1, ', text)

    # Expand common abbreviations for better pronunciation
    abbreviations = {
        r'\bDr\.': 'Doctor',
        r'\bMr\.': 'Mister',
        r'\bMrs\.': 'Missus',
        r'\bMs\.': 'Miss',
        r'\bSt\.': 'Street',
        r'\bAve\.': 'Avenue',
        r'\betc\.': 'etcetera',
        r'\bvs\.': 'versus',
        r'\bw/': 'with',
        r'\bw/o': 'without',
    }
    for abbr, expansion in abbreviations.items():
        text = re.sub(abbr, expansion, text, flags=re.IGNORECASE)

    # Add breath pause (comma) before conjunctions in long sentences
    text = re.sub(r'(\w{20,})\s+(and|but|or)\s+', r'\1, \2 ', text)

    # Ensure proper spacing after punctuation
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)

    return text

# StyleTTS2 reference voice files (place .wav files in voices/ directory for voice cloning)
# Maps voice_id to reference audio filename - if file doesn't exist, uses default voice
STYLETTS2_VOICES = {
    # Male voices
    "VR6AewLTigWG4xSOukaG": "tony.wav",     # Tony
    "TxGEqnHWrfWFTfGW9XjX": "rick.wav",     # Rick
    "pNInz6obpgDQGcFmaJgB": "dennis.wav",   # Dennis
    "ODq5zmih8GrVes37Dizd": "earl.wav",     # Earl
    "IKne3meq5aSn9XLyUdCD": "marcus.wav",   # Marcus
    # Female voices
    "jBpfuIE2acCO8z3wKNLl": "jasmine.wav",  # Jasmine
    "EXAVITQu4vr4xnSDxMaL": "megan.wav",    # Megan
    "21m00Tcm4TlvDq8ikWAM": "tanya.wav",    # Tanya
    "XB0fDUnXU5powFXDhCwa": "carla.wav",    # Carla
    "pFZP5JQG7iQjIQuC4Bku": "brenda.wav",   # Brenda
}

# F5-TTS reference voices (same files as StyleTTS2, reuses voices/ directory)
# Requires: mono, 24kHz, 5-10 seconds, with transcript in .txt file
F5TTS_VOICES = STYLETTS2_VOICES.copy()

# ChatTTS speaker seeds - different seeds produce different voices
# These are used to generate consistent speaker embeddings
CHATTTS_SEEDS = {
    # Male voices
    "VR6AewLTigWG4xSOukaG": 42,     # Tony - deep voice
    "TxGEqnHWrfWFTfGW9XjX": 123,    # Rick
    "pNInz6obpgDQGcFmaJgB": 456,    # Dennis
    "ODq5zmih8GrVes37Dizd": 789,    # Earl
    "IKne3meq5aSn9XLyUdCD": 1011,   # Marcus
    # Female voices
    "jBpfuIE2acCO8z3wKNLl": 2024,   # Jasmine
    "EXAVITQu4vr4xnSDxMaL": 3033,   # Megan
    "21m00Tcm4TlvDq8ikWAM": 4042,   # Tanya
    "XB0fDUnXU5powFXDhCwa": 5051,   # Carla
    "pFZP5JQG7iQjIQuC4Bku": 6060,   # Brenda
}
DEFAULT_CHATTTS_SEED = 42


def get_elevenlabs_client():
    """Get or create ElevenLabs client"""
    global _elevenlabs_client
    if _elevenlabs_client is None:
        from elevenlabs.client import ElevenLabs
        _elevenlabs_client = ElevenLabs(api_key=settings.elevenlabs_api_key)
    return _elevenlabs_client


def get_vits_tts():
    """Get or create VITS VCTK TTS instance"""
    global _vits_tts
    if _vits_tts is None:
        from TTS.api import TTS
        _vits_tts = TTS("tts_models/en/vctk/vits")
    return _vits_tts


def get_kokoro_model():
    """Get or create Kokoro MLX model"""
    global _kokoro_model
    if _kokoro_model is None:
        from mlx_audio.tts.utils import load_model
        _kokoro_model = load_model(model_path='mlx-community/Kokoro-82M-bf16')
        print("Kokoro MLX model loaded")
    return _kokoro_model


def ensure_bark_loaded():
    """Ensure Bark models are loaded on GPU"""
    global _bark_loaded
    if not _bark_loaded:
        os.environ['SUNO_USE_SMALL_MODELS'] = '1'

        # Force Bark to use MPS (Apple Silicon GPU)
        if torch.backends.mps.is_available():
            os.environ['SUNO_OFFLOAD_CPU'] = '0'
            os.environ['SUNO_ENABLE_MPS'] = '1'

        from bark import preload_models
        preload_models()
        _bark_loaded = True
        print(f"Bark loaded on device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")


def get_styletts2_model():
    """Get or create StyleTTS2 model"""
    global _styletts2_model
    if _styletts2_model is None:
        from styletts2 import tts
        _styletts2_model = tts.StyleTTS2()
        print("StyleTTS2 model loaded")
    return _styletts2_model


def get_f5tts_generate():
    """Get F5-TTS generate function (lazy load)"""
    global _f5tts_model
    if _f5tts_model is None:
        # Disable tqdm progress bars to avoid BrokenPipeError in server context
        import os
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        os.environ['TQDM_DISABLE'] = '1'

        from f5_tts_mlx.generate import generate
        _f5tts_model = generate
        print("F5-TTS MLX loaded")
    return _f5tts_model


def get_chattts_model():
    """Get or create ChatTTS model"""
    global _chattts_model
    if _chattts_model is None:
        import ChatTTS
        _chattts_model = ChatTTS.Chat()
        _chattts_model.load(compile=False)
        print("ChatTTS model loaded")
    return _chattts_model


def get_chattts_speaker(voice_id: str):
    """Get or create a consistent speaker embedding for a voice"""
    global _chattts_speakers
    if voice_id not in _chattts_speakers:
        chat = get_chattts_model()
        seed = CHATTTS_SEEDS.get(voice_id, DEFAULT_CHATTTS_SEED)
        # Set seed for reproducible speaker
        torch.manual_seed(seed)
        _chattts_speakers[voice_id] = chat.sample_random_speaker()
        print(f"[ChatTTS] Created speaker for voice {voice_id} with seed {seed}")
    return _chattts_speakers[voice_id]


def phone_filter(audio: np.ndarray, sample_rate: int = 24000, quality: str = "normal") -> np.ndarray:
    """Apply phone filter with variable quality."""
    audio = audio.flatten()

    presets = {
        "good": (200, 7000, 1.0, 0.0),
        "normal": (300, 3400, 1.5, 0.005),
        "bad": (400, 2800, 2.0, 0.015),
        "terrible": (500, 2200, 2.5, 0.03),
    }

    low_hz, high_hz, distortion, noise = presets.get(quality, presets["normal"])

    low = low_hz / (sample_rate / 2)
    high = high_hz / (sample_rate / 2)
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, audio)

    filtered = np.tanh(filtered * distortion) * 0.8

    if noise > 0:
        static = np.random.normal(0, noise, len(filtered)).astype(np.float32)
        static_envelope = np.random.random(len(filtered) // 1000 + 1)
        static_envelope = np.repeat(static_envelope, 1000)[:len(filtered)]
        static *= (static_envelope > 0.7).astype(np.float32)
        filtered = filtered + static

    return filtered.astype(np.float32)


async def generate_speech_elevenlabs(text: str, voice_id: str) -> tuple[np.ndarray, int]:
    """Generate speech using ElevenLabs"""
    client = get_elevenlabs_client()

    audio_gen = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_v3",
        output_format="pcm_24000"
    )

    audio_bytes = b"".join(audio_gen)
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    return audio, 24000


async def generate_speech_kokoro(text: str, voice_id: str) -> tuple[np.ndarray, int]:
    """Generate speech using MLX Kokoro (fast, good quality, Apple Silicon optimized)"""
    import librosa
    from mlx_audio.tts.generate import generate_audio

    model = get_kokoro_model()
    voice = KOKORO_VOICES.get(voice_id, DEFAULT_KOKORO_VOICE)
    speed = KOKORO_SPEEDS.get(voice_id, DEFAULT_KOKORO_SPEED)

    # Preprocess text for better prosody
    text = preprocess_text_for_kokoro(text)

    # Determine lang_code from voice prefix (a=American, b=British)
    lang_code = 'b' if voice.startswith('b') else 'a'

    with tempfile.TemporaryDirectory() as tmpdir:
        generate_audio(
            text,
            model=model,
            voice=voice,
            speed=speed,
            lang_code=lang_code,
            output_path=tmpdir,
            file_prefix='tts',
            verbose=False
        )

        # Read the generated audio file
        audio_file = Path(tmpdir) / 'tts_000.wav'
        if not audio_file.exists():
            raise RuntimeError("Kokoro failed to generate audio")

        audio, sr = librosa.load(str(audio_file), sr=None, mono=True)

        # Resample to 24kHz if needed
        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)

        return audio.astype(np.float32), 24000


async def generate_speech_vits(text: str, voice_id: str) -> tuple[np.ndarray, int]:
    """Generate speech using VITS VCTK (fast, multiple speakers)"""
    import librosa

    tts = get_vits_tts()
    speaker = VITS_SPEAKERS.get(voice_id, DEFAULT_VITS_SPEAKER)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        tts.tts_to_file(text=text, file_path=tmp_path, speaker=speaker)
        audio, sr = librosa.load(tmp_path, sr=None, mono=True)

        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)

        return audio.astype(np.float32), 24000
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def generate_speech_bark(text: str, voice_id: str) -> tuple[np.ndarray, int]:
    """Generate speech using Bark (slow but expressive, supports emotes like [laughs])"""
    import librosa
    from bark import SAMPLE_RATE, generate_audio

    ensure_bark_loaded()

    # Generate audio with Bark
    audio = generate_audio(text)

    # Normalize to prevent clipping (Bark can exceed [-1, 1])
    max_val = np.abs(audio).max()
    if max_val > 0.95:
        audio = audio * (0.95 / max_val)

    # Resample to 24kHz if needed
    if SAMPLE_RATE != 24000:
        audio = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=24000)

    return audio.astype(np.float32), 24000


async def generate_speech_styletts2(text: str, voice_id: str) -> tuple[np.ndarray, int]:
    """Generate speech using StyleTTS2 (high quality, supports voice cloning)"""
    import librosa

    model = get_styletts2_model()

    # Check for reference voice file
    voice_file = STYLETTS2_VOICES.get(voice_id)
    voice_path = None
    if voice_file:
        voice_path = settings.base_dir / "voices" / voice_file
        if not voice_path.exists():
            voice_path = None  # Use default voice if file doesn't exist

    # Generate audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if voice_path:
            print(f"[StyleTTS2] Using voice clone: {voice_path}")
            audio = model.inference(
                text,
                target_voice_path=str(voice_path),
                output_wav_file=tmp_path,
                output_sample_rate=24000,
                diffusion_steps=5,  # Balance quality/speed
                alpha=0.3,  # More voice-like than text-like
                beta=0.7,   # Good prosody
            )
        else:
            print("[StyleTTS2] Using default voice")
            audio = model.inference(
                text,
                output_wav_file=tmp_path,
                output_sample_rate=24000,
                diffusion_steps=5,
            )

        # Load the generated audio
        audio, sr = librosa.load(tmp_path, sr=None, mono=True)

        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)

        return audio.astype(np.float32), 24000
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def generate_speech_f5tts(text: str, voice_id: str) -> tuple[np.ndarray, int]:
    """Generate speech using F5-TTS MLX (very natural, supports voice cloning)"""
    import librosa

    generate = get_f5tts_generate()

    # Check for reference voice file and transcript
    voice_file = F5TTS_VOICES.get(voice_id)
    ref_audio_path = None
    ref_text = None

    if voice_file:
        voice_path = settings.base_dir / "voices" / voice_file
        txt_path = voice_path.with_suffix('.txt')

        if voice_path.exists() and txt_path.exists():
            ref_audio_path = str(voice_path)
            ref_text = txt_path.read_text().strip()
            print(f"[F5-TTS] Using voice clone: {voice_path}")

    if not ref_audio_path:
        print("[F5-TTS] Using default voice")

    # Generate audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        generate(
            generation_text=text,
            ref_audio_path=ref_audio_path,
            ref_audio_text=ref_text,
            steps=8,
            speed=1.0,
            output_path=tmp_path,
        )

        # Load the generated audio
        audio, sr = librosa.load(tmp_path, sr=None, mono=True)

        # Resample to 24kHz if needed
        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)

        return audio.astype(np.float32), 24000
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def generate_speech_chattts(text: str, voice_id: str) -> tuple[np.ndarray, int]:
    """Generate speech using ChatTTS (natural conversational speech, multiple speakers)"""
    import ChatTTS

    chat = get_chattts_model()

    # Ensure text is not empty and has reasonable content
    text = text.strip()
    if not text:
        text = "Hello."

    print(f"[ChatTTS] Generating speech for: {text[:50]}...")

    # Get consistent speaker for this voice
    seed = CHATTTS_SEEDS.get(voice_id, DEFAULT_CHATTTS_SEED)
    torch.manual_seed(seed)

    # Configure inference parameters
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        temperature=0.3,
        top_P=0.7,
        top_K=20,
    )

    # Generate audio (skip text refinement to avoid narrow() error with this version)
    wavs = chat.infer(
        [text],
        params_infer_code=params_infer_code,
        skip_refine_text=True,
    )

    if wavs is None or len(wavs) == 0:
        raise RuntimeError("ChatTTS failed to generate audio")

    audio = wavs[0]

    # Handle different output shapes
    if audio.ndim > 1:
        audio = audio.squeeze()

    # Normalize
    max_val = np.abs(audio).max()
    if max_val > 0.95:
        audio = audio * (0.95 / max_val)

    return audio.astype(np.float32), 24000


async def generate_speech_inworld(text: str, voice_id: str) -> tuple[np.ndarray, int]:
    """Generate speech using Inworld TTS API (high quality, natural voices)"""
    import httpx
    import base64
    import librosa

    # voice_id is now the Inworld voice name directly (e.g. "Edward")
    # Fall back to legacy mapping if it's an ElevenLabs ID
    if voice_id in INWORLD_VOICES:
        voice = INWORLD_VOICES[voice_id]
    else:
        voice = voice_id

    api_key = settings.inworld_api_key
    if not api_key:
        raise RuntimeError("INWORLD_API_KEY not set in environment")

    speed = INWORLD_SPEED_OVERRIDES.get(voice, DEFAULT_INWORLD_SPEED)
    print(f"[Inworld TTS] Voice: {voice}, Speed: {speed}, Text: {text[:50]}...")

    url = "https://api.inworld.ai/tts/v1/voice"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Basic {api_key}",
    }
    payload = {
        "text": text,
        "voiceId": voice,
        "modelId": "inworld-tts-1.5-max",
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": 48000,
            "speakingRate": speed,
        },
    }

    async with httpx.AsyncClient(timeout=25.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    # Decode base64 audio
    audio_b64 = data.get("audioContent")
    if not audio_b64:
        raise RuntimeError("Inworld TTS returned no audio content")

    audio_bytes = base64.b64decode(audio_b64)

    # Parse audio using soundfile (handles WAV, MP3, etc.)
    import soundfile as sf
    import io

    # soundfile can read WAV, FLAC, OGG, and with ffmpeg: MP3
    # MP3 files start with ID3 tag or 0xff sync bytes
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception as e:
        print(f"[Inworld TTS] soundfile failed: {e}, trying raw PCM")
        # Fallback to raw PCM
        if len(audio_bytes) % 2 != 0:
            audio_bytes = audio_bytes[:-1]
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sr = 48000

    # Resample to 24kHz to match other providers
    if sr != 24000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)

    return audio.astype(np.float32), 24000


_TTS_PROVIDERS = {
    "kokoro": lambda text, vid: generate_speech_kokoro(text, vid),
    "f5tts": lambda text, vid: generate_speech_f5tts(text, vid),
    "inworld": lambda text, vid: generate_speech_inworld(text, vid),
    "chattts": lambda text, vid: generate_speech_chattts(text, vid),
    "styletts2": lambda text, vid: generate_speech_styletts2(text, vid),
    "bark": lambda text, vid: generate_speech_bark(text, vid),
    "vits": lambda text, vid: generate_speech_vits(text, vid),
    "elevenlabs": lambda text, vid: generate_speech_elevenlabs(text, vid),
}

TTS_MAX_RETRIES = 3
TTS_RETRY_DELAYS = [1.0, 2.0, 4.0]  # seconds between retries


async def generate_speech(
    text: str,
    voice_id: str,
    phone_quality: str = "normal",
    apply_filter: bool = True
) -> bytes:
    """
    Generate speech from text with automatic retry on failure.

    Args:
        text: Text to speak
        voice_id: ElevenLabs voice ID (mapped to local voice if using local TTS)
        phone_quality: Quality of phone filter ("none" to disable)
        apply_filter: Whether to apply phone filter

    Returns:
        Raw PCM audio bytes (16-bit signed int, 24kHz)
    """
    import asyncio

    provider = settings.tts_provider
    print(f"[TTS] Provider: {provider}, Text: {text[:50]}...")

    gen_fn = _TTS_PROVIDERS.get(provider)
    if not gen_fn:
        raise ValueError(f"Unknown TTS provider: {provider}")

    last_error = None
    for attempt in range(TTS_MAX_RETRIES):
        try:
            audio, sample_rate = await gen_fn(text, voice_id)
            if attempt > 0:
                print(f"[TTS] Succeeded on retry {attempt}")
            break
        except Exception as e:
            last_error = e
            if attempt < TTS_MAX_RETRIES - 1:
                delay = TTS_RETRY_DELAYS[attempt]
                print(f"[TTS] {provider} attempt {attempt + 1} failed: {e} — retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                print(f"[TTS] {provider} failed after {TTS_MAX_RETRIES} attempts: {e}")
                raise

    # Apply phone filter if requested
    # Skip filter for Bark - it already has rough audio quality
    if apply_filter and phone_quality not in ("none", "studio") and provider != "bark":
        audio = phone_filter(audio, sample_rate, phone_quality)

    # Convert to bytes
    audio_int16 = (audio * 32768).clip(-32768, 32767).astype(np.int16)
    return audio_int16.tobytes()


# Voice IDs for cohost and announcer
COHOST_VOICE_ID = "nPczCjzI2devNBz1zQrb"
ANNOUNCER_VOICE_ID = "ErXwobaYiN019PkySvjV"


async def generate_cohost_speech(text: str) -> bytes:
    """Generate speech for cohost Bobby (no phone filter)"""
    return await generate_speech(text, COHOST_VOICE_ID, apply_filter=False)


async def generate_announcer_speech(text: str) -> bytes:
    """Generate speech for announcer (no phone filter)"""
    return await generate_speech(text, ANNOUNCER_VOICE_ID, apply_filter=False)
