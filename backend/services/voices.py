"""Voice configuration and TTS provider management"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class TTSProvider(str, Enum):
    ELEVENLABS = "elevenlabs"
    EDGE = "edge"  # Microsoft Edge TTS (free)
    PIPER = "piper"  # Local Piper via sherpa-onnx (free, fast)


@dataclass
class Voice:
    """Voice configuration"""
    id: str
    name: str
    provider: TTSProvider
    provider_voice_id: str  # The actual ID used by the provider
    description: str = ""
    language: str = "en"
    gender: str = "neutral"


# ElevenLabs voices
ELEVENLABS_VOICES = [
    Voice("el_tony", "Tony (ElevenLabs)", TTSProvider.ELEVENLABS, "IKne3meq5aSn9XLyUdCD",
          "Male, New York accent, expressive", "en", "male"),
    Voice("el_jasmine", "Jasmine (ElevenLabs)", TTSProvider.ELEVENLABS, "FGY2WhTYpPnrIDTdsKH5",
          "Female, confident, direct", "en", "female"),
    Voice("el_rick", "Rick (ElevenLabs)", TTSProvider.ELEVENLABS, "JBFqnCBsd6RMkjVDRZzb",
          "Male, Texas accent, older", "en", "male"),
    Voice("el_megan", "Megan (ElevenLabs)", TTSProvider.ELEVENLABS, "XrExE9yKIg1WjnnlVkGX",
          "Female, young, casual", "en", "female"),
    Voice("el_dennis", "Dennis (ElevenLabs)", TTSProvider.ELEVENLABS, "cjVigY5qzO86Huf0OWal",
          "Male, middle-aged, anxious", "en", "male"),
    Voice("el_tanya", "Tanya (ElevenLabs)", TTSProvider.ELEVENLABS, "N2lVS1w4EtoT3dr4eOWO",
          "Female, Miami, sassy", "en", "female"),
    Voice("el_earl", "Earl (ElevenLabs)", TTSProvider.ELEVENLABS, "EXAVITQu4vr4xnSDxMaL",
          "Male, elderly, Southern", "en", "male"),
    Voice("el_carla", "Carla (ElevenLabs)", TTSProvider.ELEVENLABS, "CwhRBWXzGAHq8TQ4Fs17",
          "Female, Jersey, sharp", "en", "female"),
    Voice("el_marcus", "Marcus (ElevenLabs)", TTSProvider.ELEVENLABS, "bIHbv24MWmeRgasZH58o",
          "Male, young, urban", "en", "male"),
    Voice("el_brenda", "Brenda (ElevenLabs)", TTSProvider.ELEVENLABS, "Xb7hH8MSUJpSbSDYk0k2",
          "Female, middle-aged, worried", "en", "female"),
    Voice("el_jake", "Jake (ElevenLabs)", TTSProvider.ELEVENLABS, "SOYHLrjzK2X1ezoPC6cr",
          "Male, Boston, insecure", "en", "male"),
    Voice("el_diane", "Diane (ElevenLabs)", TTSProvider.ELEVENLABS, "cgSgspJ2msm6clMCkdW9",
          "Female, mature, conflicted", "en", "female"),
    Voice("el_bobby", "Bobby (ElevenLabs)", TTSProvider.ELEVENLABS, "nPczCjzI2devNBz1zQrb",
          "Male, sidekick, wisecracking", "en", "male"),
    Voice("el_announcer", "Announcer (ElevenLabs)", TTSProvider.ELEVENLABS, "ErXwobaYiN019PkySvjV",
          "Male, radio announcer", "en", "male"),
]

# Edge TTS voices (Microsoft, free)
EDGE_VOICES = [
    # US voices
    Voice("edge_jenny", "Jenny (Edge)", TTSProvider.EDGE, "en-US-JennyNeural",
          "Female, American, friendly", "en", "female"),
    Voice("edge_guy", "Guy (Edge)", TTSProvider.EDGE, "en-US-GuyNeural",
          "Male, American, casual", "en", "male"),
    Voice("edge_aria", "Aria (Edge)", TTSProvider.EDGE, "en-US-AriaNeural",
          "Female, American, professional", "en", "female"),
    Voice("edge_davis", "Davis (Edge)", TTSProvider.EDGE, "en-US-DavisNeural",
          "Male, American, calm", "en", "male"),
    Voice("edge_amber", "Amber (Edge)", TTSProvider.EDGE, "en-US-AmberNeural",
          "Female, American, warm", "en", "female"),
    Voice("edge_andrew", "Andrew (Edge)", TTSProvider.EDGE, "en-US-AndrewNeural",
          "Male, American, confident", "en", "male"),
    Voice("edge_ashley", "Ashley (Edge)", TTSProvider.EDGE, "en-US-AshleyNeural",
          "Female, American, cheerful", "en", "female"),
    Voice("edge_brian", "Brian (Edge)", TTSProvider.EDGE, "en-US-BrianNeural",
          "Male, American, narrator", "en", "male"),
    Voice("edge_christopher", "Christopher (Edge)", TTSProvider.EDGE, "en-US-ChristopherNeural",
          "Male, American, reliable", "en", "male"),
    Voice("edge_cora", "Cora (Edge)", TTSProvider.EDGE, "en-US-CoraNeural",
          "Female, American, older", "en", "female"),
    Voice("edge_elizabeth", "Elizabeth (Edge)", TTSProvider.EDGE, "en-US-ElizabethNeural",
          "Female, American, elegant", "en", "female"),
    Voice("edge_eric", "Eric (Edge)", TTSProvider.EDGE, "en-US-EricNeural",
          "Male, American, friendly", "en", "male"),
    Voice("edge_jacob", "Jacob (Edge)", TTSProvider.EDGE, "en-US-JacobNeural",
          "Male, American, young", "en", "male"),
    Voice("edge_michelle", "Michelle (Edge)", TTSProvider.EDGE, "en-US-MichelleNeural",
          "Female, American, clear", "en", "female"),
    Voice("edge_monica", "Monica (Edge)", TTSProvider.EDGE, "en-US-MonicaNeural",
          "Female, American, expressive", "en", "female"),
    Voice("edge_roger", "Roger (Edge)", TTSProvider.EDGE, "en-US-RogerNeural",
          "Male, American, mature", "en", "male"),
    Voice("edge_steffan", "Steffan (Edge)", TTSProvider.EDGE, "en-US-SteffanNeural",
          "Male, American, formal", "en", "male"),
    Voice("edge_tony", "Tony (Edge)", TTSProvider.EDGE, "en-US-TonyNeural",
          "Male, American, conversational", "en", "male"),
    # UK voices
    Voice("edge_sonia", "Sonia (Edge UK)", TTSProvider.EDGE, "en-GB-SoniaNeural",
          "Female, British, professional", "en", "female"),
    Voice("edge_ryan", "Ryan (Edge UK)", TTSProvider.EDGE, "en-GB-RyanNeural",
          "Male, British, clear", "en", "male"),
    Voice("edge_libby", "Libby (Edge UK)", TTSProvider.EDGE, "en-GB-LibbyNeural",
          "Female, British, warm", "en", "female"),
    Voice("edge_thomas", "Thomas (Edge UK)", TTSProvider.EDGE, "en-GB-ThomasNeural",
          "Male, British, friendly", "en", "male"),
    # Australian voices
    Voice("edge_natasha", "Natasha (Edge AU)", TTSProvider.EDGE, "en-AU-NatashaNeural",
          "Female, Australian, friendly", "en", "female"),
    Voice("edge_william", "William (Edge AU)", TTSProvider.EDGE, "en-AU-WilliamNeural",
          "Male, Australian, casual", "en", "male"),
]

# Piper voices (local, via sherpa-onnx)
PIPER_VOICES = [
    Voice("piper_amy", "Amy (Piper)", TTSProvider.PIPER, "amy",
          "Female, American, clear", "en", "female"),
    Voice("piper_joe", "Joe (Piper)", TTSProvider.PIPER, "joe",
          "Male, American, natural", "en", "male"),
    Voice("piper_lessac", "Lessac (Piper)", TTSProvider.PIPER, "lessac",
          "Female, American, expressive", "en", "female"),
    Voice("piper_alan", "Alan (Piper)", TTSProvider.PIPER, "alan",
          "Male, British, clear", "en", "male"),
]

# All voices combined
ALL_VOICES = ELEVENLABS_VOICES + EDGE_VOICES + PIPER_VOICES

# Voice lookup by ID
VOICES_BY_ID = {v.id: v for v in ALL_VOICES}

# Default voice assignments for callers (maps caller key to voice ID)
DEFAULT_CALLER_VOICES = {
    "1": "el_tony",      # Tony from Staten Island
    "2": "el_jasmine",   # Jasmine from Atlanta
    "3": "el_rick",      # Rick from Texas
    "4": "el_megan",     # Megan from Portland
    "5": "el_dennis",    # Dennis from Long Island
    "6": "el_tanya",     # Tanya from Miami
    "7": "el_earl",      # Earl from Tennessee
    "8": "el_carla",     # Carla from Jersey
    "9": "el_marcus",    # Marcus from Detroit
    "0": "el_brenda",    # Brenda from Phoenix
    "-": "el_jake",      # Jake from Boston
    "=": "el_diane",     # Diane from Chicago
    "bobby": "el_bobby",
    "announcer": "el_announcer",
}


class VoiceManager:
    """Manages voice assignments and TTS provider selection"""

    def __init__(self):
        # Current voice assignments (can be modified at runtime)
        self.caller_voices = DEFAULT_CALLER_VOICES.copy()

    def get_voice(self, voice_id: str) -> Optional[Voice]:
        """Get voice by ID"""
        return VOICES_BY_ID.get(voice_id)

    def get_caller_voice(self, caller_key: str) -> Voice:
        """Get the voice assigned to a caller"""
        voice_id = self.caller_voices.get(caller_key, "el_tony")
        return VOICES_BY_ID.get(voice_id, ELEVENLABS_VOICES[0])

    def set_caller_voice(self, caller_key: str, voice_id: str):
        """Assign a voice to a caller"""
        if voice_id in VOICES_BY_ID:
            self.caller_voices[caller_key] = voice_id

    def get_all_voices(self) -> list[dict]:
        """Get all available voices as dicts for API"""
        return [
            {
                "id": v.id,
                "name": v.name,
                "provider": v.provider.value,
                "description": v.description,
                "gender": v.gender,
            }
            for v in ALL_VOICES
        ]

    def get_voices_by_provider(self, provider: TTSProvider) -> list[Voice]:
        """Get all voices for a specific provider"""
        return [v for v in ALL_VOICES if v.provider == provider]

    def get_caller_voice_assignments(self) -> dict[str, str]:
        """Get current caller voice assignments"""
        return self.caller_voices.copy()

    def set_caller_voice_assignments(self, assignments: dict[str, str]):
        """Set multiple caller voice assignments"""
        for caller_key, voice_id in assignments.items():
            if voice_id in VOICES_BY_ID:
                self.caller_voices[caller_key] = voice_id


# Global instance
voice_manager = VoiceManager()
