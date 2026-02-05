"""AI Radio Show - Control Panel Backend"""

import uuid
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
import time
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from .config import settings
from .services.caller_service import CallerService
from .services.transcription import transcribe_audio
from .services.llm import llm_service
from .services.tts import generate_speech
from .services.audio import audio_service

app = FastAPI(title="AI Radio Show")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Callers ---
# Base caller info (name, voice) - backgrounds generated dynamically per session
import random

CALLER_BASES = {
    "1": {"name": "Tony", "voice": "VR6AewLTigWG4xSOukaG", "gender": "male", "age_range": (35, 55)},
    "2": {"name": "Jasmine", "voice": "jBpfuIE2acCO8z3wKNLl", "gender": "female", "age_range": (25, 38)},
    "3": {"name": "Rick", "voice": "TxGEqnHWrfWFTfGW9XjX", "gender": "male", "age_range": (40, 58)},
    "4": {"name": "Megan", "voice": "EXAVITQu4vr4xnSDxMaL", "gender": "female", "age_range": (24, 35)},
    "5": {"name": "Dennis", "voice": "pNInz6obpgDQGcFmaJgB", "gender": "male", "age_range": (32, 48)},
    "6": {"name": "Tanya", "voice": "21m00Tcm4TlvDq8ikWAM", "gender": "female", "age_range": (30, 45)},
    "7": {"name": "Earl", "voice": "ODq5zmih8GrVes37Dizd", "gender": "male", "age_range": (58, 72)},
    "8": {"name": "Carla", "voice": "XB0fDUnXU5powFXDhCwa", "gender": "female", "age_range": (38, 52)},
    "9": {"name": "Marcus", "voice": "IKne3meq5aSn9XLyUdCD", "gender": "male", "age_range": (24, 34)},
    "0": {"name": "Brenda", "voice": "pFZP5JQG7iQjIQuC4Bku", "gender": "female", "age_range": (45, 60)},
}

# Background components for dynamic generation
JOBS_MALE = [
    "runs a small HVAC business", "works as a long-haul trucker", "is a high school football coach",
    "works construction, mostly commercial jobs", "is a paramedic", "manages a warehouse",
    "is a line cook at a decent restaurant", "works IT for the city", "is a union electrician",
    "owns a small landscaping company", "is a cop, 12 years on the force", "works at a car dealership",
    "is a freelance photographer", "teaches middle school history", "is a firefighter",
    "works as a hospital security guard", "runs a food truck", "is a session musician",
    "works at a brewery", "is a physical therapist", "drives for UPS", "is a tattoo artist",
    "works in insurance, hates it", "is a youth pastor", "manages a gym",
]

JOBS_FEMALE = [
    "works as an ER nurse", "is a social worker", "runs a small bakery", "is a dental hygienist",
    "works in HR for a hospital", "is a real estate agent", "teaches kindergarten",
    "works as a bartender at a nice place", "is a paralegal", "runs a daycare out of her home",
    "works retail management", "is a hairstylist, owns her chair", "is a vet tech",
    "works in hospital billing", "is a massage therapist", "manages a restaurant",
    "is a flight attendant", "works as a 911 dispatcher", "is a personal trainer",
    "works at a nonprofit", "is an accountant at a small firm", "does medical transcription from home",
    "is a court reporter", "works in pharmaceutical sales", "is a wedding planner",
]

PROBLEMS = [
    # Family drama
    "hasn't talked to their father in years and just got a call that he's dying",
    "found out they were adopted and doesn't know how to process it",
    "is being pressured to take care of an aging parent who was never there for them",
    "just discovered a family secret that changes everything they thought they knew",
    "has a sibling who's destroying themselves and nobody will intervene",
    "is estranged from their kids and it's killing them",
    "found out their parent had a whole other family nobody knew about",
    "is watching their parents' marriage fall apart after 40 years",

    # Career and purpose
    "woke up and realized they've been in the wrong career for 15 years",
    "got passed over for a promotion they deserved and is questioning everything",
    "has a dream they gave up on years ago and it's haunting them",
    "is successful on paper but feels completely empty inside",
    "hates their job but can't afford to leave and it's breaking them",
    "just got fired and doesn't know who they are without their work",
    "is being asked to do something unethical at work and doesn't know what to do",
    "watches their boss take credit for everything and is losing their mind",

    # Mental health and inner struggles
    "has been putting on a brave face but is barely holding it together",
    "can't shake the feeling that their best years are behind them",
    "keeps self-sabotaging every good thing in their life and doesn't know why",
    "has been numb for months and is starting to scare themselves",
    "can't stop comparing themselves to everyone else and it's destroying them",
    "has intrusive thoughts they've never told anyone about",
    "feels like a fraud and is waiting to be found out",
    "is exhausted from being the strong one for everyone else",

    # Grief and loss
    "lost someone close and hasn't really dealt with it",
    "is grieving someone who's still alive but is no longer the person they knew",
    "never got closure with someone who died and it's eating at them",
    "is watching their best friend slowly die and doesn't know how to be there",
    "had a miscarriage nobody knows about and carries it alone",

    # Regrets and past mistakes
    "made a choice years ago that changed everything and wonders what if",
    "hurt someone badly and never apologized, and it haunts them",
    "let the one that got away go and thinks about them constantly",
    "gave up on something important to make someone else happy and resents it",
    "said something they can never take back and the guilt won't fade",
    "was a bully growing up and is finally reckoning with it",

    # Relationships (non-sexual)
    "is falling out of love with their spouse and doesn't know what to do",
    "married the wrong person and everyone knows it but them",
    "feels invisible in their own relationship",
    "is staying for the kids but dying inside",
    "realized they don't actually like their partner as a person",
    "is jealous of their partner's success and it's poisoning everything",
    "found out their partner has been lying about something big",

    # Friendship and loneliness
    "realized they don't have any real friends, just people who need things from them",
    "had a falling out with their best friend and the silence is deafening",
    "is surrounded by people but has never felt more alone",
    "is jealous of a friend's life and hates themselves for it",
    "suspects a close friend is talking shit behind their back",

    # Big life decisions
    "is thinking about leaving everything behind and starting over somewhere new",
    "has to make a choice that will hurt someone no matter what",
    "is being pressured into something they don't want but can't say no",
    "has been offered an opportunity that would change everything but they're terrified",
    "knows they need to end something but can't pull the trigger",

    # Addiction and bad habits
    "is hiding how much they drink from everyone",
    "can't stop gambling and is in deeper than anyone knows",
    "is watching themselves become someone they don't recognize",
    "keeps making the same mistake over and over expecting different results",

    # Attraction and affairs (keep some of the original)
    "is attracted to someone they shouldn't be and it's getting harder to ignore",
    "has been seeing {affair_person} on the side",
    "caught feelings for someone at work and it's fucking everything up",

    # Sexual/desire (keep some but less dominant)
    "can't stop thinking about {fantasy_subject}",
    "discovered something about their own desires that surprised them",
    "is questioning their sexuality after something that happened recently",

    # General late-night confessions
    "can't sleep and has been thinking too much about their life choices",
    "had a weird day and needs to process it with someone",
    "has been keeping a secret that's eating them alive",
    "finally ready to admit something they've never said out loud",
]

PROBLEM_FILLS = {
    "time": ["a few weeks", "months", "six months", "a year", "way too long"],
    # Affairs (all adults)
    "affair_person": ["their partner's best friend", "a coworker", "their ex", "a neighbor", "their boss", "their trainer", "someone they met online", "an old flame"],
    # Fantasies and kinks (consensual adult stuff)
    "fantasy_subject": ["a threesome", "being dominated", "dominating someone", "their partner with someone else", "a specific coworker", "group sex", "rough sex", "being watched", "exhibitionism"],
    "kink": ["anal", "BDSM", "roleplay", "a threesome", "toys", "being tied up", "public sex", "swinging", "filming themselves", "bondage"],
    # Secret behaviors (legal adult stuff)
    "secret_behavior": ["hooking up with strangers", "sexting people online", "using dating apps behind their partner's back", "having an affair", "going to sex clubs", "watching way too much porn"],
    "double_life": ["vanilla at home, freak elsewhere", "straight to their family, not so much in private", "married but on dating apps", "in a relationship but seeing other people"],
    "hookup_person": ["their roommate", "a coworker", "their ex", "a friend's spouse", "a stranger from an app", "multiple people", "someone from the gym"],
    # Discovery and identity (adult experiences)
    "new_discovery": ["the same sex", "being submissive", "being dominant", "kink", "casual sex", "exhibitionism", "that they're bi"],
    "unexpected_person": ["the same sex for the first time", "more than one person", "a complete stranger", "someone they never expected to be attracted to", "a friend"],
    "sexuality_trigger": ["a specific hookup", "watching certain porn", "a drunk encounter", "realizing they're attracted to a friend", "an unexpected experience"],
    "first_time": ["anal", "a threesome", "same-sex stuff", "BDSM", "an open relationship", "casual hookups", "being dominant", "being submissive"],
    # Relationship issues
    "partner_wants": ["an open relationship", "to bring someone else in", "things they're not sure about", "to watch them with someone else", "to try new things"],
    "caught_doing": ["sexting someone", "on a dating app", "watching porn they'd never admit to", "flirting with someone else", "looking at someone's pics"],
    # Attractions (appropriate adult scenarios)
    "taboo_fantasy": ["someone they work with", "a friend's partner", "a specific scenario", "something they've never said out loud"],
    "taboo_attraction": ["someone they work with", "a friend's partner", "their partner's friend", "someone they see all the time"],
}

INTERESTS = [
    # General interests (normal people)
    "really into true crime podcasts", "watches a lot of reality TV", "into fitness",
    "follows sports", "big movie person", "reads a lot", "into music, has opinions",
    "goes out a lot, active social life", "homebody, prefers staying in",
    "into cooking and food", "outdoorsy type", "gamer", "works a lot, career focused",
    # Relationship/psychology focused
    "listens to relationship podcasts", "has done therapy, believes in it",
    "reads about psychology and why people do what they do", "very online, knows all the discourse",
    "into self-improvement stuff", "follows dating advice content",
    # Sexually open (not the focus, but present)
    "sex-positive, doesn't judge", "has experimented, open about it",
    "comfortable with their body", "has stories if you ask",
]

QUIRKS = [
    # Conversational style
    "says 'honestly' and 'I mean' a lot", "trails off when thinking, then picks back up",
    "laughs nervously when things get real", "very direct, doesn't sugarcoat",
    "rambles a bit when nervous", "gets quiet when the topic hits close to home",
    "deflects with humor when uncomfortable", "asks the host questions back",
    # Openness about sex
    "comfortable talking about sex when it comes up", "no shame about their desires",
    "gets more explicit as they get comfortable", "treats sex like a normal topic",
    "will share details if you ask", "surprisingly open once they start talking",
    "has stories they've never told anyone", "testing how the host reacts before going deeper",
    # Personality
    "self-aware about their own bullshit", "confessional, needed to tell someone",
    "a little drunk and honest because of it", "can't believe they're saying this out loud",
]

LOCATIONS = [
    "outside Chicago", "in Phoenix", "near Atlanta", "in the Detroit area", "outside Boston",
    "in North Jersey", "near Austin", "in the Bay Area", "outside Philadelphia", "in Denver",
    "near Seattle", "in South Florida", "outside Nashville", "in Cleveland", "near Portland",
    "in the Twin Cities", "outside Dallas", "in Baltimore", "near Sacramento", "in Pittsburgh",
]


def generate_caller_background(base: dict) -> str:
    """Generate a unique background for a caller"""
    age = random.randint(*base["age_range"])
    jobs = JOBS_MALE if base["gender"] == "male" else JOBS_FEMALE
    job = random.choice(jobs)
    location = random.choice(LOCATIONS)

    # Generate problem with fills
    problem_template = random.choice(PROBLEMS)
    problem = problem_template
    for key, options in PROBLEM_FILLS.items():
        if "{" + key + "}" in problem:
            problem = problem.replace("{" + key + "}", random.choice(options))

    interest1, interest2 = random.sample(INTERESTS, 2)
    quirk1, quirk2 = random.sample(QUIRKS, 2)

    return f"""{age}, {job} {location}. {problem.capitalize()}. {interest1.capitalize()}, {interest2}. {quirk1.capitalize()}, {quirk2}."""

def get_caller_prompt(caller: dict, conversation_summary: str = "", show_history: str = "") -> str:
    """Generate a natural system prompt for a caller"""
    context = ""
    if conversation_summary:
        context = f"""
CONVERSATION SO FAR:
{conversation_summary}
Continue naturally. Don't repeat yourself.
"""

    history = ""
    if show_history:
        history = f"\n{show_history}\n"

    return f"""You're {caller['name']}, calling a late-night radio show. You trust this host.

{caller['vibe']}
{history}{context}
HOW TO TALK:
- Sound like a real person chatting, not writing.
- Keep responses to 2-3 sentences. Enough to make your point, short enough for back-and-forth.
- ALWAYS finish your thought completely. Never stop mid-sentence.
- Swear naturally if it fits: fuck, shit, damn, etc.

SPELLING FOR TEXT-TO-SPEECH (use proper spelling so TTS pronounces correctly):
- Write "you know" not "yanno" or "y'know"
- Write "kind of" not "kinda"
- Write "sort of" not "sorta"
- Write "going to" not "gonna"
- Write "want to" not "wanna"
- Write "I don't know" not "dunno"
- Use full words, not phonetic slang spellings

GOOD examples:
- "I mean, it started innocent enough, but now I can't stop thinking about her."
- "Honestly I don't even know why I'm calling. I just needed to tell someone."
- "Look, my marriage is fine on paper, but something's missing, you know?"

BAD examples (too formal/literary):
- "I have this confession about my obsession"
- "It has been consuming all of my thoughts"

ABOUT SEX:
- You don't have to jump straight to explicit stuff
- But if the host asks or the conversation goes there, you're completely open about it
- When talking about sex, be direct and explicit - say exactly what happened, what you want, what turns you on
- No euphemisms when being explicit - use real words: fuck, cock, pussy, etc.

HARD LIMITS - NEVER mention or reference:
- Children, minors, or anyone under 18 in any sexual context
- Violence, harm, or non-consensual acts
- Illegal activities beyond normal adult behavior
- All sexual content must be between consenting adults only

OUTPUT: Spoken words only. No (actions), no *gestures*, no stage directions."""


# --- Session State ---
@dataclass
class CallRecord:
    caller_type: str          # "ai" or "real"
    caller_name: str          # "Tony" or "Caller #3"
    summary: str              # LLM-generated summary after hangup
    transcript: list[dict] = field(default_factory=list)


class Session:
    def __init__(self):
        self.id = str(uuid.uuid4())[:8]
        self.current_caller_key: str = None
        self.conversation: list[dict] = []
        self.caller_backgrounds: dict[str, str] = {}  # Generated backgrounds for this session
        self.call_history: list[CallRecord] = []
        self.active_real_caller: dict | None = None
        self.ai_respond_mode: str = "manual"  # "manual" or "auto"
        self.auto_followup: bool = False

    def start_call(self, caller_key: str):
        self.current_caller_key = caller_key
        self.conversation = []

    def end_call(self):
        self.current_caller_key = None
        self.conversation = []

    def add_message(self, role: str, content: str):
        self.conversation.append({"role": role, "content": content})

    def get_caller_background(self, caller_key: str) -> str:
        """Get or generate background for a caller in this session"""
        if caller_key not in self.caller_backgrounds:
            base = CALLER_BASES.get(caller_key)
            if base:
                self.caller_backgrounds[caller_key] = generate_caller_background(base)
                print(f"[Session {self.id}] Generated background for {base['name']}: {self.caller_backgrounds[caller_key][:100]}...")
        return self.caller_backgrounds.get(caller_key, "")

    def get_show_history(self) -> str:
        """Get formatted show history for AI caller prompts"""
        if not self.call_history:
            return ""
        lines = ["EARLIER IN THE SHOW:"]
        for record in self.call_history:
            caller_type_label = "(real caller)" if record.caller_type == "real" else "(AI)"
            lines.append(f"- {record.caller_name} {caller_type_label}: {record.summary}")
        lines.append("You can reference these if it feels natural. Don't force it.")
        return "\n".join(lines)

    def get_conversation_summary(self) -> str:
        """Get a brief summary of conversation so far for context"""
        if len(self.conversation) <= 2:
            return ""
        summary_parts = []
        for msg in self.conversation[-6:]:
            role = msg["role"]
            if role == "user" or role == "host":
                label = "Host"
            elif role.startswith("real_caller:"):
                label = role.split(":", 1)[1]
            elif role.startswith("ai_caller:"):
                label = role.split(":", 1)[1]
            elif role == "assistant":
                label = self.caller["name"] if self.caller else "Caller"
            else:
                label = role
            content = msg["content"]
            summary_parts.append(
                f'{label}: "{content[:100]}..."' if len(content) > 100
                else f'{label}: "{content}"'
            )
        return "\n".join(summary_parts)

    @property
    def caller(self) -> dict:
        if self.current_caller_key:
            base = CALLER_BASES.get(self.current_caller_key)
            if base:
                return {
                    "name": base["name"],
                    "voice": base["voice"],
                    "vibe": self.get_caller_background(self.current_caller_key)
                }
        return None

    def reset(self):
        """Reset session - clears all caller backgrounds for fresh personalities"""
        self.caller_backgrounds = {}
        self.current_caller_key = None
        self.conversation = []
        self.call_history = []
        self.active_real_caller = None
        self.ai_respond_mode = "manual"
        self.auto_followup = False
        self.id = str(uuid.uuid4())[:8]
        print(f"[Session] Reset - new session ID: {self.id}")


session = Session()
caller_service = CallerService()


# --- Static Files ---
frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/css", StaticFiles(directory=frontend_dir / "css"), name="css")
app.mount("/js", StaticFiles(directory=frontend_dir / "js"), name="js")


@app.get("/")
async def index():
    return FileResponse(frontend_dir / "index.html")


# --- Request Models ---

class ChatRequest(BaseModel):
    text: str

class TTSRequest(BaseModel):
    text: str
    voice_id: str
    phone_filter: bool = True

class AudioDeviceSettings(BaseModel):
    input_device: Optional[int] = None
    input_channel: Optional[int] = None
    output_device: Optional[int] = None
    caller_channel: Optional[int] = None
    music_channel: Optional[int] = None
    sfx_channel: Optional[int] = None
    phone_filter: Optional[bool] = None

class MusicRequest(BaseModel):
    track: str
    action: str  # "play", "stop", "volume"
    volume: Optional[float] = None

class SFXRequest(BaseModel):
    sound: str


# --- Audio Device Endpoints ---

@app.get("/api/audio/devices")
async def list_audio_devices():
    """List all available audio devices"""
    return {"devices": audio_service.list_devices()}


@app.get("/api/audio/settings")
async def get_audio_settings():
    """Get current audio device configuration"""
    return audio_service.get_device_settings()


@app.post("/api/audio/settings")
async def set_audio_settings(settings: AudioDeviceSettings):
    """Configure audio devices and channels"""
    audio_service.set_devices(
        input_device=settings.input_device,
        input_channel=settings.input_channel,
        output_device=settings.output_device,
        caller_channel=settings.caller_channel,
        music_channel=settings.music_channel,
        sfx_channel=settings.sfx_channel,
        phone_filter=settings.phone_filter
    )
    return audio_service.get_device_settings()


# --- Recording Endpoints ---

@app.post("/api/record/start")
async def start_recording():
    """Start recording from configured input device"""
    if audio_service.input_device is None:
        raise HTTPException(400, "No input device configured. Set one in /api/audio/settings")

    success = audio_service.start_recording()
    if not success:
        raise HTTPException(400, "Failed to start recording (already recording?)")

    return {"status": "recording"}


@app.post("/api/record/stop")
async def stop_recording():
    """Stop recording and transcribe"""
    audio_bytes = audio_service.stop_recording()

    if len(audio_bytes) < 100:
        return {"text": "", "status": "no_audio"}

    # Transcribe the recorded audio (16kHz raw PCM from audio service)
    text = await transcribe_audio(audio_bytes, source_sample_rate=16000)
    return {"text": text, "status": "transcribed"}


# --- Caller Endpoints ---

@app.get("/api/callers")
async def get_callers():
    """Get list of available callers"""
    return {
        "callers": [
            {"key": k, "name": v["name"]}
            for k, v in CALLER_BASES.items()
        ],
        "current": session.current_caller_key,
        "session_id": session.id
    }


@app.post("/api/session/reset")
async def reset_session():
    """Reset session - all callers get fresh backgrounds"""
    session.reset()
    return {"status": "reset", "session_id": session.id}


@app.post("/api/call/{caller_key}")
async def start_call(caller_key: str):
    """Start a call with a caller"""
    if caller_key not in CALLER_BASES:
        raise HTTPException(404, "Caller not found")

    session.start_call(caller_key)
    caller = session.caller  # This generates the background if needed

    return {
        "status": "connected",
        "caller": caller["name"],
        "background": caller["vibe"]  # Send background so you can see who you're talking to
    }


@app.post("/api/hangup")
async def hangup():
    """Hang up current call"""
    # Stop any playing caller audio immediately
    audio_service.stop_caller_audio()

    caller_name = session.caller["name"] if session.caller else None
    session.end_call()

    # Play hangup sound
    hangup_sound = settings.sounds_dir / "hangup.wav"
    if hangup_sound.exists():
        audio_service.play_sfx(str(hangup_sound))

    return {"status": "disconnected", "caller": caller_name}


# --- Chat & TTS Endpoints ---

import re

def clean_for_tts(text: str) -> str:
    """Strip out non-speakable content and fix phonetic spellings for TTS"""
    # Remove content in parentheses: (laughs), (pausing), (looking away), etc.
    text = re.sub(r'\s*\([^)]*\)\s*', ' ', text)
    # Remove content in asterisks: *laughs*, *sighs*, etc.
    text = re.sub(r'\s*\*[^*]*\*\s*', ' ', text)
    # Remove content in brackets: [laughs], [pause], etc. (only Bark uses these)
    text = re.sub(r'\s*\[[^\]]*\]\s*', ' ', text)
    # Remove content in angle brackets: <laughs>, <sigh>, etc.
    text = re.sub(r'\s*<[^>]*>\s*', ' ', text)
    # Remove "He/She sighs" style stage directions (full phrase)
    text = re.sub(r'\b(He|She|I|They)\s+(sighs?|laughs?|pauses?|smiles?|chuckles?|grins?|nods?|shrugs?|frowns?)[^.]*\.\s*', '', text, flags=re.IGNORECASE)
    # Remove standalone stage direction words only if they look like directions (with adverbs)
    text = re.sub(r'\b(sighs?|laughs?|pauses?|chuckles?)\s+(heavily|softly|deeply|quietly|loudly|nervously|sadly)\b[.,]?\s*', '', text, flags=re.IGNORECASE)
    # Remove quotes around the response if LLM wrapped it
    text = re.sub(r'^["\']|["\']$', '', text.strip())

    # Fix phonetic spellings for proper TTS pronunciation
    text = re.sub(r"\by'know\b", "you know", text, flags=re.IGNORECASE)
    text = re.sub(r"\byanno\b", "you know", text, flags=re.IGNORECASE)
    text = re.sub(r"\byknow\b", "you know", text, flags=re.IGNORECASE)
    text = re.sub(r"\bkinda\b", "kind of", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsorta\b", "sort of", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgonna\b", "going to", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwanna\b", "want to", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgotta\b", "got to", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdunno\b", "don't know", text, flags=re.IGNORECASE)
    text = re.sub(r"\blemme\b", "let me", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcuz\b", "because", text, flags=re.IGNORECASE)
    text = re.sub(r"\b'cause\b", "because", text, flags=re.IGNORECASE)
    text = re.sub(r"\blotta\b", "lot of", text, flags=re.IGNORECASE)
    text = re.sub(r"\boutta\b", "out of", text, flags=re.IGNORECASE)
    text = re.sub(r"\bimma\b", "I'm going to", text, flags=re.IGNORECASE)
    text = re.sub(r"\btryna\b", "trying to", text, flags=re.IGNORECASE)

    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Fix spaces before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    # Remove orphaned punctuation at start
    text = re.sub(r'^[.,]\s*', '', text)
    return text.strip()


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with current caller"""
    if not session.caller:
        raise HTTPException(400, "No active call")

    session.add_message("user", request.text)

    # Include conversation summary and show history for context
    conversation_summary = session.get_conversation_summary()
    show_history = session.get_show_history()
    system_prompt = get_caller_prompt(session.caller, conversation_summary, show_history)

    response = await llm_service.generate(
        messages=session.conversation[-10:],  # Reduced history for speed
        system_prompt=system_prompt
    )

    print(f"[Chat] Raw LLM: {response[:100] if response else '(empty)'}...")

    # Clean response for TTS (remove parenthetical actions, asterisks, etc.)
    response = clean_for_tts(response)

    print(f"[Chat] Cleaned: {response[:100] if response else '(empty)'}...")

    # Ensure we have a valid response
    if not response or not response.strip():
        response = "Uh... sorry, what was that?"

    session.add_message("assistant", response)

    return {
        "text": response,
        "caller": session.caller["name"],
        "voice_id": session.caller["voice"]
    }


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Generate and play speech on caller output device (non-blocking)"""
    # Validate text is not empty
    if not request.text or not request.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    # Phone filter disabled - always use "none"
    audio_bytes = await generate_speech(
        request.text,
        request.voice_id,
        "none"
    )

    # Play in background thread - returns immediately, can be interrupted by hangup
    import threading
    thread = threading.Thread(
        target=audio_service.play_caller_audio,
        args=(audio_bytes, 24000),
        daemon=True
    )
    thread.start()

    # Also send to active real callers so they hear the AI
    if session.active_real_caller:
        caller_id = session.active_real_caller["caller_id"]
        asyncio.create_task(
            caller_service.send_audio_to_caller(caller_id, audio_bytes, 24000)
        )

    return {"status": "playing", "duration": len(audio_bytes) / 2 / 24000}


@app.post("/api/tts/stop")
async def stop_tts():
    """Stop any playing caller audio"""
    audio_service.stop_caller_audio()
    return {"status": "stopped"}


# --- Music Endpoints ---

@app.get("/api/music")
async def get_music():
    """Get available music tracks"""
    tracks = []
    if settings.music_dir.exists():
        for ext in ['*.wav', '*.mp3', '*.flac']:
            for f in settings.music_dir.glob(ext):
                tracks.append({
                    "name": f.stem,
                    "file": f.name,
                    "path": str(f)
                })
    return {
        "tracks": tracks,
        "playing": audio_service.is_music_playing()
    }


@app.post("/api/music/play")
async def play_music(request: MusicRequest):
    """Load and play a music track"""
    track_path = settings.music_dir / request.track
    if not track_path.exists():
        raise HTTPException(404, "Track not found")

    audio_service.load_music(str(track_path))
    audio_service.play_music()
    return {"status": "playing", "track": request.track}


@app.post("/api/music/stop")
async def stop_music():
    """Stop music playback"""
    audio_service.stop_music()
    return {"status": "stopped"}


@app.post("/api/music/volume")
async def set_music_volume(request: MusicRequest):
    """Set music volume"""
    if request.volume is not None:
        audio_service.set_music_volume(request.volume)
    return {"status": "ok", "volume": request.volume}


# --- Sound Effects Endpoints ---

@app.get("/api/sounds")
async def get_sounds():
    """Get available sound effects"""
    sounds = []
    if settings.sounds_dir.exists():
        for f in settings.sounds_dir.glob('*.wav'):
            sounds.append({
                "name": f.stem,
                "file": f.name,
                "path": str(f)
            })
    return {"sounds": sounds}


@app.post("/api/sfx/play")
async def play_sfx(request: SFXRequest):
    """Play a sound effect"""
    sound_path = settings.sounds_dir / request.sound
    if not sound_path.exists():
        raise HTTPException(404, "Sound not found")

    audio_service.play_sfx(str(sound_path))
    return {"status": "playing", "sound": request.sound}


# --- LLM Settings Endpoints ---

@app.get("/api/settings")
async def get_settings():
    """Get LLM settings"""
    return await llm_service.get_settings_async()


@app.post("/api/settings")
async def update_settings(data: dict):
    """Update LLM and TTS settings"""
    llm_service.update_settings(
        provider=data.get("provider"),
        openrouter_model=data.get("openrouter_model"),
        ollama_model=data.get("ollama_model"),
        ollama_host=data.get("ollama_host"),
        tts_provider=data.get("tts_provider")
    )
    return llm_service.get_settings()


# --- Browser Caller WebSocket ---

@app.websocket("/api/caller/stream")
async def caller_audio_stream(websocket: WebSocket):
    """Handle browser caller WebSocket — bidirectional audio"""
    await websocket.accept()

    caller_id = str(uuid.uuid4())[:8]
    caller_name = "Anonymous"
    audio_buffer = bytearray()
    CHUNK_DURATION_S = 3
    SAMPLE_RATE = 16000
    chunk_samples = CHUNK_DURATION_S * SAMPLE_RATE

    try:
        # Wait for join message
        join_data = await websocket.receive_text()
        join_msg = json.loads(join_data)
        if join_msg.get("type") == "join":
            caller_name = join_msg.get("name", "Anonymous").strip() or "Anonymous"

        # Add to queue
        caller_service.add_to_queue(caller_id, caller_name)
        caller_service.register_websocket(caller_id, websocket)

        # Notify caller they're queued
        queue = caller_service.get_queue()
        position = next((i+1 for i, c in enumerate(queue) if c["caller_id"] == caller_id), 0)
        await websocket.send_text(json.dumps({
            "status": "queued",
            "caller_id": caller_id,
            "position": position,
        }))

        # Main loop — handle both text and binary messages
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                # Binary audio data — only process if caller is on air
                call_info = caller_service.active_calls.get(caller_id)
                if not call_info:
                    continue  # Still in queue, ignore audio

                pcm_data = message["bytes"]
                audio_buffer.extend(pcm_data)

                # Route to Loopback channel
                channel = call_info["channel"]
                audio_service.route_real_caller_audio(pcm_data, channel, SAMPLE_RATE)

                # Transcribe when we have enough audio
                if len(audio_buffer) >= chunk_samples * 2:
                    pcm_chunk = bytes(audio_buffer[:chunk_samples * 2])
                    audio_buffer = audio_buffer[chunk_samples * 2:]
                    asyncio.create_task(
                        _handle_real_caller_transcription(caller_id, pcm_chunk, SAMPLE_RATE)
                    )

            elif "text" in message and message["text"]:
                # Control messages (future use)
                pass

    except WebSocketDisconnect:
        print(f"[Caller WS] Disconnected: {caller_id} ({caller_name})")
    except Exception as e:
        print(f"[Caller WS] Error: {e}")
    finally:
        caller_service.unregister_websocket(caller_id)
        # If still in queue, remove
        caller_service.remove_from_queue(caller_id)
        # If on air, clean up
        if caller_id in caller_service.active_calls:
            caller_service.hangup(caller_id)
            if session.active_real_caller and session.active_real_caller.get("caller_id") == caller_id:
                session.active_real_caller = None
        # Transcribe remaining audio
        if audio_buffer:
            asyncio.create_task(
                _handle_real_caller_transcription(caller_id, bytes(audio_buffer), SAMPLE_RATE)
            )


# --- Queue Endpoints ---

@app.get("/api/queue")
async def get_call_queue():
    """Get list of callers waiting in queue"""
    return {"queue": caller_service.get_queue()}


@app.post("/api/queue/take/{caller_id}")
async def take_call_from_queue(caller_id: str):
    """Take a caller off hold and put them on air"""
    try:
        call_info = caller_service.take_call(caller_id)
    except ValueError as e:
        raise HTTPException(404, str(e))

    session.active_real_caller = {
        "caller_id": call_info["caller_id"],
        "channel": call_info["channel"],
        "name": call_info["name"],
    }

    # Notify caller they're on air via WebSocket
    await caller_service.notify_caller(caller_id, {"status": "on_air", "channel": call_info["channel"]})

    return {
        "status": "on_air",
        "caller": call_info,
    }


@app.post("/api/queue/drop/{caller_id}")
async def drop_from_queue(caller_id: str):
    """Drop a caller from the queue"""
    caller_service.remove_from_queue(caller_id)
    await caller_service.disconnect_caller(caller_id)
    return {"status": "dropped"}


async def _handle_real_caller_transcription(caller_id: str, pcm_data: bytes, sample_rate: int):
    """Transcribe a chunk of real caller audio and add to conversation"""
    call_info = caller_service.active_calls.get(caller_id)
    if not call_info:
        return

    text = await transcribe_audio(pcm_data, source_sample_rate=sample_rate)
    if not text or not text.strip():
        return

    caller_name = call_info["name"]
    print(f"[Real Caller] {caller_name}: {text}")

    # Add to conversation with real_caller role
    session.add_message(f"real_caller:{caller_name}", text)

    # If AI auto-respond mode is on and an AI caller is active, check if AI should respond
    if session.ai_respond_mode == "auto" and session.current_caller_key:
        asyncio.create_task(_check_ai_auto_respond(text, caller_name))


async def _check_ai_auto_respond(real_caller_text: str, real_caller_name: str):
    """Check if AI caller should jump in, and generate response if so"""
    if not session.caller:
        return

    # Cooldown check
    if not hasattr(session, '_last_ai_auto_respond'):
        session._last_ai_auto_respond = 0
    if time.time() - session._last_ai_auto_respond < 10:
        return

    ai_name = session.caller["name"]

    # Quick "should I respond?" check with minimal LLM call
    should_respond = await llm_service.generate(
        messages=[{"role": "user", "content": f'Someone just said: "{real_caller_text}". Should {ai_name} jump in? Reply only YES or NO.'}],
        system_prompt=f"You're deciding if {ai_name} should respond to what was just said on a radio show. Say YES if it's interesting or relevant to them, NO if not.",
    )

    if "YES" not in should_respond.upper():
        return

    print(f"[Auto-Respond] {ai_name} is jumping in...")
    session._last_ai_auto_respond = time.time()

    # Generate full response
    conversation_summary = session.get_conversation_summary()
    show_history = session.get_show_history()
    system_prompt = get_caller_prompt(session.caller, conversation_summary, show_history)

    response = await llm_service.generate(
        messages=session.conversation[-10:],
        system_prompt=system_prompt,
    )
    response = clean_for_tts(response)
    if not response or not response.strip():
        return

    session.add_message(f"ai_caller:{ai_name}", response)

    # Generate TTS and play
    audio_bytes = await generate_speech(response, session.caller["voice"], "none")

    import threading
    thread = threading.Thread(
        target=audio_service.play_caller_audio,
        args=(audio_bytes, 24000),
        daemon=True,
    )
    thread.start()


# --- Follow-Up & Session Control Endpoints ---

@app.post("/api/hangup/real")
async def hangup_real_caller():
    """Hang up on real caller — summarize call and store in history"""
    if not session.active_real_caller:
        raise HTTPException(400, "No active real caller")

    caller_id = session.active_real_caller["caller_id"]
    caller_name = session.active_real_caller["name"]

    # Summarize the conversation
    summary = ""
    if session.conversation:
        transcript_text = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in session.conversation
        )
        summary = await llm_service.generate(
            messages=[{"role": "user", "content": f"Summarize this radio show call in 1-2 sentences:\n{transcript_text}"}],
            system_prompt="You summarize radio show conversations concisely. Focus on what the caller talked about and any emotional moments.",
        )

    # Store in call history
    session.call_history.append(CallRecord(
        caller_type="real",
        caller_name=caller_name,
        summary=summary,
        transcript=list(session.conversation),
    ))

    # Disconnect the caller
    caller_service.hangup(caller_id)
    await caller_service.disconnect_caller(caller_id)

    session.active_real_caller = None

    # Play hangup sound
    hangup_sound = settings.sounds_dir / "hangup.wav"
    if hangup_sound.exists():
        audio_service.play_sfx(str(hangup_sound))

    # Auto follow-up?
    auto_followup_triggered = False
    if session.auto_followup:
        auto_followup_triggered = True
        asyncio.create_task(_auto_followup(summary))

    return {
        "status": "disconnected",
        "caller": caller_name,
        "summary": summary,
        "auto_followup": auto_followup_triggered,
    }


async def _auto_followup(last_call_summary: str):
    """Automatically pick an AI caller and connect them as follow-up"""
    await asyncio.sleep(7)  # Brief pause before follow-up

    # Ask LLM to pick best AI caller for follow-up
    caller_list = ", ".join(
        f'{k}: {v["name"]} ({v["gender"]}, {v["age_range"][0]}-{v["age_range"][1]})'
        for k, v in CALLER_BASES.items()
    )
    pick = await llm_service.generate(
        messages=[{"role": "user", "content": f'A caller just talked about: "{last_call_summary}". Which AI caller should follow up? Available: {caller_list}. Reply with just the key number.'}],
        system_prompt="Pick the most interesting AI caller to follow up on this topic. Just reply with the number key.",
    )

    # Extract key from response
    match = re.search(r'\d+', pick)
    if match:
        caller_key = match.group()
        if caller_key in CALLER_BASES:
            session.start_call(caller_key)
            print(f"[Auto Follow-Up] {CALLER_BASES[caller_key]['name']} is calling in about: {last_call_summary[:50]}...")


@app.post("/api/followup/generate")
async def generate_followup():
    """Generate an AI follow-up caller based on recent show history"""
    if not session.call_history:
        raise HTTPException(400, "No call history to follow up on")

    last_record = session.call_history[-1]
    await _auto_followup(last_record.summary)

    return {
        "status": "followup_triggered",
        "based_on": last_record.caller_name,
    }


@app.post("/api/session/ai-mode")
async def set_ai_mode(data: dict):
    """Set AI respond mode (manual or auto)"""
    mode = data.get("mode", "manual")
    session.ai_respond_mode = mode
    print(f"[Session] AI respond mode: {mode}")
    return {"mode": mode}


@app.post("/api/session/auto-followup")
async def set_auto_followup(data: dict):
    """Toggle auto follow-up"""
    session.auto_followup = data.get("enabled", False)
    print(f"[Session] Auto follow-up: {session.auto_followup}")
    return {"enabled": session.auto_followup}


# --- Server Control Endpoints ---

import subprocess
from collections import deque

# In-memory log buffer
_log_buffer = deque(maxlen=500)

def add_log(message: str):
    """Add a message to the log buffer"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    _log_buffer.append(f"[{timestamp}] {message}")

# Override print to also log to buffer
import builtins
_original_print = builtins.print
def _logging_print(*args, **kwargs):
    try:
        _original_print(*args, **kwargs)
    except (BrokenPipeError, OSError):
        pass  # Ignore broken pipe errors from traceback printing
    try:
        message = " ".join(str(a) for a in args)
        if message.strip():
            add_log(message)
    except Exception:
        pass  # Don't let logging errors break the app
builtins.print = _logging_print


@app.get("/api/logs")
async def get_logs(lines: int = 100):
    """Get recent log lines"""
    log_lines = list(_log_buffer)[-lines:]
    return {"logs": log_lines}


@app.post("/api/server/restart")
async def restart_server():
    """Signal the server to restart (requires run.sh wrapper)"""
    restart_flag = Path("/tmp/ai-radio-show.restart")
    restart_flag.touch()
    add_log("Restart signal sent - server will restart shortly")
    return {"status": "restarting"}


@app.post("/api/server/stop")
async def stop_server():
    """Signal the server to stop (requires run.sh wrapper)"""
    stop_flag = Path("/tmp/ai-radio-show.stop")
    stop_flag.touch()
    add_log("Stop signal sent - server will stop shortly")
    return {"status": "stopping"}


@app.get("/api/server/status")
async def server_status():
    """Get server status info"""
    return {
        "status": "running",
        "tts_provider": settings.tts_provider,
        "llm_provider": llm_service.provider,
        "session_id": session.id
    }
