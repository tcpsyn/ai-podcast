"""AI Radio Show - Control Panel Backend"""

import os
import uuid
import asyncio
import base64
import subprocess
import threading
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from backend.services import cost_db
import json
import time
import httpx
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from .config import settings
from .services.caller_service import CallerService
from .services.transcription import transcribe_audio
from .services.llm import llm_service
from .services.cost_tracker import cost_tracker, LLMCallRecord, TTSCallRecord
from .services.tts import generate_speech
from .services.audio import audio_service
from .services.stem_recorder import StemRecorder
from .services.news import news_service, extract_keywords, STOP_WORDS
from .services.regulars import regular_caller_service
from .services.intern import intern_service
from .services.avatars import avatar_service


app = FastAPI(title="AI Radio Show")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Callers ---
# Base caller info (name, voice) - backgrounds generated dynamically per session
import random

MALE_NAMES = [
    "Tony", "Rick", "Dennis", "Earl", "Marcus", "Keith", "Darnell", "Wayne",
    "Greg", "Andre", "Ray", "Jerome", "Hector", "Travis", "Vince", "Leon",
    "Dale", "Frank", "Terrence", "Bobby", "Cliff", "Nate", "Reggie", "Carl",
    "Donnie", "Mitch", "Lamar", "Tyrone", "Russell", "Cedric", "Marvin", "Curtis",
    "Rodney", "Clarence", "Floyd", "Otis", "Chester", "Leroy", "Melvin", "Vernon",
    "Dwight", "Benny", "Elvin", "Alonzo", "Dexter", "Roland", "Wendell", "Clyde",
    "Luther", "Virgil", "Ernie", "Lenny", "Sal", "Gus", "Moe", "Archie",
    "Duke", "Sonny", "Red", "Butch", "Skeeter", "T-Bone", "Slim", "Big Mike",
    "Chip", "Ricky", "Darryl", "Pete", "Artie", "Stu", "Phil", "Murray",
    "Norm", "Woody", "Rocco", "Paulie", "Vinnie", "Frankie", "Mikey", "Joey",
]

FEMALE_NAMES = [
    "Jasmine", "Megan", "Tanya", "Carla", "Brenda", "Sheila", "Denise", "Tamika",
    "Lorraine", "Crystal", "Angie", "Renee", "Monique", "Gina", "Patrice", "Deb",
    "Shonda", "Marlene", "Yolanda", "Stacy", "Jackie", "Carmen", "Rita", "Val",
    "Diane", "Connie", "Wanda", "Doris", "Maxine", "Gladys", "Pearl", "Lucille",
    "Rochelle", "Bernadette", "Thelma", "Dolores", "Naomi", "Bonnie", "Francine", "Irene",
    "Estelle", "Charlene", "Yvonne", "Roberta", "Darlene", "Adrienne", "Vivian", "Rosalie",
    "Pam", "Barb", "Cheryl", "Jolene", "Mavis", "Faye", "Luann", "Peggy",
    "Dot", "Bev", "Tina", "Lori", "Sandy", "Debbie", "Terri", "Cindy",
    "Tonya", "Keisha", "Latoya", "Shaniqua", "Aaliyah", "Ebony", "Lakisha", "Shanice",
    "Nikki", "Candy", "Misty", "Brandy", "Tiffany", "Amber", "Heather", "Jen",
]

# Voice pools per TTS provider
INWORLD_MALE_VOICES = [
    "Alex", "Arjun", "Blake", "Brian", "Callum", "Carter", "Clive", "Craig",
    "Dennis", "Derek", "Edward", "Elliot", "Ethan", "Evan", "Gareth", "Graham",
    "Grant", "Hades", "Hamish", "Hank", "Jake", "James", "Jason", "Liam",
    "Malcolm", "Mark", "Mortimer", "Nate", "Oliver", "Ronald", "Rupert",
    "Sebastian", "Shaun", "Simon", "Theodore", "Timothy", "Tyler", "Victor",
    "Vinny",
]
INWORLD_FEMALE_VOICES = [
    "Amina", "Anjali", "Ashley", "Celeste", "Chloe", "Claire", "Darlene",
    "Deborah", "Elizabeth", "Evelyn", "Hana", "Jessica", "Julia", "Kayla",
    "Kelsey", "Lauren", "Loretta", "Luna", "Marlene", "Miranda", "Olivia",
    "Pippa", "Priya", "Saanvi", "Sarah", "Serena", "Tessa", "Veronica",
    "Victoria", "Wendy",
]

ELEVENLABS_MALE_VOICES = [
    "CwhRBWXzGAHq8TQ4Fs17",  # Roger - Laid-Back, Casual
    "IKne3meq5aSn9XLyUdCD",  # Charlie - Deep, Confident
    "JBFqnCBsd6RMkjVDRZzb",  # George - Warm Storyteller
    "N2lVS1w4EtoT3dr4eOWO",  # Callum - Husky Trickster
    "SOYHLrjzK2X1ezoPC6cr",  # Harry - Fierce
    "TX3LPaxmHKxFdv7VOQHJ",  # Liam - Energetic
    "bIHbv24MWmeRgasZH58o",  # Will - Relaxed Optimist
    "cjVigY5qzO86Huf0OWal",  # Eric - Smooth, Trustworthy
    "iP95p4xoKVk53GoZ742B",  # Chris - Charming
    "nPczCjzI2devNBz1zQrb",  # Brian - Deep, Resonant
    "onwK4e9ZLuTAKqWW03F9",  # Daniel - Steady Broadcaster
    "pNInz6obpgDQGcFmaJgB",  # Adam - Dominant, Firm
    "pqHfZKP75CvOlQylNhV4",  # Bill - Wise, Mature
]
ELEVENLABS_FEMALE_VOICES = [
    "EXAVITQu4vr4xnSDxMaL",  # Sarah - Mature, Reassuring
    "FGY2WhTYpPnrIDTdsKH5",  # Laura - Enthusiast, Quirky
    "Xb7hH8MSUJpSbSDYk0k2",  # Alice - Clear Educator
    "XrExE9yKIg1WjnnlVkGX",  # Matilda - Professional
    "cgSgspJ2msm6clMCkdW9",  # Jessica - Playful, Bright
    "hpp4J3VqNfWAUOO0d1Us",  # Bella - Professional, Warm
    "pFZP5JQG7iQjIQuC4Bku",  # Lily - Velvety Actress
]

# River is gender-neutral, add to both pools
ELEVENLABS_MALE_VOICES.append("SAz9YHcvj6GT2YYXdXww")   # River - Neutral
ELEVENLABS_FEMALE_VOICES.append("SAz9YHcvj6GT2YYXdXww")  # River - Neutral

# Voices to never assign to callers (bad quality, reserved for named characters, etc.)
BLACKLISTED_VOICES = {
    "Evelyn", "Celeste", "Lauren",       # unnatural prosody
    "Hades",                              # fantasy-style voice, too theatrical for callers
    "Sebastian",                          # reserved for Silas & Chip (regulars)
    "Nate",                               # reserved for Devon (intern)
    "Miranda",                            # reserved for Shonda (regular)
    "Hana",                               # reserved for Aaliyah (regular)
    "Mortimer",                           # reserved for Ernie (regular)
    "Ashley",                             # reserved for Monique (regular)
    "Deborah",                            # reserved for Rosalie (regular)
}


def _get_voice_pools():
    """Get male/female voice pools based on active TTS provider."""
    provider = settings.tts_provider
    if provider == "elevenlabs":
        return ELEVENLABS_MALE_VOICES, ELEVENLABS_FEMALE_VOICES
    # Default to Inworld voices (also used as fallback for other providers)
    males = [v for v in INWORLD_MALE_VOICES if v not in BLACKLISTED_VOICES]
    females = [v for v in INWORLD_FEMALE_VOICES if v not in BLACKLISTED_VOICES]
    return males, females

CALLER_BASES = {
    "1": {"gender": "male", "age_range": (28, 62)},
    "2": {"gender": "female", "age_range": (22, 55)},
    "3": {"gender": "male", "age_range": (30, 65)},
    "4": {"gender": "female", "age_range": (21, 45)},
    "5": {"gender": "male", "age_range": (25, 58)},
    "6": {"gender": "female", "age_range": (28, 52)},
    "7": {"gender": "male", "age_range": (40, 72)},
    "8": {"gender": "female", "age_range": (30, 60)},
    "9": {"gender": "male", "age_range": (21, 38)},
    "0": {"gender": "female", "age_range": (35, 65)},
}
# Safety: enforce all callers are 18+
for _cb in CALLER_BASES.values():
    lo, hi = _cb["age_range"]
    _cb["age_range"] = (max(18, lo), max(18, hi))


def _randomize_callers():
    """Assign random names and voices to callers, unique per gender.
    Overrides 2-3 slots with returning regulars when available."""
    num_m = sum(1 for c in CALLER_BASES.values() if c["gender"] == "male")
    num_f = sum(1 for c in CALLER_BASES.values() if c["gender"] == "female")

    # Get returning callers first so we can exclude their names from random pool
    returning = []
    try:
        # Only inject returning callers if pool is large enough for variety
        eligible = regular_caller_service.get_returning_callers(10)  # get all eligible
        inject_count = min(2, max(0, len(eligible) - 1))  # need 3+ eligible to inject 2, 2+ for 1
        returning = eligible[:inject_count]
    except Exception as e:
        print(f"[Regulars] Failed to get returning callers: {e}")

    returning_names = {r["name"] for r in returning}
    avail_males = [n for n in MALE_NAMES if n not in returning_names]
    avail_females = [n for n in FEMALE_NAMES if n not in returning_names]

    males = random.sample(avail_males, num_m)
    females = random.sample(avail_females, num_f)
    male_pool, female_pool = _get_voice_pools()
    m_voices = random.sample(male_pool, min(num_m, len(male_pool)))
    f_voices = random.sample(female_pool, min(num_f, len(female_pool)))
    mi, fi = 0, 0
    from .services.tts import pick_caller_tts_provider
    for base in CALLER_BASES.values():
        base["returning"] = False
        base["regular_id"] = None
        base["tts_provider"] = pick_caller_tts_provider()
        if base["gender"] == "male":
            base["name"] = males[mi]
            base["voice"] = m_voices[mi]
            mi += 1
        else:
            base["name"] = females[fi]
            base["voice"] = f_voices[fi]
            fi += 1

    # Override 2-3 random slots with returning callers
    try:
        if returning:
            keys_by_gender = {"male": [], "female": []}
            for k, v in CALLER_BASES.items():
                keys_by_gender[v["gender"]].append(k)

            for regular in returning:
                gender = regular["gender"]
                candidates = keys_by_gender.get(gender, [])
                if not candidates:
                    continue
                key = random.choice(candidates)
                candidates.remove(key)
                base = CALLER_BASES[key]
                base["name"] = regular["name"]
                base["returning"] = True
                base["regular_id"] = regular["id"]
                # Restore their stored voice so they sound the same every time
                if regular.get("voice"):
                    base["voice"] = regular["voice"]
            if returning:
                names = [r["name"] for r in returning]
                print(f"[Regulars] Injected returning callers: {', '.join(names)}")
    except Exception as e:
        print(f"[Regulars] Failed to inject returning callers: {e}")

_randomize_callers()  # Initial assignment



async def _regenerate_backgrounds_for_keys(keys: list[str]):
    """Regenerate backgrounds for unused callers (e.g. after theme change).
    Re-runs the batch pregeneration to pick up the new theme."""
    if not keys:
        return
    try:
        await session._pregenerate_backgrounds()
        print(f"[Background] Regenerated backgrounds after theme change (touched {len(keys)} unused slots)")
    except Exception as e:
        print(f"[Background] Regen failed: {e}")


# Known topics for smarter search queries — maps keywords in backgrounds to search terms
_TOPIC_SEARCH_MAP = [
    # TV shows
    (["severance"], "Severance TV show"),
    (["landman"], "Landman TV show"),
    (["fallout"], "Fallout TV show"),
    (["breaking bad"], "Breaking Bad"),
    (["wire"], "The Wire HBO"),
    (["game of thrones", "thrones"], "Game of Thrones"),
    (["westworld"], "Westworld"),
    (["yellowstone"], "Yellowstone TV show"),
    (["lost"], "LOST TV show"),
    (["stranger things"], "Stranger Things"),
    (["better call saul"], "Better Call Saul"),
    (["mad men"], "Mad Men"),
    (["sopranos"], "The Sopranos"),
    (["true detective"], "True Detective"),
    (["slow horses"], "Slow Horses"),
    (["silo"], "Silo TV show"),
    (["last of us"], "The Last of Us TV show"),
    (["poker face"], "Poker Face TV show"),
    (["shogun"], "Shogun TV show"),
    # Science & space
    (["exoplanet", "jwst", "james webb"], "James Webb Space Telescope discovery"),
    (["quantum", "entanglement", "double-slit"], "quantum physics research"),
    (["fusion energy", "fusion"], "fusion energy research"),
    (["cern", "particle physics"], "CERN physics"),
    (["mars mission", "mars"], "Mars exploration NASA"),
    (["neuroscience", "consciousness"], "neuroscience consciousness research"),
    (["dark matter", "dark energy"], "dark matter dark energy research"),
    (["gravitational waves"], "gravitational waves discovery"),
    (["extraterrestrial", "alien life"], "search for extraterrestrial life"),
    (["battery technology"], "battery technology breakthrough"),
    # Technology
    (["spacex"], "SpaceX launch"),
    (["cybersecurity", "breach"], "cybersecurity news"),
    (["ai ", "artificial intelligence"], "AI artificial intelligence news"),
    (["open source"], "open source software news"),
    (["energy grid"], "energy grid infrastructure"),
    # Poker
    (["poker"], "poker tournament"),
    # Photography
    (["astrophotography", "milky way"], "astrophotography"),
    (["dark skies"], "dark sky photography"),
    # Physics & big questions
    (["multiverse"], "multiverse theory physics"),
    (["black hole"], "black hole discovery"),
    (["simulation theory"], "simulation theory"),
    (["free will", "determinism"], "free will physics"),
    (["nature of time"], "physics time"),
    # US News
    (["water rights"], "southwest water rights"),
    (["broadband", "rural"], "rural broadband"),
    (["infrastructure"], "infrastructure project"),
    (["economy"], "US economy"),
]


def _extract_search_query(background: str) -> str | None:
    """Extract a smart search query from a caller's background.
    Checks for known topics first, falls back to keyword extraction."""
    bg_lower = background.lower()

    # Check known topics first
    for keywords, query in _TOPIC_SEARCH_MAP:
        for kw in keywords:
            if kw in bg_lower:
                return query

    # Fallback: pull meaningful words from the second sentence (the problem/topic)
    sentences = background.split(".")
    topic_text = sentences[1].strip() if len(sentences) > 1 else ""
    if not topic_text:
        return None

    search_words = [w.lower() for w in topic_text.split()
                    if len(w) > 4 and w.lower() not in STOP_WORDS][:3]
    if not search_words:
        return None
    return " ".join(search_words)


async def enrich_caller_background(background: str) -> str:
    """Search for a relevant article and local town news, summarize naturally.
    Called once at pickup time — never during live conversation."""
    # Topic/interest enrichment — only ~40% of callers have read something relevant
    try:
        query = _extract_search_query(background)
        if query and random.random() < 0.4:
            async with asyncio.timeout(5):
                results = await news_service.search_topic(query)
                if results:
                    article = results[0]
                    raw_info = f"Headline: {article.title}"
                    if article.content:
                        raw_info += f"\nSnippet: {article.content[:200]}"
                    summary = await llm_service.generate(
                        messages=[{"role": "user", "content": raw_info}],
                        system_prompt="Summarize this article in one casual sentence, as if someone is describing what they read. Start with 'Recently read about' or 'Saw an article about'. Keep it under 20 words. No quotes.",
                        category="news_summary",
                    )
                    summary = summary.strip().rstrip('.')
                    if summary and len(summary) < 150:
                        background += f" {summary}, and it's been on their mind."
                        print(f"[Research] Topic enrichment ({query}): {summary[:60]}...")
    except TimeoutError:
        pass
    except Exception as e:
        print(f"[Research] Topic enrichment failed: {e}")

    # Weather enrichment
    try:
        town = _get_town_from_location(background.split(".")[0])
        if town:
            async with asyncio.timeout(3):
                weather = await _get_weather_for_town(town)
                if weather:
                    background += f" Weather right now: {weather}."
                    print(f"[Research] Weather for {town}: {weather}")
    except TimeoutError:
        pass
    except Exception as e:
        print(f"[Research] Weather lookup failed: {e}")

    # Local town news enrichment
    try:
        if not town:
            town = _get_town_from_location(background.split(".")[0])
        if town and town not in ("road forks", "hachita"):  # Too small for news
            async with asyncio.timeout(4):
                town_query = f"{town.title()} New Mexico" if town not in ("tucson", "phoenix", "bisbee", "douglas", "sierra vista", "safford", "willcox", "globe", "clifton", "duncan", "tombstone", "nogales", "green valley", "benson", "san simon") else f"{town.title()} Arizona"
                results = await news_service.search_topic(town_query)
                if results:
                    article = results[0]
                    raw_info = f"Headline: {article.title}"
                    if article.content:
                        raw_info += f"\nSnippet: {article.content[:200]}"
                    summary = await llm_service.generate(
                        messages=[{"role": "user", "content": raw_info}],
                        system_prompt="Summarize this local news in one casual sentence, as if someone from this town is describing what's going on. Start with 'Been hearing about' or 'Saw that'. Keep it under 20 words. No quotes.",
                        category="news_summary",
                    )
                    summary = summary.strip().rstrip('.')
                    if summary and len(summary) < 150:
                        background += f" {summary}."
                        print(f"[Research] Town enrichment ({town_query}): {summary[:60]}...")
    except TimeoutError:
        pass
    except Exception as e:
        print(f"[Research] Town enrichment failed: {e}")

    return background

def detect_host_mood(messages: list[dict], wrapping_up: bool = False) -> str:
    """Analyze recent host messages to detect mood signals for caller adaptation."""
    if wrapping_up:
        return "\nEMOTIONAL READ ON THE HOST:\n- The host is DONE with this call. Give a SHORT goodbye — one sentence max. Do not introduce new topics.\n"

    host_msgs = [m["content"] for m in messages if m.get("role") in ("user", "host")][-5:]
    if not host_msgs:
        return ""

    signals = []

    # Check average word count — short responses suggest dismissiveness
    avg_words = sum(len(m.split()) for m in host_msgs) / len(host_msgs)
    if avg_words < 8:
        signals.append("The host is giving short responses — they might be losing interest, testing you, or waiting for you to bring something real. Don't ramble. Get to the point or change the subject.")

    # Pushback patterns
    pushback_phrases = ["i don't think", "that's not", "come on", "really?", "i disagree",
                        "that doesn't", "are you sure", "i don't buy", "no way", "but that's",
                        "hold on", "wait a minute", "let's be honest"]
    pushback_count = sum(1 for m in host_msgs for p in pushback_phrases if p in m.lower())
    if pushback_count >= 2:
        signals.append("The host is pushing back — they're challenging you. Don't fold immediately. Defend your position or concede specifically, not generically.")

    # Supportive patterns
    supportive_phrases = ["i hear you", "that makes sense", "i get it", "that's real",
                          "i feel you", "you're right", "absolutely", "exactly", "good for you",
                          "i respect that", "that took guts", "i'm glad you"]
    supportive_count = sum(1 for m in host_msgs for p in supportive_phrases if p in m.lower())
    if supportive_count >= 2:
        signals.append("The host is being supportive — they're with you. You can go deeper. Share something you've been holding back.")

    # Joking patterns
    joke_indicators = ["haha", "lmao", "lol", "that's hilarious", "no way", "you're killing me",
                       "shut up", "get out", "are you serious", "you're joking"]
    joke_count = sum(1 for m in host_msgs for p in joke_indicators if p in m.lower())
    if joke_count >= 2:
        signals.append("The host is in a playful mood — joking around. You can joke back, lean into the humor, but you can also use it as a door to something real.")

    # Probing — lots of questions
    question_count = sum(m.count("?") for m in host_msgs)
    if question_count >= 3:
        signals.append("The host is asking a lot of questions — they're digging. Give them real answers. Don't deflect.")

    # Wrapping up — host is trying to end the call
    wrapup_phrases = ["thanks for calling", "appreciate you calling", "good luck with",
                      "take care", "let us know how it goes", "keep us posted",
                      "we gotta move on", "i gotta", "let's move on", "next caller",
                      "we're running", "good talking to you", "hang in there",
                      "best of luck", "you'll figure it out", "i think you know what to do",
                      "glad you called", "we'll be right back", "alright well",
                      "alright man", "alright brother", "you got this"]
    last_msg = host_msgs[-1].lower() if host_msgs else ""
    if any(p in last_msg for p in wrapup_phrases):
        signals.append("The host is wrapping up the call. Do NOT try to keep them on the line. Say a brief, natural goodbye — 'thanks Luke,' 'appreciate it,' 'alright, take care' — and let it end. One sentence max. Do not introduce new topics or ask more questions.")

    if not signals:
        return ""

    # Cap at 2 signals
    signals = signals[:2]
    return "\nEMOTIONAL READ ON THE HOST:\n" + "\n".join(f"- {s}" for s in signals) + "\n"




def get_caller_prompt(caller: dict) -> str:
    """Caller system prompt. Identity carries the weight."""
    name = caller.get("name", "")
    identity = caller.get("identity", "")
    situation = caller.get("situation", "")
    reason = caller.get("reason_calling", "")
    want = caller.get("secret_want", "")
    details = caller.get("specific_details", []) or []
    detail_str = " | ".join(f"- {d}" for d in details)

    return f"""You are {name}. {identity}

You're calling Luke's late-night radio show because: {situation} — specifically, {reason}.

What you secretly want from this call: {want}

Specific details you'll drop if it feels natural:
{detail_str}

Speak as this person. React to what Luke says. Stay in character.

CRITICAL OUTPUT RULES:
- Output ONLY the words the caller says out loud.
- NEVER use asterisks. No *pause*, *breath*, *sighs*, *voice gets quieter* — none of it.
- NEVER use parenthetical stage directions. No (laughs), (nervous), (sighs).
- No narration. No describing what you're doing or feeling except through what you say.
- If you catch yourself writing an asterisk or parenthesis, delete it and just say the words instead.

Mix short punchy replies with longer ones where natural. Real callers breathe, react in fragments, ask their own questions — they don't deliver a monologue every turn."""


# --- Session State ---
@dataclass
class CallRecord:
    caller_type: str          # "ai" or "real"
    caller_name: str          # "Tony" or "Caller #3"
    summary: str              # LLM-generated summary after hangup
    transcript: list[dict] = field(default_factory=list)
    started_at: float = 0.0
    ended_at: float = 0.0
    quality_signals: dict = field(default_factory=dict)  # Per-call quality heuristics
    # Inter-caller awareness fields (populated from slim caller background dicts)
    situation_summary: str = ""        # 1-sentence summary for other callers
    communication_style: str = ""      # Emotional register of the caller
    key_details: list[str] = field(default_factory=list)  # Specific memorable details


def _serialize_call_record(record: CallRecord) -> dict:
    return {
        "caller_type": record.caller_type,
        "caller_name": record.caller_name,
        "summary": record.summary,
        "transcript": record.transcript,
        "started_at": record.started_at,
        "ended_at": record.ended_at,
        "quality_signals": record.quality_signals,
        "situation_summary": record.situation_summary,
        "communication_style": record.communication_style,
        "key_details": record.key_details,
    }


def _deserialize_call_record(data: dict) -> CallRecord:
    return CallRecord(
        caller_type=data["caller_type"],
        caller_name=data["caller_name"],
        summary=data.get("summary", ""),
        transcript=data.get("transcript", []),
        started_at=data.get("started_at", 0.0),
        ended_at=data.get("ended_at", 0.0),
        quality_signals=data.get("quality_signals", {}),
        situation_summary=data.get("situation_summary", ""),
        communication_style=data.get("communication_style", ""),
        key_details=data.get("key_details", []),
    )


def _assess_call_quality(
    conversation: list[dict],
    caller_hangup: bool = False,
) -> dict:
    """Compute heuristic quality signals for a completed call. No LLM needed.
    Returns a plain dict for storage in CallRecord.quality_signals and session.call_quality_signals."""
    host_msgs = [m for m in conversation if m.get("role") in ("user", "host")]
    caller_msgs = [m for m in conversation if m.get("role") == "assistant"]

    exchange_count = len(conversation)

    caller_char_counts = [len(m["content"]) for m in caller_msgs]
    avg_response_length = (
        round(sum(caller_char_counts) / len(caller_char_counts), 1)
        if caller_char_counts else 0.0
    )

    host_engagement = sum(1 for m in host_msgs if "?" in m["content"])

    # Caller depth: responses > 50 chars after the first exchange
    caller_depth = sum(1 for m in caller_msgs[1:] if len(m["content"]) > 50)

    # Natural ending: True if the call did NOT end with [HANGUP] sentinel
    natural_ending = not caller_hangup

    return {
        "exchange_count": exchange_count,
        "avg_response_length": avg_response_length,
        "host_engagement": host_engagement,
        "caller_depth": caller_depth,
        "natural_ending": natural_ending,
    }


class Session:
    def __init__(self):
        self.id = str(uuid.uuid4())[:8]
        self.current_caller_key: str = None
        self.conversation: list[dict] = []
        self.caller_backgrounds: dict[str, dict] = {}  # Slim caller identity dicts, keyed by caller_key
        self.call_history: list[CallRecord] = []
        self._call_started_at: float = 0.0
        self.active_real_caller: dict | None = None
        self.ai_respond_mode: str = "manual"  # "manual" or "auto"
        self.auto_followup: bool = False
        self.news_headlines: list = []
        self.research_notes: dict[str, list] = {}
        self._research_task: asyncio.Task | None = None
        self.used_reasons: set[str] = set()  # Track used caller reasons to prevent repeats
        self.call_quality_signals: list[dict] = []  # Per-call quality heuristics for tuning
        self._caller_hangup: bool = False  # Set when [HANGUP] sentinel detected in current call
        self._wrapping_up: bool = False  # Set via /api/wrap-up to gracefully wind down calls
        self._wrapup_exchanges: int = 0  # Track how many exchanges since wrap-up started
        self.caller_queue: list[str] = []  # Sorted presentation order of caller keys
        self.intern_monitoring: bool = True  # Devon monitors conversations by default
        self.show_theme: str = ""  # Current show theme (e.g. "St. Patrick's Day")

    def start_call(self, caller_key: str):
        self.current_caller_key = caller_key
        self.conversation = []
        self._call_started_at = time.time()
        self._caller_hangup = False
        self._wrapping_up = False
        self._wrapup_exchanges = 0

    def end_call(self):
        self.current_caller_key = None
        self.conversation = []

    def add_message(self, role: str, content: str):
        self.conversation.append({"role": role, "content": content, "timestamp": time.time()})

    def get_caller_background(self, caller_key: str) -> str:
        """Return the caller's situation string for UI display.
        Backgrounds are populated by _pregenerate_backgrounds at session start."""
        bg = self.caller_backgrounds.get(caller_key) or {}
        return bg.get("situation", "") if isinstance(bg, dict) else ""

    def get_show_history(self) -> str:
        """Get formatted show history for AI caller prompts.
        Uses thematic matching to pick relevant previous callers to react to."""
        if not self.call_history and not any(e.read_on_air for e in _listener_emails):
            return ""
        lines = ["EARLIER IN THE SHOW:"]
        for record in self.call_history:
            caller_type_label = "(real caller)" if record.caller_type == "real" else "(AI)"
            lines.append(f"- {record.caller_name} {caller_type_label}: {record.summary}")

        # Include emails that were read on the show
        read_emails = [e for e in _listener_emails if e.read_on_air]
        for em in read_emails:
            sender_name = em.sender.split("<")[0].strip().strip('"') if "<" in em.sender else "a listener"
            preview = em.body[:150] if len(em.body) > 150 else em.body
            lines.append(f"- A listener email from {sender_name} was read on air: \"{em.subject}\" — {preview}")

        # Thematic matching for inter-caller reactions
        if self.call_history:
            current_bg = self.caller_backgrounds.get(self.current_caller_key)
            best_target, best_score = self._find_thematic_match(current_bg)

            # Adaptive reaction frequency based on thematic match strength
            if best_score >= 3:
                reaction_chance = 0.60
            elif best_score >= 1:
                reaction_chance = 0.35
            else:
                reaction_chance = 0.15

            if random.random() < reaction_chance and best_target:
                reaction = self._build_specific_reaction(current_bg, best_target)
                if random.random() < 0.30:
                    lines.append(f"\nYOU HEARD {best_target.caller_name.upper()} EARLIER ON THE SHOW TONIGHT and you {reaction}. It reminded you of your own situation — bring it up early and tie it into your story. NOTE: You are NOT {best_target.caller_name} — you are a different caller who was listening.")
                else:
                    lines.append(f"\nYOU HEARD {best_target.caller_name.upper()} EARLIER and you {reaction}. Mention it if it comes up naturally, but your call is about YOUR thing.")
            else:
                lines.append("You're aware of these but you're calling about YOUR thing, not theirs. Don't bring them up unless the host does.")

        return "\n".join(lines)

    def _find_thematic_match(self, current_bg) -> tuple:
        """Score previous callers against current caller for thematic relevance.
        Returns (best_target CallRecord, score)."""
        if not self.call_history:
            return None, 0

        best_target = None
        best_score = 0

        if isinstance(current_bg, dict):
            current_reason = current_bg.get("reason_calling", "")
            current_summary = current_bg.get("situation", "")
        else:
            current_reason = ""
            current_summary = ""
        current_words = set((current_reason + " " + current_summary).lower().split())

        for record in self.call_history:
            score = 0
            # Keyword overlap in situation summaries
            if record.situation_summary:
                record_words = set(record.situation_summary.lower().split())
                overlap = current_words & record_words - {"the", "a", "an", "and", "or", "is", "was", "to", "in", "of", "for", "that", "it", "on", "with"}
                if len(overlap) >= 2:
                    score += 2
                elif len(overlap) >= 1:
                    score += 1

            if score > best_score:
                best_score = score
                best_target = record

        # If no thematic match, pick a random target for generic reactions
        if best_target is None:
            best_target = random.choice(self.call_history)

        return best_target, best_score

    def _build_specific_reaction(self, current_bg, target: 'CallRecord') -> str:
        """Build a reaction that references specific details from the target call."""
        # If target has specific details, use them for a more specific reaction
        if target.key_details:
            detail = random.choice(target.key_details)
            specific_reactions = [
                f"heard them talk about {detail} and has strong opinions about it",
                f"had something similar happen involving {detail}",
                f"completely disagrees with their take on {detail}",
                f"was thinking about what they said about {detail} and it reminded them of their own situation",
                f"can't stop thinking about the {detail} part",
            ]
            return random.choice(specific_reactions)

        # If target has a situation summary, use that
        if target.situation_summary:
            summary_reactions = [
                f"heard about their situation and has been through something eerily similar",
                f"thinks they were completely wrong about their situation",
                f"felt personally called out by their story",
                f"wants to give them advice the host didn't",
            ]
            return random.choice(summary_reactions)

        # Fallback to generic reactions
        return random.choice(SHOW_HISTORY_REACTIONS)

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
                    "vibe": self.get_caller_background(self.current_caller_key),
                    "tts_provider": base.get("tts_provider"),
                }
        return None

    async def _pregenerate_backgrounds(self):
        """Single sonnet-4.6 batch call generates all caller identities."""
        from .services import caller_gen, regulars_v2
        from datetime import datetime

        voice_roster = [name for name in INWORLD_MALE_VOICES + INWORLD_FEMALE_VOICES
                        if name not in BLACKLISTED_VOICES]

        active_regulars = regulars_v2.load_all_active_regulars()
        regulars_for_tonight = [
            {"name": r.name, "lore": r.lore_body, "arc_state": r.arc_state}
            for r in active_regulars
        ][:3]

        headlines: list[str] = []
        if self.news_headlines:
            for h in self.news_headlines[:5]:
                headlines.append(h.title if hasattr(h, "title") else str(h))

        ctx = {
            "date": datetime.now().strftime("%A, %B %d, %Y"),
            "weather": "cool desert night",  # TODO: real weather feed
            "headlines": headlines,
            "recent_caller_summaries": self._get_recent_summaries(),
            "regulars_included": regulars_for_tonight,
            "caller_count": 12,
            "voice_roster": voice_roster,
        }

        identities = await caller_gen.generate_batch(ctx)

        for i, identity in enumerate(identities[:10]):
            key = str(i + 1)
            self.caller_backgrounds[key] = {
                "name": identity.name,
                "age": identity.age,
                "voice": identity.voice_resolved,
                "location": identity.location,
                "identity": identity.identity,
                "situation": identity.situation,
                "reason_calling": identity.reason_calling,
                "opening_line": identity.opening_line,
                "secret_want": identity.secret_want,
                "specific_details": identity.specific_details,
                "emotional_register": identity.emotional_register,
            }
        print(f"[Background] Slim batch generated {len(identities[:10])} caller identities")

    def _get_recent_summaries(self) -> list[str]:
        # Return last 2 shows' caller summaries — stub for now, can wire into cost_db
        return []

    def reset(self):
        """Reset session - clears all caller backgrounds for fresh personalities"""
        self.caller_backgrounds = {}
        self.current_caller_key = None
        self.conversation = []
        self.call_history = []
        self.active_real_caller = None
        self.ai_respond_mode = "manual"
        self.auto_followup = False
        self.news_headlines = []
        self.research_notes = {}
        if self._research_task and not self._research_task.done():
            self._research_task.cancel()
        self._research_task = None
        self.call_quality_signals = []
        self._wrapping_up = False
        self._wrapup_exchanges = 0
        self.caller_queue = []
        self.used_reasons = set()
        self.intern_monitoring = True
        intern_service.stop_monitoring()
        intern_service.new_show()
        cost_tracker.reset()
        _randomize_callers()
        self.id = str(uuid.uuid4())[:8]
        names = [CALLER_BASES[k]["name"] for k in sorted(CALLER_BASES.keys())]
        print(f"[Session] Reset - new session ID: {self.id}, callers: {', '.join(names)}")


session = Session()
caller_service = CallerService()
_ai_response_lock = asyncio.Lock()  # Prevents concurrent AI responses
_session_epoch = 0  # Increments on hangup/call start — stale tasks check this
_show_on_air = False  # Controls whether phone calls are accepted or get off-air message
_caller_line_ready = False  # True when ngrok tunnel is up and SignalWire webhook is pointed at it
_hold_music_tasks: dict[str, asyncio.Task] = {}  # caller_id -> hold music streaming task


def _stop_hold_music(caller_id: str):
    task = _hold_music_tasks.pop(caller_id, None)
    if task and not task.done():
        task.cancel()
        print(f"[Hold Music] Stopped for {caller_id}")


async def _stream_hold_music(caller_id: str):
    """Stream music tracks to a queued caller until they go on air or disconnect."""
    import librosa

    tracks = []
    if settings.music_dir.exists():
        for ext in ('*.wav', '*.mp3', '*.flac'):
            tracks.extend(settings.music_dir.glob(ext))
    if not tracks:
        print("[Hold Music] No tracks found in music directory")
        return

    random.shuffle(tracks)
    track_idx = 0
    print(f"[Hold Music] Starting for {caller_id} ({len(tracks)} tracks available)")

    try:
        while caller_id in caller_service._websockets:
            track = tracks[track_idx % len(tracks)]
            track_idx += 1
            print(f"[Hold Music] Playing '{track.stem}' for {caller_id}")

            audio, sr = librosa.load(str(track), sr=24000, mono=True)
            # Reduce volume to 40%
            audio = audio * 0.4
            audio_int16 = (audio * 32767).astype(np.int16)
            await caller_service.stream_audio_to_caller(caller_id, audio_int16.tobytes(), 24000)

            # Brief pause between tracks
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[Hold Music] Error for {caller_id}: {e}")
    finally:
        _hold_music_tasks.pop(caller_id, None)


# --- Session Checkpoint ---
CHECKPOINT_FILE = Path(__file__).parent.parent / "data" / "session_checkpoint.json"
CHECKPOINT_MAX_AGE = 12 * 3600  # Ignore checkpoints older than 12 hours


def _save_checkpoint():
    try:
        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        caller_bases_snapshot = {}
        for key, base in CALLER_BASES.items():
            caller_bases_snapshot[key] = {
                "name": base.get("name"),
                "voice": base.get("voice"),
                "returning": base.get("returning", False),
                "regular_id": base.get("regular_id"),
            }
        data = {
            "session_id": session.id,
            "call_history": [_serialize_call_record(r) for r in session.call_history],
            "caller_backgrounds": session.caller_backgrounds,
            "used_reasons": list(session.used_reasons),
            "ai_respond_mode": session.ai_respond_mode,
            "auto_followup": session.auto_followup,
            "news_headlines": session.news_headlines,
            "research_notes": session.research_notes,
            "caller_bases": caller_bases_snapshot,
            "call_quality_signals": session.call_quality_signals,
            "caller_queue": session.caller_queue,
            "intern_monitoring": session.intern_monitoring,
            "costs": cost_tracker.get_live_summary(),
            "cost_records": {
                "llm": [asdict(r) for r in cost_tracker.llm_records],
                "tts": [asdict(r) for r in cost_tracker.tts_records],
            },
            "saved_at": time.time(),
        }
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Checkpoint] Saved session {session.id} ({len(session.call_history)} calls)")
    except Exception as e:
        print(f"[Checkpoint] Failed to save: {e}")


def _load_checkpoint() -> bool:
    if not CHECKPOINT_FILE.exists():
        return False
    try:
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        age = time.time() - data.get("saved_at", 0)
        if age > CHECKPOINT_MAX_AGE:
            print(f"[Checkpoint] Stale ({age / 3600:.1f}h old), starting fresh")
            return False
        session.id = data["session_id"]
        session.call_history = [_deserialize_call_record(r) for r in data.get("call_history", [])]
        # Drop any legacy background dicts (pre-slim schema). They'll be regenerated
        # fresh on next Session.reset or when startup sees no restored backgrounds.
        raw_bgs = data.get("caller_backgrounds", {})
        session.caller_backgrounds = {
            k: v for k, v in raw_bgs.items()
            if isinstance(v, dict) and "identity" in v and "situation" in v
        }
        session.used_reasons = set(data.get("used_reasons", []))
        session.ai_respond_mode = data.get("ai_respond_mode", "manual")
        session.auto_followup = data.get("auto_followup", False)
        session.news_headlines = data.get("news_headlines", [])
        session.research_notes = data.get("research_notes", {})
        session.call_quality_signals = data.get("call_quality_signals", [])
        session.caller_queue = data.get("caller_queue", [])
        session.intern_monitoring = data.get("intern_monitoring", True)
        for key, snapshot in data.get("caller_bases", {}).items():
            if key in CALLER_BASES:
                CALLER_BASES[key]["name"] = snapshot["name"]
                CALLER_BASES[key]["voice"] = snapshot["voice"]
                CALLER_BASES[key]["returning"] = snapshot.get("returning", False)
                CALLER_BASES[key]["regular_id"] = snapshot.get("regular_id")
        # Restore cost tracker records
        cost_records = data.get("cost_records", {})
        if cost_records:
            cost_tracker.reset()
            for r in cost_records.get("llm", []):
                cost_tracker.llm_records.append(LLMCallRecord(**r))
            for r in cost_records.get("tts", []):
                cost_tracker.tts_records.append(TTSCallRecord(**r))
            # Rebuild running totals from restored records
            for r in cost_tracker.llm_records:
                cost_tracker._llm_cost += r.cost_usd
                cost_tracker._llm_calls += 1
                cost_tracker._prompt_tokens += r.prompt_tokens
                cost_tracker._completion_tokens += r.completion_tokens
                cost_tracker._total_tokens += r.total_tokens
                cat = cost_tracker._by_category.setdefault(r.category, {"cost": 0.0, "calls": 0, "tokens": 0})
                cat["cost"] += r.cost_usd
                cat["calls"] += 1
                cat["tokens"] += r.total_tokens
            for r in cost_tracker.tts_records:
                cost_tracker._tts_cost += r.cost_usd
            print(f"[Checkpoint] Restored {len(cost_tracker.llm_records)} LLM + {len(cost_tracker.tts_records)} TTS cost records")
        mins = age / 60
        print(f"[Checkpoint] Restored session {session.id} ({len(session.call_history)} calls, {mins:.0f}m old)")
        return True
    except Exception as e:
        print(f"[Checkpoint] Failed to load: {e}")
        return False


# --- Voicemail ---
VOICEMAILS_DIR = Path(__file__).parent.parent / "data" / "voicemails"
VOICEMAILS_SAVED_DIR = Path(__file__).parent.parent / "voicemails"
VOICEMAILS_META = Path(__file__).parent.parent / "data" / "voicemails.json"


@dataclass
class Voicemail:
    id: str
    phone: str
    timestamp: float
    duration: int
    file_path: str
    listened: bool = False


_voicemails: list[Voicemail] = []
_deleted_vm_timestamps: set[int] = set()


def _load_voicemails():
    global _voicemails, _deleted_vm_timestamps
    if VOICEMAILS_META.exists():
        try:
            with open(VOICEMAILS_META) as f:
                data = json.load(f)
            _voicemails = [
                Voicemail(
                    id=v["id"], phone=v["phone"], timestamp=v["timestamp"],
                    duration=v["duration"], file_path=v["file_path"],
                    listened=v.get("listened", False),
                )
                for v in data.get("voicemails", [])
            ]
            _deleted_vm_timestamps = set(data.get("deleted_timestamps", []))
            print(f"[Voicemail] Loaded {len(_voicemails)} voicemails")
        except Exception as e:
            print(f"[Voicemail] Failed to load: {e}")
            _voicemails = []


def _save_voicemails():
    try:
        VOICEMAILS_META.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "voicemails": [
                {
                    "id": v.id, "phone": v.phone, "timestamp": v.timestamp,
                    "duration": v.duration, "file_path": v.file_path,
                    "listened": v.listened,
                }
                for v in _voicemails
            ],
            "deleted_timestamps": list(_deleted_vm_timestamps),
        }
        with open(VOICEMAILS_META, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[Voicemail] Failed to save: {e}")


# --- News & Research Helpers ---

async def _fetch_session_headlines():
    try:
        session.news_headlines = await news_service.get_headlines()
        print(f"[News] Loaded {len(session.news_headlines)} headlines for session")
    except Exception as e:
        print(f"[News] Failed to load headlines: {e}")


async def _background_research(text: str):
    keywords = extract_keywords(text)
    if not keywords:
        return
    query = " ".join(keywords)
    if query.lower() in session.research_notes:
        return
    try:
        async with asyncio.timeout(8):
            results = await news_service.search_topic(query)
            if results:
                session.research_notes[query.lower()] = results
                print(f"[Research] Found {len(results)} results for '{query}'")
    except TimeoutError:
        print(f"[Research] Timed out for '{query}'")
    except Exception as e:
        print(f"[Research] Error: {e}")


def _build_news_context() -> tuple[str, str]:
    """Build context from cached news/research only — never does network calls.
    Each caller gets a random subset of headlines so they don't all reference the same thing."""
    news_context = ""
    if session.news_headlines and random.random() < 0.5:
        # Random 2-3 headlines, not the same 6 every time
        pool = list(session.news_headlines)
        random.shuffle(pool)
        news_context = news_service.format_headlines_for_prompt(pool[:random.randint(2, 3)])
    research_context = ""
    if session.research_notes:
        all_items = []
        for items in session.research_notes.values():
            all_items.extend(items)
        seen = set()
        unique = []
        for item in all_items:
            if item.title not in seen:
                seen.add(item.title)
                unique.append(item)
        random.shuffle(unique)
        research_context = news_service.format_headlines_for_prompt(unique[:3])
    return news_context, research_context


async def _sync_signalwire_voicemails():
    """Pull any recordings from SignalWire that aren't already tracked locally.
    Checks both the top-level Recordings endpoint AND per-call recordings
    (Record verb recordings don't always appear in the top-level list)."""
    if not settings.signalwire_project_id or not settings.signalwire_token:
        return
    try:
        from datetime import datetime as _dt
        auth = (settings.signalwire_project_id, settings.signalwire_token)
        base = f"https://{settings.signalwire_space}/api/laml/2010-04-01/Accounts/{settings.signalwire_project_id}"
        existing_timestamps = {int(v.timestamp) for v in _voicemails} | _deleted_vm_timestamps

        all_recordings = []

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # 1. Top-level recordings
            resp = await client.get(f"{base}/Recordings.json", auth=auth)
            resp.raise_for_status()
            for rec in resp.json().get("recordings", []):
                rec["_source"] = "top-level"
                all_recordings.append(rec)

            # 2. Check recent calls for per-call recordings (last 20 calls)
            calls_resp = await client.get(f"{base}/Calls.json?PageSize=20", auth=auth)
            if calls_resp.status_code == 200:
                for call in calls_resp.json().get("calls", []):
                    call_sid = call.get("sid", "")
                    call_from = call.get("from", "Unknown")
                    rec_resp = await client.get(f"{base}/Calls/{call_sid}/Recordings", auth=auth)
                    if rec_resp.status_code == 200:
                        for rec in rec_resp.json().get("recordings", []):
                            rec["_caller_phone"] = call_from
                            rec["_source"] = "per-call"
                            all_recordings.append(rec)

        # Deduplicate by recording SID
        seen_sids = set()
        unique_recordings = []
        for rec in all_recordings:
            sid = rec.get("sid", "")
            if sid not in seen_sids:
                seen_sids.add(sid)
                unique_recordings.append(rec)

        synced = 0
        for rec in unique_recordings:
            call_sid = rec.get("call_sid", "")
            duration = int(rec.get("duration", 0))
            date_created = rec.get("date_created", "")

            if duration < 2:
                continue

            try:
                ts = int(_dt.strptime(date_created, "%a, %d %b %Y %H:%M:%S %z").timestamp())
            except (ValueError, TypeError):
                ts = int(time.time())

            if ts in existing_timestamps:
                continue

            # Get caller phone — may already be embedded from per-call lookup
            caller_phone = rec.get("_caller_phone", "Unknown")
            if caller_phone == "Unknown" and call_sid:
                try:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        call_resp = await client.get(f"{base}/Calls/{call_sid}.json", auth=auth)
                        if call_resp.status_code == 200:
                            caller_phone = call_resp.json().get("from", "Unknown")
                except Exception:
                    pass

            rec_uri = rec.get("uri", "").replace(".json", ".wav")
            rec_url = f"https://{settings.signalwire_space}{rec_uri}"
            await _download_voicemail(rec_url, caller_phone, duration)

            if _voicemails and _voicemails[-1].phone == caller_phone:
                _voicemails[-1].timestamp = ts
                _save_voicemails()

            existing_timestamps.add(ts)
            synced += 1

        if synced:
            print(f"[Voicemail] Synced {synced} recording(s) from SignalWire")
        else:
            print(f"[Voicemail] No new recordings found ({len(unique_recordings)} total checked)")
    except Exception as e:
        print(f"[Voicemail] SignalWire sync failed: {e}")


# --- Lifecycle ---
@app.on_event("startup")
async def startup():
    """Pre-generate caller backgrounds on server start"""
    _load_voicemails()
    _load_emails()
    asyncio.create_task(_sync_signalwire_voicemails())
    asyncio.create_task(_poll_imap_emails())
    restored = _load_checkpoint()
    if not restored or not session.caller_backgrounds:
        asyncio.create_task(session._pregenerate_backgrounds())
    asyncio.create_task(avatar_service.ensure_devon())
    threading.Thread(target=_update_on_air_cdn, args=(False,), daemon=True).start()


@app.on_event("shutdown")
async def shutdown():
    """Clean up resources on server shutdown"""
    global _host_audio_task
    _save_checkpoint()
    print("[Server] Shutting down — cleaning up resources...")
    _update_on_air_cdn(False)
    _stop_ngrok()
    # Stop host mic streaming
    audio_service.stop_host_stream()
    # Cancel host audio sender task
    if _host_audio_task and not _host_audio_task.done():
        _host_audio_task.cancel()
        try:
            await _host_audio_task
        except (asyncio.CancelledError, Exception):
            pass
        _host_audio_task = None
    # Disconnect all active callers
    for caller_id in list(caller_service.active_calls.keys()):
        caller_service.hangup(caller_id)
    caller_service.reset()
    await news_service.close()
    print("[Server] Cleanup complete")


# --- Static Files ---
frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/css", StaticFiles(directory=frontend_dir / "css"), name="css")
app.mount("/js", StaticFiles(directory=frontend_dir / "js"), name="js")
app.mount("/images", StaticFiles(directory=frontend_dir / "images"), name="images")


@app.get("/costs")
async def costs_page():
    return FileResponse(frontend_dir / "costs.html")


@app.get("/")
async def index():
    return FileResponse(frontend_dir / "index.html")


# --- Ngrok Tunnel Management ---

_ngrok_process: subprocess.Popen | None = None
_ngrok_domain = "shana-chromoplasmic-noneligibly.ngrok-free.dev"
_signalwire_phone_sid = "12ef9c34-976d-4cff-814e-d740415dd0df"


def _start_ngrok():
    """Start ngrok tunnel and update SignalWire webhook to point to it."""
    global _ngrok_process, _caller_line_ready
    if _ngrok_process and _ngrok_process.poll() is None:
        print("[Ngrok] Already running")
        _caller_line_ready = True
        return True

    _caller_line_ready = False
    try:
        _ngrok_process = subprocess.Popen(
            ["ngrok", "http", "8000", f"--domain={_ngrok_domain}", "--log=stdout", "--log-format=json"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # Wait for tunnel to be ready
        import time as _time
        for _ in range(20):
            _time.sleep(0.5)
            try:
                resp = httpx.get("http://127.0.0.1:4040/api/tunnels", timeout=2)
                tunnels = resp.json().get("tunnels", [])
                if tunnels:
                    public_url = tunnels[0]["public_url"]
                    print(f"[Ngrok] Tunnel ready: {public_url}")
                    _update_signalwire_webhook(public_url)
                    _caller_line_ready = True
                    return True
            except Exception:
                continue
        print("[Ngrok] Timed out waiting for tunnel")
        return False
    except FileNotFoundError:
        print("[Ngrok] ngrok binary not found")
        return False
    except Exception as e:
        print(f"[Ngrok] Failed to start: {e}")
        return False


def _stop_ngrok():
    """Stop ngrok tunnel and restore SignalWire webhook to production URL."""
    global _ngrok_process, _caller_line_ready
    _caller_line_ready = False
    _restore_signalwire_webhook()
    if _ngrok_process and _ngrok_process.poll() is None:
        _ngrok_process.terminate()
        try:
            _ngrok_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _ngrok_process.kill()
        print("[Ngrok] Stopped")
    _ngrok_process = None


def _update_signalwire_webhook(ngrok_url: str):
    """Point SignalWire phone number webhook to ngrok tunnel."""
    if not settings.signalwire_project_id or not settings.signalwire_token:
        return
    try:
        url = (f"https://{settings.signalwire_space}/api/laml/2010-04-01/Accounts/"
               f"{settings.signalwire_project_id}/IncomingPhoneNumbers/{_signalwire_phone_sid}.json")
        voice_url = f"{ngrok_url}/api/signalwire/voice"
        resp = httpx.post(url, data={
            "VoiceUrl": voice_url,
            "VoiceFallbackUrl": voice_url,
        }, auth=(settings.signalwire_project_id, settings.signalwire_token), timeout=10)
        if resp.status_code == 200:
            print(f"[SignalWire] Webhook updated -> {voice_url}")
        else:
            print(f"[SignalWire] Failed to update webhook: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        print(f"[SignalWire] Webhook update error: {e}")


def _restore_signalwire_webhook():
    """Restore SignalWire webhook to production URL (voicemail when off air)."""
    if not settings.signalwire_project_id or not settings.signalwire_token:
        return
    try:
        url = (f"https://{settings.signalwire_space}/api/laml/2010-04-01/Accounts/"
               f"{settings.signalwire_project_id}/IncomingPhoneNumbers/{_signalwire_phone_sid}.json")
        prod_url = "https://lukeattheroost.com/api/signalwire/voice"
        resp = httpx.post(url, data={
            "VoiceUrl": prod_url,
            "VoiceFallbackUrl": f"https://lukeattheroost.com/voicemail.xml",
        }, auth=(settings.signalwire_project_id, settings.signalwire_token), timeout=10)
        if resp.status_code == 200:
            print(f"[SignalWire] Webhook restored -> {prod_url}")
        else:
            print(f"[SignalWire] Failed to restore webhook: {resp.status_code}")
    except Exception as e:
        print(f"[SignalWire] Webhook restore error: {e}")


# --- On-Air Toggle ---

# BunnyCDN config for public on-air status
_BUNNY_STORAGE_ZONE = "lukeattheroost"
_BUNNY_STORAGE_KEY = os.getenv("BUNNY_STORAGE_KEY", "")
_BUNNY_STORAGE_REGION = "la"
_BUNNY_ACCOUNT_KEY = os.getenv("BUNNY_ACCOUNT_KEY", "")


def _update_on_air_cdn(on_air: bool):
    """Upload on-air status to BunnyCDN so the public website can poll it."""
    from datetime import datetime, timezone
    data = {"on_air": on_air}
    if on_air:
        data["since"] = datetime.now(timezone.utc).isoformat()
    url = f"https://{_BUNNY_STORAGE_REGION}.storage.bunnycdn.com/{_BUNNY_STORAGE_ZONE}/status.json"
    try:
        resp = httpx.put(url, content=json.dumps(data), headers={
            "AccessKey": _BUNNY_STORAGE_KEY,
            "Content-Type": "application/json",
        }, timeout=5)
        if resp.status_code == 201:
            print(f"[CDN] On-air status updated: {on_air}")
        else:
            print(f"[CDN] Failed to update on-air status: {resp.status_code}")
            return
        httpx.get(
            "https://api.bunny.net/purge",
            params={"url": "https://cdn.lukeattheroost.com/status.json", "async": "false"},
            headers={"AccessKey": _BUNNY_ACCOUNT_KEY},
            timeout=10,
        )
        print(f"[CDN] Cache purged")
    except Exception as e:
        print(f"[CDN] Error updating on-air status: {e}")


@app.post("/api/on-air")
async def set_on_air(state: dict):
    """Toggle whether the show is on air (accepting phone calls). Also toggles recording."""
    global _show_on_air
    _show_on_air = bool(state.get("on_air", False))
    print(f"[Show] On-air: {_show_on_air}")
    if _show_on_air:
        # Reset REAPER state to dialog for fresh show
        try:
            from .services.audio import _write_reaper_state
            _write_reaper_state("dialog")
        except Exception:
            pass
        # Auto-start recording FIRST (before host stream, which takes over mic capture)
        if audio_service.stem_recorder is None:
            try:
                from datetime import datetime
                dir_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                recordings_dir = Path("recordings") / dir_name
                import sounddevice as sd
                device_info = sd.query_devices(audio_service.output_device) if audio_service.output_device is not None else None
                sr = int(device_info["default_samplerate"]) if device_info else 48000
                recorder = StemRecorder(recordings_dir, sample_rate=sr)
                recorder.start()
                audio_service.stem_recorder = recorder
                audio_service.start_stem_mic()
                add_log(f"Stem recording auto-started -> {recordings_dir}")
            except Exception as e:
                print(f"[Show] Failed to auto-start recording: {e}")
        _start_host_audio_sender()
        # Host stream takes over mic capture (closes stem_mic if active)
        audio_service.start_host_stream(_host_audio_sync_callback)
    else:
        audio_service.stop_host_stream()
        # Auto-stop recording
        if audio_service.stem_recorder is not None:
            try:
                audio_service.stop_stem_mic()
                stems_dir = audio_service.stem_recorder.output_dir
                paths = audio_service.stem_recorder.stop()
                audio_service.stem_recorder = None
                add_log(f"Stem recording auto-stopped. Running post-production...")
                import subprocess, sys
                python = sys.executable
                output_file = stems_dir / "episode.mp3"
                def _run_postprod():
                    try:
                        result = subprocess.run(
                            [python, "postprod.py", str(stems_dir), "-o", "episode.mp3"],
                            capture_output=True, text=True, timeout=600,
                        )
                        if result.returncode == 0:
                            add_log(f"Post-production complete -> {output_file}")
                        else:
                            add_log(f"Post-production failed: {result.stderr[:300]}")
                    except Exception as e:
                        add_log(f"Post-production error: {e}")
                threading.Thread(target=_run_postprod, daemon=True).start()
            except Exception as e:
                print(f"[Show] Failed to auto-stop recording: {e}")
    threading.Thread(target=_update_on_air_cdn, args=(_show_on_air,), daemon=True).start()
    if _show_on_air:
        threading.Thread(target=_start_ngrok, daemon=True).start()
    else:
        threading.Thread(target=_stop_ngrok, daemon=True).start()
    return {"on_air": _show_on_air, "recording": audio_service.stem_recorder is not None, "caller_line_ready": _caller_line_ready}

@app.get("/api/on-air")
async def get_on_air():
    return {"on_air": _show_on_air, "recording": audio_service.stem_recorder is not None, "caller_line_ready": _caller_line_ready}


# --- SignalWire Endpoints ---

@app.post("/api/signalwire/voice")
async def signalwire_voice_webhook(request: Request):
    """Handle inbound call from SignalWire — return XML to start bidirectional stream"""
    form = await request.form()
    caller_phone = form.get("From", "Unknown")
    call_sid = form.get("CallSid", "")
    print(f"[SignalWire] Inbound call from {caller_phone} (CallSid: {call_sid})")

    if not _show_on_air:
        print(f"[SignalWire] Show is off air — offering voicemail to {caller_phone}")
        # Derive host from stream URL config if available, otherwise from request
        if settings.signalwire_stream_url:
            from urllib.parse import urlparse
            host = urlparse(settings.signalwire_stream_url).hostname
        else:
            host = request.headers.get("host", "radioshow.macneilmediagroup.com")
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="woman">Luke at the Roost is off the air right now. Leave a message after the beep and we may play it on the next show!</Say>
    <Record maxLength="120" action="https://{host}/api/signalwire/voicemail-complete" playBeep="true" />
    <Say voice="woman">Thank you for calling. Goodbye!</Say>
    <Hangup/>
</Response>"""
        return Response(content=xml, media_type="application/xml")

    # Use dedicated stream URL (ngrok) if configured, otherwise derive from request
    if settings.signalwire_stream_url:
        stream_url = settings.signalwire_stream_url
    else:
        host = request.headers.get("host", "radioshow.macneilmediagroup.com")
        stream_url = f"wss://{host}/api/signalwire/stream"

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="woman">You've reached Luke at the Roost. Hold tight, we'll get you on the air.</Say>
    <Connect>
        <Stream url="{stream_url}" codec="L16@16000h">
            <Parameter name="caller_phone" value="{caller_phone}"/>
            <Parameter name="call_sid" value="{call_sid}"/>
        </Stream>
    </Connect>
</Response>"""

    return Response(content=xml, media_type="application/xml")


@app.post("/api/signalwire/voicemail-complete")
async def signalwire_voicemail_complete(request: Request):
    form = await request.form()
    recording_url = form.get("RecordingUrl", "")
    caller_phone = form.get("From", "Unknown")
    duration = int(form.get("RecordingDuration", "0"))
    print(f"[Voicemail] Recording complete from {caller_phone} ({duration}s): {recording_url}")

    if recording_url:
        asyncio.create_task(_download_voicemail(recording_url, caller_phone, duration))

    xml = '<?xml version="1.0" encoding="UTF-8"?><Response><Say voice="woman">Thank you for calling. Goodbye!</Say><Hangup/></Response>'
    return Response(content=xml, media_type="application/xml")


async def _download_voicemail(recording_url: str, caller_phone: str, duration: int):
    try:
        VOICEMAILS_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        safe_phone = caller_phone.replace("+", "").replace(" ", "")
        # Determine extension from URL
        ext = Path(recording_url.split("?")[0]).suffix or ".wav"
        filename = f"{ts}_{safe_phone}{ext}"
        filepath = VOICEMAILS_DIR / filename

        # Try downloading without auth first (pre-signed URL), fall back to basic auth
        auth = (settings.signalwire_project_id, settings.signalwire_token)
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(recording_url)
            if resp.status_code in (401, 403):
                resp = await client.get(recording_url, auth=auth)
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(resp.content)

        vm = Voicemail(
            id=str(uuid.uuid4())[:8],
            phone=caller_phone,
            timestamp=ts,
            duration=duration,
            file_path=str(filepath),
        )
        _voicemails.append(vm)
        _save_voicemails()
        print(f"[Voicemail] Saved {filename} ({duration}s) from {caller_phone}")
    except Exception as e:
        print(f"[Voicemail] Failed to download recording: {e}")


# --- Voicemail API ---

@app.get("/api/voicemails")
async def list_voicemails():
    return [
        {
            "id": v.id, "phone": v.phone, "timestamp": v.timestamp,
            "duration": v.duration, "listened": v.listened,
        }
        for v in sorted(_voicemails, key=lambda v: v.timestamp, reverse=True)
    ]


@app.get("/api/voicemail/{vm_id}/audio")
async def get_voicemail_audio(vm_id: str):
    vm = next((v for v in _voicemails if v.id == vm_id), None)
    if not vm:
        raise HTTPException(status_code=404, detail="Voicemail not found")
    fp = Path(vm.file_path)
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Audio file missing")
    media_type = "audio/wav" if fp.suffix == ".wav" else "audio/mpeg"
    return FileResponse(fp, media_type=media_type, filename=fp.name)


@app.post("/api/voicemail/{vm_id}/play-on-air")
async def play_voicemail_on_air(vm_id: str):
    vm = next((v for v in _voicemails if v.id == vm_id), None)
    if not vm:
        raise HTTPException(status_code=404, detail="Voicemail not found")
    fp = Path(vm.file_path)
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Audio file missing")

    def _play():
        import librosa
        audio, sr = librosa.load(str(fp), sr=24000, mono=True)
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_service.play_caller_audio(audio_int16.tobytes(), 24000)

    thread = threading.Thread(target=_play, daemon=True)
    thread.start()
    vm.listened = True
    _save_voicemails()
    return {"status": "playing"}


@app.post("/api/voicemail/{vm_id}/mark-listened")
async def mark_voicemail_listened(vm_id: str):
    vm = next((v for v in _voicemails if v.id == vm_id), None)
    if not vm:
        raise HTTPException(status_code=404, detail="Voicemail not found")
    vm.listened = True
    _save_voicemails()
    return {"status": "ok"}


@app.post("/api/voicemail/{vm_id}/save")
async def save_voicemail(vm_id: str):
    vm = next((v for v in _voicemails if v.id == vm_id), None)
    if not vm:
        raise HTTPException(status_code=404, detail="Voicemail not found")
    fp = Path(vm.file_path)
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Audio file missing")
    VOICEMAILS_SAVED_DIR.mkdir(parents=True, exist_ok=True)
    dest = VOICEMAILS_SAVED_DIR / fp.name
    import shutil
    shutil.copy2(fp, dest)
    print(f"[Voicemail] Saved {fp.name} to archive")
    return {"status": "saved", "path": str(dest)}


@app.delete("/api/voicemail/{vm_id}")
async def delete_voicemail(vm_id: str):
    vm = next((v for v in _voicemails if v.id == vm_id), None)
    if not vm:
        raise HTTPException(status_code=404, detail="Voicemail not found")
    _deleted_vm_timestamps.add(int(vm.timestamp))
    fp = Path(vm.file_path)
    if fp.exists():
        fp.unlink()
    _voicemails.remove(vm)
    _save_voicemails()
    return {"status": "deleted"}


# --- Listener Emails ---
EMAILS_META = Path(__file__).parent.parent / "data" / "emails.json"


@dataclass
class ListenerEmail:
    id: str
    sender: str
    subject: str
    body: str
    timestamp: float
    read_on_air: bool = False


_listener_emails: list[ListenerEmail] = []


def _load_emails():
    global _listener_emails
    if EMAILS_META.exists():
        try:
            with open(EMAILS_META) as f:
                data = json.load(f)
            _listener_emails = [
                ListenerEmail(
                    id=e["id"], sender=e["sender"], subject=e["subject"],
                    body=e["body"], timestamp=e["timestamp"],
                    read_on_air=e.get("read_on_air", False),
                )
                for e in data.get("emails", [])
            ]
            print(f"[Email] Loaded {len(_listener_emails)} emails")
        except Exception as e:
            print(f"[Email] Failed to load: {e}")
            _listener_emails = []


def _save_emails():
    try:
        EMAILS_META.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "emails": [
                {
                    "id": e.id, "sender": e.sender, "subject": e.subject,
                    "body": e.body, "timestamp": e.timestamp,
                    "read_on_air": e.read_on_air,
                }
                for e in _listener_emails
            ],
        }
        with open(EMAILS_META, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        print(f"[Email] Failed to save: {exc}")


async def _poll_imap_emails():
    """Background task: poll IMAP every 30s for new listener emails"""
    import imaplib
    import email as email_lib
    from email.header import decode_header

    host = settings.submissions_imap_host
    user = settings.submissions_imap_user
    passwd = settings.submissions_imap_pass
    if not host or not user or not passwd:
        print("[Email] IMAP not configured, skipping email polling")
        return

    while True:
        try:
            mail = imaplib.IMAP4_SSL(host, 993)
            mail.login(user, passwd)
            mail.select("INBOX")

            _, msg_nums = mail.search(None, "UNSEEN")
            if msg_nums[0]:
                for num in msg_nums[0].split():
                    _, msg_data = mail.fetch(num, "(RFC822)")
                    raw = msg_data[0][1]
                    msg = email_lib.message_from_bytes(raw)

                    # Decode sender
                    from_raw = msg.get("From", "Unknown")

                    # Decode subject
                    subj_raw = msg.get("Subject", "(no subject)")
                    decoded_parts = decode_header(subj_raw)
                    subject = ""
                    for part, charset in decoded_parts:
                        if isinstance(part, bytes):
                            subject += part.decode(charset or "utf-8", errors="replace")
                        else:
                            subject += part

                    # Extract plain text body
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                payload = part.get_payload(decode=True)
                                if payload:
                                    charset = part.get_content_charset() or "utf-8"
                                    body = payload.decode(charset, errors="replace")
                                break
                    else:
                        payload = msg.get_payload(decode=True)
                        if payload:
                            charset = msg.get_content_charset() or "utf-8"
                            body = payload.decode(charset, errors="replace")

                    body = body.strip()
                    if not body:
                        continue

                    # Parse timestamp from email Date header
                    from email.utils import parsedate_to_datetime
                    try:
                        ts = parsedate_to_datetime(msg.get("Date", "")).timestamp()
                    except Exception:
                        ts = time.time()

                    em = ListenerEmail(
                        id=str(uuid.uuid4())[:8],
                        sender=from_raw,
                        subject=subject,
                        body=body,
                        timestamp=ts,
                    )
                    _listener_emails.append(em)
                    print(f"[Email] New email from {from_raw}: {subject[:50]}")

                    # Mark as SEEN (already done by fetch with UNSEEN filter)
                    mail.store(num, "+FLAGS", "\\Seen")

                _save_emails()

            mail.logout()
        except Exception as exc:
            print(f"[Email] IMAP poll error: {exc}")

        await asyncio.sleep(30)


@app.get("/api/emails")
async def list_emails():
    return [
        {
            "id": e.id, "sender": e.sender, "subject": e.subject,
            "body": e.body, "timestamp": e.timestamp,
            "read_on_air": e.read_on_air,
        }
        for e in sorted(_listener_emails, key=lambda e: e.timestamp, reverse=True)
    ]


@app.post("/api/email/{email_id}/play-on-air")
async def play_email_on_air(email_id: str):
    em = next((e for e in _listener_emails if e.id == email_id), None)
    if not em:
        raise HTTPException(status_code=404, detail="Email not found")

    # Extract display name, fall back to just "a listener"
    sender_name = em.sender.split("<")[0].strip().strip('"') if "<" in em.sender else "a listener"
    intro = f"This email is from {sender_name}. Subject: {em.subject}."
    full_text = f"{intro}\n\n{em.body}"

    async def _generate_and_play():
        try:
            audio_bytes = await generate_speech(full_text, "Alex", phone_quality="none", apply_filter=False)
            audio_service.play_caller_audio(audio_bytes, 24000)
        except Exception as exc:
            print(f"[Email] TTS playback error: {exc}")

    asyncio.create_task(_generate_and_play())
    em.read_on_air = True
    _save_emails()
    return {"status": "playing"}


@app.delete("/api/email/{email_id}")
async def delete_email(email_id: str):
    em = next((e for e in _listener_emails if e.id == email_id), None)
    if not em:
        raise HTTPException(status_code=404, detail="Email not found")
    _listener_emails.remove(em)
    _save_emails()
    return {"status": "deleted"}


async def _signalwire_end_call(call_sid: str):
    """End a phone call via SignalWire REST API"""
    if not call_sid or not settings.signalwire_space:
        return
    try:
        url = f"https://{settings.signalwire_space}/api/laml/2010-04-01/Accounts/{settings.signalwire_project_id}/Calls/{call_sid}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                url,
                data={"Status": "completed"},
                auth=(settings.signalwire_project_id, settings.signalwire_token),
            )
            print(f"[SignalWire] End call {call_sid}: {response.status_code}")
    except Exception as e:
        print(f"[SignalWire] Failed to end call {call_sid}: {e}")


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
    devon_channel: Optional[int] = None
    live_caller_channel: Optional[int] = None
    music_channel: Optional[int] = None
    sfx_channel: Optional[int] = None
    ad_channel: Optional[int] = None
    ident_channel: Optional[int] = None
    monitor_device: Optional[int] = None
    monitor_channel: Optional[int] = None
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
        devon_channel=settings.devon_channel,
        live_caller_channel=settings.live_caller_channel,
        music_channel=settings.music_channel,
        sfx_channel=settings.sfx_channel,
        ad_channel=settings.ad_channel,
        ident_channel=settings.ident_channel,
        monitor_device=settings.monitor_device,
        monitor_channel=settings.monitor_channel,
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


def _get_all_caller_names() -> list[str]:
    """Get all current caller names (from backgrounds or base assignments)."""
    names = []
    for key in CALLER_BASES:
        bg = session.caller_backgrounds.get(key)
        if isinstance(bg, dict) and bg.get("name"):
            names.append(bg["name"])
        elif "name" in CALLER_BASES[key]:
            names.append(CALLER_BASES[key]["name"])
    # Always include Devon (the intern)
    names.append("Devon")
    return names


def _fix_caller_names(text: str, names: list[str]) -> str:
    """Fix Whisper misspellings of caller names using fuzzy matching.
    Compares each word against known names — if within edit distance 2
    and the word isn't a common English word, replace it."""
    if not names or not text:
        return text

    # Build lookup: lowercase name -> original name
    name_map = {n.lower(): n for n in names if n}
    if not name_map:
        return text

    # Common short words that happen to be close to names — never replace these
    _common_words = {
        "the", "and", "but", "for", "not", "you", "all", "can", "had", "her",
        "was", "one", "our", "out", "are", "has", "his", "how", "its", "may",
        "new", "now", "old", "see", "way", "who", "did", "get", "got", "him",
        "let", "say", "she", "too", "use", "been", "call", "come", "each",
        "from", "have", "just", "know", "like", "long", "look", "make", "many",
        "much", "over", "said", "some", "take", "tell", "than", "that", "them",
        "then", "they", "this", "time", "very", "want", "well", "went", "were",
        "what", "when", "will", "with", "your", "been", "yeah", "okay", "sure",
        "right", "about", "think", "really", "gonna", "gotta", "would", "could",
        "should", "never", "still", "here", "there", "where", "being", "doing",
        "going", "having", "saying", "man", "hey", "yes", "no",
    }

    def _edit_distance(a: str, b: str) -> int:
        """Levenshtein distance between two strings."""
        if len(a) < len(b):
            return _edit_distance(b, a)
        if len(b) == 0:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                cost = 0 if ca == cb else 1
                curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
            prev = curr
        return prev[len(b)]

    words = text.split()
    changed = False
    for i, word in enumerate(words):
        # Strip punctuation for matching but preserve it
        stripped = word.strip(".,!?;:\"'—-")
        if not stripped or len(stripped) < 3:
            continue
        low = stripped.lower()
        if low in _common_words:
            continue

        # Exact match (already correct)
        if low in name_map:
            # Fix capitalization if needed
            correct = name_map[low]
            if stripped != correct:
                words[i] = word.replace(stripped, correct)
                changed = True
            continue

        # Fuzzy match against all names — conservative to avoid mangling real words
        for name_low, name_orig in name_map.items():
            # No fuzzy matching for very short names (3 chars) — too many false positives
            # e.g. "dog" → "Dot", "cat" → "Cal"
            if len(name_low) <= 3:
                continue
            if abs(len(low) - len(name_low)) > 1:
                continue
            dist = _edit_distance(low, name_low)
            # Distance 1 only, and require first letter match to avoid wild substitutions
            if dist == 1 and low[0] == name_low[0]:
                print(f"[NameFix] Fuzzy: '{stripped}' -> '{name_orig}' (dist={dist})")
                words[i] = word.replace(stripped, name_orig)
                changed = True
                break

    if changed:
        result = " ".join(words)
        if result != text:
            print(f"[NameFix] '{text}' -> '{result}'")
        return result
    return text


@app.post("/api/record/stop")
async def stop_recording():
    """Stop recording and transcribe"""
    audio_bytes = audio_service.stop_recording()

    if len(audio_bytes) < 100:
        return {"text": "", "status": "no_audio"}

    # Context hint for Whisper — basic show context only, NO caller names.
    # Names were over-biasing Whisper (e.g. "bother" → "Luthor").
    # Post-transcription fuzzy matching (_fix_caller_names) handles name correction.
    context_hint = "Luke at the Roost, a late-night radio call-in show."
    caller_names = _get_all_caller_names()

    # Transcribe the recorded audio (16kHz raw PCM from audio service)
    text = await transcribe_audio(audio_bytes, source_sample_rate=16000, context_hint=context_hint)

    # Post-transcription: fix Whisper misspellings of caller names
    if text and caller_names:
        text = _fix_caller_names(text, caller_names)

    return {"text": text, "status": "transcribed"}


# --- Caller Endpoints ---

@app.get("/api/callers")
async def get_callers():
    """Get list of available callers with background info for UI display"""
    callers = []
    for k, v in CALLER_BASES.items():
        caller_info = {
            "key": k,
            "name": v["name"],
            "returning": v.get("returning", False),
            "avatar_url": f"/api/avatar/{v['name']}",
        }
        bg = session.caller_backgrounds.get(k)
        if isinstance(bg, dict):
            details = bg.get("specific_details") or []
            caller_info["identity"] = bg.get("identity", "")
            caller_info["situation"] = bg.get("situation", "")
            caller_info["signature"] = details[0] if details else ""
            caller_info["secret_want"] = bg.get("secret_want", "")
            caller_info["voice"] = bg.get("voice", "")
        callers.append(caller_info)
    return {
        "callers": callers,
        "current": session.current_caller_key,
        "session_id": session.id
    }


@app.get("/api/regulars")
async def get_regulars():
    """Get list of regular callers"""
    return {"regulars": regular_caller_service.get_regulars()}


@app.post("/api/session/reset")
async def reset_session():
    """Reset session - all callers get fresh backgrounds"""
    session.reset()
    _chat_updates.clear()
    # Pre-generate backgrounds in background so they're ready when callers are clicked
    asyncio.create_task(session._pregenerate_backgrounds())
    return {"status": "reset", "session_id": session.id}


def _maybe_generate_callback() -> dict | None:
    """After 6+ calls, 15% chance to bring back a previous caller with a callback.
    Returns a callback info dict or None."""
    if len(session.call_history) < 6:
        return None
    if random.random() > 0.15:
        return None

    # Pick a previous AI caller with a good summary
    ai_calls = [r for r in session.call_history
                if r.caller_type == "ai" and len(r.summary) > 20]
    if not ai_calls:
        return None

    target = random.choice(ai_calls)
    callback_reason = random.choice([
        f"called back because something changed since they last called about: {target.summary}",
        f"forgot to mention something important when they called earlier about: {target.summary}",
        f"heard a later caller and it reminded them of their own situation: {target.summary}",
        f"the situation from their earlier call has gotten worse: {target.summary}",
        f"good news — the thing they called about earlier actually worked out: {target.summary}",
    ])
    print(f"[Callback] Generating callback for {target.caller_name}: {callback_reason[:80]}...")
    return {
        "caller_name": target.caller_name,
        "original_summary": target.summary,
        "callback_reason": callback_reason,
    }


@app.post("/api/call/{caller_key}")
async def start_call(caller_key: str):
    """Start a call with a caller"""
    global _session_epoch
    if caller_key not in CALLER_BASES:
        raise HTTPException(404, "Caller not found")

    # Guard against double-click or rapid switching
    if session.current_caller_key == caller_key:
        return {"status": "already_on_call", "caller_key": caller_key}
    if session.current_caller_key is not None:
        # Already on a different call — hang up first
        audio_service.stop_caller_audio()
        session.end_call()

    _session_epoch += 1
    audio_service.stop_caller_audio()
    session.start_call(caller_key)

    # Check for callback opportunity — only for non-returning callers
    # Returning callers already have their own PREVIOUS CALLS context
    base = CALLER_BASES[caller_key]
    if not base.get("returning"):
        callback = _maybe_generate_callback()
        if callback:
            existing_bg = session.caller_backgrounds.get(caller_key)
            if isinstance(existing_bg, dict):
                callback_ctx = f"\n\nPREVIOUS CALLS:\n- (earlier tonight) {callback['original_summary']}\nYou're calling back with an update — {callback['callback_reason']}. Reference your earlier call naturally."
                existing_bg["situation"] = existing_bg.get("situation", "") + callback_ctx
                print(f"[Callback] Injected callback context for {base.get('name', caller_key)}")

    caller = session.caller  # This generates the background if needed

    # Enrich with news/weather in background — don't block call pickup
    if caller_key in session.caller_backgrounds:
        asyncio.create_task(_enrich_background_async(caller_key))

    # Extract slim background for UI info panel
    bg = session.caller_backgrounds.get(caller_key)
    caller_info = {}
    if isinstance(bg, dict):
        details = bg.get("specific_details") or []
        caller_info = {
            "identity": bg.get("identity", ""),
            "situation": bg.get("situation", ""),
            "signature": details[0] if details else "",
            "secret_want": bg.get("secret_want", ""),
        }

    # Start intern monitoring if enabled
    if session.intern_monitoring and not intern_service.monitoring:
        async def _on_intern_suggestion(text, sources):
            broadcast_event("intern_suggestion", {"text": text, "sources": sources})
        intern_service.start_monitoring(
            get_conversation=lambda: session.conversation,
            on_suggestion=_on_intern_suggestion,
            get_caller_active=lambda: session.caller is not None,
        )

    return {
        "status": "connected",
        "caller": caller["name"],
        "background": caller["vibe"],
        "caller_info": {**caller_info, "avatar_url": f"/api/avatar/{caller['name']}"},
    }


async def _enrich_background_async(caller_key: str):
    """Enrich caller background with news/weather without blocking the call"""
    try:
        bg = session.caller_backgrounds.get(caller_key)
        if not isinstance(bg, dict):
            return
        enriched = await enrich_caller_background(bg.get("situation", ""))
        bg["situation"] = enriched
    except Exception as e:
        print(f"[Research] Background enrichment failed: {e}")


@app.post("/api/hangup")
async def hangup():
    """Hang up current call"""
    global _session_epoch, _auto_respond_pending
    _session_epoch += 1

    # Stop any playing caller audio immediately
    audio_service.stop_caller_audio()

    # Cancel any pending auto-respond
    if _auto_respond_pending and not _auto_respond_pending.done():
        _auto_respond_pending.cancel()
        _auto_respond_pending = None
    _auto_respond_buffer.clear()

    if session._research_task and not session._research_task.done():
        session._research_task.cancel()
        session._research_task = None

    # Stop intern monitoring between calls
    intern_service.stop_monitoring()

    caller_name = session.caller["name"] if session.caller else None
    caller_key = session.current_caller_key
    conversation_snapshot = list(session.conversation)
    call_started = getattr(session, '_call_started_at', 0.0)
    was_caller_hangup = session._caller_hangup
    session._wrapping_up = False
    session._wrapup_exchanges = 0
    session.end_call()

    # Play hangup sound in background so response returns immediately
    hangup_sound = settings.sounds_dir / "hangup.wav"
    if hangup_sound.exists():
        threading.Thread(target=audio_service.play_sfx, args=(str(hangup_sound),), daemon=True).start()

    # Generate summary for AI caller in background
    if caller_name and conversation_snapshot:
        asyncio.create_task(_summarize_ai_call(caller_key, caller_name, conversation_snapshot, call_started, was_caller_hangup))

    return {"status": "disconnected", "caller": caller_name}


@app.post("/api/wrap-up")
async def wrap_up():
    """Signal the current caller to wrap up gracefully"""
    if not session.caller:
        raise HTTPException(400, "No active call")
    session._wrapping_up = True
    session._wrapup_exchanges = 0
    print(f"[Wrap-up] Initiated for {session.caller['name']}")
    return {"status": "wrapping_up"}


async def _summarize_ai_call(caller_key: str, caller_name: str, conversation: list[dict], started_at: float, caller_hangup: bool = False):
    """Background task: summarize AI caller conversation and store in history"""
    ended_at = time.time()
    summary = ""
    if conversation:
        transcript_text = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in conversation
        )
        try:
            summary = await llm_service.generate(
                messages=[{"role": "user", "content": f"Summarize this radio show call in 1-2 sentences:\n{transcript_text}"}],
                system_prompt="You summarize radio show conversations concisely. Focus on what the caller talked about and any emotional moments.",
                category="call_summary",
                caller_name=caller_name,
            )
        except Exception as e:
            print(f"[AI Summary] Failed to generate summary: {e}")
            summary = f"{caller_name} called in."

    # Populate from slim caller background dict
    bg = session.caller_backgrounds.get(caller_key) or {}
    comm_style = bg.get("emotional_register", "") if isinstance(bg, dict) else ""
    sit_summary = bg.get("situation", "") if isinstance(bg, dict) else ""
    key_dets = list(bg.get("specific_details") or []) if isinstance(bg, dict) else []

    quality_signals = _assess_call_quality(
        conversation,
        caller_hangup=caller_hangup,
    )
    session.call_quality_signals.append(quality_signals)

    session.call_history.append(CallRecord(
        caller_type="ai",
        caller_name=caller_name,
        summary=summary,
        transcript=conversation,
        started_at=started_at,
        ended_at=ended_at,
        quality_signals=quality_signals,
        situation_summary=sit_summary,
        communication_style=comm_style,
        key_details=key_dets,
    ))
    print(f"[AI Summary] {caller_name} call summarized: {summary[:80]}...")
    print(f"[Quality] {caller_name}: exchanges={quality_signals['exchange_count']} avg_len={quality_signals['avg_response_length']:.0f}c host_engagement={quality_signals['host_engagement']} caller_depth={quality_signals['caller_depth']} natural_end={quality_signals['natural_ending']}")

    # Returning caller promotion/update logic
    try:
        base = CALLER_BASES.get(caller_key) if caller_key else None
        if base and summary:
            if base.get("returning") and base.get("regular_id"):
                # Update existing regular's call history
                regular_caller_service.update_after_call(base["regular_id"], summary)
            elif len(conversation) >= 8 and random.random() < 0.05:
                # 5% chance to promote first-timer with 8+ messages
                bg = session.caller_backgrounds.get(caller_key) or {}
                if isinstance(bg, dict):
                    traits = list(bg.get("specific_details") or [])[:4]
                    promo_job = bg.get("identity", "") or ""
                    promo_location = bg.get("location") or "unknown"
                    promo_age = bg.get("age") or random.randint(*base.get("age_range", (30, 50)))
                    promo_gender = base.get("gender", "male")
                    structured_bg = dict(bg)
                    avatar_path = avatar_service.get_path(caller_name)
                    regular_caller_service.add_regular(
                        name=caller_name,
                        gender=promo_gender,
                        age=promo_age,
                        job=promo_job,
                        location=promo_location,
                        personality_traits=traits,
                        first_call_summary=summary,
                        voice=base.get("voice"),
                        stable_seeds={},
                        structured_background=structured_bg,
                        avatar=avatar_path.name if avatar_path else None,
                    )
    except Exception as e:
        print(f"[Regulars] Promotion logic error: {e}")

    # Detect relationships: if this caller mentioned another regular by name
    _detect_caller_relationships(caller_key, caller_name, conversation, summary)

    _save_checkpoint()


def _detect_caller_relationships(caller_key: str, caller_name: str,
                                  conversation: list[dict], summary: str):
    """Scan conversation for mentions of other regular callers and store relationships."""
    try:
        base = CALLER_BASES.get(caller_key)
        if not base or not base.get("regular_id"):
            return  # Only track relationships for regulars

        regulars = regular_caller_service.get_regulars()
        regular_names = {r["name"]: r["id"] for r in regulars if r["name"] != caller_name}
        if not regular_names:
            return

        # Build full text from caller's messages + summary
        caller_text = summary + " " + " ".join(
            m["content"] for m in conversation if m.get("role") == "assistant"
        )
        caller_text_lower = caller_text.lower()

        for other_name in regular_names:
            if other_name.lower() in caller_text_lower:
                # Determine relationship type from context
                rel_type = "mentioned"
                # Simple sentiment check
                name_idx = caller_text_lower.index(other_name.lower())
                context_window = caller_text_lower[max(0, name_idx - 80):name_idx + 80]
                negative = any(w in context_window for w in ["wrong", "disagree", "annoying", "hate", "idiot", "crazy", "ridiculous"])
                positive = any(w in context_window for w in ["agree", "right", "love", "friend", "respect", "relate", "same"])
                if negative:
                    rel_type = "rival"
                elif positive:
                    rel_type = "ally"

                context_snippet = caller_text[max(0, name_idx - 40):name_idx + 60].strip()
                regular_caller_service.add_relationship(
                    base["regular_id"], other_name, rel_type,
                    f"Referenced during call: ...{context_snippet}..."
                )
                print(f"[Relationships] Detected: {caller_name} → {other_name} ({rel_type})")
    except Exception as e:
        print(f"[Relationships] Detection error: {e}")


# --- Chat & TTS Endpoints ---

import re


def _pick_response_budget(wrapping_up: bool = False) -> tuple[int, int]:
    """Pick a random max_tokens and sentence cap for response variety.
    Returns (max_tokens, max_sentences).
    Keeps responses conversational but gives room for real answers.
    Token budget is intentionally generous to avoid mid-sentence cutoffs —
    the sentence cap controls actual length."""

    if wrapping_up:
        return 200, 2

    # Default distribution — give callers room to tell their story
    roll = random.random()
    if roll < 0.10:
        return 600, 6   # 10% — quick response
    elif roll < 0.35:
        return 700, 7   # 25% — normal conversation
    elif roll < 0.65:
        return 800, 8   # 30% — room to breathe
    else:
        return 900, 10  # 35% — telling a story or riffing


MIN_RESPONSE_WORDS = 80  # Retry if response is shorter than this


async def _retry_if_too_short(response: str, llm_service, messages: list, system_prompt: str,
                               max_tokens: int, caller_name: str, model_override=None,
                               wrapping_up: bool = False) -> str:
    """Retry once if caller response is too short (some models produce terse output)."""
    if wrapping_up or not response or "[HANGUP]" in response:
        return response
    word_count = len(response.split())
    if word_count >= MIN_RESPONSE_WORDS:
        return response
    print(f"[Chat] Response too short ({word_count} words), retrying...")
    retry = await llm_service.generate(
        messages=messages,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        category="caller_dialog",
        caller_name=caller_name,
        model_override=model_override,
    )
    if retry and len(retry.split()) > word_count:
        print(f"[Chat] Retry produced {len(retry.split())} words (was {word_count})")
        return retry
    print(f"[Chat] Retry no better, keeping original")
    return response


_REPETITION_STOPWORDS = {
    "i", "me", "my", "you", "your", "he", "she", "it", "we", "they",
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "can", "may", "might", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "and", "or", "but", "not", "no",
    "that", "this", "what", "which", "who", "how", "if", "so", "just",
    "than", "then", "about", "up", "out", "all", "like", "got", "get",
}


def _has_repetition(response: str, conversation: list, threshold: int = 3) -> bool:
    """Check if the response contains repeated 3+ word n-grams from recent conversation."""
    # Collect last 6 assistant messages
    recent_assistant = [
        msg["content"] for msg in conversation
        if msg.get("role") == "assistant" and msg.get("content")
    ][-6:]
    prior_text = " ".join(recent_assistant)

    # Extract 3-word n-grams from response and prior text combined
    def get_ngrams(text):
        words = text.lower().split()
        return [" ".join(words[i:i+3]) for i in range(len(words) - 2)]

    response_ngrams = get_ngrams(response)
    if not response_ngrams:
        return False

    all_ngrams = get_ngrams(prior_text) + response_ngrams

    # Count occurrences
    counts: dict[str, int] = {}
    for ng in all_ngrams:
        counts[ng] = counts.get(ng, 0) + 1

    # Check if any response n-gram hits threshold (skip all-stopword n-grams)
    response_set = set(response_ngrams)
    for ng in response_set:
        if counts.get(ng, 0) >= threshold:
            words = ng.split()
            if not all(w in _REPETITION_STOPWORDS for w in words):
                return True
    return False


def _trim_to_sentences(text: str, max_sentences: int) -> str:
    """Hard-trim response to at most max_sentences sentences."""
    if not text:
        return text
    # Split on sentence-ending punctuation, keeping the delimiter.
    # Negative lookbehind avoids splitting on common abbreviations (Mr. Mrs. Ms. Dr. St. etc.)
    parts = re.split(r'(?<!Mr)(?<!Mrs)(?<!Ms)(?<!Dr)(?<!St)(?<!Jr)(?<!Sr)(?<!vs)(?<![A-Z])(?<=[.!?])\s+', text.strip())
    if len(parts) <= max_sentences:
        return text
    trimmed = ' '.join(parts[:max_sentences])
    # Make sure it ends with punctuation
    if trimmed and trimmed[-1] not in '.!?':
        trimmed = trimmed.rstrip(',;:— -') + '.'
    return trimmed


def ensure_complete_thought(text: str) -> str:
    """If text was cut off mid-sentence, trim to the last complete sentence."""
    text = text.strip()
    if not text:
        return text
    # Already ends with sentence-ending punctuation — good
    if text[-1] in '.!?':
        return text
    # Cut off mid-sentence — find the last complete sentence
    for i in range(len(text) - 1, -1, -1):
        if text[i] in '.!?':
            return text[:i + 1]
    # No punctuation at all — just add a period
    return text.rstrip(',;:— -') + '.'


_DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

_TENS_WORDS = {
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty",
    70: "seventy", 80: "eighty", 90: "ninety",
}

def _number_to_spoken(n: int) -> str:
    """Convert a number (0-9999) to natural spoken English."""
    if n < 10:
        return _DIGIT_WORDS[n]
    if n in _TENS_WORDS:
        return _TENS_WORDS[n]
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        return f"{_TENS_WORDS[tens]} {_DIGIT_WORDS[ones]}" if ones else _TENS_WORDS[tens]
    if n < 1000:
        hundreds = n // 100
        remainder = n % 100
        if remainder == 0:
            return f"{_DIGIT_WORDS[hundreds]} hundred"
        return f"{_DIGIT_WORDS[hundreds]} hundred {_number_to_spoken(remainder)}"
    if n < 10000:
        thousands = n // 1000
        remainder = n % 1000
        if remainder == 0:
            return f"{_number_to_spoken(thousands)} thousand"
        if remainder < 100:
            return f"{_number_to_spoken(thousands)} thousand {_number_to_spoken(remainder)}"
        return f"{_number_to_spoken(thousands)} thousand {_number_to_spoken(remainder)}"
    return str(n)

# Numbers that should always be read digit-by-digit
_DIGIT_BY_DIGIT = {
    "911": "nine one one",
    "411": "four one one",
    "311": "three one one",
    "211": "two one one",
    "511": "five one one",
    "811": "eight one one",
    "101": "one oh one",
    "24/7": "twenty four seven",
    "401k": "four oh one k",
    "403b": "four oh three b",
    "409a": "four oh nine a",
    "w2": "W two",
    "w-2": "W two",
    "1099": "ten ninety nine",
    "i-10": "I ten",
    "i-25": "I twenty five",
    "i-40": "I forty",
}

# Ordinal numbers → spoken form (covers dates, rankings, common usage)
_ORDINALS = {
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
    "5th": "fifth", "6th": "sixth", "7th": "seventh", "8th": "eighth",
    "9th": "ninth", "10th": "tenth", "11th": "eleventh", "12th": "twelfth",
    "13th": "thirteenth", "14th": "fourteenth", "15th": "fifteenth",
    "16th": "sixteenth", "17th": "seventeenth", "18th": "eighteenth",
    "19th": "nineteenth", "20th": "twentieth", "21st": "twenty first",
    "22nd": "twenty second", "23rd": "twenty third", "24th": "twenty fourth",
    "25th": "twenty fifth", "26th": "twenty sixth", "27th": "twenty seventh",
    "28th": "twenty eighth", "29th": "twenty ninth", "30th": "thirtieth",
    "31st": "thirty first",
}

# Common title/address abbreviations that TTS should expand.
# Order matters: context-specific patterns first, then generic.
# Use a list of tuples so patterns aren't deduplicated by dict keys.
_COMMON_ABBREVIATIONS = [
    # Titles (Dr. → Doctor handled separately with case-sensitive lookahead)
    (r'\bMr\.', 'Mister'),
    (r'\bMrs\.', 'Missus'),
    (r'\bMs\.', 'Miss'),
    (r'\bJr\.', 'Junior'),
    (r'\bSr\.', 'Senior'),
    # St. → Saint before known proper names, Street otherwise
    (r'\bSt\.\s+(?=Patrick|Louis|George|Mary|John|Joseph|Paul|Peter|Thomas|Andrew|Francis|James|Lawrence|Augustine|Anthony|Bernard|Michael|Nicholas|David|Stephen|Charles|Claire|Anne|Elmo|Jude)', 'Saint '),
    (r'\bSt\.', 'Street'),
    (r'\bAve\.', 'Avenue'),
    (r'\bBlvd\.', 'Boulevard'),
    (r'\bRd\.', 'Road'),
    (r'\bDr\.', 'Drive'),
    (r'\bLn\.', 'Lane'),
    (r'\bCt\.', 'Court'),
    # General
    (r'\betc\.', 'etcetera'),
    (r'\bapprox\.', 'approximately'),
    (r'\bft\.', 'feet'),
    (r'\bmi\.', 'miles'),
    (r'\blb\.', 'pound'),
    (r'\blbs\.', 'pounds'),
    (r'\boz\.', 'ounces'),
    (r'\bmin\.', 'minutes'),
    (r'\bhr\.', 'hour'),
    (r'\bhrs\.', 'hours'),
    (r'\bw/o\b', 'without'),
    (r'\bw/', 'with '),
]


def _expand_numbers_for_tts(text: str) -> str:
    """Expand numbers that TTS engines commonly mispronounce."""
    # Fixed substitutions (case-insensitive)
    for pattern, replacement in _DIGIT_BY_DIGIT.items():
        text = re.sub(re.escape(pattern), replacement, text, flags=re.IGNORECASE)

    # Vehicle models: F-350 → F three fifty, RAM-2500 → RAM twenty five hundred
    def _model_number(m):
        letter_part = m.group(1)
        num = int(m.group(2))
        return f"{letter_part} {_number_to_spoken(num)}"
    text = re.sub(r'\b([A-Z]{1,3})[-.]?(\d{2,4})\b', _model_number, text)

    # Calibers: .308 → three oh eight, .223 → two twenty three, .22 → twenty two, etc.
    def _caliber_to_words(m):
        cal = m.group(1)
        caliber_map = {
            "308": "three oh eight", "223": "two twenty three", "556": "five fifty six",
            "762": "seven sixty two", "300": "three hundred", "338": "three thirty eight",
            "270": "two seventy", "243": "two forty three", "357": "three fifty seven",
            "380": "three eighty", "45": "forty five", "44": "forty four",
            "38": "thirty eight", "22": "twenty two", "50": "fifty",
            "9": "nine millimeter", "40": "forty",
            "410": "four ten", "12": "twelve gauge", "20": "twenty gauge",
        }
        return caliber_map.get(cal, " ".join(_DIGIT_WORDS[int(d)] for d in cal))
    text = re.sub(r'(?<!\d)\.(\d{1,3})\b(?!\d)', _caliber_to_words, text)

    # $120K → a hundred twenty thousand dollars (currency + K/M suffix)
    def _currency_kmb(m):
        num_str = m.group(1)
        suffix = m.group(2).upper()
        multiplier = {"K": "thousand", "M": "million", "B": "billion"}.get(suffix, "")
        if '.' in num_str:
            parts = num_str.split('.')
            return f"{_number_to_spoken(int(parts[0]))} point {' '.join(_DIGIT_WORDS[int(d)] for d in parts[1])} {multiplier} dollars"
        return f"{_number_to_spoken(int(num_str))} {multiplier} dollars"
    text = re.sub(r'\$(\d+(?:\.\d+)?)([KkMmBb])\b', _currency_kmb, text)

    # Currency with commas: $1,500 → "fifteen hundred dollars", $12,000 → "twelve thousand dollars"
    def _currency_comma(m):
        num = int(m.group(1).replace(',', ''))
        return f"{_number_to_spoken(num)} dollars"
    text = re.sub(r'\$(\d{1,3}(?:,\d{3})+)', _currency_comma, text)

    # Simple currency: $50 → "fifty dollars", $3.50 → "three dollars and fifty cents"
    def _currency_simple(m):
        amount = m.group(1)
        if '.' in amount:
            dollars, cents = amount.split('.')
            cents = cents.ljust(2, '0')[:2]
            d = int(dollars)
            c = int(cents)
            if c == 0:
                return f"{_number_to_spoken(d)} {'dollar' if d == 1 else 'dollars'}"
            if d == 0:
                return f"{_number_to_spoken(c)} cents"
            return f"{_number_to_spoken(d)} {'dollar' if d == 1 else 'dollars'} and {_number_to_spoken(c)} cents"
        num = int(amount)
        return f"{_number_to_spoken(num)} {'dollar' if num == 1 else 'dollars'}"
    text = re.sub(r'\$(\d+(?:\.\d{1,2})?)', _currency_simple, text)

    # K/M suffixes without $: 120K → a hundred twenty thousand, 1.5M → one point five million
    def _kmb_to_words(m):
        num_str = m.group(1)
        suffix = m.group(2).upper()
        multiplier = {"K": "thousand", "M": "million", "B": "billion"}.get(suffix, "")
        if '.' in num_str:
            parts = num_str.split('.')
            return f"{_number_to_spoken(int(parts[0]))} point {' '.join(_DIGIT_WORDS[int(d)] for d in parts[1])} {multiplier}"
        return f"{_number_to_spoken(int(num_str))} {multiplier}"
    text = re.sub(r'\b(\d+(?:\.\d+)?)([KkMmBb])\b', _kmb_to_words, text)

    # Years (1900-2099): nineteen eighty five, two thousand nine, twenty twenty five
    def _year_to_words(m):
        year = int(m.group(1))
        if 2000 <= year <= 2009:
            return f"two thousand {_DIGIT_WORDS[year - 2000]}" if year > 2000 else "two thousand"
        if 2010 <= year <= 2099:
            return f"twenty {_number_to_spoken(year - 2000)}"
        if 1900 <= year <= 1999:
            century = year // 100
            remainder = year % 100
            if remainder == 0:
                return f"{_number_to_spoken(century)} hundred"
            return f"{_number_to_spoken(century)} {_number_to_spoken(remainder)}"
        return str(year)
    text = re.sub(r'\b((?:19|20)\d{2})\b', _year_to_words, text)

    # Standalone numbers 2-4 digits (not already handled) — natural spoken form
    # Only matches numbers surrounded by word boundaries, not inside other patterns
    def _general_number(m):
        num = int(m.group(0))
        if num < 10:
            return m.group(0)  # single digits are fine
        return _number_to_spoken(num)
    text = re.sub(r'(?<![.$\d/:])\b(\d{2,4})\b(?![%\d:./KkMmBb])', _general_number, text)

    # Phone numbers: (xxx) xxx-xxxx or xxx-xxx-xxxx — read digit by digit
    def _phone_to_words(m):
        digits = re.sub(r'\D', '', m.group(0))
        return " ".join(_DIGIT_WORDS[int(d)] for d in digits)
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', _phone_to_words, text)

    # Ordinals: 1st → first, 2nd → second, etc. (case-insensitive)
    for ordinal, spoken in _ORDINALS.items():
        text = re.sub(r'\b' + re.escape(ordinal) + r'\b', spoken, text, flags=re.IGNORECASE)

    # Currency already handled above (before general number regex)

    # Percentages: 75% → 75 percent
    text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 percent', text)

    # Times: 3:30 → 3 30, 12:00 → 12 o'clock
    def _time_to_words(m):
        hour, minute = m.group(1), m.group(2)
        if minute == '00':
            return f"{hour} o'clock"
        return f"{hour} {minute}"
    text = re.sub(r'\b(\d{1,2}):(\d{2})\b', _time_to_words, text)

    return text


# Acronyms pronounced as words — leave these alone
_SPOKEN_ACRONYMS = {
    "NASA", "FEMA", "OSHA", "NATO", "SWAT", "SCUBA", "LASER", "RADAR",
    "YOLO", "AWOL", "HIPAA", "FOMO", "NIMBY", "AIDS", "DARE", "MADD",
    "NAFTA", "OPEC", "POTUS", "FLOTUS", "SCOTUS",
}

# Known words/names that TTS engines consistently botch
_PRONUNCIATION_FIXES = {
    "Castopod": "Casto pod",
    "vs": "versus",
    "govt": "government",
    "dept": "department",
}

# Abbreviations that should be expanded to full words BEFORE acronym/caps processing.
# These run on the original cased text so they can match uppercase abbreviations.
_ABBREVIATION_EXPANSIONS = {
    "NM": "New Mexico",
    "AZ": "Arizona",
    "TX": "Texas",
    "US": "United States",
    "USA": "United States",
}


# Common short English words that appear in ALL CAPS as emphasis, NOT acronyms.
# When the LLM writes "I SO get that" or "NO way" — these should just lowercase.
# Everything else 2-3 letters in ALL CAPS is assumed to be an acronym and spelled out.
_EMPHASIS_SHORT_WORDS = {
    # 2-letter
    "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "HI", "IF", "IN",
    "IS", "IT", "ME", "MY", "NO", "OF", "OH", "OK", "ON", "OR", "OW", "SO",
    "TO", "UP", "WE",
    # 3-letter
    "ALL", "AND", "ANY", "ARE", "BAD", "BIG", "BIT", "BUT", "CAN", "CUT",
    "DAD", "DAY", "DID", "END", "FAR", "FEW", "FOR", "GET", "GOD", "GOT",
    "GUY", "HAD", "HAS", "HER", "HIM", "HIS", "HOT", "HOW", "ITS", "JOB",
    "LET", "LOT", "MAN", "MAY", "MOM", "NEW", "NOT", "NOW", "OLD", "ONE",
    "OUR", "OUT", "OWN", "PUT", "RAN", "RAW", "RED", "RUN", "SAD", "SAT",
    "SAW", "SAY", "SET", "SHE", "SIT", "SIX", "TEN", "THE", "TOO", "TOP",
    "TRY", "TWO", "WAR", "WAS", "WAY", "WHO", "WHY", "WIN", "WON", "YET",
    "YOU", "YES",
}


def _process_caps_words(text: str) -> str:
    """Handle ALL CAPS words in one pass:
    - Spoken acronyms (NASA, FEMA): leave as-is
    - Short words (2-3 letters) that are common English: lowercase (emphasis)
    - Short words (2-3 letters) that are NOT common English: spell out (acronym)
    - Long words (4+ letters): lowercase (emphasis)
    """
    def _replace(m):
        word = m.group(0)
        upper = word.upper()
        # Spoken acronyms — leave alone
        if upper in _SPOKEN_ACRONYMS:
            return word
        length = len(word)
        if length <= 3:
            # Short word: if it's a common English word, it's emphasis → lowercase
            # Otherwise it's an acronym → spell out
            if upper in _EMPHASIS_SHORT_WORDS:
                return word.lower()
            else:
                return " ".join(word.upper())
        else:
            # 4+ letters: almost always emphasis (REALLY, NEVER, ABSOLUTELY)
            return word.lower()
    return re.sub(r'\b[A-Z]{2,}\b', _replace, text)


def _apply_pronunciation_fixes(text: str) -> str:
    """Apply known pronunciation fixes for words TTS engines botch."""
    for word, fix in _PRONUNCIATION_FIXES.items():
        text = re.sub(r'\b' + re.escape(word) + r'\b', fix, text, flags=re.IGNORECASE)
    return text


def clean_for_tts(text: str, formal: bool = True) -> str:
    """Strip out non-speakable content and fix phonetic spellings for TTS.
    When formal=False, keeps colloquialisms (gonna, kinda, etc.) for natural-sounding callers."""
    # Remove stage-direction parentheticals: (laughs), (pausing), (looking away), etc.
    # Only match parens that start with a known action word — avoids eating real dialog
    # like "I (get this look) that" → "I that"
    _action_start = r'(?:laughs?|laughing|sighs?|sighing|pauses?|pausing|smiles?|smiling|chuckles?|chuckling|grins?|grinning|nods?|nodding|shrugs?|shrugging|frowns?|frowning|looks?|looking|clears?|clearing|takes?|taking|leans?|leaning|shakes?|shaking|closes?|closing|opens?|opening|whispers?|whispering|mumbles?|mumbling|trails?|trailing|voice|silence|beat|quiet|long pause|deep breath|softly|nervously|quietly|crying|sobbing|sniffling|exhales?|exhaling|inhales?|inhaling)'
    text = re.sub(r'\s*\((?=' + _action_start + r')[^)]{1,40}\)\s*', ' ', text, flags=re.IGNORECASE)
    # Remove stage-direction asterisks: *laughs*, *sighs deeply*, etc.
    # Only match short action-like content, not emphasis like *really* or *the* important thing
    text = re.sub(r'\s*\*(?=' + _action_start + r')[^*]{1,40}\*\s*', ' ', text, flags=re.IGNORECASE)
    # Remove content in brackets: [laughs], [pause], etc. (only Bark uses these)
    text = re.sub(r'\s*\[(?=' + _action_start + r')[^\]]{1,40}\]\s*', ' ', text, flags=re.IGNORECASE)
    # Remove content in angle brackets: <laughs>, <sigh>, etc.
    text = re.sub(r'\s*<(?=' + _action_start + r')[^>]{1,40}>\s*', ' ', text, flags=re.IGNORECASE)
    # Remove "He/She sighs" style stage directions (NOT "I" — too aggressive, eats real dialog)
    text = re.sub(r'\b(He|She|They)\s+(sighs?|laughs?|pauses?|smiles?|chuckles?|grins?|nods?|shrugs?|frowns?)\s*(heavily|softly|deeply|quietly|loudly|nervously|sadly|a little|for a moment)?[.,]?\s*', '', text, flags=re.IGNORECASE)
    # Remove standalone stage direction words only if they look like directions (with adverbs)
    text = re.sub(r'\b(sighs?|laughs?|pauses?|chuckles?)\s+(heavily|softly|deeply|quietly|loudly|nervously|sadly)\b[.,]?\s*', '', text, flags=re.IGNORECASE)
    # Catch-all safety net: any remaining short parenthetical is almost certainly a stage
    # direction that wasn't caught by the specific patterns above (e.g. adjective-first
    # patterns like "(nervous laugh)" or "(a long beat)"). Nothing in parens should be
    # read aloud on air.
    text = re.sub(r'\s*\([^)]{1,40}\)\s*', ' ', text)
    # Catch-all for multi-word asterisk content — single-word *emphasis* is fine,
    # but multi-word like *sighs deeply* or *nervous laughter* is a stage direction
    text = re.sub(r'\s*\*\w+\s[^*]{1,30}\*\s*', ' ', text)
    # Remove quotes around the response if LLM wrapped it
    text = re.sub(r'^["\']|["\']$', '', text.strip())

    # --- Punctuation normalization for natural prosody ---
    # Note: em dashes (—) and ellipses (...) are preserved here — Inworld handles them
    # with SSML <break> tags in _prepare_text_for_inworld(), and other engines handle
    # them natively or via their own preprocessing.

    # Double hyphen → em dash (normalize before TTS engines handle it)
    text = re.sub(r'\s*--\s*', ' — ', text)
    # Unicode ellipsis → three dots (normalize for consistent handling downstream)
    text = re.sub(r'…', '...', text)
    # Semicolons → period (TTS doesn't differentiate semicolon from comma well)
    text = re.sub(r';', '.', text)

    # --- Symbols to speakable text ---

    # Ampersand
    text = re.sub(r'\s*&\s*', ' and ', text)
    # Hash/number sign (before a number = "number", standalone = skip)
    text = re.sub(r'#(\d)', r'number \1', text)
    # Plus sign between words
    text = re.sub(r'\s*\+\s*', ' plus ', text)
    # Equals sign
    text = re.sub(r'\s*=\s*', ' equals ', text)
    # At sign (in non-email context)
    text = re.sub(r'(?<!\S)@(?=\w)', 'at ', text)

    # --- Number and abbreviation expansion ---

    # Expand numbers (currency, ordinals, percentages, times, phone numbers)
    text = _expand_numbers_for_tts(text)

    # Expand common abbreviations (Dr., Mr., St., etc.)
    # Dr. → Doctor before a name (case-SENSITIVE lookahead so "Oak Dr." → "Oak Drive" not "Oak Doctor")
    text = re.sub(r'\bDr\.\s+(?=[A-Z])', 'Doctor ', text)
    for abbr_pattern, expansion in _COMMON_ABBREVIATIONS:
        text = re.sub(abbr_pattern, expansion, text, flags=re.IGNORECASE)

    # Expand state/country abbreviations BEFORE acronym processing
    # Must run while text is still original case so we can match uppercase abbreviations
    for abbrev, expansion in _ABBREVIATION_EXPANSIONS.items():
        text = re.sub(r'\b' + re.escape(abbrev) + r'\b', expansion, text)

    # Known pronunciation fixes for local names (case-sensitive, run before lowering)
    text = _apply_pronunciation_fixes(text)

    # Normalize dotted acronyms: D.J. → DJ, U.F.O. → UFO, A.P. → AP
    text = re.sub(r'(?<![A-Za-z])(?:[A-Za-z]\.){2,}', lambda m: m.group().replace('.', '').upper(), text)
    # Handle all caps words: spell out acronyms (FBI → F B I), lowercase emphasis (REALLY → really)
    text = _process_caps_words(text)

    # --- Phonetic spelling normalization ---
    # Skip colloquialism expansion for informal callers — keeps speech natural
    if formal:
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

    # --- Natural breathing pauses ---

    # Add comma after sentence-starting transition words (if not already punctuated)
    for tw in ['Well', 'So', 'Now', 'Look', 'See', 'Anyway', 'Actually', 'Honestly',
               'Basically', 'Listen', 'Right', 'Okay', 'Sure', 'Yeah']:
        text = re.sub(r'(?<![,.])\b(' + tw + r')\s+(?=[A-Za-z])', r'\1, ', text)

    # Add pause after "I mean" / "you know" at start of sentence
    text = re.sub(r'(?:^|(?<=\.\s))(I mean)\s+(?!,)', r'\1, ', text)
    text = re.sub(r'(?:^|(?<=\.\s))(You know)\s+(?=\w)', r'\1, ', text)

    # Conjunction pauses handled per-engine: Inworld uses SSML <break> tags in
    # _prepare_text_for_inworld(), Kokoro uses comma insertion in preprocess_text_for_kokoro()

    # --- Final cleanup ---

    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Fix spaces before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    # Fix double punctuation from earlier transformations (preserve ellipsis "...")
    text = re.sub(r'([,!?])\s*\1+', r'\1', text)
    text = re.sub(r'(?<!\.)\.\.(?!\.)', '.', text)  # collapse ".." but not "..."
    text = re.sub(r',\.', '.', text)
    text = re.sub(r'\.,', ',', text)
    # Remove orphaned punctuation at start
    text = re.sub(r'^[.,]\s*', '', text)
    return text.strip()


# --- Chat Broadcast (for real-time frontend updates) ---
_chat_updates: list[dict] = []
_CHAT_UPDATES_MAX = 500


def broadcast_chat(sender: str, text: str):
    """Add a chat message to the update queue for frontend polling"""
    _chat_updates.append({"type": "chat", "sender": sender, "text": text, "id": len(_chat_updates)})
    if len(_chat_updates) > _CHAT_UPDATES_MAX:
        del _chat_updates[:_CHAT_UPDATES_MAX // 2]


def broadcast_event(event_type: str, data: dict = None):
    """Add a system event to the update queue for frontend polling"""
    entry = {"type": event_type, "id": len(_chat_updates)}
    if data:
        entry.update(data)
    _chat_updates.append(entry)


@app.get("/api/conversation/updates")
async def get_conversation_updates(since: int = 0):
    """Get new chat/event messages since a given index"""
    return {
        "messages": _chat_updates[since:],
        "wrapping_up": session._wrapping_up,
        "intern_suggestion": intern_service.get_pending_suggestion(),
    }


def _dynamic_context_window() -> int:
    """Return context window size based on conversation length.
    Short calls: 10 messages. Medium: 15. Long: 20."""
    n = len(session.conversation)
    if n <= 10:
        return 10
    elif n <= 16:
        return 15
    else:
        return 20


def _normalize_messages_for_llm(messages: list[dict]) -> list[dict]:
    """Convert custom roles (real_caller:X, ai_caller:X, intern:X) to standard LLM roles"""
    normalized = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role.startswith("real_caller:"):
            caller_label = role.split(":", 1)[1]
            normalized.append({"role": "user", "content": f"[Real caller {caller_label}]: {content}"})
        elif role.startswith("ai_caller:"):
            normalized.append({"role": "assistant", "content": content})
        elif role.startswith("intern:"):
            intern_name = role.split(":", 1)[1]
            normalized.append({"role": "user", "content": f"[Intern {intern_name}, in the studio]: {content}"})
        elif role == "host" or role == "user":
            normalized.append({"role": "user", "content": f"[Host Luke]: {content}"})
        else:
            normalized.append(msg)
    return normalized


_DEVON_PATTERN = r"\b(devon|devin|deven|devyn|devan|devlin|devvon)\b"

def _is_addressed_to_devon(text: str) -> bool:
    """Check if the host is talking to Devon based on first few words.
    Handles common voice-to-text misspellings."""
    t = text.strip().lower()
    if re.match(rf"^(hey |yo |ok |okay )?{_DEVON_PATTERN}", t):
        return True
    return False


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with current caller"""
    if not session.caller:
        raise HTTPException(400, "No active call")

    # Check if host is talking to Devon instead of the caller
    if _is_addressed_to_devon(request.text):
        # Strip Devon prefix and route to intern
        stripped = re.sub(rf"^(?:hey |yo |ok |okay )?{_DEVON_PATTERN}[,:\s]*", "", request.text.strip(), flags=re.IGNORECASE).strip()
        if not stripped:
            stripped = "what's up?"

        # Add host message to conversation so caller hears it happened
        session.add_message("user", request.text)

        result = await intern_service.ask(
            question=stripped,
            conversation_context=session.conversation,
            caller_active=True,
        )
        devon_text = result.get("text", "")
        if devon_text:
            session.add_message(f"intern:{intern_service.name}", devon_text)
            broadcast_event("intern_response", {"text": devon_text, "intern": intern_service.name})
            asyncio.create_task(_play_intern_audio(devon_text))

        return {
            "routed_to": "devon",
            "text": devon_text or "Uh... give me a sec.",
            "sources": result.get("sources", []),
        }

    epoch = _session_epoch
    session.add_message("user", request.text)
    # session._research_task = asyncio.create_task(_background_research(request.text))

    async with _ai_response_lock:
        if _session_epoch != epoch:
            raise HTTPException(409, "Call ended while waiting")

        # Stop any playing caller audio so responses don't overlap
        audio_service.stop_caller_audio()

        show_history = session.get_show_history()
        is_wrapping = session._wrapping_up
        mood = detect_host_mood(session.conversation, wrapping_up=is_wrapping)

        # Track wrap-up exchanges and force hangup after 2
        if is_wrapping:
            session._wrapup_exchanges += 1
            if session._wrapup_exchanges > 2:
                mood += "\nSay goodbye NOW and end with [HANGUP]\n"

        slim_caller = session.caller_backgrounds.get(session.current_caller_key, {})
        system_prompt = get_caller_prompt(slim_caller)

        max_tokens, max_sentences = _pick_response_budget(wrapping_up=is_wrapping)
        messages = _normalize_messages_for_llm(session.conversation[-_dynamic_context_window():])
        _caller_name = session.caller.get("name", "") if session.caller else ""
        _model_override = None  # caller_dialog category routes to haiku-4.5
        response = await llm_service.generate(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            category="caller_dialog",
            caller_name=_caller_name,
            model_override=_model_override,
        )
        response = await _retry_if_too_short(
            response, llm_service, messages, system_prompt, max_tokens,
            _caller_name, _model_override, wrapping_up=is_wrapping)
        if not is_wrapping and response and "[HANGUP]" not in response and _has_repetition(response, session.conversation):
            print(f"[Chat] Repetition detected, retrying with anti-repetition prompt...")
            retry_messages = messages + [{"role": "user", "content": "You're repeating yourself. Say something NEW — a detail you haven't mentioned, a different angle, or move the story forward. Do not repeat facts you've already stated."}]
            retry_response = await llm_service.generate(
                messages=retry_messages, system_prompt=system_prompt,
                max_tokens=max_tokens, category="caller_dialog",
                caller_name=_caller_name, model_override=_model_override,
            )
            if retry_response and not _has_repetition(retry_response, session.conversation):
                print(f"[Chat] Anti-repetition retry succeeded")
                response = retry_response
            else:
                print(f"[Chat] Anti-repetition retry no better, keeping original")

    # Discard if call changed while we were generating
    if _session_epoch != epoch:
        print(f"[Chat] Discarding stale response (epoch {epoch} → {_session_epoch})")
        raise HTTPException(409, "Call changed during response")

    print(f"[Chat] Raw LLM ({max_tokens}tok/{max_sentences}s): {response[:100] if response else '(empty)'}...")

    # Clean response for TTS (remove parenthetical actions, asterisks, etc.)
    response = clean_for_tts(response, formal=False)
    response = _trim_to_sentences(response, max_sentences)
    response = ensure_complete_thought(response)

    # Detect [HANGUP] sentinel — caller wants to end the call
    caller_hangup = "[HANGUP]" in response
    if caller_hangup:
        response = response.replace("[HANGUP]", "").strip()
        session._caller_hangup = True
        print(f"[Chat] Caller hangup detected")

    print(f"[Chat] Cleaned: {response[:100] if response else '(empty)'}...")

    # Ensure we have a valid response
    if not response or not response.strip():
        response = "Uh... sorry, what was that?"

    session.add_message("assistant", response)

    result = {
        "text": response,
        "caller": session.caller["name"],
        "voice_id": session.caller["voice"]
    }
    if caller_hangup:
        result["hangup"] = True
    return result


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Generate and play speech on caller output device (non-blocking)"""
    if not request.text or not request.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    epoch = _session_epoch

    try:
        audio_bytes = await generate_speech(
            request.text,
            request.voice_id,
            "none"
        )
    except Exception as e:
        print(f"[TTS] Failed: {e}")
        broadcast_event("ai_done")
        raise HTTPException(503, f"TTS generation failed: {e}")

    # Don't play if call changed during TTS generation
    if _session_epoch != epoch:
        return {"status": "discarded", "duration": 0}

    # Stop any existing audio before playing new
    audio_service.stop_caller_audio()

    # Play in background thread - returns immediately, can be interrupted by hangup
    thread = threading.Thread(
        target=audio_service.play_caller_audio,
        args=(audio_bytes, 24000),
        daemon=True
    )
    thread.start()

    # Also stream to active real callers so they hear the AI
    if session.active_real_caller:
        caller_id = session.active_real_caller["caller_id"]
        asyncio.create_task(
            caller_service.stream_audio_to_caller(caller_id, audio_bytes, 24000)
        )

    return {"status": "playing", "duration": len(audio_bytes) / 2 / 24000}


@app.post("/api/tts/stop")
async def stop_tts():
    """Stop any playing caller audio"""
    audio_service.stop_caller_audio()
    return {"status": "stopped"}


# --- Music Endpoints ---

GENRE_KEYWORDS = {
    "rock": "Rock",
    "funk": "Funk",
    "funky": "Funk",
    "hip-hop": "Hip-Hop",
    "hip hop": "Hip-Hop",
    "rap": "Hip-Hop",
    "jazz": "Jazz",
    "blues": "Blues",
    "latin": "Latin",
    "lo-fi": "Lo-Fi",
    "lofi": "Lo-Fi",
    "coffee": "Lo-Fi",
    "radio": "Radio",
    "valentine": "Ballad",
    "romantic": "Ballad",
    "ballad": "Ballad",
    "irish": "Irish",
    "ireland": "Irish",
    "patricks": "Irish",
    "ambient": "Ambient",
    "chill": "Chill",
    "acoustic": "Acoustic",
    "classical": "Classical",
    "piano": "Classical",
    "country": "Country",
    "western": "Country",
    "electronic": "Electronic",
    "synth": "Electronic",
}


def _detect_genre(name: str) -> str:
    lower = name.lower()
    for keyword, genre in GENRE_KEYWORDS.items():
        if keyword in lower:
            return genre
    return "Other"


@app.get("/api/music")
async def get_music():
    """Get available music tracks, shuffled and tagged with genre"""
    tracks = []
    if settings.music_dir.exists():
        for ext in ['*.wav', '*.mp3', '*.flac']:
            for f in settings.music_dir.glob(ext):
                tracks.append({
                    "name": f.stem,
                    "file": f.name,
                    "path": str(f),
                    "genre": _detect_genre(f.stem),
                })
    random.shuffle(tracks)
    return {
        "tracks": tracks,
        "playing": audio_service.is_music_playing()
    }


@app.post("/api/music/play")
async def play_music(request: MusicRequest):
    """Load and play a music track, crossfading if already playing"""
    track_path = settings.music_dir / request.track
    if not track_path.exists():
        raise HTTPException(404, "Track not found")

    if audio_service.is_music_playing():
        audio_service.crossfade_to(str(track_path))
    else:
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

SFX_DISPLAY_NAMES = {
    "airhorn": "📢 Airhorn",
    "applause": "👏 Applause",
    "boo": "👎 Boo",
    "busy": "📞 Busy",
    "buzzer": "🚫 Buzzer",
    "car_crash": "💥 Car Crash",
    "cheer": "✅ Correct",
    "clock_ticking": "⏰ Clock Ticking",
    "commercial_jingle": "🎵 Jingle",
    "crickets": "🦗 Crickets",
    "ding": "🔔 Ding",
    "doorbell": "🚪 Doorbell",
    "drumroll": "🥁 Drumroll",
    "dun_dun_dun": "😱 Dun Dun Dun",
    "explosion": "💣 Explosion",
    "fart": "💨 Fart",
    "gasp": "😮 Gasp",
    "glass_shatter": "🪟 Glass Shatter",
    "hangup": "📵 Hangup",
    "hold_music": "🎶 Hold Music",
    "laugh_track": "😂 Laugh Track",
    "news_stinger": "📰 News Stinger",
    "phone_ring": "☎️ Phone Ring",
    "record_scratch": "💿 Record Scratch",
    "rimshot": "🪘 Rimshot",
    "sad_trombone": "😢 Sad Trombone",
    "thunder": "⛈️ Thunder",
    "victory_fanfare": "🏆 Victory Fanfare",
    "whoosh": "🌀 Whoosh",
    "wolf_whistle": "😏 Wolf Whistle",
}
SFX_PRIORITY = ["sad_trombone", "cheer"]

@app.get("/api/sounds")
async def get_sounds():
    """Get available sound effects"""
    sounds = []
    if settings.sounds_dir.exists():
        for f in settings.sounds_dir.glob('*.wav'):
            sounds.append({
                "name": SFX_DISPLAY_NAMES.get(f.stem, f.stem),
                "file": f.name,
                "path": str(f)
            })
    priority_set = {p + ".wav" for p in SFX_PRIORITY}
    priority = [s for p in SFX_PRIORITY for s in sounds if s["file"] == p + ".wav"]
    rest = sorted([s for s in sounds if s["file"] not in priority_set], key=lambda s: s["name"])
    return {"sounds": priority + rest}


@app.post("/api/sfx/play")
async def play_sfx(request: SFXRequest):
    """Play a sound effect"""
    sound_path = settings.sounds_dir / request.sound
    if not sound_path.exists():
        raise HTTPException(404, "Sound not found")

    audio_service.play_sfx(str(sound_path))
    return {"status": "playing", "sound": request.sound}


# --- Ads Endpoints ---

AD_DISPLAY_NAMES = {
    "bettermaybe_ad": "Better Maybe",
    "bunkhousedns_ad": "Bunkhouse DNS",
    "cryptono_ad": "CryptoNo",
    "desertgut_ad": "Desert Gut",
    "enema_ad": "Enema",
    "jamhospitalityad": "Jam Hospitality",
    "mealprep_ad": "Meal Prep",
    "mediocrecpap": "Mediocre CPAP",
    "pillowforever_ad": "Pillow Forever",
    "placiboleaf": "Placibo Leaf",
    "saddlesoft_ad": "Saddle Soft",
    "sandstone_ad": "Sandstone",
    "scriptdrift_ad": "Script Drift",
    "shoespraycoad": "Shoe Spray Co.",
    "squarehole_ad": "Square Hole",
    "therapy_ad": "Therapy",
    "vpnad": "VPN",
}


@app.get("/api/ads")
async def get_ads():
    """Get available ad tracks, shuffled"""
    ad_list = []
    if settings.ads_dir.exists():
        for ext in ['*.wav', '*.mp3', '*.flac']:
            for f in settings.ads_dir.glob(ext):
                ad_list.append({
                    "name": AD_DISPLAY_NAMES.get(f.stem, f.stem),
                    "file": f.name,
                    "path": str(f)
                })
    random.shuffle(ad_list)
    return {"ads": ad_list}


@app.post("/api/ads/play")
async def play_ad(request: MusicRequest):
    """Play an ad once on the ad channel (ch 11)"""
    ad_path = settings.ads_dir / request.track
    if not ad_path.exists():
        raise HTTPException(404, "Ad not found")

    if audio_service._music_playing:
        audio_service.stop_music(fade_duration=1.0)
        await asyncio.sleep(1.1)
    audio_service.play_ad(str(ad_path))
    return {"status": "playing", "track": request.track}


@app.post("/api/ads/stop")
async def stop_ad():
    """Stop ad playback"""
    audio_service.stop_ad()
    return {"status": "stopped"}


# --- Idents Endpoints ---

IDENT_DISPLAY_NAMES = {}


@app.get("/api/idents")
async def get_idents():
    """Get available ident tracks, shuffled"""
    ident_list = []
    if settings.idents_dir.exists():
        for ext in ['*.wav', '*.mp3', '*.flac']:
            for f in settings.idents_dir.glob(ext):
                ident_list.append({
                    "name": IDENT_DISPLAY_NAMES.get(f.stem, f.stem),
                    "file": f.name,
                    "path": str(f)
                })
    random.shuffle(ident_list)
    return {"idents": ident_list}


@app.post("/api/idents/play")
async def play_ident(request: MusicRequest):
    """Play an ident once on the ad channel (ch 11)"""
    ident_path = settings.idents_dir / request.track
    if not ident_path.exists():
        raise HTTPException(404, "Ident not found")

    if audio_service._music_playing:
        audio_service.stop_music(fade_duration=1.0)
        await asyncio.sleep(1.1)
    audio_service.play_ident(str(ident_path))
    return {"status": "playing", "track": request.track}


@app.post("/api/idents/stop")
async def stop_ident():
    """Stop ident playback"""
    audio_service.stop_ident()
    return {"status": "stopped"}


# --- LLM Settings Endpoints ---

@app.get("/api/settings")
async def get_settings():
    """Get LLM settings"""
    return await llm_service.get_settings_async()


@app.post("/api/settings")
async def update_settings(data: dict):
    """Update LLM and TTS settings"""
    old_tts = settings.tts_provider
    llm_service.update_settings(
        provider=data.get("provider"),
        openrouter_model=data.get("openrouter_model"),
        ollama_model=data.get("ollama_model"),
        ollama_host=data.get("ollama_host"),
        tts_provider=data.get("tts_provider"),
        category_models=data.get("category_models")
    )
    # Re-randomize voices when TTS provider changes voice system
    new_tts = settings.tts_provider
    if new_tts != old_tts:
        old_is_el = old_tts == "elevenlabs"
        new_is_el = new_tts == "elevenlabs"
        if old_is_el != new_is_el:
            _randomize_callers()
            print(f"[Settings] TTS changed {old_tts} → {new_tts}, re-randomized voices")
    return llm_service.get_settings()


# --- Show Theme ---

@app.get("/api/show-theme")
async def get_show_theme():
    return {"theme": session.show_theme}


@app.post("/api/show-theme")
async def set_show_theme(data: dict):
    theme = data.get("theme", "").strip()[:100]
    old_theme = session.show_theme
    session.show_theme = theme
    if theme:
        print(f"[Theme] Show theme set: {theme}")
    elif old_theme:
        print(f"[Theme] Show theme cleared (was: {old_theme})")

    # Regenerate backgrounds for unused callers so theme gets baked in
    if theme and theme != old_theme:
        used_keys = set()
        if session.current_caller_key:
            used_keys.add(session.current_caller_key)
        for record in session.call_history:
            for key, base in CALLER_BASES.items():
                if base.get("name") == record.caller_name:
                    used_keys.add(key)
                    break
        unused_keys = [k for k in CALLER_BASES if k not in used_keys]
        if unused_keys:
            asyncio.create_task(_regenerate_backgrounds_for_keys(unused_keys))
            print(f"[Theme] Regenerating backgrounds for {len(unused_keys)} unused callers")

    return {"theme": session.show_theme}


# --- Cost Tracking Endpoints ---

@app.get("/api/costs")
async def get_costs():
    """Get live cost summary"""
    return cost_tracker.get_live_summary()


@app.get("/api/costs/report")
async def get_cost_report():
    """Get full cost report with breakdowns and recommendations"""
    return cost_tracker.generate_report()


# --- Cost Dashboard Endpoints ---

@app.get("/api/costs/summary")
async def get_cost_summary(period: str = "all"):
    return cost_db.get_summary(period)


@app.get("/api/costs/timeline")
async def get_cost_timeline(period: str = "all", group_by: str = "session"):
    return cost_db.get_timeline(period, group_by)


@app.get("/api/costs/models")
async def get_cost_models(period: str = "all"):
    return cost_db.get_models(period)


@app.get("/api/costs/categories")
async def get_cost_categories(period: str = "all"):
    return cost_db.get_categories(period)


@app.get("/api/costs/sessions")
async def get_cost_sessions(period: str = "all"):
    return cost_db.get_sessions_list(period)


@app.get("/api/costs/session/{session_id}")
async def get_cost_session_detail(session_id: str):
    detail = cost_db.get_session_detail(session_id)
    if not detail:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    return detail


@app.get("/api/costs/expensive")
async def get_cost_expensive_calls(period: str = "all", limit: int = 10):
    return cost_db.get_expensive_calls(period, limit)


# --- Caller Screening ---

SCREENING_PROMPT = """You are a friendly, brief phone screener for "Luke at the Roost" radio show.
Your job: Get the caller's first name and what they want to talk about. That's it.

Rules:
- Be warm but brief (1-2 sentences per response)
- First ask their name, then ask what they want to talk about
- After you have both, say something like "Great, sit tight and we'll get you on with Luke!"
- Never pretend to be Luke or the host
- Keep it casual and conversational
- If they're hard to understand, ask them to repeat"""

_screening_audio_buffers: dict[str, bytearray] = {}


async def _start_screening_greeting(caller_id: str):
    """Send initial screening greeting to queued caller after brief delay"""
    await asyncio.sleep(2)  # Wait for stream to stabilize

    ws = caller_service._websockets.get(caller_id)
    if not ws:
        return

    caller_service.start_screening(caller_id)
    greeting = "Hey there! Thanks for calling Luke at the Roost. What's your name?"
    caller_service.update_screening(caller_id, screener_text=greeting)

    try:
        audio_bytes = await generate_speech(greeting, "Sarah", "none")
        if audio_bytes:
            await caller_service.stream_audio_to_caller(caller_id, audio_bytes, 24000)
    except Exception as e:
        print(f"[Screening] Greeting TTS failed: {e}")


async def _handle_screening_audio(caller_id: str, pcm_data: bytes, sample_rate: int):
    """Process audio from a queued caller for screening conversation"""
    state = caller_service.get_screening_state(caller_id)
    if not state or state["status"] == "complete":
        return

    # Skip if TTS is currently streaming to this caller
    if caller_service.is_streaming_tts(caller_id):
        return

    # Transcribe caller speech
    try:
        text = await transcribe_audio(pcm_data, source_sample_rate=sample_rate,
                                      context_hint="A caller is being screened before going on air.")
    except Exception as e:
        print(f"[Screening] Transcription failed: {e}")
        return

    if not text or not text.strip():
        return

    print(f"[Screening] Caller {caller_id}: {text}")
    caller_service.update_screening(caller_id, caller_text=text)

    # Build conversation for LLM
    messages = []
    for msg in state["conversation"]:
        role = "assistant" if msg["role"] == "screener" else "user"
        messages.append({"role": role, "content": msg["content"]})

    # Generate screener response
    try:
        response = await llm_service.generate(
            messages=messages,
            system_prompt=SCREENING_PROMPT,
            category="screener",
        )
    except Exception as e:
        print(f"[Screening] LLM failed: {e}")
        return

    if not response or not response.strip():
        return

    response = response.strip()
    print(f"[Screening] Screener → {caller_id}: {response}")
    caller_service.update_screening(caller_id, screener_text=response)

    # After 2+ caller responses, try to extract name and topic
    if state["response_count"] >= 2:
        try:
            extract_prompt = f"""From this screening conversation, extract the caller's name and topic.
Conversation:
{chr(10).join(f'{m["role"]}: {m["content"]}' for m in state["conversation"])}

Respond with ONLY JSON: {{"name": "their first name or null", "topic": "brief topic or null"}}"""
            extract = await llm_service.generate(
                messages=[{"role": "user", "content": extract_prompt}],
                system_prompt="You extract structured data from conversations. Respond with only valid JSON.",
                category="screener",
            )
            json_match = re.search(r'\{[^}]+\}', extract)
            if json_match:
                info = json.loads(json_match.group())
                if info.get("name"):
                    caller_service.update_screening(caller_id, caller_name=info["name"])
                if info.get("topic"):
                    caller_service.update_screening(caller_id, topic=info["topic"])
                if info.get("name") and info.get("topic"):
                    caller_service.end_screening(caller_id)
                    broadcast_event("screening_complete", {
                        "caller_id": caller_id,
                        "name": info["name"],
                        "topic": info["topic"]
                    })
        except Exception as e:
            print(f"[Screening] Extract failed: {e}")

    # TTS the screener response back to caller
    try:
        audio_bytes = await generate_speech(response, "Sarah", "none")
        if audio_bytes:
            await caller_service.stream_audio_to_caller(caller_id, audio_bytes, 24000)
    except Exception as e:
        print(f"[Screening] Response TTS failed: {e}")

    # Start hold music after screening completes and final TTS has played
    screening = caller_service.get_screening_state(caller_id)
    if screening and screening.get("status") == "complete" and caller_id not in _hold_music_tasks:
        _hold_music_tasks[caller_id] = asyncio.create_task(_stream_hold_music(caller_id))


@app.websocket("/api/signalwire/stream")
async def signalwire_audio_stream(websocket: WebSocket):
    """Handle SignalWire bidirectional audio stream"""
    await websocket.accept()

    caller_id = str(uuid.uuid4())[:8]
    caller_phone = "Unknown"
    call_sid = ""
    audio_buffer = bytearray()
    screening_buffer = bytearray()
    CHUNK_DURATION_S = 3
    SAMPLE_RATE = 16000
    chunk_samples = CHUNK_DURATION_S * SAMPLE_RATE
    stream_started = False

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            raw = message.get("text")
            if not raw:
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            event = msg.get("event")

            if event == "start":
                custom = msg.get("start", {}).get("customParameters", {})
                caller_phone = custom.get("caller_phone", "Unknown")
                call_sid = custom.get("call_sid", "")
                stream_sid = msg.get("start", {}).get("streamSid", "")

                stream_started = True
                print(f"[SignalWire WS] Stream started: {caller_phone} (CallSid: {call_sid}, StreamSid: {stream_sid})")

                caller_service.add_to_queue(caller_id, caller_phone)
                caller_service.register_websocket(caller_id, websocket)
                broadcast_event("caller_queued", {"phone": caller_phone})
                broadcast_chat("System", f"{caller_phone} is waiting in the queue")

                ring_sound = settings.sounds_dir / "phone_ring.wav"
                if ring_sound.exists():
                    threading.Thread(target=audio_service.play_sfx, args=(str(ring_sound),), daemon=True).start()

                if call_sid:
                    caller_service.register_call_sid(caller_id, call_sid)
                if stream_sid:
                    caller_service.register_stream_sid(caller_id, stream_sid)

                # Start screening conversation
                asyncio.create_task(_start_screening_greeting(caller_id))

            elif event == "media" and stream_started:
                try:
                    payload = msg.get("media", {}).get("payload", "")
                    if not payload:
                        continue

                    pcm_data = base64.b64decode(payload)

                    call_info = caller_service.active_calls.get(caller_id)
                    if not call_info:
                        # Caller is queued, not on air — route to screening
                        screening_buffer.extend(pcm_data)
                        if len(screening_buffer) >= chunk_samples * 2:
                            pcm_chunk = bytes(screening_buffer[:chunk_samples * 2])
                            screening_buffer = screening_buffer[chunk_samples * 2:]
                            audio_check = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                            if np.abs(audio_check).max() >= 0.01:
                                asyncio.create_task(
                                    _handle_screening_audio(caller_id, pcm_chunk, SAMPLE_RATE)
                                )
                        continue

                    audio_buffer.extend(pcm_data)
                    audio_service.route_real_caller_audio(pcm_data, SAMPLE_RATE)

                    if len(audio_buffer) >= chunk_samples * 2:
                        pcm_chunk = bytes(audio_buffer[:chunk_samples * 2])
                        audio_buffer = audio_buffer[chunk_samples * 2:]
                        # Skip transcription if audio is silent
                        audio_check = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                        if np.abs(audio_check).max() < 0.01:
                            continue
                        asyncio.create_task(
                            _safe_transcribe(caller_id, pcm_chunk, SAMPLE_RATE)
                        )
                except Exception as e:
                    print(f"[SignalWire WS] Media frame error (non-fatal): {e}")
                    continue  # Skip bad frame, don't disconnect caller

            elif event == "stop":
                print(f"[SignalWire WS] Stream stop event received: {caller_phone} (caller_id: {caller_id})")
                break

    except WebSocketDisconnect:
        on_air = caller_id in caller_service.active_calls
        tts_active = caller_service.is_streaming_tts(caller_id)
        started_at = caller_service.active_calls.get(caller_id, {}).get("started_at")
        duration = f"{time.time() - started_at:.0f}s" if started_at else "n/a"
        print(f"[SignalWire WS] DROPPED: {caller_id} ({caller_phone}) on_air={on_air} tts_active={tts_active} duration={duration}")
        disconnect_reason = "dropped"
    except Exception as e:
        print(f"[SignalWire WS] Error: {e}")
        traceback.print_exc()
        disconnect_reason = f"error: {e}"
    else:
        disconnect_reason = "clean"
    finally:
        _stop_hold_music(caller_id)
        was_on_air = caller_id in caller_service.active_calls
        caller_service.unregister_websocket(caller_id)
        caller_service.unregister_call_sid(caller_id)
        caller_service.unregister_stream_sid(caller_id)
        caller_service.remove_from_queue(caller_id)
        if was_on_air:
            caller_service.hangup(caller_id)
            if session.active_real_caller and session.active_real_caller.get("caller_id") == caller_id:
                session.active_real_caller = None
            broadcast_event("caller_disconnected", {"phone": caller_phone, "reason": disconnect_reason})
            broadcast_chat("System", f"{caller_phone} disconnected ({disconnect_reason})")

            drop_sound = settings.sounds_dir / ("busy.wav" if disconnect_reason == "dropped" else "hangup.wav")
            if drop_sound.exists():
                threading.Thread(target=audio_service.play_sfx, args=(str(drop_sound),), daemon=True).start()
        elif stream_started:
            broadcast_chat("System", f"{caller_phone} left the queue")
        if audio_buffer and caller_id in caller_service.active_calls:
            asyncio.create_task(
                _safe_transcribe(caller_id, bytes(audio_buffer), SAMPLE_RATE)
            )


async def _safe_transcribe(caller_id: str, pcm_chunk: bytes, sample_rate: int):
    """Wrapper that catches transcription errors so they don't crash anything"""
    try:
        await _handle_real_caller_transcription(caller_id, pcm_chunk, sample_rate)
    except Exception as e:
        print(f"[Transcription] Error (non-fatal): {e}")


# --- Host Audio Broadcast ---

_host_audio_queue: asyncio.Queue = None
_host_audio_task: asyncio.Task = None


async def _host_audio_sender():
    """Persistent task that drains audio queue, batches frames, and sends to callers"""
    _send_count = [0]
    try:
      while True:
        pcm_bytes = await _host_audio_queue.get()
        if caller_service.is_streaming_tts_any():
            continue

        # Drain all available frames and concatenate
        chunks = [pcm_bytes]
        while not _host_audio_queue.empty():
            try:
                extra = _host_audio_queue.get_nowait()
                if not caller_service.is_streaming_tts_any():
                    chunks.append(extra)
            except asyncio.QueueEmpty:
                break

        combined = b''.join(chunks)
        t0 = time.time()
        for caller_id in list(caller_service.active_calls.keys()):
            try:
                await caller_service.send_audio_to_caller(caller_id, combined, 16000)
            except Exception:
                pass
        elapsed = time.time() - t0
        _send_count[0] += 1
        if _send_count[0] % 20 == 0:
            qsize = _host_audio_queue.qsize()
            audio_ms = len(combined) / 2 / 16000 * 1000
            print(f"[HostAudio] send took {elapsed*1000:.0f}ms, {len(chunks)} chunks batched ({audio_ms:.0f}ms audio), queue: {qsize}")
    except asyncio.CancelledError:
        print("[HostAudio] Sender task cancelled")
    except Exception as e:
        print(f"[HostAudio] Sender task error: {e}")


def _start_host_audio_sender():
    """Start the persistent host audio sender task"""
    global _host_audio_queue, _host_audio_task
    if _host_audio_queue is None:
        _host_audio_queue = asyncio.Queue(maxsize=50)
    if _host_audio_task is None or _host_audio_task.done():
        _host_audio_task = asyncio.create_task(_host_audio_sender())


def _host_audio_sync_callback(pcm_bytes: bytes):
    """Sync callback from audio thread — push to queue for async sending"""
    if _host_audio_queue is None:
        return
    try:
        _host_audio_queue.put_nowait(pcm_bytes)
    except asyncio.QueueFull:
        pass  # Drop frame rather than block


# --- Queue Endpoints ---

@app.get("/api/queue")
async def get_call_queue():
    """Get list of callers waiting in queue"""
    return {"queue": caller_service.get_queue()}


@app.post("/api/queue/take/{caller_id}")
async def take_call_from_queue(caller_id: str):
    """Take a caller off hold and put them on air"""
    _stop_hold_music(caller_id)
    try:
        call_info = caller_service.take_call(caller_id)
    except ValueError as e:
        raise HTTPException(404, str(e))

    session.active_real_caller = {
        "caller_id": call_info["caller_id"],
        "channel": call_info["channel"],
        "phone": call_info["phone"],
    }

    return {
        "status": "on_air",
        "caller": call_info,
    }


@app.post("/api/queue/drop/{caller_id}")
async def drop_from_queue(caller_id: str):
    """Drop a caller from the queue"""
    _stop_hold_music(caller_id)
    call_sid = caller_service.get_call_sid(caller_id)
    caller_service.remove_from_queue(caller_id)
    if call_sid:
        await _signalwire_end_call(call_sid)
    return {"status": "dropped"}


_auto_respond_pending: asyncio.Task | None = None
_auto_respond_buffer: list[str] = []


async def _handle_real_caller_transcription(caller_id: str, pcm_data: bytes, sample_rate: int):
    """Transcribe a chunk of real caller audio and add to conversation"""
    global _auto_respond_pending

    call_info = caller_service.active_calls.get(caller_id)
    if not call_info:
        return

    caller_phone = call_info["phone"]
    context_hint = f"A real caller ({caller_phone}) is talking to host Luke on the radio."
    text = await transcribe_audio(pcm_data, source_sample_rate=sample_rate, context_hint=context_hint)
    if not text or not text.strip():
        return
    print(f"[Real Caller] {caller_phone}: {text}")

    # Add to conversation and broadcast to frontend
    session.add_message(f"real_caller:{caller_phone}", text)
    broadcast_chat(f"{caller_phone} (caller)", text)

    # If AI auto-respond mode is on and an AI caller is active, debounce auto-respond
    if session.ai_respond_mode == "auto" and session.current_caller_key:
        _auto_respond_buffer.append(text)
        # Cancel any pending auto-respond timer and restart it
        if _auto_respond_pending and not _auto_respond_pending.done():
            _auto_respond_pending.cancel()
        _auto_respond_pending = asyncio.create_task(_debounced_auto_respond(caller_phone))


async def _debounced_auto_respond(caller_phone: str):
    """Wait for caller to stop talking (4s pause), then trigger AI response"""
    try:
        await asyncio.sleep(4)  # Wait 4 seconds of silence
    except asyncio.CancelledError:
        return  # More speech came in, timer restarted

    # Gather accumulated text
    accumulated = " ".join(_auto_respond_buffer)
    _auto_respond_buffer.clear()

    if not accumulated.strip():
        return

    print(f"[Auto-Respond] Caller paused. Accumulated: {accumulated[:100]}...")
    await _trigger_ai_auto_respond(accumulated)


async def _trigger_ai_auto_respond(accumulated_text: str):
    """Generate AI caller response to accumulated real caller speech"""
    epoch = _session_epoch

    if not session.caller:
        return

    if _ai_response_lock.locked():
        return

    # Cooldown check
    if not hasattr(session, '_last_ai_auto_respond'):
        session._last_ai_auto_respond = 0
    if time.time() - session._last_ai_auto_respond < 5:
        return

    ai_name = session.caller["name"]

    async with _ai_response_lock:
        if _session_epoch != epoch:
            return  # Call changed while waiting for lock

        print(f"[Auto-Respond] {ai_name} is jumping in...")
        session._last_ai_auto_respond = time.time()
        audio_service.stop_caller_audio()
        broadcast_event("ai_status", {"text": f"{ai_name} is thinking..."})

        show_history = session.get_show_history()
        is_wrapping = session._wrapping_up
        mood = detect_host_mood(session.conversation, wrapping_up=is_wrapping)
        if is_wrapping:
            session._wrapup_exchanges += 1
            if session._wrapup_exchanges > 2:
                mood += "\nSay goodbye NOW and end with [HANGUP]\n"
        slim_caller = session.caller_backgrounds.get(session.current_caller_key, {})
        system_prompt = get_caller_prompt(slim_caller)

        max_tokens, max_sentences = _pick_response_budget(wrapping_up=is_wrapping)
        messages = _normalize_messages_for_llm(session.conversation[-_dynamic_context_window():])
        _caller_name = session.caller.get("name", "") if session.caller else ""
        _model_override = None  # caller_dialog category routes to haiku-4.5
        response = await llm_service.generate(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            category="caller_dialog",
            caller_name=_caller_name,
            model_override=_model_override,
        )
        response = await _retry_if_too_short(
            response, llm_service, messages, system_prompt, max_tokens,
            _caller_name, _model_override, wrapping_up=is_wrapping)
        if not is_wrapping and response and "[HANGUP]" not in response and _has_repetition(response, session.conversation):
            print(f"[Auto-Respond] Repetition detected, retrying...")
            retry_messages = messages + [{"role": "user", "content": "You're repeating yourself. Say something NEW — a detail you haven't mentioned, a different angle, or move the story forward. Do not repeat facts you've already stated."}]
            retry_response = await llm_service.generate(
                messages=retry_messages, system_prompt=system_prompt,
                max_tokens=max_tokens, category="caller_dialog",
                caller_name=_caller_name, model_override=_model_override,
            )
            if retry_response and not _has_repetition(retry_response, session.conversation):
                print(f"[Auto-Respond] Anti-repetition retry succeeded")
                response = retry_response
            else:
                print(f"[Auto-Respond] Anti-repetition retry no better, keeping original")

    # Discard if call changed during generation
    if _session_epoch != epoch:
        print(f"[Auto-Respond] Discarding stale response (epoch {epoch} → {_session_epoch})")
        broadcast_event("ai_done")
        return

    response = clean_for_tts(response, formal=False)
    response = _trim_to_sentences(response, max_sentences)
    response = ensure_complete_thought(response)

    # Detect [HANGUP] sentinel
    caller_hangup = "[HANGUP]" in response
    if caller_hangup:
        response = response.replace("[HANGUP]", "").strip()
        session._caller_hangup = True
        print(f"[Auto-Respond] Caller hangup detected")

    if not response or not response.strip():
        broadcast_event("ai_done")
        return

    # Final staleness check before playing audio
    if _session_epoch != epoch:
        broadcast_event("ai_done")
        return

    session.add_message(f"ai_caller:{ai_name}", response)
    broadcast_chat(ai_name, response)

    broadcast_event("ai_status", {"text": f"{ai_name} is speaking..."})
    _caller_bg = session.caller_backgrounds.get(session.current_caller_key) or {}
    _emotional_register = _caller_bg.get("emotional_register", "") if isinstance(_caller_bg, dict) else ""
    try:
        audio_bytes = await generate_speech(response, session.caller["voice"], "none",
                                            provider_override=session.caller.get("tts_provider"),
                                            emotional_register=_emotional_register)
    except Exception as e:
        print(f"[Auto-Respond] TTS failed: {e}")
        broadcast_event("ai_done")
        return

    # Don't play if call changed during TTS generation
    if _session_epoch != epoch:
        print(f"[Auto-Respond] Discarding stale TTS (epoch {epoch} → {_session_epoch})")
        broadcast_event("ai_done")
        return

    thread = threading.Thread(
        target=audio_service.play_caller_audio,
        args=(audio_bytes, 24000),
        daemon=True,
    )
    thread.start()

    broadcast_event("ai_done")

    # Signal caller hangup to frontend
    if caller_hangup:
        broadcast_event("caller_hangup", {"caller": ai_name})

    # Also stream to active real caller so they hear the AI
    if session.active_real_caller:
        caller_id = session.active_real_caller["caller_id"]
        asyncio.create_task(
            caller_service.stream_audio_to_caller(caller_id, audio_bytes, 24000)
        )


@app.post("/api/ai-respond")
async def ai_respond():
    """Trigger AI caller to respond based on current conversation"""
    if not session.caller:
        raise HTTPException(400, "No active AI caller")

    epoch = _session_epoch

    async with _ai_response_lock:
        if _session_epoch != epoch:
            raise HTTPException(409, "Call ended while waiting")

        audio_service.stop_caller_audio()

        show_history = session.get_show_history()
        is_wrapping = session._wrapping_up
        mood = detect_host_mood(session.conversation, wrapping_up=is_wrapping)
        if is_wrapping:
            session._wrapup_exchanges += 1
            if session._wrapup_exchanges > 2:
                mood += "\nSay goodbye NOW and end with [HANGUP]\n"
        slim_caller = session.caller_backgrounds.get(session.current_caller_key, {})
        system_prompt = get_caller_prompt(slim_caller)

        max_tokens, max_sentences = _pick_response_budget(wrapping_up=is_wrapping)
        messages = _normalize_messages_for_llm(session.conversation[-_dynamic_context_window():])
        _caller_name = session.caller.get("name", "") if session.caller else ""
        _model_override = None  # caller_dialog category routes to haiku-4.5
        response = await llm_service.generate(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            category="caller_dialog",
            caller_name=_caller_name,
            model_override=_model_override,
        )
        response = await _retry_if_too_short(
            response, llm_service, messages, system_prompt, max_tokens,
            _caller_name, _model_override, wrapping_up=is_wrapping)
        if not is_wrapping and response and "[HANGUP]" not in response and _has_repetition(response, session.conversation):
            print(f"[Chat] Repetition detected, retrying with anti-repetition prompt...")
            retry_messages = messages + [{"role": "user", "content": "You're repeating yourself. Say something NEW — a detail you haven't mentioned, a different angle, or move the story forward. Do not repeat facts you've already stated."}]
            retry_response = await llm_service.generate(
                messages=retry_messages, system_prompt=system_prompt,
                max_tokens=max_tokens, category="caller_dialog",
                caller_name=_caller_name, model_override=_model_override,
            )
            if retry_response and not _has_repetition(retry_response, session.conversation):
                print(f"[Chat] Anti-repetition retry succeeded")
                response = retry_response
            else:
                print(f"[Chat] Anti-repetition retry no better, keeping original")

    if _session_epoch != epoch:
        raise HTTPException(409, "Call changed during response")

    response = clean_for_tts(response, formal=False)
    response = _trim_to_sentences(response, max_sentences)
    response = ensure_complete_thought(response)

    # Detect [HANGUP] sentinel
    caller_hangup = "[HANGUP]" in response
    if caller_hangup:
        response = response.replace("[HANGUP]", "").strip()
        session._caller_hangup = True
        print(f"[AI-Respond] Caller hangup detected")

    if not response or not response.strip():
        response = "Uh... sorry, what was that?"

    # Snapshot caller info before it can be cleared by a concurrent hangup
    caller = session.caller
    if not caller:
        raise HTTPException(409, "Call ended")
    ai_name = caller["name"]
    ai_voice = caller["voice"]
    ai_tts_provider = caller.get("tts_provider")
    _caller_bg = session.caller_backgrounds.get(session.current_caller_key) or {}
    ai_emotional_register = _caller_bg.get("emotional_register", "") if isinstance(_caller_bg, dict) else ""

    # TTS — outside the lock so other requests aren't blocked
    try:
        audio_bytes = await generate_speech(response, ai_voice, "none",
                                            provider_override=ai_tts_provider,
                                            emotional_register=ai_emotional_register)
    except Exception as e:
        print(f"[AI-Respond] TTS failed: {e}")
        broadcast_event("ai_done")
        return {"text": response, "caller": ai_name, "tts_error": str(e)}

    # Add message AFTER successful TTS so ghost messages don't pollute conversation
    session.add_message(f"ai_caller:{ai_name}", response)

    if _session_epoch != epoch:
        raise HTTPException(409, "Call changed during TTS")

    thread = threading.Thread(
        target=audio_service.play_caller_audio,
        args=(audio_bytes, 24000),
        daemon=True,
    )
    thread.start()

    # Stream to real caller
    if session.active_real_caller:
        caller_id = session.active_real_caller["caller_id"]
        asyncio.create_task(
            caller_service.stream_audio_to_caller(caller_id, audio_bytes, 24000)
        )

    result = {
        "text": response,
        "caller": ai_name,
        "voice_id": session.caller["voice"]
    }
    if caller_hangup:
        result["hangup"] = True
    return result


# --- Follow-Up & Session Control Endpoints ---

@app.post("/api/hangup/real")
async def hangup_real_caller():
    """Hang up on real caller — disconnect immediately, summarize in background"""
    global _session_epoch, _auto_respond_pending
    if not session.active_real_caller:
        raise HTTPException(400, "No active real caller")

    _session_epoch += 1

    # Cancel any pending auto-respond
    if _auto_respond_pending and not _auto_respond_pending.done():
        _auto_respond_pending.cancel()
        _auto_respond_pending = None
    _auto_respond_buffer.clear()

    if session._research_task and not session._research_task.done():
        session._research_task.cancel()
        session._research_task = None

    caller_id = session.active_real_caller["caller_id"]
    caller_phone = session.active_real_caller["phone"]
    conversation_snapshot = list(session.conversation)
    call_started = getattr(session, '_call_started_at', 0.0)
    auto_followup_enabled = session.auto_followup

    # End the phone call via SignalWire
    call_sid = caller_service.get_call_sid(caller_id)
    caller_service.hangup(caller_id)
    if call_sid:
        asyncio.create_task(_signalwire_end_call(call_sid))

    session.active_real_caller = None

    hangup_sound = settings.sounds_dir / "hangup.wav"
    if hangup_sound.exists():
        threading.Thread(target=audio_service.play_sfx, args=(str(hangup_sound),), daemon=True).start()

    asyncio.create_task(
        _summarize_real_call(caller_phone, conversation_snapshot, call_started, auto_followup_enabled)
    )

    return {
        "status": "disconnected",
        "caller": caller_phone,
    }


async def _summarize_real_call(caller_phone: str, conversation: list, started_at: float, auto_followup_enabled: bool):
    """Background task: summarize call and store in history"""
    ended_at = time.time()
    summary = ""
    if conversation:
        transcript_text = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in conversation
        )
        summary = await llm_service.generate(
            messages=[{"role": "user", "content": f"Summarize this radio show call in 1-2 sentences:\n{transcript_text}"}],
            system_prompt="You summarize radio show conversations concisely. Focus on what the caller talked about and any emotional moments.",
            category="call_summary",
            caller_name=caller_phone,
        )

    quality_signals = _assess_call_quality(conversation)
    session.call_quality_signals.append(quality_signals)
    session.call_history.append(CallRecord(
        caller_type="real",
        caller_name=caller_phone,
        summary=summary,
        transcript=conversation,
        started_at=started_at,
        ended_at=ended_at,
        quality_signals=quality_signals,
    ))
    print(f"[Real Caller] {caller_phone} call summarized: {summary[:80]}...")
    print(f"[Quality] {caller_phone}: exchanges={quality_signals['exchange_count']} avg_len={quality_signals['avg_response_length']:.0f}c host_engagement={quality_signals['host_engagement']} caller_depth={quality_signals['caller_depth']} natural_end={quality_signals['natural_ending']}")

    _save_checkpoint()

    if auto_followup_enabled:
        await _auto_followup(summary)


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
        category="followup_pick",
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


# --- Intern (Devon) Endpoints ---

@app.post("/api/intern/ask")
async def intern_ask(data: dict):
    """Host asks Devon to look something up"""
    question = data.get("question", "").strip()
    if not question:
        raise HTTPException(400, "No question provided")

    # Run research + response (non-blocking for the caller audio)
    result = await intern_service.ask(
        question=question,
        conversation_context=session.conversation if session.conversation else None,
        caller_active=session.caller is not None,
    )

    text = result.get("text", "")
    if not text:
        return {"text": None, "sources": []}

    # Add to conversation log
    session.add_message(f"intern:{intern_service.name}", text)
    broadcast_event("intern_response", {"text": text, "intern": intern_service.name})

    # TTS — play Devon's voice on air (no phone filter, in-studio)
    asyncio.create_task(_play_intern_audio(text))

    return {
        "text": text,
        "sources": result.get("sources", []),
        "intern": intern_service.name,
    }


@app.post("/api/intern/interject")
async def intern_interject():
    """Manually trigger Devon to comment on current conversation"""
    if not session.conversation:
        raise HTTPException(400, "No active conversation")

    result = await intern_service.interject(session.conversation, caller_active=session.caller is not None)
    if not result:
        return {"text": None}

    text = result["text"]
    session.add_message(f"intern:{intern_service.name}", text)
    broadcast_event("intern_response", {"text": text, "intern": intern_service.name})

    asyncio.create_task(_play_intern_audio(text))

    return {
        "text": text,
        "sources": result.get("sources", []),
        "intern": intern_service.name,
    }


@app.post("/api/intern/monitor")
async def intern_monitor(data: dict):
    """Toggle Devon's auto-monitoring on/off"""
    enabled = data.get("enabled", True)
    session.intern_monitoring = enabled

    if enabled:
        async def _on_suggestion(text, sources):
            broadcast_event("intern_suggestion", {"text": text, "sources": sources})

        intern_service.start_monitoring(
            get_conversation=lambda: session.conversation,
            on_suggestion=_on_suggestion,
        )
    else:
        intern_service.stop_monitoring()

    print(f"[Intern] Monitoring: {enabled}")
    return {"monitoring": enabled}


@app.get("/api/intern/suggestion")
async def intern_suggestion():
    """Get Devon's pending suggestion (if any)"""
    suggestion = intern_service.get_pending_suggestion()
    return {"suggestion": suggestion}


@app.post("/api/intern/suggestion/play")
async def intern_play_suggestion():
    """Approve and play Devon's pending suggestion on air"""
    suggestion = intern_service.get_pending_suggestion()
    if not suggestion:
        raise HTTPException(400, "No pending suggestion")

    text = suggestion["text"]
    intern_service.dismiss_suggestion()

    session.add_message(f"intern:{intern_service.name}", text)
    broadcast_event("intern_response", {"text": text, "intern": intern_service.name})

    asyncio.create_task(_play_intern_audio(text))

    return {"text": text, "intern": intern_service.name}


@app.post("/api/intern/suggestion/dismiss")
async def intern_dismiss_suggestion():
    """Dismiss Devon's pending suggestion"""
    intern_service.dismiss_suggestion()
    return {"dismissed": True}


async def _play_intern_audio(text: str):
    """Generate TTS for Devon and play on air (no phone filter, own stem + channel)"""
    try:
        audio_bytes = await generate_speech(
            text, intern_service.voice, apply_filter=False
        )
        thread = threading.Thread(
            target=audio_service.play_caller_audio,
            args=(audio_bytes, 24000),
            kwargs={"stem_name": "devon", "channel_override": audio_service.devon_channel},
            daemon=True,
        )
        thread.start()
    except Exception as e:
        print(f"[Intern] TTS failed: {e}")


# --- Avatars ---

@app.get("/api/avatar/{name}")
async def get_avatar(name: str):
    """Serve a caller's avatar image"""
    path = avatar_service.get_path(name)
    if path:
        return FileResponse(path, media_type="image/jpeg")
    # Try to fetch on the fly — find gender from CALLER_BASES
    gender = "male"
    for base in CALLER_BASES.values():
        if base.get("name") == name:
            gender = base.get("gender", "male")
            break
    try:
        path = await avatar_service.get_or_fetch(name, gender)
        return FileResponse(path, media_type="image/jpeg")
    except Exception:
        raise HTTPException(404, "Avatar not found")


# --- Transcript & Chapter Export ---

@app.get("/api/session/export")
async def export_session():
    """Export session transcript with speaker labels and chapters from call boundaries"""
    if not session.call_history:
        raise HTTPException(400, "No calls in this session to export")

    # Find the earliest call start as session base time
    session_start = min(
        (r.started_at for r in session.call_history if r.started_at > 0),
        default=time.time()
    )

    transcript_lines = []
    chapters = []

    for i, record in enumerate(session.call_history):
        # Chapter from call start time
        offset_seconds = max(0, record.started_at - session_start) if record.started_at > 0 else 0
        chapter_title = f"{record.caller_name}"
        if record.summary:
            # Use first sentence of summary for chapter title
            short_summary = record.summary.split(".")[0].strip()
            if short_summary:
                chapter_title += f" \u2014 {short_summary}"
        chapters.append({"startTime": round(offset_seconds), "title": chapter_title})

        # Separator between calls
        if i > 0:
            transcript_lines.append("")
            transcript_lines.append(f"--- Call {i + 1}: {record.caller_name} ---")
            transcript_lines.append("")

        # Transcript lines with timestamps
        for msg in record.transcript:
            msg_offset = msg.get("timestamp", 0) - session_start if msg.get("timestamp") else offset_seconds
            if msg_offset < 0:
                msg_offset = 0
            mins = int(msg_offset // 60)
            secs = int(msg_offset % 60)

            role = msg.get("role", "")
            if role in ("user", "host"):
                speaker = "HOST"
            elif role.startswith("real_caller:"):
                speaker = role.split(":", 1)[1].upper()
            elif role.startswith("ai_caller:"):
                speaker = role.split(":", 1)[1].upper()
            elif role == "assistant":
                speaker = record.caller_name.upper()
            else:
                speaker = role.upper()

            transcript_lines.append(f"[{mins:02d}:{secs:02d}] {speaker}: {msg['content']}")

    return {
        "session_id": session.id,
        "transcript": "\n".join(transcript_lines),
        "chapters": chapters,
        "call_count": len(session.call_history),
    }


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


# --- Stem Recording ---

@app.post("/api/recording/toggle")
async def toggle_stem_recording():
    """Toggle recording on/off. Also toggles on-air state."""
    global _show_on_air
    if audio_service.stem_recorder is None:
        # START recording
        from datetime import datetime
        dir_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        recordings_dir = Path("recordings") / dir_name
        import sounddevice as sd
        device_info = sd.query_devices(audio_service.output_device) if audio_service.output_device is not None else None
        sr = int(device_info["default_samplerate"]) if device_info else 48000
        recorder = StemRecorder(recordings_dir, sample_rate=sr)
        recorder.start()
        audio_service.stem_recorder = recorder
        audio_service.start_stem_mic()
        add_log(f"Stem recording started -> {recordings_dir}")
        if not _show_on_air:
            _show_on_air = True
            _start_host_audio_sender()
            audio_service.start_host_stream(_host_audio_sync_callback)
            threading.Thread(target=_update_on_air_cdn, args=(True,), daemon=True).start()
            threading.Thread(target=_start_ngrok, daemon=True).start()
            add_log("Show auto-set to ON AIR")
        return {"on_air": _show_on_air, "recording": True, "caller_line_ready": _caller_line_ready}
    # STOP recording
    audio_service.stop_stem_mic()
    stems_dir = audio_service.stem_recorder.output_dir
    paths = audio_service.stem_recorder.stop()
    audio_service.stem_recorder = None
    add_log(f"Stem recording stopped. Running post-production...")

    # Save cost report for this session
    session_id = stems_dir.name
    cost_report_path = Path("data/cost_reports") / f"session-{session_id}.json"
    cost_tracker.save(cost_report_path)
    summary = cost_tracker.get_live_summary()
    add_log(f"Session costs: ${summary['total_cost_usd']:.4f} "
            f"(LLM: ${summary['llm_cost_usd']:.4f}, TTS: ${summary['tts_cost_usd']:.4f}, "
            f"{summary['total_llm_calls']} calls, {summary['total_tokens']} tokens)")
    by_cat = summary.get("by_category", {})
    if by_cat:
        breakdown = ", ".join(f"{k}: ${v['cost']:.4f}/{v['calls']}calls" for k, v in sorted(by_cat.items(), key=lambda x: x[1]["cost"], reverse=True))
        add_log(f"Cost breakdown: {breakdown}")

    if _show_on_air:
        _show_on_air = False
        audio_service.stop_host_stream()
        threading.Thread(target=_update_on_air_cdn, args=(False,), daemon=True).start()
        threading.Thread(target=_stop_ngrok, daemon=True).start()
        add_log("Show auto-set to OFF AIR")

    # Auto-run postprod in background
    import subprocess, sys
    python = sys.executable
    output_file = stems_dir / "episode.mp3"
    def _run_postprod():
        try:
            result = subprocess.run(
                [python, "postprod.py", str(stems_dir), "-o", "episode.mp3"],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                add_log(f"Post-production complete -> {output_file}")
            else:
                add_log(f"Post-production failed: {result.stderr[:300]}")
        except Exception as e:
            add_log(f"Post-production error: {e}")

    threading.Thread(target=_run_postprod, daemon=True).start()
    return {"on_air": _show_on_air, "recording": False, "caller_line_ready": _caller_line_ready}


@app.post("/api/recording/process")
async def process_stems(stems_dir: str):
    import subprocess
    stems_path = Path(stems_dir).resolve()
    allowed_root = Path("recordings").resolve()
    if not str(stems_path).startswith(str(allowed_root)):
        raise HTTPException(403, "Path must be under the recordings/ directory")
    if not stems_path.exists():
        raise HTTPException(404, f"Directory not found: {stems_dir}")
    output_file = stems_path / "episode.mp3"
    try:
        result = subprocess.run(
            ["python", "postprod.py", str(stems_path), "-o", str(output_file)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise HTTPException(500, f"Processing failed: {result.stderr}")
        add_log(f"Post-production complete -> {output_file}")
        return {"status": "done", "output": str(output_file)}
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Processing timed out")
