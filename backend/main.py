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
from fastapi.responses import FileResponse
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


# --- Structured Caller Background (must be defined before functions that use it) ---
@dataclass
class CallerBackground:
    name: str
    age: int
    gender: str
    job: str
    location: str | None
    reason_for_calling: str
    pool_name: str
    communication_style: str
    energy_level: str              # low / medium / high / very_high
    emotional_state: str           # nervous, excited, angry, vulnerable, calm, etc.
    signature_detail: str          # The memorable thing about them
    situation_summary: str         # 1-sentence summary for other callers to reference
    natural_description: str       # 3-5 sentence prose for the prompt
    seeds: list[str] = field(default_factory=list)
    verbal_fluency: str = "medium"
    calling_from: str = ""


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

# Voices to never assign to callers (annoying, bad quality, etc.)
BLACKLISTED_VOICES = {"Evelyn", "Sebastian", "Celeste"}  # Sebastian reserved for Silas


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
        returning = regular_caller_service.get_returning_callers(2)
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

# Background components for dynamic generation
JOBS_MALE = [
    # Trades & blue collar
    "runs a small HVAC business", "works as a long-haul trucker", "works construction",
    "is a union electrician", "owns a small landscaping company", "drives for UPS",
    "is a welder at a shipyard", "works as a diesel mechanic", "does roofing",
    "is a plumber, runs his own crew", "works at a grain elevator", "is a ranch hand",
    # Service & public
    "is a paramedic", "is a cop, 12 years on the force", "is a firefighter",
    "works as a hospital security guard", "is a corrections officer", "drives a city bus",
    # Food & hospitality
    "is a line cook at a decent restaurant", "runs a food truck", "manages a bar",
    "works the night shift at a gas station", "delivers pizza, has for years",
    # White collar & tech
    "works IT for the city", "is an insurance adjuster, hates it", "is a bank teller",
    "does accounting for a small firm", "sells cars at a dealership", "works in a call center",
    "is a project manager at a mid-size company", "works in logistics",
    # Creative & education
    "teaches middle school history",
    "is a freelance photographer", "is a session musician", "is a tattoo artist",
    "works at a brewery", "is a youth pastor", "does standup comedy on the side",
    # Healthcare (not just women)
    "works as an ER nurse, been doing it 10 years", "is a home health aide",
    "is a physical therapist", "works as an EMT",
    # Odd & specific
    "works at a pawn shop", "is a repo man", "runs a junkyard", "is a locksmith",
    "works overnight stocking shelves", "is a pest control guy", "drives a tow truck",
    "is a bouncer at a club", "works at a cemetery", "is a crop duster pilot",
    "manages a storage facility", "is a hunting guide", "works on an oil rig, two weeks on two off",
]

JOBS_FEMALE = [
    # Healthcare
    "works as an ER nurse", "is a dental hygienist", "is a vet tech",
    "works in hospital billing", "is a home health aide", "is a phlebotomist",
    "works as a traveling nurse", "is a midwife",
    # Service & public
    "works as a 911 dispatcher", "is a social worker", "works retail management",
    "works as a bartender at a dive bar", "is a flight attendant",
    "manages a restaurant", "works the front desk at a hotel",
    # Trades & blue collar (not just men)
    "works as a diesel mechanic, learned from her dad", "is an electrician, runs her own jobs",
    "drives a long-haul truck, been on the road for years", "works construction management",
    "is a welder, one of two women at the shop",
    # Education & office
    "teaches kindergarten", "is a paralegal", "is an accountant at a small firm",
    "works in HR", "is a court reporter", "does data entry from home",
    "is a school bus driver", "works at the DMV",
    # Creative & entrepreneurial
    "is a hairstylist, owns her chair", "runs a small bakery",
    "runs a daycare out of her home", "is a real estate agent",
    "is a wedding planner", "does nails, has a loyal clientele",
    "sells stuff on Etsy full time", "is a dog groomer",
    # Odd & specific
    "works at a truck stop diner", "is a bail bonds agent",
    "works at a tribal casino", "manages a laundromat",
    "works overnight at a group home", "is a park ranger",
    "drives an ambulance", "works at a thrift store",
    "is a taxidermist", "cleans houses, runs her own business",
    "works at a gun range", "is a cop, five years on",
    "works the night shift at Waffle House", "is a funeral home director",
]

PROBLEMS = [
    # Family drama
    "hasn't talked to their father in years and just got a voicemail from a number they didn't recognize — turns out it was their dad's new wife asking them to come say goodbye before the surgery",
    "got a bill in the mail for $14,000 from a hospital in a city they've never been to — for a surgery under their name and social security number that happened three weeks ago",
    "is being pressured to take care of an aging parent who was never there for them",
    "found their dad's second driver's license with a different name while cleaning out his truck after he died — and the address on it is a house forty minutes away with a family in it",
    "caught their brother selling tools from their dead father's workshop on Facebook Marketplace and when they confronted him he said 'dad would've wanted me to have the money'",
    "saw their estranged daughter's wedding photos on Facebook — outdoor ceremony, beautiful dress, the whole thing — and realized nobody told them it happened",
    "came home to find their landlord had entered their apartment and rearranged the furniture — not stolen anything, just moved everything six inches to the left, and now denies it happened",
    "is watching their parents' marriage fall apart after 40 years",
    "their kid just got arrested and they don't know what to do",
    "found out their teenager has been lying about where they go at night",
    "their in-laws are trying to take over their life and their spouse won't say anything",

    # Career and purpose
    "walked out of their job today after 15 years with no plan and is sitting in their truck in a parking lot",
    "got passed over for a promotion they trained their replacement for and their boss didn't even tell them personally",
    "found their old demo tapes in a box in the garage and spent all night listening to how good they used to be",
    "makes six figures but just sat in their driveway for 45 minutes tonight because they couldn't make themselves go inside",
    "got fired today by email — not even a phone call — after seven years at the company",
    "their boss asked them to sign off on safety reports they know are falsified and gave them a week to decide",
    "watched their boss present their project to the board word-for-word as his own idea and nobody batted an eye",
    "sunk their retirement into a restaurant that's been open three months and they're already hemorrhaging money",
    "got offered their dream job in Portland but their kid just started high school and their spouse said absolutely not",
    "found out they're getting laid off next Friday but can't tell anyone at work because they signed an NDA",
    "pulled up their coworker's offer letter by accident on a shared drive and found out they make $40k more for the same title",

    # Money and survival
    "is drowning in debt and can't see a way out",
    "just found out their spouse has been hiding massive credit card debt",
    "lost their savings in a bad investment and is too ashamed to tell anyone",
    "can't make rent and is about to be evicted",
    "lent a family member a ton of money and they won't pay it back",
    "is working three jobs and still barely making it",
    "inherited money and it's tearing the family apart",
    "their car broke down and they can't afford to fix it and need it for work",

    # Health scares
    "got told today they have MS and they're sitting in their car in the hospital parking lot trying to figure out how to tell their family",
    "found a lump three months ago and has been too scared to go to the doctor — their partner doesn't know",
    "their wife was just diagnosed with stage 3 breast cancer and he had to hold it together for the kids all day and is falling apart now",
    "collapsed at work last week and the ER doctor told them their blood pressure is so high they could stroke out any day — they're 38",
    "has been to four different doctors for back pain and they all say nothing's wrong but they can barely walk some days",
    "just got back from the fertility clinic and found out their chances are basically zero — they've been trying for three years",

    # Mental health and inner struggles
    "had a full-blown panic attack at the grocery store today and had to leave their cart in the aisle",
    "just turned 50 and realized they've spent their whole adult life doing what other people expected",
    "got blackout drunk at their kid's birthday party last weekend and nobody's said a word about it",
    "lied on their resume to get their current job and just found out there's an audit coming",
    "everyone comes to them with their problems and last night they screamed at their kid over nothing and realized they're running on empty",
    "woke up on the kitchen floor at 3am and doesn't remember how they got there — this is the third time",
    "hasn't answered a phone call in two months and just lets everything go to voicemail",
    "their therapist dropped them as a patient and they don't know what they did wrong",
    "caught themselves crying in the work bathroom for the second time this week over absolutely nothing they can identify",

    # Grief and loss
    "their mom died six months ago and they just found a voicemail from her they never listened to",
    "their dad has Alzheimer's and called them by their dead brother's name tonight",
    "best friend from high school died in a car wreck last month and they hadn't talked in three years because of a stupid argument",
    "their buddy from the service is in hospice and they can't bring themselves to visit",
    "their dog of 14 years died yesterday and they called in sick to work because they can't stop crying",
    "their house burned down three weeks ago and they just found out insurance won't cover it because of a technicality",

    # Regrets and past mistakes
    "turned down a chance to buy into a business 10 years ago that's now worth millions",
    "said something horrible to their mom the last time they saw her alive and she died two days later",
    "ran into their ex from 20 years ago at the store today and realized they never got over them",
    "gave up a music scholarship to stay home and take care of family and just heard the kid they lost the spot to is touring nationally",
    "found out from a classmate reunion that the kid they bullied in school attempted suicide because of them",
    "got a DUI last month and their mugshot is on the local news site — everyone in town has seen it",
    "ghosted the best friend they ever had over a $200 debt and just saw them posting about battling cancer",

    # Relationships
    "realized at dinner tonight that they haven't had a real conversation with their spouse in months — they just sat there in silence for an hour",
    "their spouse's family keeps telling them they married the wrong person and last Thanksgiving someone actually said it to their face",
    "their wife planned a whole vacation without asking them and booked it — they found out from a credit card notification",
    "has been sleeping in separate rooms for four months 'because of snoring' but neither of them is buying that excuse anymore",
    "caught feelings for a coworker and is terrified because they've been married for 12 years and have three kids",
    "found out their partner has been sending money to an ex every month for the past two years",
    "found their partner's Tinder profile — with photos from their vacation together — and hasn't said anything yet",
    "has been seeing two people for six months and both just asked them to move in on the same week",
    "their ex showed up at their job today with flowers and their current partner saw the whole thing",
    "signed a lease with someone after dating for three weeks and now they realize this person is a completely different human behind closed doors",

    # Friendship and loneliness
    "threw a birthday party last weekend and only one person showed up — and they left early",
    "overheard their best friend of 20 years making fun of them at a barbecue last Saturday",
    "moved to town two years ago and still doesn't know a single person well enough to call in an emergency",
    "found a group text between friends that they weren't included in, planning stuff they were never invited to",
    "went to their 20-year reunion and realized everyone moved on except them",
    "their best friend started dating their ex-wife and just asked them to be cool about it",

    # Neighbor and community drama
    "is in a feud with their neighbor that's gotten way out of hand",
    "found out something sketchy is going on next door and doesn't know if they should say something",
    "got into it with someone at their kid's school and now it's a whole thing",
    "someone at church said something that made them question their entire faith",

    # Big life decisions
    "packed a bag tonight and it's sitting by the front door — they're one argument away from driving to their sister's in Tucson and not coming back",
    "their aging mother needs full-time care and the only option is selling their house to pay for it, but their kids grew up there",
    "got accepted to nursing school at 44 and their spouse said if they quit their job to go back to school it's over",
    "found out they're pregnant and they already have four kids and no money",
    "their kid just enlisted in the Marines without telling anyone and ships out in three weeks",
    "has been sitting on a resignation letter for two months and every morning they almost turn it in",

    # Addiction and bad habits
    "poured out every bottle in the house last night at 2am and woke up this morning and drove straight to the liquor store",
    "lost $8,000 at the casino last weekend and told their wife they got robbed at a gas station",
    "went to pick up their kid from school today and realized they were still drunk from the night before",
    "just did 90 days in rehab and came home to find their roommate left a six-pack in the fridge as a 'welcome home'",
    "found oxy in their 16-year-old's backpack and recognized the pills because they used to take the same ones",

    # Legal trouble
    "is being sued by their former business partner for $200k and just got served at their daughter's soccer game",
    "got caught shoplifting at Walmart — not because they needed to, they have no idea why they did it — and now they have a court date",
    "in a custody battle where their ex is telling the kids daddy doesn't want them and the judge seems to be buying it",
    "has had a warrant for a missed court date for six months and tonight a deputy showed up at their neighbor's house asking about them",

    # Attraction and affairs
    "has been seeing {affair_person} for months and something happened tonight that means it's about to come out — they need to figure out what to do before morning",

    # Sexual/desire
    "their partner found their browser history and now they have to have a conversation they've been avoiding for years",
    "went to a party last month and something happened that made them realize they might not be as straight as they thought",

    # General late-night confessions
    "found a letter their dead father wrote but never sent — it's addressed to a woman who isn't their mother",
    "accidentally overheard their doctor on the phone saying something about their test results and now they're spiraling",
    "has been stealing from the register at work for eight months and the new cameras just went up today",
    "saw their spouse's car parked outside someone's house at 11pm when they said they were working late",
    "ran into someone from their past at the gas station today — someone who knows something about them that nobody else does",
    "yelled at their kid tonight the exact same way their father used to yell at them, same words and everything",

    # Weird situations
    "found a camera hidden in their Airbnb rental and confronted the owner who denied it — they have it in their hand right now",
    "their neighbor's been flying a drone over their property every evening and when they asked about it the guy said 'just keeping an eye on things'",
    "accidentally sent a text trashing their boss to their boss instead of their friend and now they're waiting for Monday",
    "woke up to find someone had left a box of their stuff on their porch — stuff they lost years ago — with no note",
    "their smart doorbell recorded someone standing on their porch at 3am just staring at the door for ten minutes then walking away",
    "found a journal in a used car they bought and it's full of detailed entries about a life falling apart — they can't stop reading it",
    "a stranger at a gas station handed them an envelope with $500 and said 'you look like you need this more than me' and drove off",

    # Parenting nightmares
    "their 14-year-old came home with a tattoo and won't say where they got it",
    "found out their kid has been bullying other kids at school and the principal wants a meeting tomorrow",
    "their adult child just moved back home for the third time and their spouse is ready to change the locks",
    "caught their 17-year-old sneaking out at 2am and followed them — they were going to a house where adults were partying",
    "their kid just told them they're dropping out of college to become a TikTok creator and they already quit their campus job",
    "got a call from the school that their kid punched a teacher — kid says the teacher grabbed them first",
    "their 19-year-old just announced they're engaged to someone they met online three weeks ago",
    "found out their kid has been skipping school for two months and the school never called",

    # Money situations with stakes
    "just got a $14,000 medical bill in the mail for a procedure insurance said was covered",
    "their business partner cleaned out the company account and disappeared — they found out when payroll bounced",
    "won $50,000 on a scratch ticket and hasn't told a soul yet because they know everyone's going to come asking",
    "co-signed a loan for their brother who stopped making payments six months ago and now the bank is coming after them",
    "found out the house they just bought has a lien on it that the seller didn't disclose",
    "their identity got stolen and someone opened three credit cards in their name and ran up $30k",
    "just calculated that they've spent over $60,000 on their kid's college and the kid just failed out",
    "the IRS sent them a letter saying they owe $22,000 from three years ago and they have no idea what it's about",

    # Confrontations
    "finally told their mother-in-law to stop coming over unannounced and now their spouse isn't speaking to them",
    "got into a screaming match with their landlord in front of the whole apartment complex about black mold they've been complaining about for a year",
    "punched their brother-in-law at a family barbecue last weekend and now the family is picking sides",
    "told their best friend that their spouse is cheating on them and the friend chose the spouse's side",
    "stood up to their abusive father for the first time at age 40 and now half the family says they're the problem",
    "reported their employer to OSHA and now everyone at work knows it was them",

    # Small town drama
    "the only mechanic in town ripped them off and when they posted about it on Facebook the whole town turned on them",
    "caught the mayor's son dumping trash on their property and when they reported it the sheriff told them to drop it",
    "someone spray-painted something on their garage door and they know exactly who did it but can't prove it",
    "got banned from the only bar in town for something they didn't do and now they have nowhere to go on weekends",
    "their ex started dating the bartender at their regular spot and now they have to find a new place to drink",
    "someone's been stealing packages off their porch and they set up a camera and it's their neighbor's kid",

    # Moral dilemmas
    "found a wallet with $3,000 in cash and the ID of someone who lives in their town — been sitting on it for two days",
    "knows their best friend's husband is cheating because they saw him at a restaurant in Las Cruces with someone, and the friend just posted about their anniversary",
    "their coworker confessed to stealing from the company and asked them to keep quiet — it's been eating at them for weeks",
    "witnessed a hit and run in a parking lot and got the plate number but the car belongs to someone they know from church",
    "their elderly neighbor asked them to help write a new will leaving everything to a caregiver they just met — something feels off",
    "found out their landlord has been renting to them without permits and now they have leverage but also might lose their home",

    # Identity and life changes
    "just turned 60 and realized they have no hobbies, no friends outside work, and retire in five years with nothing to do",
    "their spouse of 20 years just came out and is asking to stay together as co-parents",
    "caught their elderly neighbor burying something in the backyard at 2am and now they can't decide whether to ask about it or call someone",
    "moved back to their hometown after 25 years and doesn't recognize anything or anyone",
    "just became a grandparent and it's bringing up every mistake they made as a parent",
    "retired three months ago and has called their old office twice pretending to need something just to talk to someone",

    # Work situations
    "found out their company is about to do a massive layoff and their name is on the list — they saw the spreadsheet on their manager's screen",
    "their new boss is 25 years old and just told the whole team they need to 'align on synergies' and they almost quit on the spot",
    "has been working remote for two years and just got told to come back to the office or lose their job — office is 90 minutes away",
    "accidentally replied-all to a company email with something they definitely should not have said",
    "their coworker has been taking credit for their work for a year and just got promoted because of it",
    "found evidence that their company has been dumping waste illegally and doesn't know whether to report it or keep their head down",

    # Relationships with specific incidents
    "found a second phone in their partner's car hidden under the seat — it's locked and they haven't mentioned it yet",
    "their partner proposed in front of both families at Thanksgiving and they said yes but they don't want to marry them",
    "woke up to their partner packing a bag at 4am — they said they're 'going to their mom's for a few days' but wouldn't make eye contact",
    "their ex just published a thinly-veiled novel where they're clearly the villain and it's getting good reviews locally",
    "caught their partner in a lie about where they were last Tuesday and the story keeps changing every time they ask",
    "their partner just told them they gambled away their vacation savings — the trip is in two weeks",

    # Unexpected discoveries
    "was cleaning out their dead uncle's house and found a room full of journals describing a completely different life than anyone knew about",
    "got pulled over and the cop ran their plates and told them the truck was reported stolen — it's their truck, they bought it cash from a guy in a Walmart parking lot two years ago and never got it titled",
    "their kid's school project about family history turned up the fact that their grandfather was someone fairly notorious",
    "discovered that the 'family cabin' they've been going to for 30 years actually belongs to a stranger who never knew they were using it",
    "found their late mother's journal and the last entry is about a decision she made that contradicts everything she ever told them about why she left their father",
    "found out the house they grew up in is about to be demolished and it hit them way harder than they expected",

    # Animal situations
    "their neighbor's dog bit their kid and the neighbor says the kid provoked it — now animal control is involved",
    "found a stray dog three months ago, nursed it back to health, and now the original owner showed up wanting it back",
    "their HOA says they have to get rid of their chickens and they're ready to go to war over it",
    "hit a deer on the highway tonight and it's still alive on the shoulder and they don't know what to do",
    "their cat brought home a wallet and they found a missing person flyer with the wallet owner's face on it",

    # Veterans and service
    "got a letter from the VA denying their disability claim for the third time and they can barely function some days",
    "ran into their old sergeant at the hardware store and had a full panic attack in the parking lot afterward",
    "their civilian friends keep thanking them for their service and they want to scream because they feel like they failed the people next to them",
    "just found out the buddy they served with who they lost contact with died two years ago and nobody told them",

    # Technology and modern life
    "their kid showed them a deepfake video of them saying things they never said and it's circulating at school",
    "got catfished for four months and sent the person $3,000 before figuring it out",
    "their teenager posted something online that went viral for the wrong reasons and now strangers are showing up at their house",
    "found out their ex has been tracking their location through a shared app they forgot to turn off",
    "someone made a fake social media profile using their photos and has been messaging people they know",

    # Layered / morally ambiguous / weird-but-real
    "has been pretending to be a widower for sympathy at a grief support group but they actually just got divorced — and now they've made real friends there and don't know how to come clean",
    "accidentally got cc'd on an email chain where their entire friend group is planning an intervention for them and they don't think they have a problem",
    "their therapist ran into them at a bar and they had a totally normal conversation for 20 minutes before it got weird — now they feel like they can't go back to sessions",
    "has been writing letters to their dead wife every week for three years and mailing them to her old address — the new tenant just wrote back",
    "overheard two coworkers in the break room planning to frame a third coworker for something they did — now they have to decide whether to get involved",
    "works as a 911 dispatcher and took a call last week from someone in a situation almost identical to one they went through — they froze up and can't stop replaying it",
    "has been tipping a waitress at a diner $100 every Friday for a year because she reminds them of their daughter they haven't spoken to — the waitress just asked them why",
    "found out the guy they've been playing online chess with every night for two years is their estranged brother — recognized a phrase he used in the chat",
    "coached their kid's little league team to an undefeated season but just found out the other parents have been complaining they're too intense and the league isn't renewing them",
    "got a thank-you card from someone they don't remember — it says they saved their life ten years ago at a gas station in Tucson and they have no memory of it",
    "has been secretly paying their adult kid's rent for six months because they're too proud to admit they're struggling — spouse just found the bank statements",
    "went to their high school reunion and the person who bullied them for four years came up and apologized in tears — and they felt nothing, which scares them more than the bullying did",
    "started a small business selling furniture they build by hand and just got a huge order from a company that turns out to be owned by their ex-wife's new husband",
    "volunteers at a soup kitchen every Saturday and just realized one of the regulars is their old college roommate who ghosted everyone 15 years ago",
    "kept their grandmother's house exactly the way she left it after she died — they go there and sit in her chair every Sunday — and now their siblings want to sell it",
    "has been lying about being bilingual on their resume for years and just got assigned to lead a project in Mexico City next month",
    "ran a red light last month and caused a fender bender — nobody was hurt but they drove off, and now they keep seeing the other car around town with the damage they caused",
    "their elderly neighbor asked them to be their emergency contact because they have no family — it's been six months and they're basically this person's whole support system now and it's a lot",
    "found their dad's old ham radio in the attic, got it working, and has been talking to strangers at 2am — one of them just said something that makes them think it's someone they know",
    "won a local chili cookoff with their dead mother's recipe and now everyone wants it — but sharing it feels like giving away the last private thing they have of hers",

    # Secrets and double lives
    "has been pretending to go to work every day for three weeks but they actually got fired — they sit in their car at the library until 5pm",
    "won a radio contest for a free vacation and brought their partner — except the resort lost the reservation and the only room left is a honeymoon suite, and now their partner thinks it's a proposal setup and is acting weird",
    "has been living under a fake name for 15 years and their spouse doesn't know their real one",
    "their spouse thinks they're sober but they've been keeping a bottle in their truck toolbox and drinking in parking lots after work",
    "has been telling everyone they went to college but they dropped out after one semester — now their kid wants to go to the same school",
    "got a DM from someone claiming to be their father's other kid — there are apparently four of them across three states",

    # Escalated neighbor/community situations
    "their neighbor built a fence six inches onto their property and when they brought it up the guy pulled out a surveyor's report that might actually prove it's his land",
    "woke up to find their truck on blocks with all four tires stolen and the security camera footage shows their cousin's boyfriend doing it",
    "someone has been leaving dead animals on their porch once a week for a month and the cops say there's nothing they can do",
    "their neighbor has been running an unlicensed daycare with 15 kids and the noise is destroying their life but calling the city feels wrong",
    "got into a road rage incident and the other driver followed them home — now they see the same truck drive past their house every night",

    # Workplace chaos
    "walked in on their boss crying in the bathroom and now the boss won't make eye contact with them and they think they're about to get fired for it",
    "accidentally discovered their company has been billing clients for work that was never done and they have the receipts on a USB drive in their glove box",
    "their coworker died on the job last month and the company hasn't changed a single safety protocol — they're scared to go back in",
    "just found out the 'charity' their company donates to every year is a shell company owned by the CEO's wife",
    "has been sleeping in their office for two weeks because they can't afford an apartment in the city they transferred to",
    "got promoted to manage their former peers and two of them have made their life a living hell since — one of them is their best friend",

    # Unhinged confessions
    "has been going to open houses every weekend for three years pretending to be a buyer — it's the only time they feel like they have a future",
    "ate their roommate's leftovers and found a note in the fridge the next day that said 'I know it was you. This isn't over.'",
    "has been anonymously sending flowers to a coworker every week for six months and the coworker just announced at a meeting that they're filing a police report about it",
    "stole a garden gnome from someone's yard as a joke ten years ago and has been moving it around their house ever since — their spouse thinks they bought it",
    "has been writing one-star Yelp reviews for their ex's business under fake names and just found out their ex figured out it's them",
    "catfished their own spouse to see if they'd cheat — and they did, immediately",
    "created a fake dating profile using their own photos but a completely different name and life story — the fake version of them gets ten times the matches and they're starting to resent their actual life",
    "has been breaking into their own workplace after hours to sleep there because their home life is so unbearable — they've been doing it for three months and security hasn't noticed because they know the camera blind spots",
    "joined a grief support group for widows even though their spouse is alive — they just wanted to feel something and now they're the most popular person in the group and they've started believing their own lies",
    "has been mailing anonymous letters to people in their town telling them secrets about their neighbors that are all true — they call it a public service and they've sent over forty letters",
    "keyed their own car and filed an insurance claim blaming a coworker they hate — the insurance company investigated, the coworker got fired, and they got a brand new paint job",
    "has been calling in fake pizza orders to their ex's address three times a week for six months — the ex posted about it on social media begging for it to stop and they watched the post go viral while eating one of the pizzas",

    # Existential and philosophical crises
    "woke up from anesthesia during a routine knee surgery and heard the surgeon making fun of their weight — now they have to go back for a follow-up with the same guy",
    "went to their own high school reunion and the guy who peaked in 10th grade pulled out a yearbook and showed everyone a photo of them they'd completely blocked out — and it's been eating at them why",
    "found a journal they kept in 2018 and realized they had completely different political beliefs, different friends, and were planning to move to Montana — they have zero memory of any of it",
    "their kid's teacher called them by their ex's name at parent-teacher night and their current spouse was sitting right there",

    # Outrageous situations
    "got a cease and desist letter from Disney because their kid's birthday party decorations went viral on TikTok",
    "found out their Airbnb guest has been living in their rental for three months and won't leave — and legally they might be a tenant now",
    "their ex started a podcast specifically to talk about their relationship and it's getting popular in their town",
    "accidentally RSVP'd yes to their ex's wedding thinking it was a joke and now they have a seat at table 6",
    "their HOA fined them $500 for having a 'non-approved shade of beige' on their front door and they're ready to burn the whole neighborhood down",
    "bid on a storage unit at auction and found a box of love letters between their mother and a man who isn't their father",

    # Betrayal and trust
    "their business partner has been siphoning money for a year and when they confronted them, the partner said 'prove it' and smiled",
    "trusted their brother to housesit and came back to find he'd thrown a party that caused $8,000 in damage and he's acting like nothing happened",
    "found out their therapist has been discussing their sessions with a mutual friend",
    "lent their car to a friend who got a DUI in it and now they're being sued by the other driver",
    "their best friend of 30 years slept with their ex-wife the week the divorce was finalized and just told them 'it's been six months, you should be over it'",
    "discovered their financial advisor put their retirement into investments that benefit the advisor's other company",

    # Crossroads moments
    "got two job offers on the same day — one pays double but means moving away from their dying father, the other keeps them close but they'll be broke",
    "their teenage kid just told them they want to go live with their other parent and they have to decide whether to fight it or let go",
    "a developer offered them $800,000 for their family ranch and their siblings want to sell but they'd rather die than let it go",
    "accidentally wore the same exact outfit as their boss to a client meeting — same shirt, same pants, same shoes — and the boss pulled them aside after and said 'this can never happen again' with complete seriousness",
    "has to testify against their childhood best friend in court next week and the friend's family has been calling them a traitor",

    # Dark humor situations
    "accidentally liked their ex's Instagram photo from 2019 at 2am and the panic spiral has been going for six hours",
    "their date went so badly the restaurant comped the meal and the waiter said 'I'm sorry' on the way out",
    "got a fortune cookie that said 'it's too late' and nothing else — no lucky numbers, no smiley face, just those three words",
    "tried to surprise their spouse for their anniversary and walked in on a surprise party for themselves that their spouse forgot to tell them about — for a birthday that was two months ago",
    "got pulled over doing 90 in a 45 and the cop turned out to be the kid they used to babysit — who let them off with a warning and a lecture that felt worse than a ticket",
    "went to confront someone who keyed their car and it turned out to be their own wife who did it during an argument she says they should remember but they genuinely don't",

    # --- Morally ambiguous (Am I the bad guy?) ---
    "has been reporting their neighbor's code violations to the city anonymously and the neighbor just got a $12,000 fine — they feel terrible but also their property value went up",
    "told their teenage daughter's boyfriend's parents about the kid's drug use and now their daughter won't speak to them — they'd do it again but it's killing them",
    "found their elderly mother's will and she's leaving everything to a church she just started going to — they moved it to a drawer she won't find and are pretending they never saw it",
    "secretly recorded their boss making racist jokes and sent it to HR — the boss got fired but now the whole office suspects them and nobody will talk to them",
    "their best friend asked them to be a character witness in a custody battle and they honestly think the friend is a bad parent — they said yes and don't know what to say on the stand",
    "stopped talking to their brother after he voted for someone they hate — it's been two years and their mom is dying and she wants them all together and they don't know if they can fake it",
    "adopted a dog from a family who couldn't keep it and now they see missing dog posters around the neighborhood — the family's kids made the posters and they feel like a monster",
    "caught their teenage son shoplifting and instead of telling the store they just paid for the item and left — their spouse says they're enabling him and they think their spouse might be right",
    "has been using their dead father's handicapped parking placard for three years and just got confronted by someone in a wheelchair in the parking lot",
    "their elderly neighbor gave them power of attorney and now the neighbor's kids are accusing them of financial exploitation — they've been paying the neighbor's bills out of their own pocket",
    "ghosted someone they were dating for six months because they didn't know how to break up — the person just showed up at their job asking what happened and they can't even explain it to themselves",
    "has been feeding a stray cat for a year that turns out to belong to their neighbor — the neighbor just put up a passive-aggressive sign saying 'STOP FEEDING MY CAT' and now there's a full-blown neighborhood feud",
    "tipped off immigration about an employer using undocumented workers because the employer was paying them nothing — now those workers have no income at all and they feel responsible",
    "inherited their grandparents' house and their cousins expected them to share the proceeds but the will only named them — they kept the house and now the whole family thinks they're greedy",
    "has been secretly attending their ex's church just to see their kids during the service because the custody agreement doesn't give them enough time",
    "told their friend's fiancé about the friend's cheating history and now the wedding is off and they've lost the friend — they think they did the right thing but nobody agrees",
    "found out their kid's teacher is an old college friend they had a falling out with — they requested a class transfer and the school wants to know why and they can't tell the truth",
    "lied on their dying grandmother's behalf and told her that her estranged son was sorry and loved her — the son never said any of that and now the grandmother died at peace with a lie",
    "their spouse wants to put their dog down because of mounting vet bills but the dog still seems happy — they took out a secret credit card to keep paying and the balance is $4,000",
    "turned down a promotion because it would mean managing their best friend and they knew it would ruin the friendship — now someone terrible got the job and everyone blames them",

    # --- Ethically impossible / no right answer ---
    "is a nurse who accidentally gave a patient the wrong dosage three months ago — nothing happened, the patient is fine, but they never reported it and now there's a safety audit and they have to decide whether to come clean or let it go",
    "has been the anonymous donor keeping a local family afloat after their house fire — just found out the father set the fire for insurance money and the family doesn't know they know",
    "their dying mother confessed that their father isn't their biological dad — now they have to decide whether to tell their father, who has been the best parent imaginable, or carry this secret to protect him",
    "caught their best friend's husband hitting on a 19-year-old at a bar — the friend is eight months pregnant and they don't know if telling her right now would do more harm than good",
    "runs a small landlord operation and has a single mom tenant who's three months behind on rent — they need the money to pay their own mortgage but evicting her means her kids end up in a shelter",
    "found out the charity they've been volunteering at for five years has been skimming donations — the charity still does genuine good in the community and reporting it would shut the whole thing down",
    "is a teacher who caught their best student cheating on the exam that determines a full-ride scholarship — the kid comes from nothing and this is their only way out, but other kids played fair",
    "their brother got sober and is making amends, but one of the people he hurt most refuses to forgive him and it's derailing his recovery — they're thinking about confronting the person and they know it's not their place",
    "was the driver in a car accident that killed their best friend fifteen years ago — they were sober, it wasn't their fault, but they've never been able to shake the feeling that they could have reacted faster, and they just got invited to the friend's daughter's wedding",
    "discovered their terminally ill spouse has been secretly hoarding painkillers and they're pretty sure they know why — they can't bring themselves to confront it because part of them understands",
    "has been pretending to still be religious for twenty years because their entire family, social life, and community is built around their church — their spouse just asked them to become a deacon and they feel like they're drowning in the lie",
    "their adult daughter cut them off two years ago and they genuinely don't understand why — they just got a letter from her therapist explaining the ways they caused harm and they recognize some of it but think half of it is unfair",
    "accidentally saw a coworker's medical results on a shared printer — terminal diagnosis, maybe six months — the coworker hasn't told anyone at work and keeps talking about their five-year plan",
    "raised their grandchild since birth because their own kid was a mess — the kid got clean and wants the child back and legally has every right, but the grandchild calls them mom and doesn't really know the biological parent",
    "is a cop who pulled over a fellow officer driving drunk with his kids in the car — if they report it the guy loses everything including custody, if they don't report it and something happens it's on them",
    "their kid's school called and said their 8-year-old has been running a black market candy operation out of their locker — buying in bulk at Costco and marking up 400% — and they're honestly kind of proud but have to pretend to be upset",
    "put their mother in a memory care facility and she begs to come home every visit — the doctors say she needs to be there, the guilt is destroying them, and last week she looked at them and said 'I thought you loved me'",
    "their teenage kid came out to them and they said all the right things but they're struggling with it privately and they feel like a fraud for performing acceptance they haven't fully gotten to yet — and they hate themselves for that",
    "was the whistleblower who shut down a factory that was poisoning the water supply — did the right thing, saved lives, but 200 people lost their jobs in a town with nothing else, and those people's kids are the ones who suffer",
    "forgave the drunk driver who killed their son because their faith demanded it — went public with the forgiveness, everyone called them a saint — but at 2am they fantasize about hurting the driver and they think the forgiveness was a performance they can't take back",
    "runs a family business and just realized their father has been cooking the books for decades — reporting it means their dad goes to prison at 74, not reporting it means they're now complicit, and the money paid for their college and their house",
    "rear-ended someone at a stop light and when they got out to exchange info it was their ex's new partner — the same person they've been trash-talking to everyone for six months — and the person was incredibly nice about it which somehow made it worse",
    "has been the sole caretaker for their disabled sibling for fifteen years and they're burned out, resentful, and starting to hate someone they love — they fantasize about leaving and the shame of that thought is eating them alive",
    "their father was a genuinely terrible person who hurt a lot of people — he died last week and they're grieving hard and everyone around them keeps saying 'you're better off' and they want to scream because grief doesn't work that way",
    "testified against a man who went to prison for twelve years — they were certain at the time but now they're not sure anymore and the man just got out and they saw him at the grocery store",
    "is a doctor who has to decide whether to be honest with a patient about a prognosis that will destroy their will to live — the patient specifically asked for the truth and the truth is there's almost no hope, and they've seen patients who don't know do better",
    "secretly agrees with the person everyone in their life hates — a family member did something unforgivable and the whole family rallied against them, but they've heard the other side and it's more complicated than anyone wants to admit",
    "their spouse's best friend made a pass at them two years ago and they've never told their spouse — not because they're hiding it but because they know their spouse will lose their closest friend and they're not sure the truth is worth that cost",
    "got locked out of their house at 3am in their underwear and had to break into their own home — the neighbor called the cops and now they have to go to court to prove they live there",
    "mentored a kid from a rough neighborhood for three years, got them into college, changed their life — just found out the kid has been dealing drugs the entire time and using the college acceptance as cover, and they're the character reference on the kid's record",

    # --- Dark and compelling confessions ---
    "has been visiting their ex-wife's grave every week for six years and leaving flowers — the problem is they're remarried and their current wife doesn't know and they don't think it's wrong but they know how it looks",
    "was a prison guard for twenty years and did things they were told to do that they now understand were cruel — they followed orders, kept their pension, and retired comfortably while the people they guarded suffered, and they can't sleep anymore",
    "watched someone drown when they were seventeen — there were other people around, everyone froze, and they've told themselves for thirty years that they couldn't have done anything but they've never actually believed it",
    "their spouse died by suicide and everyone treats them like a victim but there are things about the marriage that make them wonder if they contributed — they've never said this out loud because people get angry when you suggest a survivor might carry some responsibility",
    "stole a business idea from a friend's notebook twenty years ago — built a successful career on it — the friend never knew and died last year still wondering why they never got their break",
    "was bullied mercilessly as a kid and grew up to become successful — ran into their childhood bully working a dead-end job and felt genuine joy about it, then went home and cried because they didn't recognize the person they've become",
    "is raising a child they know isn't biologically theirs — they figured it out years ago but the child doesn't know and neither does their spouse, and they love this kid completely but the lie is the foundation of their entire family",
    "pulled the plug on their father's life support against the wishes of half the family — the doctors said there was no hope, they had power of attorney, they made the call, and two of their siblings haven't spoken to them since and it's been four years",
    "survived something terrible and wrote a memoir about it that became locally famous — except they changed key details to make themselves look better and left out the part where they made choices that made things worse, and now they're seen as an inspiration based on a version of events that isn't fully true",

    # --- Outrageous but believable ---
    "just found out their landlord has been entering their apartment while they're at work — they set up a hidden camera and have two weeks of footage of the guy just sitting on their couch watching TV",
    "got a call from a hospital saying they were listed as emergency contact for someone they've never heard of — went to the hospital and the person looks exactly like them, same age, same build",
    "their neighbor installed a surveillance camera that points directly into their bedroom window and when they complained the neighbor said 'then close your blinds'",
    "went to pick up their car from the mechanic and it had 400 more miles on it than when they dropped it off — mechanic says it's a calibration error",
    "found out the house they've been renting for five years isn't owned by their landlord — it belongs to an old woman in a nursing home and the 'landlord' is just some guy collecting rent",
    "woke up to find a full Thanksgiving dinner set up on their front porch — table, chairs, turkey, the works — and nobody in the neighborhood knows anything about it",
    "got a letter from a law firm saying they're a beneficiary in the will of someone they went on one date with 20 years ago — the person left them a boat",
    "their kid's school called to say someone else has been picking up their child using their name and ID — the school let it happen three times before noticing",
    "walked into their garage and found a man living in the crawl space above the ceiling — he'd been there for at least a month based on the setup",
    "received a package addressed to them containing a USB drive with hundreds of photos of them taken over the past year from across the street — no note, no return address",
    "their dentist found a tracking device embedded in a crown they got done at a different practice five years ago",
    "caught their Uber driver taking the long way around and when they mentioned it the driver said 'you don't want to go down that street right now' and wouldn't explain why",
    "just discovered that the 'organic eggs' they've been buying from a coworker for two years are just regular grocery store eggs repackaged in a basket with straw",
    "found a fully furnished room behind a false wall in their basement that wasn't on the original house plans — the previous owner died and nobody knows what it was for",
    "their mail carrier has been writing them anonymous love poems for months — they figured it out because one was delivered with no stamp and had the mail carrier's fingerprints in ink",
    "found a shrine dedicated to them in their ex's closet — photos, old clothes they thought they lost, a candle, and a journal with entries dated from after the breakup that read like they're still together",
    "their neighbor has been collecting their trash and sorting through it — they found out because the neighbor confronted them about a receipt for something 'they shouldn't be buying'",
    "woke up and their car was in a completely different spot in the driveway — it's happened four times now and their spouse says they're imagining it but the odometer doesn't lie",
    "hired a private investigator to follow their spouse and the PI came back and said 'you should probably sit down' — but then said the spouse isn't cheating, they're living an entirely different life during the day than what they've described",
    "got a notification from their home security camera at 3am — it's their spouse, in the backyard, burying something, and when they asked about it in the morning the spouse acted like they had no idea what they were talking about",
    "their dead relative's phone number got reassigned and the new owner has been texting them pretending to be the dead person — they fell for it for two weeks before figuring it out",

    # --- Sex/kink calls (Loveline style) ---
    "just discovered their partner has a {fetish_detail} kink and walked in on them {sex_situation} — they're not disgusted, they're confused about why they're kind of into it too",
    "has been hiding their {fetish_detail} fetish for their entire marriage and their spouse just found their browser history — {partner_reaction}",
    "went to a sex club for the first time with their partner and {partner_reaction} — now they can't stop thinking about going back but their partner pretends it never happened",
    "started an OnlyFans as a joke with their spouse and now they're making $4,000 a month and {partner_reaction} — the money is great but it's changing their relationship",
    "matched with their spouse's sibling on a dating app — they were both supposedly in monogamous relationships and now they share this horrible secret",
    "their partner wants to try {fetish_detail} and they said yes to be supportive but {partner_reaction} — they need to figure out how to have this conversation",
    "found out their quiet, conservative partner had a wild past involving {fetish_detail} and {sex_situation} — they don't care about the past but they want to know why the partner feels they can't be that person anymore",
    "has been having the best sex of their life since they opened up about their {fetish_detail} interest — the problem is it's with someone who isn't their partner",
    "caught their roommate {sex_situation} and now they can't make eye contact — the roommate acts like nothing happened but it was extremely {fetish_detail}-adjacent",
    "went to a couples therapist about their dead bedroom and the therapist suggested {fetish_detail} exploration — they tried it and now they can't go back to vanilla and their partner feels pressured",
    "their ex keeps texting them explicit stuff about {fetish_detail} fantasies they used to do together and they haven't blocked the number because honestly they miss it",
    "just realized they might be into {fetish_detail} after a very specific {sex_situation} experience and they don't know how to bring it up with anyone",
    "has been lying about their number — their actual body count is way higher than what they told their partner and a mutual friend knows the truth and keeps making comments",
    "caught their partner watching porn that features {fetish_detail} content and they're worried it means something about what their partner actually wants",
    "their new partner is incredible in every way except sexually — they're completely incompatible in bed and they've tried {fetish_detail} and it made things worse",
    "accidentally sent a very explicit photo meant for their partner to their work group chat — the photo involved {fetish_detail} context and HR wants to 'have a conversation'",
    "their spouse suggested swinging and they reluctantly agreed — the first experience was {sex_situation} and now the spouse wants to stop but they want to keep going",
    "has a {fetish_detail} kink they've never told anyone about because they're afraid people will think they're weird — but it's consuming their fantasy life",
    "went on a date that started normal and ended up {sex_situation} — they had the time of their life but now they're questioning everything they thought they knew about themselves",
    "their partner found the drawer — the one with the {fetish_detail} stuff in it — and {partner_reaction}",
    "just found out the person they've been sexting for three months is someone from their friend group — the conversation involved detailed {fetish_detail} scenarios",
    "their partner asked them 'what's your biggest fantasy' and they told the truth about {fetish_detail} and the silence that followed was the longest ten seconds of their life",
    "hooked up with someone at a wedding and it got {fetish_detail} fast — the problem is it was their spouse's cousin and now every family gathering is going to be a nightmare",
    "tried {fetish_detail} with their partner for the first time and it was so good they're worried they're addicted — they think about it constantly and normal intimacy feels boring now",
    "their partner confessed to a {fetish_detail} fantasy involving {sex_situation} and they're trying to be open-minded but they have a lot of questions",
    "has been in a secret friends-with-benefits arrangement that involves {fetish_detail} stuff they'd never do in a relationship — the compartmentalization is starting to crack",
    "realized during a very awkward moment {sex_situation} that they have zero chemistry with the person they just moved in with",
    "their couples therapist told them their sex life issues stem from unaddressed {fetish_detail} desires and now the drive home from therapy is incredibly silent",
    "found out their partner has been faking it for years and only admitted it because a conversation about {fetish_detail} finally made them honest about what they actually want",
    "hooked up with their personal trainer and the power dynamic has made every gym session since then unbearably weird — they can't switch trainers because it's a small town",

    # --- Shocking / unhinged / morally reprehensible confessions ---
    "has been sleeping with their spouse's therapist — the therapist started it, they know it's insane, and the worst part is the therapist uses things their spouse said in sessions as pillow talk",
    "bought a used couch off Craigslist and found $11,000 cash sewn into the cushion — the seller won't return their calls and they can't decide if it's theirs now or if keeping it makes them a thief",
    "has been stealing prescription pads from the clinic they clean at night and selling them — they need the money for their kid's medical bills and they know exactly how wrong it is",
    "paid someone to take a lie detector test for them during a custody hearing — passed it, got custody, and now they have to live with the fact that their entire relationship with their kid is built on fraud",
    "slept with their best friend's spouse at that friend's funeral reception — they were both grief-drunk and now they see each other every week because they're both in the dead friend's will as co-executors",
    "has a second family in another state that neither family knows about — two mortgages, two sets of holidays, two birthdays for kids who don't know about each other — and a work trip schedule that's entirely fabricated",
    "got road rage so bad they followed someone home and sat outside their house for an hour — they didn't do anything but the fact that they WANTED to scared them more than anything in their life",
    "got a wrong-number text meant for someone else that contained extremely detailed plans to surprise their spouse with a divorce — and the phone number is one digit off from their own spouse's number, so now they're spiraling",
    "has been pocketing cash from their elderly mother's social security checks for three years — they tell themselves it's payment for caregiving but they know it's theft and their siblings would destroy them if they found out",
    "accidentally killed their neighbor's dog with rat poison they put out — the neighbor thinks it was someone else and they've been helping the neighbor search for who did it",
    "their spouse is in prison and they started sleeping with someone three months in — they drive to visitation every Sunday, hold hands through the glass, and go home to someone else's bed",
    "walked in on their parent having sex with someone who is not their other parent — the parent looked them dead in the eye and said 'we'll talk about this later' and it's been six months and they haven't",
    "got so drunk at a work conference that they slept with two different coworkers on the same night in the same hotel — one of them was their direct report and the other was married to someone in their department",
    "has been pretending to have cancer to get out of family obligations — it started as a small lie and now people are doing fundraisers and shaving their heads for them",
    "planted drugs in their roommate's car and called in an anonymous tip because the roommate owed them $8,000 and wouldn't pay — the roommate did six months in county and just got out",
    "is sleeping with the person their ex left them for — not for revenge, they genuinely caught feelings, and now the three of them are in an impossible triangle where everyone's cheating on everyone",
    "recorded their spouse confessing to an affair during a fight and has been holding the recording as leverage for two years — they haven't played it, they just need to know they COULD",
    "hit someone with their car, panicked, and drove away — the person wasn't badly hurt, they checked the news, but they've been having nightmares every night since and they can't tell anyone because it's a felony",
    "discovered their pastor has been embezzling from the church — but the pastor also paid for their kid's rehab out of pocket and they literally owe this man their child's life",

    # --- More sex/kink/shocking sexual situations ---
    "went to a massage parlor that turned out to be one of THOSE massage parlors — they didn't leave, and now they've been going back every two weeks and their spouse thinks they have a chiropractor",
    "has been having phone sex with a stranger they met on a late-night chat line for six months — they know the person's voice better than their spouse's and they've started comparing the two out loud by accident",
    "slept with someone at a party and found out afterward it was their cousin's ex who looks completely different now — the cousin doesn't know and the sex was honestly the best they've ever had and they want to see them again",
    "got caught having sex in their car by a cop who turned out to be a guy they went to high school with — the cop let them go but now the whole town seems to know",
    "started going to a sex addiction support group as a joke and realized halfway through the first meeting that they actually belong there — they haven't missed a meeting since and their partner has no idea",
    "their spouse found a burner phone with hundreds of explicit texts to multiple people — none of them were physical affairs, all sexting, and they genuinely don't understand why their spouse is acting like it's the same thing",
    "has been paying for a premium subscription to a cam site where the performer turned out to be someone from their neighborhood — they recognized the bedroom furniture and now they can't look at this person at the HOA meeting",
    "agreed to an open marriage thinking it would save things — their spouse immediately started seeing someone and is clearly happier, and they haven't been able to find a single person interested in them, so they just sit home alone while their spouse is out",
    "had a threesome with their partner and a friend — the friend and the partner clearly had better chemistry with each other than with them, and now the partner keeps suggesting they invite the friend over for dinner",
    "found their parent's sex tape while cleaning out the attic — it was labeled with a date and a name, and the name isn't their other parent's, and the date is roughly nine months before they were born",

    # --- Absurd/unhinged comedy situations ---
    "has been pretending to be left-handed at their new job for six months because someone assumed they were on day one and they never corrected it — now there's a company softball game and they can't bat",
    "accidentally started a rumor at the gym that they're a black belt and now three people have asked them to train them and one guy challenged them to spar",
    "told a stranger at a bar they were a pilot and the stranger turned out to be their new neighbor — they've been maintaining the lie for four months including faking phone calls to 'the tower'",
    "bought a metal detector as a joke and found a class ring from 1986 in their backyard — they tracked down the owner who says they've never been to that town and now everyone thinks they're haunted",
    "lied on a dating profile about loving hiking and now they're three months into a relationship with someone who hikes every weekend — they can barely do stairs",
    "their roomba escaped out the front door and someone two streets over found it and won't give it back because they say it 'chose to leave'",
    "replied-all to a company-wide email with a complaint about their boss that was meant for one friend — it's been four hours and they're hiding in the bathroom",
    "signed up for a 5K charity run thinking it was a 5K walk and just found out it's actually a mud run obstacle course — it's this Saturday and they sold $400 in sponsor pledges",
    "got caught talking to their plants by their apartment maintenance guy who now tells everyone in the building they're a 'plant psychic' — two neighbors have asked for consultations",
    "started a fake book club as an excuse to avoid family dinners and now it has twelve members who actually read the books and they haven't read a single one",
    "told their dentist they floss every day for so long that the dentist uses them as an example for other patients — they've never flossed once in their adult life",
    "accidentally won a chili cookoff with canned chili they doctored up — now they're being asked to cater a church fundraiser and they don't know how to make chili",
    "has been waving at someone in their neighborhood for two years thinking they know them — the person finally came up and said 'who are you?' and they panicked and said 'I'm your cousin' and now they're invited to Thanksgiving",
    "their parrot learned to mimic their phone's alarm sound and now they can't tell what's real — they've been late to work three times this month",
    "made up a fake allergy to avoid their coworker's terrible cooking at potlucks and now HR has put a special protocol in place and the cafeteria has a warning sign with their name on it",
    "bet a coworker fifty bucks they could eat a whole jar of pickles in one sitting — they did it but now they physically cannot look at a pickle without gagging and their coworker puts pickles on their desk every day",

    # --- Family drama (expanded) ---
    "their mother told them at dinner tonight that she never wanted kids and it was just something she said but they can't stop hearing it",
    "found out their parents secretly remortgaged the house to bail their sibling out of debt — nobody told them and now the house might be lost",
    "their sister's wedding is next month and their mom just told them they're not invited because the sister's fiancé doesn't like them",
    "just found out they have a half-brother in another state because their dad had an affair 25 years ago — the brother reached out on Facebook and wants to meet",
    "their family has been fighting over their grandmother's china set for six months and nobody speaks to each other anymore over dishes",
    "caught their mother lying about having cancer to get attention from the family — she's done this before with other illnesses and nobody believes the caller",
    "their father remarried three months after their mother's death and the new wife is redecorating the house and throwing away their mom's things",
    "their kid just asked why grandma and grandpa won't come to their house anymore and the real answer is a family grudge the kid shouldn't know about",
    "their sibling keeps borrowing their identity for credit applications and when they confronted them the sibling said 'family helps family'",
    "found an old family video and realized their childhood was nothing like they remember — everyone looks miserable and their parents are fighting in the background of every birthday",

    # --- Career & work (expanded) ---
    "just realized they've been at the same job for 15 years and haven't learned a single new skill — they could be replaced by anyone with two weeks of training",
    "their company just got acquired and the new owners are gutting everything — half their department is already gone and they're waiting for the axe",
    "found out their reference from their last job has been sabotaging their applications — a hiring manager let it slip",
    "started a side business that's taking off but their employer has a non-compete clause and they can't figure out if it applies",
    "got written up at work for something everyone does and they're pretty sure it's retaliation for filing a complaint last month",
    "their internship ended and they were promised a full-time offer that never came — they turned down two other jobs waiting for it",
    "realized their dream career pays so little they can't afford rent and health insurance at the same time",
    "works the graveyard shift and hasn't seen their kids awake in three weeks — their spouse is running on empty and resents them for it",
    "got promoted into a role they're not qualified for because nobody else wanted it and now they're drowning",
    "just discovered their company has been classifying them as a contractor to avoid benefits — for six years",

    # --- Money & debt (expanded) ---
    "their spouse opened a credit card in their kid's name to cover bills and the kid just found out when they tried to get a student loan",
    "inherited a house with $40,000 in back taxes and a lien from a contractor who did work they never approved",
    "loaned their entire tax refund to a friend who swore they'd pay it back by June — it's now March and the friend is avoiding their calls",
    "their bank account got hacked and $6,000 is gone — the bank is investigating but they can't pay rent in the meantime",
    "found out they owe $18,000 in back child support they didn't know about because the letters went to an old address",
    "their car got repossessed at work and everyone watched it get towed out of the parking lot",
    "just calculated they've spent $30,000 on lottery tickets over the past decade and won a total of $600",
    "cosigned a student loan for their niece who dropped out after one semester and moved to another state",
    "their spouse has been paying their ex alimony from a secret bank account and just ran out of money",
    "got a notice that their property taxes tripled because of a reassessment and they literally cannot pay the new amount",

    # --- Health & body (expanded) ---
    "their doctor told them they have six months to make serious changes or they're looking at a heart attack before 50 — they drove straight to a fast food place afterward and hate themselves for it",
    "found out their kid needs surgery and the insurance company denied it — they're on hold for the fourth time today trying to get an override",
    "has been having migraines so bad they see spots and forget words and every test comes back normal — they're starting to think they're going crazy",
    "their spouse was misdiagnosed for two years and the condition has now progressed to a stage that could have been prevented",
    "just got their hospital bill from an emergency appendectomy and even with insurance they owe $23,000",
    "their parent fell and broke a hip and the assisted living facility wants $7,000 a month and they don't have it",
    "found out the medication they've been on for ten years was recalled and the side effects explain everything they've been feeling",
    "their kid has been complaining of chest pains and two doctors said it's anxiety but a parent knows when something's wrong",
    "went to the dentist for the first time in seven years and needs $12,000 in work — they don't have dental insurance",
    "their partner just got diagnosed with something degenerative and they're trying to be strong but they don't know how to plan for a future that looks completely different now",

    # --- Mental health (expanded) ---
    "hasn't been outside in eleven days and they're not sure if they're depressed or just done with people",
    "their medication stopped working three weeks ago and their psychiatrist can't see them for another month",
    "keeps driving past the turn for home and just driving — sometimes for hours — because pulling into the driveway fills them with dread they can't name",
    "deleted every social media app last week because comparing their life to everyone else's was making them physically sick — but now they feel completely cut off",
    "their kid's therapist called to say the sessions aren't working and recommended inpatient and the word 'inpatient' made the floor drop out",
    "realized they've said 'I'm fine' so many times that they can't tell the difference between fine and not fine anymore",
    "woke up on the bathroom floor at 4 AM with their phone dead and no memory of falling asleep there — this is becoming a pattern",
    "has been sleeping 14 hours a day and their boss thinks they're slacking but they genuinely cannot wake up no matter how many alarms they set",

    # --- Grief (expanded) ---
    "their friend's kid died and they don't know what to say — everything feels wrong and they've been avoiding the friend because of it and the guilt is eating them alive",
    "lost their house in a fire two months ago and the insurance is stalling — they're living in a motel with their family and running out of savings",
    "their dog died on the operating table during what was supposed to be a routine procedure and the vet won't return their calls about what happened",
    "found out their childhood home was demolished last week and even though they haven't lived there in 20 years it hit them like a truck",
    "miscarried last month and went back to work after three days because they didn't have any more PTO — nobody at work knows",
    "their grandmother's nursing home closed without notice and they had to scramble to find placement — she's confused and scared and keeps asking to go home",
    "lost both parents within six months of each other and people keep telling them 'at least they're together now' and it makes them want to scream",

    # --- Addiction (expanded) ---
    "counted their drinks last week for the first time and the number scared them — they've been averaging 28 a week and didn't even realize",
    "their kid came home from college with a vape addiction that's costing $200 a month and they didn't even know vaping was addictive",
    "has been taking their spouse's Adderall for three months to keep up at work and just realized they can't function without it",
    "their friend keeps suggesting they 'just have one drink' at every gathering even though they know about the sobriety and it's getting harder to say no",
    "relapsed last week after two years clean and the shame of telling their sponsor feels worse than the relapse itself",
    "found a hidden stash of pills in their teenager's room and recognized the brand because they used to buy from the same person",
    "has been gambling online every night after their family goes to bed — they're down $14,000 and the account is linked to their joint savings",

    # --- Parenting (expanded) ---
    "their kid told the school counselor something about the home that got taken out of context and now CPS is investigating",
    "found out their kid has been skipping lunch every day because they're being bullied in the cafeteria — the kid has been hiding their lunchbox and saying they ate",
    "their teenager just told them they've been self-harming and showed them their arms and the caller had no idea",
    "adopted a child two years ago and the bonding hasn't happened the way the books said it would — they love this kid but it feels forced and they're terrified that's permanent",
    "their kid got expelled and the only other school is 45 minutes away — they can't drive that far and work full-time",
    "their adult child told them at dinner that they're cutting contact and handed them a letter listing everything they did wrong as a parent",
    "just found out their 12-year-old has been watching extremely graphic content online for months and they thought the parental controls were working",
    "their baby won't stop crying and it's been hours and their partner left and they're alone and they've never felt this desperate before",
    "their kid got caught cheating on a test and the school wants a meeting but the real issue is the kid has been under so much pressure they haven't slept properly in weeks",
    "their teenager wants to move in with a partner they've been dating for three weeks and says they'll run away if the caller says no",

    # --- Relationship chaos (expanded) ---
    "found airline tickets on their partner's phone for a trip next week that they weren't told about — to a city where their partner's ex lives",
    "their partner just told them they're aromantic and doesn't experience romantic love the same way — they're not breaking up but the caller doesn't know what to feel",
    "caught their spouse crying in the car at 2 AM and when they asked what was wrong, the spouse said 'everything' and drove away — they've been gone four hours",
    "their partner's therapist told the partner to set boundaries and now the partner won't do anything the caller asks without calling it a 'boundary violation'",
    "went through their partner's phone (they know they shouldn't have) and found nothing — but the relief they felt made them realize the relationship is built on suspicion and that might be worse",
    "their spouse announced at a family dinner that they're quitting their job to 'find themselves' — no discussion, no plan, three kids",
    "just found out their fiancé has a child from a previous relationship that they never mentioned — the kid is seven",
    "their partner of five years just told them they've been lying about their age — they're twelve years older than they claimed",

    # --- Legal & justice (expanded) ---
    "got pulled over for a broken taillight and the cop found something in the car that belongs to whoever they borrowed it from — now they're facing charges for someone else's stuff",
    "their landlord changed the locks while they were at work and threw their stuff on the curb — they're standing outside their apartment right now",
    "got a letter from a lawyer saying someone is suing them for a car accident they were involved in eighteen months ago that they thought was settled",
    "their ex filed a restraining order full of lies and the hearing is Monday and they don't have a lawyer",
    "just found out there's a warrant for their arrest in another state for a ticket they never knew about — from a rental car they returned twelve years ago",
    "their neighbor is suing them because a tree on their property fell during a storm and damaged the neighbor's fence — the neighbor wants $20,000",
    "witnessed a crime and the detective wants them to testify but the defendant lives on their street and knows where they live",

    # --- Community & neighbor drama (expanded) ---
    "their neighbor put up a 12-foot fence that blocks all the sunlight to their garden and the city says it's within code",
    "someone has been filing anonymous complaints with code enforcement about their property and they're pretty sure they know who it is",
    "got into it at the school board meeting last night and a video of them yelling is now on the local Facebook page",
    "their neighbor runs a business out of their garage and the traffic and noise are destroying the street — the other neighbors are afraid to say anything",
    "the HOA just voted to ban trucks with visible work equipment from the driveway — that's their livelihood parked there every night",
    "their small town is being split by a proposed mine that would bring jobs but destroy the aquifer — families who've been friends for decades are on opposite sides",
    "found out the city council rezoned their neighborhood for commercial use without a single public hearing",

    # --- Secrets & shame (expanded) ---
    "has been lying to their family about their job for two years — they got fired and have been day-trading the savings while pretending to commute",
    "cheated on a professional exam and got certified — now they're in a position where their incompetence could hurt people and the anxiety is killing them",
    "has been using a fake social media profile to check on their ex for three years — they're not a stalker, they just can't move on and they know how pathetic that sounds",
    "stole something valuable from a friend's house during a party ten years ago and the friend still mentions it sometimes as an unsolved mystery",
    "has a chronic illness they haven't told their employer about because they'd lose the position — but it's getting harder to hide",
    "was the anonymous source for a news story that ruined someone's career — the story was true but the fallout was worse than they expected",
    "has been pretending to be straight at work for five years because the industry they're in and the town they live in would make their life hell if anyone knew",
    "ghosted their entire friend group eight months ago and has been too ashamed to reach out even though they know it hurt people who cared about them",

    # --- Bizarre situations (expanded) ---
    "their Alexa ordered $800 worth of cat food and they don't have a cat — Amazon says the voice command came from their device and won't refund it",
    "got a bill from a hospital in their name for a baby that was delivered last month — they are a 62-year-old man",
    "woke up to find their front door wide open, the TV on, and a plate of food on the table — they live alone and everything was locked when they went to bed",
    "their car's GPS started giving directions in a language they don't recognize and won't switch back — they've factory-reset it twice",
    "received a birthday card every year for five years from someone they've never met — same handwriting, different postmarks, always just says 'thinking of you'",
    "their new puppy dug up a bag in the backyard with $3,000 cash and what looks like a burner phone — they just moved in three months ago",
    "got a Facebook friend request from someone with their exact name, same birthday, who looks eerily like them but lives in a different state",
    "their mail carrier delivered a handwritten apology letter addressed to them by name — it's clearly heartfelt but they have no idea who wrote it",
    "found a USB drive in the library with what appears to be an entire novel about someone whose life is almost exactly like theirs",
    "got a call from their own phone number while holding their phone — the voicemail is static with what might be their own voice underneath",

    # --- Generational & cultural ---
    "is the first person in their family to go to college and the pressure to succeed for everyone who couldn't is crushing them",
    "grew up in extreme poverty and now makes good money but can't stop hoarding food and living like they're broke — their partner is concerned",
    "their immigrant parents sacrificed everything to give them a better life and they feel guilty every day that they're not living up to what they gave up",
    "is caught between two cultures — too American for their parents' homeland, too foreign for America — and doesn't feel like they belong anywhere",
    "their family expects them to send money back to relatives in another country every month and it's bankrupting them but saying no would mean being disowned",
    "was raised in a religious cult and left five years ago but can't make normal relationships because they never learned how",

    # --- Veterans expanded ---
    "came home from deployment and their marriage was over but nobody told them — their spouse just acted different and the whole thing unraveled",
    "the fireworks on the Fourth put them in a state every year and they've started spending the holiday alone in a cabin with noise-canceling headphones",
    "their VA therapist got reassigned and the new one wants to start from scratch — they can't tell the story again",
    "got a civilian job that's nothing like what they were trained for and feels useless for the first time in their life",
    "their buddy who was struggling called them last week and they were too busy to pick up — they called back the next day and the number was disconnected",

    # --- Technology & modern life (expanded) ---
    "their teenager created an AI chatbot that talks like them and showed it to the family as a joke — but some of the things it says are things the caller has only thought, never said",
    "got deepfaked into a video that's circulating at work and nobody believes it's not really them",
    "their identity was stolen so thoroughly that the thief filed their taxes, renewed their license, and opened a P.O. box",
    "found out their smart TV has been recording audio and sending it to the manufacturer — they had a very private conversation in front of it last week",
    "their elderly parent fell for a phone scam and wired $8,000 to someone pretending to be them in jail",

    # --- Betrayal (expanded) ---
    "their mentor who got them into the industry just published an article taking credit for their biggest achievement by name",
    "trusted their accountant with their taxes for ten years and just found out the accountant never filed the last three years",
    "their childhood best friend wrote a memoir that includes private conversations they had in confidence — some of it is embellished",
    "gave their sibling a key to their house for emergencies and found out the sibling has been coming in when they're at work to eat their food for months",
    "their partner promised to stop talking to their ex and the caller just found a second phone with six months of messages",

    # --- Moral gray zones (expanded) ---
    "saw a hit and run from their porch — they have the plate number — but the driver is a kid, maybe 16, and they remember what it was like to be terrified at that age",
    "their elderly neighbor has been driving erratically and they're worried someone's going to get killed — but calling the DMV feels like taking away their last independence",
    "volunteers at a food bank and saw a coworker who makes six figures coming through the line — they know it's none of their business but it's eating at them",
    "their friend asked them to lie to a cop about being together on a specific night — the friend won't explain why but seems desperate",
    "found out their kid's favorite teacher is an undocumented immigrant and ICE has been making rounds in town",

    # --- Escalating situations ---
    "their landlord just raised rent by 40% and gave them 30 days to accept or move — they can't afford either option",
    "someone at work started a rumor that they're having an affair with the boss and the rumor is completely false but the boss hasn't denied it",
    "their ex keeps showing up at places they go and always acts surprised but it's happened twelve times in two weeks",
    "filed a noise complaint about their neighbor and the neighbor put a sign in their yard calling the caller a snitch with their name on it",
    "their kid's bully's parent confronted them at pickup and it almost turned physical — other parents filmed it and now it's on Facebook",
    "just got back from a road trip and their house smells like cigarettes — they don't smoke, nobody has a key, and nothing is missing but things have been moved",

    # --- Unhinged confessions (expanded) ---
    "has been calling in sick to work once a week for six months to sit in a park alone because it's the only time they feel like a person",
    "learned their neighbor's WiFi password and has been using it for a year — the neighbor just upgraded to gigabit and the caller is getting better speeds than the plan they're paying for",
    "started leaving anonymous compliments in people's mailboxes to cheer people up and now there's a Nextdoor thread calling them a stalker",
    "has been pretending to know how to swim for their entire adult life and their family just planned a beach vacation with snorkeling",
    "told their date they were allergic to shellfish to avoid splitting the lobster and now every date avoids seafood places — they love seafood",
    "started going to a random church every Sunday because they like the free coffee and now they've been elected to the welcoming committee",
    "has been secretly watering their neighbor's dying plant through the fence because they can't stand watching it die",

    # --- Additional problems (reaching 600+) ---
    "their kid's college fund got wiped out by a market crash and the kid is a junior in high school with early admission plans",
    "got called into a meeting and told their position is being 'restructured' — same job, lower pay, fewer hours, no benefits",
    "their spouse's old friend moved in 'for a week' three months ago and shows no sign of leaving — the friend doesn't pay rent or help with anything",
    "found out their contractor used substandard materials on their roof and the warranty is void because the contractor disappeared",
    "their insurance dropped them after a claim and now nobody will cover them — they have a pre-existing condition and a kid with asthma",
    "was told by a financial advisor to invest in something that lost 60% of its value in two months — the advisor won't return calls",
    "their adult kid maxed out a credit card in the caller's name without permission and the kid doesn't think it's a big deal",
    "got a notice from the city that their house doesn't meet current code and they have 90 days to bring it up to standard — the estimate is $35,000",
    "their neighbor's construction project flooded their basement and the neighbor's insurance says it's not their problem",
    "found out their spouse took a second mortgage on the house without telling them to fund a business idea that's already failing",
    "just realized their 401k has been going into the wrong fund for four years because of a typo nobody caught — they've lost tens of thousands in potential growth",
    "their kid's school is closing and the nearest alternative is in another district that won't accept transfers without a move",
    "got a collections call for a medical bill they thought was covered — turns out the anesthesiologist was out-of-network at an in-network hospital",
    "their dog bit a delivery driver and now there's a lawsuit — the dog has never bitten anyone before and the driver was in the yard uninvited",
    "found out their elderly mother signed over power of attorney to a neighbor she barely knows — the family is scrambling to figure out what to do",
    "their small business got a one-star review from someone they've never served and it's tanking their rating — they can't get the platform to remove it",
    "woke up to find their car vandalized with spray paint that says something personal — they don't know who did it but the message means someone knows their business",
    "their kid's prom date just canceled and the kid is devastated — they spent $600 on the outfit and the caller can't get any of it back",
    "got a cease and desist from a company claiming their small business name infringes on their trademark — hiring a lawyer would cost more than the business makes in a year",
    "their spouse wants to homeschool the kids starting next month and they fundamentally disagree but the spouse already told the kids and now they're excited",
    "found mold in their apartment and the landlord says it's their problem — they've been sick for months and just realized it might be connected",
    "their neighbor's tree roots are destroying their foundation and the neighbor refuses to pay for removal because the tree is 'historic'",
    "got rear-ended by an uninsured driver and their own insurance deductible is $2,500 they don't have",
    "their sibling just announced they're moving back to town and wants to move in — the last time they lived together it ended in a screaming match that the whole street heard",
    "found out their home inspector missed major issues when they bought the house and the inspector's liability insurance has a cap that doesn't cover the damage",
    "their kid got rejected from every college they applied to and the backup plan fell through — they're sitting across from a devastated 18-year-old with no idea what to say",
    "just discovered their spouse has been hiding a storage unit full of purchases — the bill is $300 a month and it's been going on for two years",
    "their boss told them to train their replacement and didn't say why — nobody will give them a straight answer and the anxiety is eating them alive",
    "found out their trusted babysitter has been letting strangers into the house while they're at work — the Ring camera footage is from last week and they're trying to stay calm",
    "their landlord sold the building and the new owner wants everyone out in 60 days — they've been there nine years and have nowhere to go",
    "got a DNA test as a birthday gift and the results say their dad isn't their biological father — and their mom won't talk about it",
    "their kid's coach has been playing favorites and their kid hasn't seen the field in five games — they don't want to be 'that parent' but the kid is losing confidence",
    "found out their home is in a wildfire evacuation zone they didn't know about and their insurance doesn't cover fire damage",
    "their partner spent their vacation fund on crypto without telling them — the crypto is now worth a third of what was paid",
    "their best employee just quit with no notice and they have a major deadline next week — the employee left for a competitor and took clients with them",
    "got a call from their kid's principal saying the kid brought something to school they shouldn't have — the meeting is tomorrow and they have no idea what it is",
    "their HOA is forcing them to remove a wheelchair ramp they built for their disabled spouse because it 'doesn't match the aesthetic'",
    "found out their retirement date just got pushed back five years because of a pension rule change nobody told them about",
    # Self-inflicted / ego-driven problems
    "has been pretending to know how to swim for their entire adult life and their spouse just booked a Caribbean cruise with a snorkeling excursion for their anniversary — they leave in three weeks and googled 'how to swim' last night and immediately closed the laptop when their wife walked in",
    "has been telling their coworkers they speak fluent Italian for two years as a personality thing and now the company is sending them to Rome to lead a client meeting — they've been doing Duolingo fourteen hours a day and can currently order coffee and ask where the bathroom is",
    "got into a road rage incident where they followed the other driver to a parking lot to yell at them — turns out the parking lot was a police station and the other driver was an off-duty officer who walked inside, came back out in uniform, and wrote them three tickets",
    "told their new girlfriend they own their house and now she wants to move in — they rent, the landlord lives next door, and the girlfriend just introduced herself to the landlord and said 'so nice to meet our neighbor' and the landlord looked at the caller and raised one eyebrow",
    "refused to apologize to their neighbor over a fence dispute out of principle and it's been four years — their kid and the neighbor's kid are now dating and both families have to sit at the same table for graduation dinner and nobody has acknowledged the fence once",
    "has been faking an injury at work for three months to keep getting light duty and just found out the company hired a private investigator — they saw the PI's car outside their house while they were carrying two bags of concrete mix into the backyard for a patio project",
    "got drunk and told their wife's entire family what they actually think of them at Thanksgiving dinner — called the brother-in-law a 'grown man who collects swords' and told the mother-in-law her casserole 'tastes like revenge' — and now their wife is saying she agrees with her family that he needs anger management but won't disagree about the casserole",
    "made a fake LinkedIn profile to catfish their ex and accidentally built a real professional network with it — the fake persona now has 3,000 connections and a recruiter just reached out with a six-figure offer for a person who doesn't exist and the caller is seriously considering showing up to the interview in a wig",
    "lied on their dating profile about being 6'1\" and they're 5'8\" — it worked until the woman showed up in heels and he was eye level with her chin, and when she said 'you're not six one' he said 'I am in boots' and she said 'you're wearing sneakers' and he said 'yeah'",
    "told everyone at work they ran a marathon and now there's a company team signing up for one in their name — they have never run more than the length of a driveway and the race is in six weeks and their boss already ordered shirts with their name on them",
]

STORIES = [
    # Neighbor/community weirdness
    "found out their neighbor has been watering their lawn with a hose that runs from the caller's outdoor spigot — for at least a year based on the water bills",
    "walked into the wrong house in their subdivision — same floor plan, door was unlocked — sat down on the couch before the actual homeowner came out of the bathroom",
    "their UPS driver has been leaving passive-aggressive notes about their package volume — the latest one said 'you know Amazon has lockers right'",
    "caught their neighbor's Roomba in their house — it came through the dog door and was vacuuming their kitchen at 3am",
    "has been getting someone else's mail for six months and it's increasingly personal — birthday cards, love letters, a small inheritance check — and they can't find the intended recipient",
    "their neighbor put up a 'Beware of Dog' sign but doesn't have a dog — when asked about it they winked and said 'exactly'",
    "found out the previous owner of their house buried a time capsule in the backyard — they dug it up and it's just a note that says 'don't open the wall in the basement' and now they can't stop thinking about the wall",
    "their HOA sent them a letter praising their lawn as the best in the neighborhood — they haven't mowed in two months, a neighbor has been secretly maintaining it",
    # Workplace absurdity
    "their coworker has been microwaving fish every single day for a year and when confronted said 'I will die on this hill' with complete sincerity",
    "accidentally went to the wrong job interview, got hired, and has been working there for three weeks — the job is better than the one they applied for",
    "found a hidden room at their office that nobody seems to know about — it has a couch, a mini fridge, and someone's personal photos on the wall",
    "their boss calls them by the wrong name and has for two years — they corrected him once and he said 'no, I'm pretty sure it's Steve' and they are not Steve",
    "got a performance review that was clearly written about someone else — all the accomplishments are things they didn't do but the rating was excellent so they signed it",
    "found out their quiet coworker who eats lunch alone every day is a semi-famous competitive eater who goes by a different name on the circuit",
    # Animal encounters
    "a turkey has been following them to work every morning for three weeks — it waits in the parking lot and follows them to the door",
    "their cat brought home a live snake and dropped it in their bed at 2am — they didn't find it until they felt it move under the covers",
    "found a tortoise in their backyard that wasn't there yesterday — nobody within five miles owns a tortoise and it won't leave",
    "a hawk stole their sandwich right out of their hand at a gas station and they made eye contact with it the entire time",
    "woke up to find a family of javelinas had pushed open their back gate and were sleeping in their yard like they owned the place",
    # Technology/modern life mishaps
    "their smart home went haywire and started playing mariachi music at 4am at full volume — they couldn't turn it off and had to physically unplug the speaker from the attic",
    "accidentally left their phone's live location sharing on for three months and their entire family watched them go to Taco Bell 47 times",
    "their kid's school called because their child told the class their parent was a spy — the parent is an accountant but they once jokingly told the kid that to explain a business trip",
    "got a notification that their Ring doorbell detected a person at 3am — it was a raccoon standing on its hind legs wearing what appeared to be a small hat",
    "their GPS has been routing them past the same house for three weeks on different drives and they're starting to think the universe is trying to tell them something",
    # Coincidence/bizarre timing
    "ran into their doppelganger at a restaurant — same face, same outfit, even ordered the same meal — the other person was just as freaked out",
    "found a photo of their great-grandfather at a flea market 500 miles from where the family is from — it was in a box of random photos priced at 50 cents",
    "got a wrong-number text that described their exact life situation so perfectly they responded and now they're friends with the stranger",
    "ordered food delivery and the driver turned out to be their old college professor — the professor recognized them and gave them a lecture about tipping",
    "found a voicemail on their dead phone from three years ago that they never listened to — it's from someone they had a huge falling out with and they're afraid to play it",
    # Social/dating mishaps
    "went on a blind date and realized ten minutes in that they'd already been on a date with this person five years ago — neither of them had a good time the first time either",
    "accidentally RSVP'd to the wrong funeral — realized halfway through the service but couldn't leave because they were sitting in the front row",
    "their kid's teacher just asked them out and they said yes before realizing it might be weird — parent-teacher conferences are next week",
    "showed up to a costume party that wasn't a costume party — they were dressed as a giant banana and had to commit to it for four hours",
    "got stuck in an elevator with their ex-spouse and their ex-spouse's new partner for 45 minutes — nobody had phone service",
    # Mundane that escalated
    "returned a library book 22 years late and the fine was $847 — the librarian remembered them by name",
    "has been arguing with their spouse about whether a hotdog is a sandwich for three days and it has genuinely become a relationship issue",
    "accidentally tipped 100% at a restaurant instead of 10% and was too embarrassed to say anything — the waiter cried and hugged them",
    "found $200 in a coat they hadn't worn in two years and can't remember if it's theirs or someone else's — the coat was borrowed from someone they no longer talk to",
    "their garage door opener started opening their neighbor's garage instead of theirs after a power outage and the neighbor thinks they've been snooping",
    "ordered something online that arrived in a box way too big — like 6 feet tall — and inside was their order plus an entire set of patio furniture that wasn't on the invoice",
    # --- neighbor & community absurdity ---
    "their neighbor started building something in the backyard and it's been growing for six months — it's now a three-story structure that might be a windmill and the HOA doesn't know what to do",
    "found an entire beehive inside their walls — not a small one, the bee guy said it's been there at least five years and there are 60,000 bees in it",
    "their mail carrier accidentally delivered a love letter meant for someone else — the caller read it before realizing and now they're invested in this stranger's love life",
    "got a knock on the door from a man claiming to have buried a time capsule in their yard in 1988 and asking if he could dig it up — they let him and it was full of Polaroids and a Walkman with a mixtape",
    "their neighbor's parrot escaped and spent three days on their roof yelling what turned out to be the neighbor's WiFi password, their dog's name, and 'I want a divorce'",
    "woke up to find someone had mowed a giant smiley face into their front lawn — nobody on the street will admit to it and the mowing is professional-grade",
    "their doorbell camera caught the FedEx driver doing a full celebration dance after making a tricky porch delivery — fist pumps, a spin, and a little bow to the camera",
    "found a message in a bottle washed up at a lake — it was from a kid five miles upstream who wanted to know if fish have feelings and included a stamped return envelope",
    "their kid's lemonade stand accidentally undercut the local coffee shop's iced tea price and the owner came over to 'negotiate'",
    "got a letter from the county saying their property line is six feet further east than they thought — they've been mowing their neighbor's lawn for twelve years",
    # --- workplace chaos ---
    "the office fridge has had a piece of cake in it for seven months and someone keeps refreshing the 'DO NOT EAT' sticky note — nobody knows whose cake it is or who's protecting it",
    "accidentally sent a text meant for their spouse to their entire team Slack — it said 'my boss is driving me insane' and their boss hearted the message",
    "their company hired a motivational consultant who just turned out to be a guy with a megaphone and a lot of energy — he made the IT department do trust falls in the parking lot",
    "walked into the wrong conference room and sat through an hour-long meeting for a completely different department before anyone noticed — they contributed twice and nobody questioned it",
    "found out the office 'ghost' that keeps eating people's lunches is their CEO — the janitor caught them on camera at 6 AM raiding the fridge in a hoodie",
    "their coworker has been secretly replacing the break room coffee with decaf for three months as a 'social experiment' — they only confessed because someone threatened to call a plumber about the 'broken' coffee maker",
    "accidentally wore the exact same outfit as their boss three days in a row — neither of them acknowledged it until the third day when the boss said 'one of us has to go home'",
    # --- animal adventures ---
    "their dog learned to open the fridge and has been helping himself to lunch meat — they only found out because the dog gained eight pounds in a month and the deli drawer was mysteriously empty",
    "a goat appeared in their yard and won't leave — they don't own a goat, neither do any neighbors, and animal control says it's not their jurisdiction because it's 'livestock'",
    "their cat has been bringing them increasingly valuable items — started with dead mice, moved to socks, and last week brought home a $20 bill and a car key that doesn't belong to anyone they know",
    "a raccoon broke into their truck and ate an entire bag of gas station donuts — then fell asleep in the back seat and they didn't find it until the next morning",
    "their dog made friends with a deer and now the deer comes to the backyard every morning and they just hang out — the dog doesn't bark and the deer doesn't run",
    "found a frog in their toilet three days in a row — same frog based on a distinctive spot — they've named it and stopped trying to remove it",
    "their chickens staged what can only be described as a jailbreak — dug under the fence, walked down the road, and were found in the neighbor's swimming pool a quarter mile away",
    "a hawk keeps dropping things on their porch — so far: a sock, a tennis ball, half a sandwich, and what appears to be someone's credit card",
    # --- tech & modern life ---
    "their smart speaker started answering questions they didn't ask — at 2 AM it said 'the weather tomorrow is partly cloudy' and nobody was in the room",
    "accidentally joined a Zoom call for a book club in a different time zone and they're now three books deep with people they've never met in person — the group thinks they live in Ohio",
    "their kid changed the Netflix profile names and now they can't figure out who is 'Pasta Lord' and who is 'Couch Goblin' — they've been accidentally watching each other's recommendations for a month",
    "got a notification that someone signed into their email from Brazil — turns out it was an old phone they sold on eBay and the new owner has been reading their newsletters without unsubscribing",
    "their robot vacuum has been mapping their house wrong and keeps trying to clean a room that doesn't exist — it rams into the same wall every day at the same time",
    "accidentally texted 'I love you' to their dentist and the dentist texted 'I love you too' back — neither of them has addressed it",
    # --- coincidence & bizarre timing ---
    "bought a used book online and found a photo inside of someone at the exact restaurant they had dinner at last night — same table, same booth, dated 1997",
    "wore a shirt with a specific obscure band logo and three separate strangers in one day commented on it — none of them knew each other and the band has twelve fans",
    "showed up to a potluck and someone else brought the exact same dish in the exact same Pyrex — same recipe, same garnish, and they don't know each other",
    "their kid was assigned a pen pal through school and it turned out to be their old college roommate's kid — the roommate lives 2,000 miles away",
    "booked a vacation rental sight unseen and when they walked in, it had the exact same layout and furniture as their childhood home — even the wallpaper in the bathroom",
    # --- social & dating mishaps ---
    "went to their high school reunion and nobody recognized them — they've changed so much someone thought they were a caterer and asked them to refill the water pitcher",
    "accidentally waved back at someone who was waving at the person behind them — committed to it and had a five-minute conversation pretending they knew each other, exchanged numbers, and is now too deep to explain",
    "their Uber driver turned out to be their ex-husband's new wife — neither of them said a word for eighteen minutes but the driver gave them a one-star rating",
    "showed up to a first date and the person across the table said 'wait, were you the kid who threw up on me at summer camp in 1996' and they were",
    "got seated next to their therapist at a wedding — neither acknowledged the professional relationship and they had to watch their therapist do the Macarena",
    # --- mundane escalations ---
    "bought a storage unit at auction for $50 and it was full of nothing but garden gnomes — over 200 of them, some wearing tiny outfits, and now they can't bring themselves to throw them away",
    "their Craigslist ad for a free couch got 47 responses in an hour, including one person who sent their life story and another who offered to trade a live iguana",
    "tried to return a pair of jeans without a receipt and somehow ended up in a forty-minute conversation with the manager about their divorce — the manager's divorce, not theirs",
    "accidentally brought their kid's lunch to work and their own briefcase to the school — the kid had to explain why they had a laptop and a granola bar instead of a sandwich and juice box",
    "planted what they thought was a small herb garden and one of the plants turned out to be zucchini — it's now producing twelve zucchinis a week and they've been leaving them on neighbors' porches anonymously like a vegetable vigilante",
    "entered a costume contest as a last-minute decision wearing a bedsheet ghost and won — beat someone who'd spent six months on a screen-accurate Iron Man suit and the Iron Man person has not spoken to them since",
    "started a puzzle three weeks ago and realized at 98% completion that there are two pieces missing and one extra piece from a different puzzle — they've been staring at the empty spots every night",
    # --- travel & road stories ---
    "got the wrong rental car at the airport and drove it for three days before realizing — it was a luxury upgrade and the person who got their economy car has been calling Hertz hourly",
    "stopped at a diner in the middle of nowhere and the waitress said 'oh you're back' even though they'd never been there — she insisted they were there last Tuesday and ordered the meatloaf",
    "their GPS rerouted them through a town so small it had one building that was simultaneously the post office, the bar, and the feed store — the bartender/postmaster gave them directions and a beer",
    "took a wrong turn on a road trip and ended up at a festival they'd never heard of celebrating a vegetable they couldn't identify — they stayed for two hours and won a ribbon",
    "their flight got delayed so they started talking to a stranger at the gate — eight hours later they'd covered childhood trauma, career regrets, and a business idea, and they've never spoken since",
    # --- family weirdness ---
    "found out they were named after their parent's favorite gas station attendant and they don't know how to process that information",
    "their grandmother's will specified that whoever takes care of her 23-year-old parrot inherits the house — the parrot is mean and bites everyone but they need the house",
    "discovered a family recipe that's been passed down for generations is actually just the recipe from the back of a Campbell's soup can with one ingredient changed",
    "their dad has been telling everyone he's retired for five years but they just found out he's been going to work every day at a different job he's embarrassed about",
    "found out their 'uncle' who comes to every Thanksgiving is not related to anyone in the family — he just showed up one year and nobody questioned it and now it's been 20 years",
    "their mom started a TikTok account and has more followers than them — she posts cooking videos but the comments are all about how attractive she is and the caller doesn't know what to do with that",
    "inherited a storage locker from a great-uncle they never met — it's full of clown costumes, at least 40 of them, all different, all well-worn",
    "their parents announced they're getting divorced after 40 years and both separately asked the caller to help them set up dating profiles on the same day",
    "found their dad's secret second phone and confronted him about it — turns out he's been playing Candy Crush in secret because their mom banned screen time for the whole family",
    "their sibling got a DNA test and found out they have a different father — the parents are refusing to discuss it and Thanksgiving is in two weeks",
    "their grandpa left behind a locked safe nobody could open — they finally cracked it and inside was a single post-it note that says 'ha' in his handwriting",
    "their mom has been mailing them a clipping from the local newspaper every week for three years with no note — just the clipping in an envelope and they can't figure out the pattern",
    "found out their aunt has been regifting the same candle set for fifteen years — it's made it around the family twice and someone finally recognized it",
    "their kid asked why grandma has a different last name than grandpa and it opened a can of worms that has been sealed since 1983",
    "found home movies in the attic and one of them shows their parents at a party in the 80s doing things that cannot be unseen",
    # --- purchases & consumer nightmares ---
    "bought a couch off Craigslist and when they got it home found $8,000 in cash sewn into the cushion — the seller won't return their calls",
    "ordered a custom birthday cake that was supposed to say 'Happy 40th' and it arrived saying 'Happy 40th, you're closer to death' — the bakery says that's what the order form said",
    "bought a used car and found a love letter in the glove box that's so beautiful they framed it — three months later the previous owner showed up asking for it back",
    "their contractor disappeared mid-renovation — took the deposit, ripped out the kitchen, and vanished — they've been cooking on a camping stove in their garage for two months",
    "found out the 'antique' table they paid $2,000 for at an estate sale is from IKEA — the auctioneer had rubbed shoe polish on it to make it look old",
    "accidentally bought a timeshare while drunk on vacation and the cancellation period ended while they were still hungover — they now own one week a year in Branson, Missouri",
    "their online mattress came vacuum-sealed and when they opened it in the bedroom it expanded and pinned them against the wall — they were stuck for 45 minutes until their kid came home",
    "ordered a 'slightly used' textbook online and it arrived with someone's entire semester of notes in the margins — including personal diary entries and a phone number with a heart next to it",
    "bought a vintage jacket from a thrift store and found a key in the pocket — nobody at the store knows what it opens and now they're obsessed with finding out",
    "hired a painter for their house and the painter chose a color that was slightly off from the sample — the whole house is now a shade of pink they didn't agree to and the painter says it'll 'grow on them'",
    # --- hobby & passion gone wrong ---
    "started a vegetable garden to save money and has now spent $3,000 on supplies to grow $40 worth of tomatoes — their spouse keeps a running spreadsheet",
    "got really into woodworking during the pandemic and built their spouse a bookshelf that collapsed the first time they put books on it — they'd already posted it to Instagram and got 200 likes",
    "joined a recreational softball league and tore their ACL in the first game — they're 38 and the doctor said 'this is why people your age should stretch'",
    "started homebrewing beer and the first batch exploded in their closet — it ruined twelve dress shirts and the closet still smells like hops four months later",
    "got into birdwatching and became so obsessed they called in sick to work three times to see a rare warbler that turned out to be a common sparrow with unusual coloring",
    "started a podcast about their niche hobby and the only listener is their mom — their mom has opinions about the format and gives them notes after every episode",
    "took up beekeeping and their neighbor is threatening to sue because bees keep getting in their pool — the caller says the bees were there first",
    "got into competitive barbecue and spent $4,000 on a smoker they've used twice — their spouse gave them an ultimatum: the smoker or the second parking spot",
    "got really into metal detecting and found nothing but bottle caps for three months — then found a Civil War era belt buckle and now they think they're Indiana Jones",
    "decided to learn guitar at 45 and their family staged a gentle intervention after two months of the same three chords played loudly every evening",
    "started collecting vinyl records and can't stop — they've spent $6,000 and don't actually own a record player yet",
    "took up marathon running and got so into nutrition they now bring their own food to restaurants in Tupperware — their friends have stopped inviting them to dinner",
    "signed up for a pottery class and their first piece looked like a crime scene — the instructor said 'well, it has character' and put it on the shelf where everyone could see it",
    "got into amateur astronomy and bought a telescope that's now pointed at the neighbor's house because the angle is better — the neighbors think they're being spied on",
    # --- late-night revelations ---
    "can't sleep because they just realized they've been pronouncing a word wrong their entire life and nobody ever corrected them — they used it in their wedding vows",
    "was going through old emails and found one from 2014 that would have changed their life if they'd read it — it was a job offer they never saw buried in spam",
    "just remembered something embarrassing they did in high school and the shame hit them like it happened yesterday — it's been 25 years",
    "can't stop thinking about the fact that they've been tipping wrong at restaurants for years — they just learned you're supposed to tip on the pre-tax amount, or is it post-tax, and now they don't know",
    "lying in bed replaying a conversation from earlier today where they said something they meant as a joke and the other person clearly didn't take it that way",
    "just found out a word they made up as a kid and have been using their whole life is not a real word — they used it in a work presentation last week",
    "realized at 2am that they've been telling a story about themselves for years and just realized it actually happened to someone else — they stole someone's anecdote and made it their own",
    "was cleaning out a drawer and found a to-do list from ten years ago — half the things are still undone and they're having an existential crisis about it",
    "can't sleep because they finally did the math on how much they've spent on coffee over the last decade and the number is upsetting",
    "just realized they've been waving at their neighbor every morning for two years and it might be a completely different person than who they think it is",
    # --- food & cooking disasters ---
    "tried to deep-fry a turkey and set their deck on fire — the fire department came and one of the firefighters asked if they could have some turkey because it actually smelled great",
    "brought a homemade dish to a potluck and everyone loved it — they didn't make it, they bought it from a restaurant and put it in their own container, and now people keep asking for the recipe",
    "their sourdough starter is three years old and they're more emotionally attached to it than some of their friendships — they named it and take it on vacation",
    "accidentally made the spiciest salsa anyone has ever tasted and now everyone wants more — they can't remember what they put in it and have been unable to replicate it for six months",
    "left a crockpot going while they were at work and came home to find their entire house smelling like burnt chili — the smell has been there for three weeks and nothing removes it",
    "tried making tamales from scratch for the first time — it took eleven hours, produced fourteen tamales, and their abuela tasted one and said 'maybe next time'",
    "accidentally grabbed the wrong bag at the grocery store self-checkout and didn't realize until they got home — the bag contained four avocados, a candle, and a pregnancy test",
    "made a casserole so bad even the dog wouldn't eat it — their kid said 'I'd rather have detention lunch' and their spouse quietly ordered pizza",
    "brought the wrong dish to a funeral potluck — it was a birthday cake they'd picked up for another event and they had to explain the 'Happy Birthday!' frosting to the grieving family",
    # --- cars & driving ---
    "their check engine light has been on for so long it went off and they panicked — took it to the shop and the mechanic said 'it just gave up'",
    "got rear-ended at a stoplight and the other driver got out and said 'that's for what you did at the Safeway' — the caller has never been to that Safeway",
    "their car makes a noise that sounds exactly like someone saying 'help' when they turn left — the mechanic can't find anything wrong",
    "found a note on their windshield that said 'nice parking job' with a hand-drawn diagram showing how badly they parked — the diagram is surprisingly detailed and accurate",
    "their GPS once routed them through someone's private property — they drove through a gate, past a barn, and came out on the other side of a mountain on the correct road",
    "parallel parked so perfectly one time that a stranger applauded — it's been three years and they still think about it",
    "locked their keys in the car with the engine running at a gas station — the locksmith who came said 'again?' even though they'd never met",
    "their car's Bluetooth keeps connecting to a stranger's phone and they can hear the stranger's music — it's always smooth jazz and they've started to enjoy it",
    "drove through a car wash and their side mirror got ripped off — the car wash said their sign clearly states 'fold in mirrors' but the sign is in 6-point font behind a bush",
    # --- medical oddities ---
    "sneezed so hard they threw out their back and had to call an ambulance — the paramedic said it was the third sneeze-related call that week",
    "went to the doctor for a routine checkup and the doctor said 'huh, that's interesting' and then walked out to get another doctor — both of them said 'huh' and nobody explained",
    "hiccupped for 72 hours straight and tried every remedy anyone suggested — the thing that finally stopped it was their kid jumping out of a closet and scaring them so badly they fell off a chair",
    "found out they've been allergic to something they eat every single day and the symptoms they thought were normal are not normal at all",
    "their dentist found a baby tooth they never lost — they're 42 and it's been there the whole time and now they need a plan",
    "got a splinter in their foot six months ago and just assumed it worked itself out — it did not and the doctor's reaction was memorable",
    "went to urgent care for a stomachache and the intake nurse asked 'on a scale of 1 to 10' and they said '4' and the nurse said 'you look like a 9, be honest'",
    # --- unexplained phenomena ---
    "keeps finding pennies in their shoes every morning — they live alone and the doors are locked — it's been happening for three months",
    "their clock stops at the same time every night — 3:17am — they've replaced the batteries, bought a new clock, and it still happens",
    "found handwriting on their bathroom mirror that wasn't there the night before — it says 'behind the furnace' and they haven't checked yet and they're calling because they need someone to tell them to check",
    "their deceased grandmother's phone number called them — the number has been disconnected for two years and when they answered it was just static",
    "woke up in the middle of the night to their TV playing a show that doesn't exist — they searched for it the next day and found nothing",
    "has been having the same dream every Tuesday for two months — it's incredibly mundane, just grocery shopping, but everything in the dream shows up in their actual life the next day",
    "found a photo of themselves in a place they've never been — they're clearly in the photo, it's their face, but they have no memory of it and neither does anyone they know",
    "keeps hearing a phone ring inside their walls but there's no phone line connected to the house — it rings three times every evening around 9pm",
    # --- small town life ---
    "their town's only restaurant changed the recipe for the green chile and there's a petition with 200 signatures demanding they change it back — the caller started the petition",
    "the local bar has a jukebox that someone keeps loading with $20 worth of 'What's New Pussycat' by Tom Jones every Saturday — nobody knows who it is but the bartender is losing their mind",
    "their town had a power outage and everyone went outside and hung out in the street for three hours — it was the best night they've had in years and they're weirdly hoping it happens again",
    "someone in town put up a billboard that just says 'WE KNOW WHAT YOU DID, GERALD' and nobody named Gerald will admit to anything but three Geralds have left town",
    "their small town has a feud between two competing taco trucks that's been going on for eight years — families are divided, there are bumper stickers, and the caller is a double agent eating at both",
    "the only traffic light in town has been yellow-flashing for six months and the town council keeps tabling the repair because they can't agree on whether to make it a stop sign instead",
    "their town's annual chili cook-off was won by someone using store-bought chili and the scandal is bigger than anything that's happened there in decades",
    "the local cemetery started doing 'historical ghost tours' and one of the ancestors featured is the caller's great-great-grandfather — the tour guide gets the story completely wrong every time",
    "their town's volunteer fire department calendar fundraiser accidentally featured the same guy three times because nobody else signed up",
    "the one gas station in town raised prices by a penny and it made the front page of the weekly paper — the editorial was 800 words long",
    "the town's only barber retired and now everyone drives 40 minutes for a haircut — someone suggested a co-op barbershop and the town council has been debating it for six months",
    "their town's Facebook group has devolved into a full civil war over whether the new stoplight is helping or hurting traffic — someone made a PowerPoint",
    # --- embarrassing moments ---
    "waved back at someone who wasn't waving at them and committed so hard they walked over and introduced themselves — the person was calling their dog",
    "called their teacher 'mom' in high school and somehow that became their nickname for the rest of the year — they graduated with it in the yearbook",
    "walked around all day with their shirt inside out and nobody said anything until their kid picked them up and said 'did you lose a bet'",
    "confidently answered a trivia question at a bar and was so wrong the entire bar went silent — they still go there and people bring it up",
    "sent a selfie meant for their partner to their boss — the selfie was innocent but the caption was not",
    "tripped walking into a job interview, knocked over a plant, and somehow still got the job — the interviewer said 'we admire your commitment to showing up'",
    "fell asleep at a movie theater and woke up shouting during a quiet scene — the person next to them said 'same' and they bonded",
    "accidentally put salt instead of sugar in a pie they brought to a work potluck — watched six coworkers take bites and try to be polite about it before someone finally cracked",
    # --- random life chaos ---
    "found a suitcase on the side of the highway and brought it home — it's full of bowling trophies from the 1980s all belonging to the same person",
    "their car horn started going off randomly and they can't fix it — it goes off in parking lots, at stoplights, and once during a funeral procession",
    "accidentally volunteered to coach their kid's sports team by raising their hand to ask a question at the parent meeting — they don't know the rules of the sport",
    "woke up to find their front door wide open and nothing stolen — but someone had rearranged their living room furniture and it actually looks better",
    "got a letter addressed to 'Current Resident' that was a handwritten apology for something that happened in the house in 1987 — they have no idea what it's referring to",
    "their smoke detector has been chirping for eight months and they can't figure out which one it is — they've replaced the batteries in all of them twice",
    "won a radio contest they don't remember entering and the prize is a year's supply of something they're allergic to",
    "found their car parked two blocks from where they left it with more gas than it had before — nothing was stolen but the seat was adjusted",
    "their kid told their class that their parent 'makes people disappear for a living' — the caller is a professional organizer who helps people declutter",
    "went to vote and found out someone had already voted under their name — they've been dealing with the county clerk's office for three months and nobody can explain it",
    "has a bathroom faucet that only runs hot water on Wednesdays — every plumber they've called has said that's not possible and yet",
    "their electricity bill tripled last month and the power company says there's no issue — they suspect their neighbor is running an extension cord from their outdoor outlet but can't prove it",
    "found a crawl space in their house they didn't know existed — it had a sleeping bag, a book, and a half-eaten can of beans and they're trying to figure out how recently someone was in there",
    "their garage door opens at exactly 3am every night — the remote is in a drawer and the button isn't stuck and the repair company can't explain it",
    "accidentally mailed their rent check to their cable company and their cable payment to their landlord — neither noticed for two months",
    "keeps getting someone else's prescription glasses in the mail from an online eyewear company — the prescription is almost exactly theirs and the frames are nice so they've been wearing them",
    "found a journal wedged behind a bathroom wall during a renovation — it's someone's detailed diary from 1994 and the last entry says 'if you're reading this, the closet floor isn't what it seems'",
    # Comedy writer entries
    "walked in on their roommate having a full conversation with a sex doll at the kitchen table — not a sexual situation, they were eating breakfast across from it and arguing about politics — and when the caller said 'what the hell' the roommate said 'do you mind, we're in the middle of something'",
    "got a lap dance at a strip club and halfway through realized the dancer was their kid's second-grade teacher — they made eye contact, she said 'we will never speak of this,' he said 'agreed,' and now they both pretend not to recognize each other at parent-teacher conferences",
    "accidentally liked their ex's Instagram photo from 2019 at 3am and instead of unliking it they panicked and liked every single photo going back to 2016 so it would look like they were hacked — the ex called the next morning and said 'are you okay' and they said 'I think someone got into my account' and the ex said 'whoever it was also ordered you a pizza because I can see the Domino's box on your story'",
    "clogged the toilet at their boss's dinner party and couldn't find a plunger so they reached in barehanded and fixed it — washed their hands for five minutes, came back to the table, and their boss handed them a bread roll and said 'you've got great hands, you should try the piano' and they've never told anyone until now",
    "their elderly neighbor died and they went to the estate sale and accidentally bought back their own lawnmower the neighbor had 'borrowed' seven years ago — they paid forty dollars for their own property and didn't realize until they saw the scratch mark from when their kid hit the fence with it",
    "went on a first date and the woman asked 'what do you do for fun' and they blanked so hard they said 'I collect rocks' — they don't collect rocks, have never collected rocks, but now they're six dates in and she bought them a geode for their birthday and they have a shelf of rocks they pretend to care about",
    "farted so loud during a moment of silence at a funeral that the pastor stopped and looked directly at them — the deceased's wife started laughing which made the whole front row laugh and now the family says grandpa would have loved it but the caller has not recovered",
    "got pulled over doing 95 in a 55 and when the cop asked where the fire was they accidentally said 'my wife is having a baby' — the cop gave them a full escort to the hospital with lights and sirens and they had to stand in the maternity ward explaining to nurses that nobody was actually pregnant while the cop waited to congratulate them",
    "accidentally sent a sext meant for their girlfriend to the family group chat — their dad responded 'wrong chat, son' and their mother hasn't spoken to them in three weeks but their uncle sent a thumbs up",
    "their coworker microwaved fish at work and when someone complained, the coworker brought in a laminated printout of the company handbook highlighting that there's no policy against it — then started microwaving increasingly aggressive fish every day as a form of protest and HR is now involved in what they're calling 'the fish situation'",
    "told their barber they liked the haircut when they didn't and has been going to the same barber getting the same bad haircut for four years because they can't figure out how to ask for something different without admitting they've been lying since the first visit",
    "went to their high school reunion and someone said 'you look exactly the same' and they can't figure out if it was a compliment because they were ugly in high school and they've been thinking about it every day for two months",
    "sneezed during a work video call and their camera unfroze at the exact moment their face was fully contorted — someone screenshotted it and it's been the team's Slack emoji for six months and they can't get IT to remove it",
    "their Tinder date showed up and it was their cousin's ex-wife — they both knew immediately but neither said anything and they sat through an entire dinner making small talk about the weather before she said 'this never happened' and he said 'what never happened' and they've never spoken again",
    "was at a urinal and their boss walked up to the one next to them and started a performance review — full eye contact, talked about quarterly goals, mentioned areas for improvement — and the caller didn't know whether to respond professionally or acknowledge that they were both holding their dicks",
    "went to a couples massage with their wife and accidentally moaned — not a little, a full audible moan — and the masseuse stopped, their wife sat up, and nobody has spoken about it but his wife has not booked another massage and it's been eight months",
    "got into a fender bender in a grocery store parking lot and when they got out to exchange information it was the same person they'd gotten into a fender bender with two years ago in a different parking lot — the other driver said 'you again?' and they now have each other's insurance memorized",
    "left a brutally honest Yelp review for a restaurant and the owner responded publicly with the caller's full order history — including seventeen orders of the dish they said they hated, a note that they always request extra ranch, and a reminder that they asked for a birthday discount three times in one year",
    "called in sick to work to go to a baseball game and ended up on the Jumbotron — their manager was watching the broadcast and texted them 'nice seats, see you Monday' with no further comment and now they don't know if they're fired or forgiven",
    "their smart speaker overheard them talking trash about their mother-in-law and added 'divorce lawyer' to their shopping list — their wife saw it before they did and the conversation that followed was worse than anything the lawyer could have helped with",
    "was trying to impress a date by cooking dinner and set off the smoke alarm so badly that the fire department came — one of the firefighters looked at the pan and said 'were you trying to cook this or punish it' and the date married them anyway but tells this story at every party",
    "accidentally wore their shirt inside out to a job interview, got the job, and has been wearing the same shirt inside out to work every day because they think it's lucky — a coworker finally told them after three months and they said 'I know' because admitting it was an accident felt worse",
    "got into an argument with a stranger on the internet that lasted three days and when they finally looked at the profile picture they realized they'd been arguing with their own brother using a fake account — neither of them has brought it up in person",
    "told a long, elaborate story at a dinner party and absolutely nailed the delivery — everyone laughed, people applauded — and then their spouse leaned over and whispered 'that happened to me, not you' and they've been telling this person's story as their own for so long they genuinely forgot",
    "their dog got loose and when they found him he was sitting on the porch of a house three blocks away with another family who'd already named him, bought him a bowl, and seemed genuinely upset to give him back — the dog looked at the caller like he'd been caught cheating",
]

ADVICE = [
    # Career/money forks
    "got offered a job that pays 40% more but the company does sketchy stuff — nothing illegal but ethically gray and they'd have to look the other way",
    "found out they can buy the building their business rents but it needs $60k in foundation work and they've only got $20k liquid",
    "their side hustle is now making more than their day job but has no benefits — they have a kid with a medical condition and can't risk losing insurance",
    "got accepted to two grad programs — one is prestigious but across the country, the other is local and their aging parents need them close",
    "inherited $80,000 and half the family says invest it, half says pay off debt — the debt has low interest but the weight of it is crushing them",
    "their business partner wants to bring in an investor but the investor wants 40% equity and a board seat — the money would let them grow but they'd lose control",
    "was offered early retirement at 52 with a decent package but they're not sure they can afford 30+ years without working — their spouse says take it",
    "has a chance to buy their childhood home from a family member at below market value but it needs $100k in work and they'd have to sell their current house first",
    # Family/relationship crossroads
    "aging parent wants to move in but last time they lived together it nearly ended their marriage — the alternative is a facility the parent can barely afford",
    "their spouse wants to homeschool their kids and they think it's a terrible idea — the local schools aren't great but they value socialization",
    "found out they can't have kids biologically and they're split on adoption vs. IVF vs. accepting it — their partner is leaning one way and they're leaning another",
    "their adult kid moved back home after a divorce and it was supposed to be temporary — it's been eight months and there's no plan to leave",
    "their in-laws want to spend every holiday together and their spouse agrees but they haven't seen their own family for Thanksgiving in four years",
    "best friend asked them to be a business partner and they love the idea but they've seen money ruin friendships — the friend is putting up most of the capital",
    "their teenager wants to skip college and start a business — the kid has a real plan and some traction but they can't shake the feeling it's a mistake",
    # Life decisions
    "thinking about leaving a small town they've lived in for 30 years — the town is dying but all their roots are here",
    "got a job offer in another country and they have 10 days to decide — it's a once-in-a-lifetime opportunity but they'd be leaving everything",
    "wants to blow the whistle on something at work but the company is the biggest employer in town and people will lose jobs if it goes public",
    "found out their house is in a flood zone that's getting worse every year — they can sell now at a loss or wait and risk losing everything",
    "their doctor told them they need a lifestyle change or they'll be on medication for life — they know what they need to do but can't start",
    "been offered a chance to foster a kid and they want to but their house is small and their schedule is packed — they keep saying 'someday' and wondering if today is the day",
    # Ethical dilemmas with real stakes
    "found out a close friend is cheating on their spouse — the spouse is also their friend and they have dinner with both of them next week",
    "their neighbor's tree is about to fall on their house and the neighbor refuses to deal with it — cutting it themselves would be trespassing",
    "discovered their kid is being bullied but the bully is the child of their boss — they don't know how to address it without risking their job",
    "their mechanic accidentally told them their car is worth three times what they paid — they could flip it but the seller was a family friend who didn't know the value",
    "someone they supervise at work confided in them about a mental health crisis — they should report it per company policy but reporting will get the person fired",
    "knows their landlord is violating building codes in other units but their own rent is below market — if they report it they'll probably lose their lease",
    "their kid found a wallet with $3,000 cash and the kid wants to keep it — there's an ID inside and they could return it but the kid has never had that kind of money",
    "was accidentally overpaid by $5,000 at work and nobody has noticed in three months — they need the money but they know eventually someone will catch it",

    # Gut-wrenching ethical dilemmas
    "is a social worker who has to recommend whether to remove a child from a home — the parents love the kid and are trying, but the conditions are bad and getting worse, and the foster system in their county is a nightmare",
    "runs a small business and just found out their most important employee — the one keeping the company alive — is undocumented, and there's an audit coming in two months",
    "their elderly father wants to stop dialysis and die on his own terms — the rest of the family is begging them to convince him to keep going, but they think he has the right to choose and they're being called selfish for not fighting harder",
    "was asked to write a recommendation letter for someone they think is mediocre — the person is a minority candidate and the company desperately needs diversity, and they know a lukewarm letter will tank their chances but an honest letter IS lukewarm",
    "found evidence that their kid's coach is having an inappropriate relationship with a player on another team — not their kid — but the coach is beloved, it could be misinterpreted, and if they're wrong they'll destroy an innocent person's life",
    "is a pharmacist who recognized a regular customer's prescription pattern as doctor shopping for opioids — the customer is also a friend and clearly in chronic pain, and reporting them means they lose access to any pain management",
    "inherited a gun collection worth $200k from their father — they're deeply anti-gun and want to destroy them all, but their father specifically asked that they be kept in the family, and selling them could fund their kid's entire college education",
    "their company is about to lay off 30 people and they've been asked to choose who stays — one of the people on the bubble is a single parent who's mediocre at their job, and the person who'd replace them is brilliant and hungry",
    "discovered their best friend's nonprofit is spending donor money on overhead and salaries that are technically legal but morally sketchy — the friend pays themselves $180k to run a charity that gives away $40k a year",
    "was in a hit and run twenty years ago — they were the one who ran — nobody was seriously hurt but a woman broke her arm, and they've carried it their whole life and just saw a post from the woman saying the driver who hit her ruined her ability to trust people",

    # --- career & purpose forks ---
    "their dream job just opened up but it would mean moving to a state they hate — the pay is double what they make now and they'd never get this chance again",
    "got asked to manage the team they used to be on — half the team is older than them and one of them applied for the same position and didn't get it",
    "their company is offering voluntary buyouts and the math says take it, but they've been there 22 years and don't know who they are without this job",
    "has a chance to take over the family business but the business is failing and their parents won't admit it — saying no means watching it die, saying yes means going down with the ship",
    "passed all the tests to become a firefighter at 41 and the academy starts next month — their spouse says it's reckless, their kid thinks it's the coolest thing ever",
    "got offered a position overseas that would double their salary but they'd miss their kid's last two years of high school",
    "wants to go back to school at 45 for nursing but the program is two years with no income — their spouse supports it emotionally but they'd have to drain savings",
    "was offered a partnership at their law firm but the senior partner they'd be tied to is someone they fundamentally disagree with ethically",
    "has been running a side business from their garage and a chain store wants to buy the brand name for $150k — it's good money but the brand is their identity",
    "their startup is running out of runway and an investor offered to bail them out but wants them to fire their cofounder — the cofounder is their best friend",

    # --- relationship crossroads ---
    "their partner of eight years won't commit to marriage and they've been patient but they're turning 38 and want kids — they love this person but the clock is real",
    "caught their spouse in a lie about where they were last night — nothing as bad as an affair, but they were gambling at a casino and they have a history of addiction",
    "their partner wants to open the relationship and they don't — the partner says they'll resent being 'caged' and the caller says they'll resent being 'shared'",
    "reconnected with an ex who's now sober and completely different — they're married and happy, but there's a pull they can't explain and they haven't told their spouse they've been talking",
    "their fiancé's family is demanding a huge wedding they can't afford — the fiancé sides with the family and the caller feels like they're already losing battles before the marriage starts",
    "married their high school sweetheart and they love them but they've never been with anyone else and wonder if they're missing something — is that normal or a sign",
    "their partner wants to quit their stable job to become a full-time artist — they're talented but not making money yet and the caller is the only income",
    "found out their fiancé doesn't want kids after saying they did during dating — the wedding is in three months",

    # --- family decisions ---
    "their parent with dementia is becoming dangerous — left the stove on twice, wandered into the road — but the parent is lucid enough to refuse help and gets angry when anyone suggests a facility",
    "their teenager wants to live with their other parent after the divorce — the caller has primary custody and the request feels like rejection even though they know it shouldn't",
    "adopted their nephew after their sibling went to jail — the sibling is getting out next year and wants the kid back but the kid calls the caller 'mom' now",
    "their mother-in-law wants to move in and their spouse already said yes without asking — the caller loves their spouse but can barely tolerate the mother-in-law for a weekend",
    "their adult child is in a relationship they think is abusive but every time they bring it up the child cuts them off for weeks — they don't know if saying something helps or pushes them away",
    "their sibling wants to sell the family home their parents built — the caller can't afford to buy them out and watching it go to strangers feels like losing their parents all over again",
    "their kid came out as trans and they want to be supportive but they're struggling in ways they're ashamed of — they're not transphobic, they just weren't prepared and don't know who to talk to",
    "found out their father isn't their biological father through a DNA test — their mother confessed when confronted but their father doesn't know they know",

    # --- ethical dilemmas ---
    "witnessed a coworker steal from the company but the coworker is a single parent who's about to lose their apartment — reporting them would end their career",
    "found out their friend is driving without insurance and has a suspended license — the friend drives their kid to school every morning",
    "their neighbor confided they're an undocumented immigrant who's lived here for 15 years — someone in the neighborhood has been calling ICE on people and the neighbor is terrified",
    "overheard their kid's coach making comments that border on inappropriate but aren't quite over the line — other parents think the coach is great and they don't want to be 'that parent'",
    "their friend asked them to be an alibi for something they won't specify — the friend has never asked for anything before and seems genuinely scared",
    "their boss asked them to backdate a document and it's technically not illegal but it's definitely not right — saying no means the boss holds a grudge, saying yes means they're complicit",
    "a close friend confessed to a crime from ten years ago during a late-night conversation — nobody was hurt but it's been eating at the friend and now it's eating at the caller",
    "their kid found a gun in a friend's house — an unlocked, loaded gun in a bedroom where kids play — and they want to tell the friend's parents but their own kid begged them not to because they'd lose their best friend",

    # --- money & property decisions ---
    "inherited a piece of land with emotional value to the family but the property taxes are bleeding them dry — selling it would pay off their house but their siblings would never forgive them",
    "someone rear-ended them and is begging not to go through insurance because their rates will spike — the damage is $3,000 and the person offered to pay cash in installments",
    "won $20k in a settlement and can't decide between paying off debt or investing in a business idea they've had for years — the business is risky but the debt isn't going anywhere",
    "their financial advisor is recommending they cash out a life insurance policy to invest in the market — the math makes sense but their gut says no and they can't explain why",
    "cosigned a loan for their kid and the kid stopped making payments three months ago — their own credit is tanking and confronting the kid will blow up the relationship",
    "got a lowball offer on their house but the market is dropping and waiting could mean an even lower offer — their agent says sell now, their spouse says wait",
    "their landlord offered to sell them the house they've rented for ten years at below market value but the inspection revealed foundation issues that could cost $50k to fix",

    # --- life direction ---
    "just turned 50 and realized they've never left the state they were born in — their kids are grown, their spouse is open to it, and they have just enough saved to make a move but no plan",
    "been sober for five years and their old friend group keeps inviting them to things where drinking is the main event — they don't want to lose the friendships but every invite is a risk",
    "their doctor gave them a wake-up call about their weight and they know they need to change but they've failed every diet and exercise plan they've tried — asking what actually works",
    "retired at 60 and is bored out of their mind after two months — they thought they'd love it but they have no hobbies and their spouse is annoyed they're always around",
    "got diagnosed with something manageable but chronic and it's making them rethink every priority — should they stay in the career they hate or do the thing they've been putting off",
    "their best friend is in a cult — they won't call it that, but the caller has done the research and it ticks every box — and they don't know how to have that conversation",
    "wants to report their company for an environmental violation but they're the sole provider for three kids and whistleblowers in their industry don't get hired again",
    "is 30 and has no idea what they want to do with their life — tried college, tried trades, tried the military, nothing stuck and everyone else seems to have it figured out",

    # --- neighbor & community dilemmas ---
    "their neighbor's dog barks for eight hours a day while the owner is at work — they've talked to the owner twice, filed a noise complaint, and nothing has changed",
    "found out the house next door is being turned into an Airbnb and parties have already started — property values are at stake and the city says it's legal",
    "their kid's best friend's family is going through a rough patch and the kid is practically living at their house — they're happy to help but it's been four months and nobody's talked about boundaries",
    "lives in a small town and their kid's teacher is also their neighbor — the teacher gave the kid a C they think is unfair and they can't address it without it being personal",

    # --- trust & honesty ---
    "their spouse went through their phone and found nothing but the fact that they looked has destroyed trust — now the caller is angry and the spouse is defensive",
    "lied about their education on a job application fifteen years ago and has been quietly terrified ever since — they've been promoted four times and are now in a role where someone might actually check",
    "their therapist accidentally revealed something another client said — not by name but the details were so specific the caller knows exactly who it is, and it's someone they know",
    "found their kid's diary and read it — what they found isn't dangerous but it's personal and now they can't unknow it and don't know how to act normal",
    "their friend confessed they're the one who anonymously reported the caller's other friend to CPS two years ago — the report was unfounded and nearly destroyed a family",

    # --- More career/money crossroads ---
    "was offered a partnership at a firm that does work they find morally questionable — the money would solve all their problems but they'd have to sign their name to things they don't believe in",
    "their small business landlord offered to sell the building at a fair price but they'd have to take on massive debt — if the business fails they lose everything, if it succeeds they own the whole thing",
    "got accepted to a program that would retrain them for a completely different career at 48 — it means two years of no income and starting over from the bottom",
    "their employer offered relocation to keep their job or a severance package to leave — the new city is terrible for their family but the severance only covers six months",
    "has a chance to buy a franchise that's been profitable for other owners but it requires liquidating their entire retirement — their financial advisor says no, their gut says yes",
    "got offered a book deal to write about their industry and the publisher wants dirt — telling the truth would burn every bridge they have but the advance is life-changing money",
    "their company is going public and they have stock options worth $200k — they can exercise now and pay huge taxes or wait and risk the stock tanking",
    "was asked to take a pay cut to keep their team from layoffs — the CEO hasn't taken a pay cut and the caller isn't sure solidarity goes that far",

    # --- More relationship crossroads ---
    "their ex is remarrying and wants to introduce the new spouse to the kids — the caller has full custody and the ex abandoned them three years ago",
    "discovered their partner has a gambling problem after finding the receipts — the partner denies it and says it's 'entertainment spending' but the numbers don't lie",
    "their spouse got a job offer that would triple their income but it's in a city the caller hates with a culture that's the opposite of how they want to raise their kids",
    "met someone at a conference who they connected with instantly — nothing happened but they haven't stopped thinking about this person and they're happily married",
    "their partner wants to sell everything and travel full-time in an RV — the caller likes the idea in theory but has aging parents and a kid in middle school",
    "their ex wants to reconcile after five years apart — the caller still has feelings but the reasons they left haven't changed",

    # --- More family decisions ---
    "their parent wants to give away their entire estate to charity before they die and leave nothing to the family — the parent says the kids don't need it but the caller disagrees",
    "their sibling came out of prison after eight years and wants back into the family — some members are ready, others say no, and the caller is the swing vote",
    "their kid wants to change their last name to their stepparent's name — the caller is the biological parent and it stings even though they understand",
    "their aging parent refuses to write a will and the family is too afraid to push the issue — the caller knows what's going to happen when the parent dies and it terrifies them",
    "found out their kid has been secretly visiting their estranged grandparent — the caller cut the grandparent off for good reason but the kid doesn't know the full story",
    "their partner wants to do a destination wedding that half the family can't afford to attend — saying no feels controlling but saying yes means empty seats",

    # --- More ethical dilemmas ---
    "their tenant is three months behind on rent and has a newborn — the caller needs the rental income to pay their own mortgage but can't live with themselves for evicting a baby",
    "found out a coworker falsified a safety report and nobody got hurt this time — reporting it would shut down the project and cost twenty people their jobs",
    "their friend asked them to cosign a loan and the friend has bad credit for reasons that are understandable — but the caller can't afford to cover if the friend defaults",
    "caught a kid shoplifting at their store — the kid is clearly doing it for food and the caller's policy says call the cops but everything in them says let the kid go",
    "their neighbor asked them to testify in a property dispute against another neighbor — both are friends and both think they're right, and the caller actually agrees with the side that's losing",
    "a homeless person has been sleeping in their business's parking structure and they feel for the person but customers are complaining and it's affecting revenue",

    # --- More life direction ---
    "turning 40 next month and made a list of everything they said they'd do by now — not a single thing on it is checked off and they don't know whether to grieve the plan or make a new one",
    "their therapist suggested they might have ADHD and if the diagnosis is confirmed it would explain thirty years of struggle — but they're afraid of what it means",
    "considering converting to a different religion than the one they were raised in — the spiritual pull is real but the family consequences would be severe",
    "just got laid off for the second time in three years and is questioning whether their entire career path was a mistake",
    "wants to confront their aging parent about something that happened in childhood — the statute of limitations on everything has passed but the emotional weight hasn't",
    "their whole friend group is getting married and having kids and they have zero interest in either — they're happy but starting to wonder if they're broken or just different",
    "has a chance to study abroad for a year at 55 — the kids are grown, the job would survive, but their spouse says it's selfish to leave",
    "thinking about buying land and going off-grid — they've done the math and it works but their partner thinks they're having a midlife crisis",
    "been volunteering at a hospice and it's changing how they see their own life — wants advice on whether to leave their career and do it full-time for a fraction of the pay",
    "their doctor told them stress is literally killing them and they need to make a major life change within the year — they know what needs to go but they're too scared to let it go",

    # --- More trust & honesty ---
    "told their partner a small lie years ago that has grown into a foundational assumption of their relationship — correcting it now would unravel years of shared decisions based on it",
    "their friend group has been talking behind someone's back about their behavior and the caller was asked to be the one to have 'the conversation' — they're not sure it's their place",
    "received information about a friend's spouse that could destroy their marriage — the information came from someone unreliable and the caller doesn't know if it's true",
    "their employer asked them to sign a non-disparagement agreement as part of a settlement for something that actually happened — signing means they can never tell the truth publicly",
    "their kid asked them point blank if Santa is real and they don't believe in lying to their kids but their spouse wants to keep the magic going for one more year",
    # --- reaching 150+ ---
    "their neighbor cut down a tree that was on the property line and now both yards flood when it rains — neither will pay for drainage and it's getting worse each storm",
    "was invited to join a business venture with their in-laws and their spouse thinks it's a great idea — every financial advisor says never mix family and money",
    "their coworker asked them for a kidney — a real kidney — they're a match and the coworker has no other options and they have to make a decision they never imagined facing",
    "found out their home's previous owner had a meth lab in the garage — the house passed inspection but now they're getting headaches and wondering if it was properly remediated",
    "their therapist retired and recommended someone who uses a completely different approach — they don't want to start over but they also know they still need help",
    "has been offered a spot on the city council which they've always wanted — but it means their private life becomes public in a small town where everyone talks",
    "their kid wants to take a gap year and travel instead of going straight to college — the caller sees the value but worries the kid won't go back",
    "their landlord is offering a rent-to-own deal that sounds too good to be true — the numbers work on paper but something feels off and they can't put their finger on it",
    "their best friend from childhood wants to reconnect but the friend has changed dramatically — different values, different lifestyle, different everything — and the caller isn't sure there's enough left to rebuild on",
    "their elderly parent wants to give away their house to the church and the family is split — half say it's the parent's right, half say the parent is being manipulated",
    "found out their partner lied on their resume about having a degree they don't have — the partner makes good money and does good work but could be fired if anyone checks",
    "thinking about telling their boss they're looking for other jobs as a negotiation tactic — but if the boss calls their bluff they're stuck with no backup plan",

    # --- parenting dilemmas (backend-dev) ---
    "their kid wants to drop out of college after one semester and the caller already paid the full year — the kid says they're miserable and learning nothing",
    "caught their teenager vaping and grounded them — now the kid's grades have dropped and they're questioning if the punishment is doing more harm than good",
    "their kid is being excluded by a friend group and wants to switch schools — it's mid-year and the logistics are a nightmare but the kid cries every morning before school",
    "found out their teenager has been skipping church to go to a friend's house — the caller is devout and the spouse thinks it's rebellion but the caller thinks the kid just doesn't believe anymore",
    "their kid wants to take a gap year before college and work on a farm — the caller's parents are horrified and their spouse is neutral and they genuinely don't know what's right",
    "co-parenting with their ex and the ex's new partner is undermining their rules — the kid comes back from their ex's house with later bedtimes, more screen time, and an attitude",
    "their kid asked why they got divorced and they don't know how to answer honestly without badmouthing the other parent",
    "discovered their kid has been lying about going to school and has missed 15 days this semester — the kid is smart and their grades are somehow still decent",
    "their youngest just left for college and the house is empty for the first time in 22 years — they thought they'd be relieved but they're devastated and their spouse seems fine",
    "their adult kid asked to borrow $10k and won't say what it's for — the kid has a good job and has never asked before so it's either very good or very bad",
    "their kid's teacher wants to hold them back a grade and the kid is begging not to — the caller thinks the teacher might be right but the social damage of being held back worries them more",
    "their teenager's best friend is a terrible influence and they've seen the grades drop — banning the friendship feels authoritarian but doing nothing feels negligent",
    "their kid wants to join the military and the caller is terrified — the kid has thought it through and it's a good career path but the caller can't stop thinking about worst-case scenarios",
    # --- aging parents (backend-dev) ---
    "their parent has started giving away possessions and talking about death and they can't tell if it's acceptance or depression — bringing it up feels invasive",
    "their mother fell and won't go to the doctor — she says she's fine but she's limping and the caller lives two hours away and can't check on her every day",
    "their parent is dating someone new eight months after the other parent died — the family is split between supportive and furious and nobody knows how to act at dinner",
    "their father's dementia is getting worse and he keeps asking for their mother who died ten years ago — they don't know if they should keep telling him or let him believe she's coming",
    "their parent needs to stop driving but won't — they've had three fender benders this year and the caller is terrified they'll hurt someone",
    "their parent is giving large amounts of money to a new 'friend' they met online — the caller thinks it's a scam but the parent says they're just jealous",
    "their siblings refuse to help with their mother's care and the caller is doing everything — they want to confront them but their mother begged them not to cause family drama",
    "their parent has a DNR and the caller disagrees with the decision — the parent is relatively healthy and the DNR feels premature but the parent says it's their choice",
    "their parent keeps calling them their sibling's name — the sibling died two years ago and every time it happens it breaks the caller's heart",
    "their mother's boyfriend moved in after three months of dating and the caller doesn't trust him — he's already on the bank account and the mother won't hear criticism",
    "their father refuses to move out of a house he can't maintain alone — the yard is overgrown, the roof leaks, and he won't accept help because he sees it as losing independence",
    # --- friendship crises (backend-dev) ---
    "their best friend borrowed $5,000 two years ago and has never mentioned it since — the caller needs the money back but bringing it up feels like it'll end the friendship",
    "found out their friend group has a separate group chat without them — they saw it on a friend's phone and the chat name includes an inside joke they're clearly the butt of",
    "their friend is marrying someone the caller thinks is terrible for them — everyone else seems to see it too but nobody will say anything and the wedding is in three months",
    "their lifelong best friend just told them they've been in love with them for years — the caller doesn't feel the same way and doesn't know how to keep the friendship",
    "was asked to be a bridesmaid and the wedding costs are already at $4,000 in flights, dress, bachelorette — they can't afford it but saying something now feels too late",
    "their friend ghosted them six months ago with no explanation — they ran into each other at the store and the friend acted like nothing happened and they're angry but also relieved",
    "helped a friend through a crisis last year and now the friend is going through the exact same thing again — the caller is burned out on the friendship but feels guilty walking away",
    "their friend group wants to take an expensive vacation together and the caller can't afford it — everyone else is well-off and they don't want to be the reason plans change",
    "their closest friend just got the job the caller applied for — the friend doesn't know the caller applied and keeps talking about how excited they are",
    "their oldest friend has become someone they don't recognize — different values, different politics, different priorities — and they're mourning a friendship that technically still exists",
    "their friend asked them to lie to their friend's spouse about where they were last night — nothing happened but the situation looks bad and the caller doesn't want to be part of the deception",
    # --- housing & living situations (backend-dev) ---
    "their landlord is selling the building and the new owner wants to raise rent 40% — they've lived there for eight years and can't afford anywhere else in the neighborhood",
    "bought their first house and within a month discovered mold, a cracked foundation, and the previous owner lied on the disclosure — their inspector missed everything",
    "their roommate's partner has basically moved in without paying rent — they're there every night, using the kitchen, taking showers, and the roommate says it's temporary",
    "inherited a house jointly with their siblings and nobody can agree on what to do — one wants to sell, one wants to rent it, one wants to live in it, and the house is deteriorating while they argue",
    "their downstairs neighbor plays music until 2am every night and management says they can't do anything because it's 'within acceptable hours' per the lease",
    "their HOA is threatening to fine them $500 for a fence they built with HOA approval last year — the board changed and the new president says the approval was invalid",
    "wants to build an addition on their house but the neighbor objected to the building permit — the neighbor says it'll block their view and they've been friends for a decade",
    "their kid wants to move back home with their spouse and baby — the caller's house is small but their kid is struggling financially and the caller can't say no to their grandchild",
    "found out their septic system needs a $15,000 replacement and the house is only worth $120,000 — fixing it means staying, not fixing it means they can't sell",
    "their neighbor built a shed that's two feet over the property line and won't move it — the caller doesn't want to sue but the shed blocks their garden's sunlight",
    # --- romance & relationships (backend-dev) ---
    "been with their partner for 15 years and they've never said 'I love you' — the relationship is solid and happy but the absence of those words has started to bother them",
    "their spouse wants an open relationship and the caller doesn't — the spouse says it's about trust and the caller thinks it's about wanting someone else specifically",
    "reconnected with their high school sweetheart on social media and the chemistry is still there — they're both married to other people",
    "their partner got a job across the country and they both agreed to do long distance — it's been four months and it's destroying them but neither wants to be the one to say it's not working",
    "proposed to their partner and got 'I need to think about it' — that was three weeks ago and neither of them has brought it up since",
    "their spouse makes significantly more money and has started making all the financial decisions without input — the caller feels like a dependent, not a partner",
    "found out their partner has been talking to an ex regularly — the conversations are innocent but the secrecy is what bothers the caller",
    "been dating someone for six months who is perfect on paper but the caller feels nothing — they keep hoping feelings will develop but they haven't and now they feel trapped",
    "their ex wants to get back together and they're tempted — everyone in their life says it's a terrible idea and they know the same problems will come back but they're lonely",
    "their spouse's family hates them and the spouse won't stand up for them — every holiday is miserable and the caller is starting to resent both the family and the spouse",
    "caught their spouse in a lie that isn't about cheating but is about money — a secret credit card with $12,000 on it that the caller didn't know existed",
    "their partner wants kids and they don't — they thought they'd come around but they haven't and the clock is ticking and someone has to compromise or leave",
    # --- moral gray areas (backend-dev) ---
    "found a wallet with $2,000 cash and no ID — there's no way to return it and they're three months behind on rent but keeping it feels wrong",
    "saw a shoplifter at the grocery store — the person was clearly stealing food, not luxury items, and looked like they were struggling, and the caller didn't say anything but feels weird about it",
    "their kid's friend told their kid a secret about abuse at home and the kid told the caller — the friend begged the caller's kid not to tell anyone and calling CPS could make things worse or better",
    "accidentally received a double refund from a company and they need the money — it's a big corporation and nobody will notice but they were raised to be honest",
    "a stranger accidentally Venmo'd them $500 and the transaction can't be reversed because the account was closed — they could just keep it but they know it was a mistake",
    "their company is underpaying a new hire who doesn't know the market rate — the caller could tell them and help them negotiate but it might blow back on the caller",
    "their kid's coach is clearly favoring their own child for playing time and it's affecting the team — other parents are grumbling but nobody wants to be the one to say something",
    "found out their Uber driver doesn't have insurance and they're in the car right now — do they say something, report it after, or just pretend they don't know",
    # --- health & wellness (backend-dev) ---
    "their doctor recommended surgery but a second opinion said physical therapy might work — the surgery has a faster recovery but PT has no risk of complications and they can't decide",
    "been prescribed medication for anxiety and it works but they feel like a different person — their family says they seem better but the caller misses feeling like themselves",
    "their spouse refuses to go to therapy even though they clearly need it — the caller can't force them but the untreated depression is affecting the whole family",
    "found a lump and has been putting off the appointment for two months because they're terrified of what the doctor might say — they know waiting makes it worse but they're paralyzed",
    "was told they need to quit drinking for medical reasons and they're not sure they can — they don't think they're an alcoholic but two drinks a night for twenty years is apparently a problem",
    "just got a clean bill of health after a cancer scare and instead of feeling relieved they feel worse — the anxiety hasn't gone away and they don't know why",
    "their partner snores so loudly they haven't slept properly in years — separate bedrooms would fix it but the partner takes it personally and they're both exhausted",
    "hasn't been to a dentist in seven years because of a childhood trauma and the pain is getting bad enough that they can't ignore it anymore",
    # --- identity & self-discovery (backend-dev) ---
    "just turned 40 and realized they have no close friends — they have acquaintances and work contacts but nobody they could call at 2am, and they don't know how adults make real friends",
    "grew up in a strict religious household and stopped believing years ago but haven't told their family — every family gathering involves church and prayer and they feel like a fraud",
    "spent their whole life being 'the responsible one' in the family and they're exhausted — their siblings get to make mistakes and they get to clean them up, and they want to stop but don't know how",
    "always wanted to be a writer but never tried because their family said it wasn't a real career — they're 55 and just finished a novel and don't know if showing it to anyone is brave or foolish",
    "has been faking confidence at work for so long they don't know who they actually are anymore — they got promoted because of the persona and now they're trapped in it",
    "grew up poor and now makes good money but can't stop living like they're broke — they won't buy new clothes, won't eat out, won't take vacations, and their partner thinks it's gone from frugal to pathological",
    "wants to change their name because they were named after a family member they recently found out did something terrible — the family says it's just a name but it doesn't feel that way",
    "has been the peacekeeper in their family for decades and just snapped at a dinner — said things they meant but now everyone is acting like they committed a crime and they feel both guilty and free",
    "started going to therapy six months ago and it's changing everything — the problem is they're now seeing dysfunction in every relationship they have and they can't un-see it",
    "never went to college and has done fine but their kid just got accepted to a great school and the caller feels both proud and jealous in a way they weren't expecting",
    # --- community & civic (backend-dev) ---
    "was asked to testify in a neighbor's custody case and they have to be honest — what they've seen doesn't look great for the neighbor and the neighbor considers them a friend",
    "found out the local youth sports coach has a DUI history and nobody vetted them — confronting it means potentially ending a program that 50 kids depend on",
    "their church is taking a political stance they disagree with and they've been a member for 30 years — leaving feels like losing their community but staying feels like endorsing something they can't support",
    "volunteers at a food bank and found out the director is skimming donations — not a lot, maybe $200 a month, but the food bank serves 500 families and every dollar matters",
    "their kid's school is banning books and they disagree with the list — fighting it publicly means their kid becomes 'that parent's kid' in a small town where everyone knows everyone",
    "their town's water supply tested positive for something concerning and the city council downplayed it — the caller has the test results but going public could tank property values for everyone",
    "was elected to a local board they care about but the meetings are three hours long, nobody agrees on anything, and they're starting to understand why nobody else volunteered",
    # --- unexpected life turns (backend-dev) ---
    "just found out they have a half-sibling they never knew about — the sibling reached out online and wants to meet but the caller's parent who had the affair doesn't know they know",
    "won a modest amount of money and told one person — now everyone knows and distant relatives are calling with sob stories and business pitches",
    "was falsely accused of something at work and even though they were cleared, people treat them differently now — they're considering quitting even though they did nothing wrong",
    "their estranged sibling sent them a letter after ten years of silence asking to reconnect — the caller has built a life without them and isn't sure they want to reopen that chapter",
    "their house burned down six months ago and the insurance paid out but they can't bring themselves to buy a new one — they've been staying with friends and something about starting over feels impossible",
    "just learned they were adopted after their adoptive parents died — they found the paperwork while going through the estate and now they don't know what's real",
    "their company went bankrupt and they lost their retirement savings in company stock — they're 58 and starting over with nothing and don't know where to begin",
    "got a DNA match on a genealogy site that suggests their grandfather was involved in a historical crime — the evidence is circumstantial but compelling and they don't know whether to dig deeper",
    "their house appraised for twice what they expected and now they're wondering if they should sell, downsize, and live off the difference — but it's the house their kids grew up in",
    "their identity was stolen and the thief racked up $30,000 in debt — the banks say it's not their problem and the police say it's the banks' problem and nobody is helping",
    "received a letter from a lawyer saying they're named in a will by someone they've never heard of — the inheritance is modest but the mystery is eating at them",
    # Comedy writer entries
    "has been lying about their salary to their spouse for six years — telling them they make $20k less than they do and putting the difference in a secret account — they've got $130k saved and the original reason was a surprise house down payment but now they're addicted to the secret and don't want to stop",
    "found their teenage son's search history and it's not porn — it's hours of research on how to legally emancipate yourself from your parents — the kid is 15, gets good grades, and has never once complained, and the caller doesn't know if they should confront him or just sit with the fact that their kid is quietly planning to leave them",
    "wants to know if they're a bad person for being relieved their mother-in-law's Alzheimer's is getting worse — she was cruel to them for twenty years, called them trash to their face at their own wedding, and now she smiles at them and holds their hand and they finally have the mother-in-law they always wanted and they feel sick about how good it feels",
    "needs advice on whether to tell their friend that nobody likes their friend's cooking — the friend hosts dinner parties every month, everyone pretends the food is good, and now the friend is talking about quitting their accounting job to open a restaurant and the caller is the only one who can stop it but saying something means admitting they've been lying for three years",
    "wants to know if it's okay to break up with someone because of how they chew — they've been together two years and everything else is perfect but the chewing is so loud they've started wearing earbuds at dinner and last week they had a dream about smothering their partner with a pillow and woke up feeling calm",
    "let their neighbor borrow a ladder eight months ago and the neighbor hasn't returned it — they've hinted six times, the neighbor keeps saying 'oh yeah I'll bring it over,' and last week the caller saw the neighbor lending THEIR ladder to another neighbor and they're trying to figure out at what point they can just walk into the guy's garage and take it back without it being a crime",
    "their best friend got a terrible tattoo and keeps asking if it looks good — it does not look good, it's a wolf howling at the moon but it looks like a dog having a seizure, it's on their forearm where everyone can see it, and now the friend wants the caller to get a matching one and the appointment is next Thursday",
    "is in love with their best friend's wife and has been for eight years — nothing has ever happened, they've never said a word, but the friend just asked them to be the executor of his will and the person who takes care of his wife and kids if anything happens to him, and the caller said yes immediately and hates themselves for how fast they said it",
    "secretly got a vasectomy two years ago and hasn't told their wife — she thinks they've been trying for a baby and he goes along with the fertility appointments and the ovulation tracking and watches her cry every month when the test is negative because he's too much of a coward to tell her he doesn't want another kid",
    "found out their brother has been telling people their mother died to get sympathy and free things — their mother is alive and well and living in Tucson, and the brother has used the dead mom story to get out of speeding tickets, get upgraded at hotels, and get a month of free meals from a church group",
    "has been going to the wrong therapist for three months — they mixed up the address at the first appointment, walked into a different practice, and the therapist never questioned it because the caller's name is close to an actual patient's — the therapy has been going great and they don't want to switch",
    "is agonizing over whether to tell their coworker they've been calling another coworker by the wrong name for eleven months — the wrong-name coworker has started answering to it out of politeness and now half the office uses the wrong name too and correcting it would humiliate everyone involved",
]

GOSSIP = [
    # Secret lives
    "just found out their quiet churchgoing neighbor runs an anonymous Instagram reviewing strip clubs with 40k followers — complete with ratings and detailed write-ups",
    "their coworker who brags about being sober was spotted at a bar in the next town doing karaoke, extremely drunk, singing 'Don't Stop Believin' on a table",
    "overheard their boss on speakerphone applying for a job at their company's biggest competitor — the boss was trash-talking the CEO by name",
    "found out their HOA president who is strict about lawn height has a backyard that's basically a junkyard — they saw it on Google Earth",
    "their PTA president who lectures everyone about screen time got caught letting their kids play video games for 8 hours straight at a sleepover",
    "discovered their very religious uncle has a Burning Man habit — they found photos and the man was wearing body paint and not much else",
    "found out the town's strictest health inspector eats gas station sushi every single day — they saw him three days in a row from the same Chevron",
    # Relationship revelations
    "just learned their married neighbor has been having an affair with the mail carrier — they literally watched the pattern for weeks before putting it together",
    "found out two of their friends who supposedly hate each other have been secretly dating for a year — they were making out in a parking garage",
    "their friend who constantly posts about their amazing marriage just filed for divorce and the spouse had no idea it was coming",
    "overheard their sister-in-law on the phone planning a surprise that is definitely not a surprise party — it involves a lawyer and a storage unit",
    "their buddy who claims to be a confirmed bachelor has a secret long-distance girlfriend nobody knows about — they found out because of a shared Netflix account",
    # Professional/financial secrets
    "found out their coworker who drives a new BMW and wears designer clothes is completely broke — the coworker accidentally left a bank statement on the printer showing a negative balance",
    "their neighbor who claims to be a retired executive actually works the night shift at a warehouse — they saw him going in at 11pm in a vest and hard hat",
    "discovered the local restaurant that's always empty but never closes is definitely a front for something — there's never more than two customers but they just renovated",
    "found out their financial advisor who preaches conservative investing just lost $200k on meme stocks — they saw the Robinhood app open on his phone during a meeting",
    "their friend who runs a 'successful consulting firm' just works from Starbucks all day watching YouTube — they sat three tables away for four hours and watched",
    # Unexpected discoveries
    "found out their sweet elderly neighbor was a groupie for a famous rock band in the 70s — there are photos and they are WILD",
    "just learned the crossing guard at their kid's school is a retired professional poker player who won a bracelet at the World Series of Poker",
    "their quiet librarian neighbor writes extremely explicit romance novels under a pen name — they found out because Amazon recommended one based on their address",
    "discovered their dad has a secret record collection of nothing but death metal hidden in the attic — he listens to smooth jazz around the family",
    "found out their coworker who always brings fancy lunches is actually an incredible chef who almost made it on a cooking competition show but got cut in the final round",
    "their friend who swears they've never been on a dating app has five active profiles — they know because three different friends matched with them",
    "overheard the uptight HOA vice president at Home Depot buying supplies for what is clearly an enormous illegal fireworks display",
    "found out their kid's soccer coach used to be in a punk band that opened for Green Day — there's a music video on YouTube with 2 million views",
    "their strait-laced accountant neighbor got drunk at a block party and revealed they were a competitive breakdancer in college — then proved it on the spot",
    "just discovered their coworker's 'service dog' is not a service dog — they overheard them coaching the dog to 'act sad' before walking into the office",
    "found out the guy who runs the neighborhood watch has a Ring camera pointed at everyone's house and a spreadsheet logging who comes and goes — with timestamps and notes",

    # Juicy and morally loaded gossip
    "just found out the couple everyone in town thinks has the perfect marriage are actually swingers — they know because they accidentally got invited to the same party",
    "their squeaky-clean coworker who leads the office Bible study got arrested for solicitation over the weekend — they're the only one who knows and the coworker doesn't know they know",
    "discovered their kid's beloved little league coach did time for armed robbery in another state — he's been clean for fifteen years but nobody in town knows and the parents would lose their minds",
    "overheard a city council member bragging at a bar about approving a development deal in exchange for a kitchen renovation — they have it on their phone's voice recorder",
    "found out their neighbor who runs the local 'Buy Nothing' group has been reselling the free items on eBay at a massive markup — they've been tracking it for months and have screenshots",
    "knows for a fact that the local high school principal and the vice principal are sleeping together — both are married to other people, both have kids at the same school",
    "their friend who posts constantly about being a devoted wife has a separate Instagram where she posts thirst traps and flirts with men in the DMs — they found it because a mutual friend matched with her on a dating app",

    # --- neighborhood drama ---
    "found out the guy who runs the block's annual Fourth of July cookout has been charging people a 'participation fee' and pocketing the money — the food is donated by local businesses",
    "their neighbor's 'home office' is actually a full-blown recording studio and they've been making music all day — the caller only found out because a song came on the radio and they recognized the voice",
    "discovered their HOA board has been meeting secretly at someone's house to decide fines before the official meetings — the caller got ahold of the group text",
    "their neighbor who always complains about noise was caught on Ring camera sneaking into people's yards at night to move their trash cans to generate HOA violations",
    "found out the couple next door who 'renovated their kitchen themselves' actually hired a crew and filmed fake DIY videos pretending they did the work — the videos have 200k views",
    "their neighbor's wife left him three months ago but he's been pretending she's still there — waves at nobody when he leaves, sets two places at dinner visible through the window",
    "discovered the neighborhood's most vocal anti-development activist secretly owns three rental properties through an LLC — they found the filings at the county clerk's office",
    "their neighbor who brags about his organic garden has been buying produce from the farmer's market and repotting it — the caller watched him do it through the fence",
    "found out the sweet old lady who brings cookies to every neighborhood event is the anonymous person who's been writing passive-aggressive letters to the HOA about everyone's Christmas lights",
    "their neighbor who runs a 'home daycare' is actually just letting kids watch iPads in the garage for eight hours — a parent showed up early and saw",

    # --- workplace secrets ---
    "discovered their office 'Employee of the Month' has been nominating themselves under fake names using different email addresses — they found the sent folder open on a shared computer",
    "their coworker who claims to work remotely from their home office is actually working from a beach in Mexico — they saw them on someone's Instagram story with a laptop at a resort",
    "found out the company's 'employee wellness program' is actually just the HR director's side business that she's billing the company for",
    "their boss who preaches work-life balance was seen at the office at midnight three times last week — they know because the cleaning crew told them",
    "discovered their coworker has been running a fantasy football gambling ring using the office Slack — buy-ins are $500 and the IT guy is in on it",
    "their manager who always takes credit for the team's work got caught plagiarizing a presentation from a YouTube video — someone in the meeting recognized it",
    "found out the 'motivational speaker' their company hired for $10,000 got their credentials from an online diploma mill — the caller Googled them out of boredom",
    "their coworker who brags about their MBA got it from a school that was shut down for fraud two years after they graduated — it's technically not accredited anymore",
    "overheard two managers in the bathroom planning to lay off the entire customer service team and outsource it — nobody on the team knows yet",
    "their coworker who always says they 'can't eat gluten' was caught housing a breadstick basket at Olive Garden — the caller was at the next table",

    # --- family secrets ---
    "just found out their grandmother was married twice before their grandfather — nobody in the family knew and they found the marriage certificates in a safety deposit box",
    "discovered their uncle who claims to have been in Vietnam was actually stationed in Germany the entire time — they found his actual service records",
    "their cousin who 'made it big in real estate' is actually deeply in debt and has been borrowing money from every family member separately — the caller compared notes at Thanksgiving",
    "found out their father had a pen pal for forty years that nobody knew about — the letters are deeply personal and the pen pal is a woman in another state",
    "their aunt who claims she's 58 is actually 67 — they found her real birth certificate while helping her move and she begged them not to tell anyone",
    "discovered that the 'family recipe' their mother is famous for was stolen word-for-word from a church cookbook published in 1974 — the original author's name is right there on page 43",
    "just learned that their parents almost got divorced when they were a kid — they found the filed papers in a box, signed by both parents, but it was never finalized",
    "their sibling who claims to be a vegetarian for 'ethical reasons' orders a bacon cheeseburger every time they eat out alone — the caller's friend works at the restaurant",
    "found out their brother-in-law's 'business trips' are actually poker tournaments — he's been hiding the losses by telling his wife the trips are expensed by the company",
    "their cousin who posts about their 'perfect marriage' on social media has been sleeping in their car three nights a week — the caller lives on the same street",

    # --- social circle drama ---
    "found out their friend group has a separate group chat without them in it — they saw it over someone's shoulder and it was called 'without [their name]'",
    "their friend who swears they quit drinking has a hidden mini fridge in their garage full of White Claw — the caller helped them move something and found it",
    "discovered their 'happily single' friend has been on three different dating apps under a fake name with photos from ten years ago",
    "their friend who always picks up the check and acts generous? Every single one of those dinners goes on a company card — the friend's business partner told them",
    "found out the couple in their friend group who is always giving relationship advice has been in couples therapy for two years and the therapist told them to stop advising other people",
    "their friend who claimed to write a bestselling self-help book actually used a ghostwriter — they know because the ghostwriter is their other friend and she's furious she didn't get credit",
    "discovered their gym buddy who lifts impressive weight has been sneaking extra plates on during warmups when nobody's looking — they're actually lifting 60 pounds less than they claim",
    "their friend who posts about minimalism and 'living with less' has a storage unit packed floor to ceiling — the caller rents in the same facility",
    "found out their buddy who 'never watches TV' has an absolutely insane streaming setup with six subscriptions — their wife accidentally shared the family login",
    "their friend who always talks about being broke just bought a $4,000 handbag — they saw the Amex statement because the friend was showing them something else on their phone and swiped too far",

    # --- community figures ---
    "found out the local weather guy on the news doesn't actually have a meteorology degree — he studied theater and just reads the teleprompter really convincingly",
    "their kid's school principal who sends stern emails about attendance has missed more days than any student this year — the secretary let it slip",
    "discovered the town's 'self-made' business owner got their entire startup capital from a rich parent — the owner has been telling a bootstrapping story for twenty years",
    "the local cop who gives everyone grief about speeding got three tickets himself in the next county — someone at the courthouse told them",
    "their pastor who preaches about generosity drives a $90,000 truck and the church is behind on the mortgage — a deacon told them during a heated board meeting",
    "found out the yoga instructor who preaches inner peace and clean living chain-smokes in the parking lot after every class — multiple people have seen it",
    "the local real estate agent who sells 'family-friendly neighborhoods' is being sued by three neighbors for noise complaints from their own house parties",
    "their doctor who lectures them about diet every visit was spotted buying four bags of candy at Costco — the doctor saw them seeing him and neither of them said a word",
    "found out the town librarian who runs the book club has never actually finished any of the books they've picked — they admitted it after three glasses of wine at a dinner party",
    "the local personal trainer who posts transformation photos is using the same 'before' photo for different clients — the caller recognized the same kitchen in three different success stories",

    # --- online & digital discoveries ---
    "found their coworker's secret Reddit account where they post extremely detailed stories about office drama — every single person in the office is recognizable even with fake names",
    "discovered their neighbor has been reviewing their house on Google Maps under a fake name calling it 'an eyesore that ruins the street' — the caller recognized the writing style",
    "their friend who claims to travel all the time has been photoshopping themselves into vacation photos — they noticed the shadows are wrong and the same hotel lobby appears in 'different countries'",
    "found out their kid's tutor has a TikTok where they make fun of their students (without names, but the descriptions are very specific) — the caller recognized their kid's math struggles in one of the videos",
    "their cousin who has 50k followers as a 'fitness influencer' buys most of their followers — the caller checked on one of those follower audit sites and 80% are bots",
    "discovered their elderly father has been catfished — he's been sending money to a 'woman' online for six months who is clearly using stolen photos, and he refuses to believe it",

    # --- money & financial scandals ---
    "found out their friend who organizes charity poker nights has been skimming 30% off the top — they volunteered to help count the money and the math didn't add up",
    "their coworker who's always first to suggest expensive group dinners never actually Venmos their share — the caller did the math and they owe $600 across twelve dinners",
    "discovered the neighbor who refinanced their house 'for renovations' actually used the money to buy a boat that's hidden at a marina two towns over — the wife doesn't know about the boat",
    "their landlord who claims they 'can't afford' to fix the AC is currently building a pool at their own house — the caller drove past it",
    "found out their friend's 'vintage car collection' is mostly bought with money from a personal injury lawsuit they faked — a mutual friend was the one who helped stage the accident",
    "their relative who won $50k in a lawsuit told the family it was only $15k and kept the rest — the caller works at the courthouse and saw the actual settlement",

    # --- identity & secret pasts ---
    "just found out their coworker who claims to be from Connecticut is actually from their same small town — they found a yearbook photo and the coworker had a completely different name",
    "discovered their neighbor who says he's a retired teacher was actually a minor league baseball player who had a brief moment of fame — there are baseball cards of him on eBay",
    "their friend's new boyfriend is definitely still married — the caller found the wedding registry online, it's from last year, and there are no divorce filings in the county",
    "found out their quiet neighbor used to be the lead singer of a one-hit-wonder band in the 90s — they recognized a song playing from their garage and Googled the lyrics",
    "their dentist has a secret stand-up comedy career under a different name — they found his comedy special on YouTube and half the jokes are about his patients",
    "discovered that the 'organic honey' the farmer at the market sells is actually store-bought, repackaged with a handmade label — they saw him doing it in his truck",

    # --- juicy reveals ---
    "overheard two nurses at the clinic gossiping about which doctor is the worst and the doctor they trust most was at the top of the list — they named specific incidents",
    "found out the 'from scratch' baker who wins the county fair every year uses boxed cake mix as a base — they walked in on them dumping Betty Crocker into a mixing bowl",
    "their friend who brags about being 'chemical-free' and judging others for using products has a bathroom cabinet full of every product they shame others for using",
    "discovered the organizer of the local 5K charity run spent more on their own branded merchandise than they raised for the actual cause — the financials are public and nobody's looked",
    "their daughter's dance instructor who charges $200/month has been posting to a vent account calling the parents 'delusional' and the kids 'talentless' — the caller found the account through a hashtag",
    "found out that two families in their neighborhood have been in a secret prank war for three years — escalating from moving garden gnomes to filling mailboxes with confetti to rewiring each other's sprinklers",

    # --- small town secrets ---
    "their town's beloved mailman has been reading postcards and everyone knows but nobody says anything because he's 78 and lonely",
    "found out the woman who runs the local thrift store has been keeping the best donations for herself before they hit the floor — the caller recognized their own donated jacket on the woman's Instagram",
    "the couple who always wins the chili cookoff has been using the same canned base for years — the caller's kid worked at their house and saw the evidence in the recycling",
    "their neighbor who is known for their beautiful Christmas lights hires a professional company to set them up but tells everyone they do it themselves — the caller watched the crew install them at 6 AM",
    "the town's loudest church-going family hasn't actually been to church in months — the caller knows because they sit in the same pew area and they've been empty since September",
    "found out the guy who runs the local 'organic farm stand' buys half his produce from the same wholesale distributor as the grocery store",
    "their neighbor who brags about hunting skills buys meat from the butcher and wraps it in brown paper to look like he processed it himself — his wife told the caller after too much wine",
    "the woman who runs the community garden has been secretly using commercial fertilizer while lecturing everyone else about going organic",
    "discovered the 'veteran discount' card the guy at the hardware store flashes isn't real — the caller's friend works at the store and looked it up",
    "their mechanic who claims he's the cheapest in town is actually the most expensive — the caller got three quotes and his was double",

    # --- workplace tea ---
    "found out their office's 'anonymous' suggestion box isn't anonymous — the manager photographs each submission with a UV light that shows fingerprints and matches them to employee records",
    "their coworker who's 'working from home' has actually been at Disneyland for a week — a mutual friend tagged them in photos while they were in a 'status meeting' on Zoom with their camera off",
    "discovered the company's 'random' drug testing isn't random — HR targets people who file complaints, and the caller has emails to prove it",
    "their coworker who claims to run marathons has fake race bibs hanging in their office — the caller checked the race results databases and they're not listed in any of them",
    "found out their boss has been expensing personal vacations as 'business travel' for three years — the receipts are all restaurants and resorts with no meeting notes attached",
    "their colleague who posts about their 'side hustle' making six figures is actually making $200 a month — the caller knows because they use the same platform and can see the numbers",
    "the office 'wellness champion' who leads the walking club and posts inspirational quotes smokes a pack a day in their car — three different people have caught them",
    "discovered their company's 'charity match program' hasn't actually matched any donations in two years — an accountant friend checked the books",
    "their coworker who tells everyone they got scouted by an Ivy League school for sports actually walked on to the JV team at a Division III community college — the caller found the roster",
    "found out the new hire who claims to have 'ten years of experience' graduated six months ago — their LinkedIn dates don't match their resume and nobody in HR checked",

    # --- dating & relationship gossip ---
    "their friend who has been 'dating a model' for six months has never introduced anyone — the caller is starting to think the model is an AI-generated image",
    "found out the couple everyone calls 'relationship goals' met because one of them catfished the other — they've been together five years and neither will admit how it started",
    "their friend who posts #SoBlessed anniversary posts is actually in a situationship they've been trying to DTR for three years",
    "discovered their coworker's 'long-distance boyfriend' is actually someone they've never met who might not be who they say they are — classic catfish red flags everywhere",
    "their friend who always gives dating advice and acts like a love guru has been single for nine years and hasn't been on a date in eighteen months",
    "found out the 'happily married' neighbor has profiles on three different dating apps — the caller matched with them by accident",
    "their friend who claims to have broken up 'on great terms' with their ex actually got served a no-contact order — a mutual friend saw the court filing",
    "the couple who just got engaged has been having screaming fights every weekend that the whole apartment building can hear — but the Instagram posts are all champagne and smiles",

    # --- family drama gossip ---
    "discovered their cousin's 'homemade' pies that win every Thanksgiving compliment are actually from a bakery two towns over — the cousin's kid ratted them out",
    "found out their aunt who claims to be 'independently wealthy' is actually living off their grandmother's pension checks without anyone's knowledge",
    "their uncle who always shows up to family events in a new car is drowning in lease payments — his ex-wife told the caller during a surprisingly honest conversation",
    "just learned that the family's 'heirloom ring' that's been passed down four generations was actually bought at a pawn shop by their great-grandmother — there's a receipt in a box of papers",
    "their sibling who 'moved to the city for a great job opportunity' actually moved because they owe money to half the people in town",
    "found out their mother-in-law who says she 'doesn't drink' has a wine delivery subscription — the caller saw the boxes in the recycling when they came over early",
    "their cousin who brags about their kid being a 'gifted student' is actually paying for a private tutor seven hours a week — the tutor is the caller's friend",
    "discovered that the 'beach house' their brother-in-law keeps inviting everyone to isn't his — it's his boss's, and the boss doesn't know he's been using it",
    "their relative who posts about their 'amazing home renovation' has been doing all the work without permits — the city inspector is their other relative's poker buddy and word is getting around",
    "found out their father-in-law who tells war stories at every gathering was actually stationed at a desk job in Virginia — the caller's military friend checked the records",

    # --- community figure gossip (expanded) ---
    "the HOA president who fined someone for having a brown patch in their lawn has three code violations on their own property that everyone's too scared to report",
    "found out the 'life coach' in town who charges $200/hour for sessions got their certification from a two-day online course that costs $49",
    "the crossing guard who everyone loves has been taking home lost items from the school lost-and-found — the caller's kid recognized their jacket on the guard's kid",
    "their town's most prolific letter-to-the-editor writer uses three different pen names to make it seem like multiple people agree with them",
    "discovered the local Instagram food blogger who rates restaurants with devastating reviews has been asking for free meals in exchange for positive coverage — the restaurant owners compare notes",
    "the guy who runs the neighborhood Facebook group moderates everything to favor his friends — opposing comments get deleted and the poster gets banned for 'community guidelines violations'",
    "found out the local personal injury lawyer who has billboards all over town got his own lawsuit settled for a fraction because he forgot to file paperwork on time",
    "the president of the book club has never finished a single book in two years of leading discussions — they read the SparkNotes version every time and nobody's caught on until the caller noticed specific phrases being parroted",
    "their kid's Little League umpire has been making calls that suspiciously favor the team his nephew plays on — three other parents have noticed and they're keeping a spreadsheet",
    "the neighborhood's 'master gardener' who gives everyone plant advice killed their own houseplants and replaced them with fake ones — the caller noticed during a house tour",

    # --- online & social media tea ---
    "found their super-private boss on TikTok where they do cooking videos with 80k followers and a persona completely different from work — they're bubbly and goofy and use a fake name",
    "their friend who rants about 'toxic influencer culture' has a secret account with 20k followers where they post luxury haul videos",
    "discovered their coworker runs an anonymous gossip account about the company and the posts are 100% accurate — the caller recognized details only someone in their department would know",
    "found their extremely conservative uncle's Spotify wrapped — it's almost entirely Broadway musicals and Beyoncé",
    "their neighbor's kid who got famous on TikTok for 'making it on their own' has parents paying for their apartment, car, and equipment — the parents told the caller at a BBQ",
    "found their friend's secret podcast where they tell stories about their friend group with the names barely changed — the caller recognized their own divorce story in episode four",

    # --- money & lifestyle lies ---
    "their friend who 'works from home as a consultant' has been living off their inheritance and hasn't had a client in two years — their spouse doesn't know the money is running out",
    "found out their neighbor who drives a Tesla and wears designer clothes took out a personal loan for both — the bank officer is married to the caller's coworker",
    "their friend who posts about their 'debt-free journey' still owes $60k in student loans — they just stopped counting certain debts",
    "discovered the guy who runs the 'financial freedom' workshops in town filed for bankruptcy last year — the paperwork is public and the caller checked",
    "their coworker who always brings expensive wine to parties actually just peels off the label from the cheap bottle and replaces it — the caller watched them do it at a party when they thought nobody was looking",
    "the woman who sells MLM supplements and posts about her 'six-figure income' made $400 last quarter — she accidentally showed a screenshot of her dashboard at a recruitment event",
    "their relative who claims to have 'paid off their mortgage early' actually just refinanced with a longer term and lower payments — they're not lying technically but they're definitely misleading everyone",
    "found out their neighbor's new pool was financed by a home equity loan they can barely afford — the neighbor's contractor friend told the caller over beers",

    # --- hidden talent & double lives (expanded) ---
    "discovered their boring accountant neighbor is secretly a competition barbecue champion who travels the country on weekends with a custom smoker rig — he has trophies in his garage",
    "found out the quiet school librarian runs a true crime podcast with 50k monthly listeners — she interviews former detectives using a voice modulator",
    "their neighbor who only listens to country music in public has a vinyl collection in their basement that's entirely 90s hip-hop — the caller saw it when they borrowed a ladder",
    "the woman at their gym who wears a baggy shirt and keeps to herself used to be a competitive bodybuilder — old photos surfaced at a birthday party and she's jacked in them",
    "discovered their kid's math teacher is a former professional poker player who quit when they had a family — they still play high-stakes games once a month under a screen name",
    "their elderly neighbor who shuffles to the mailbox every morning used to be a championship ballroom dancer — the caller found a trophy display in the back of a closet while helping them fix a shelf",
    "found out the gruff rancher who never says more than ten words at a time writes poetry — his wife submitted one to the local paper anonymously and it was beautiful",
    "their dentist moonlights as a jazz drummer and plays gigs in Tucson on weekends — the caller saw the dental office van parked outside a club at midnight",
    "discovered their bus driver has a master's degree in philosophy and chose driving because 'nobody asks a bus driver about Heidegger' — the caller overheard them quoting Sartre to another driver",
    "the guy who pumps gas at the station on Route 9 is an ex-MMA fighter with a record of 12-3 — a new customer tried to start something and the station owner just said 'I wouldn't' with a look that told the whole story",

    # --- health & wellness secrets ---
    "found out the guy at work who's been 'intermittent fasting' has just been eating in his car in the parking lot so nobody sees his Taco Bell habit",
    "their fitness instructor who preaches 'clean eating' was spotted at 2 AM at a Waffle House housing a plate of smothered hash browns — the caller's friend was working the late shift",
    "discovered their chiropractor doesn't believe in chiropractic — the caller overheard them telling a friend 'it's mostly placebo but the money is incredible'",
    "their friend who sells essential oils as a 'wellness consultant' takes Advil for literally everything — the caller found a Costco-size bottle in their medicine cabinet",
    "the gym owner who's always talking about natural fitness and hard work got caught ordering steroids — a delivery driver showed up at the gym asking for him by name with a suspicious package",
    "found out their coworker's 'perfect skin routine' that they sell online is actually just a filter — the caller saw them without makeup at a pool party and they look totally different",

    # --- parenting & kids secrets ---
    "their neighbor who brags about their kid's acceptance to an elite college doesn't mention the $200k in donations the grandparents made — another parent on the admissions committee let it slip",
    "found out the kid who won the science fair used their parent's work as a research scientist to basically do the project — the parent bragged about it at a dinner party thinking everyone would be impressed",
    "the mom in the friend group who always posts about 'screen-free' parenting has a TV in every room including the bathroom — the caller babysat and the kids immediately turned everything on like it was routine",
    "their friend's kid who 'taught themselves piano' actually has a tutor who comes three times a week — the tutor is the caller's cousin",
    "discovered the family who 'doesn't believe in sugar' at the school bake sale has a candy drawer the size of a filing cabinet — their kid told the caller's kid and now both families are implicated",
    "the parent who leads the anti-phone crusade at the PTA was caught by their own kid doom-scrolling for four hours — the kid put it in a school essay about hypocrisy",

    # --- hobby & skill lies ---
    "their friend who claims to catch fish every weekend buys fish at the market and takes photos with them before 'cleaning' them — the caller recognized the fish market's specific packaging in a background photo",
    "found out the guy who runs the local running club has been driving to the halfway point of their group runs and waiting for them — his watch data shows suspiciously fast splits and zero elevation gain",
    "their neighbor's 'hand-carved' fence posts are actually factory-made and ordered online — the same posts are on Amazon and the caller found the exact listing",
    "the woman in their knitting circle who makes the most impressive pieces has been buying finished items from Etsy and unraveling them slightly to make them look handmade — the caller found the original listing",
    "their friend who 'built their own deck' hired a contractor and then staged photos with tools to make it look DIY — the contractor told another client who told the caller",
    "discovered that the 'scratch-made' salsa their coworker brings to every potluck is just Pace with fresh cilantro added — they watched the preparation through a kitchen window",

    # --- financial lies & status symbols ---
    "found out their friend's 'investment portfolio' is actually just $500 in a savings account — they accidentally showed a bank notification while showing something on their phone",
    "their neighbor's 'vacation home' is actually a timeshare they can only use two weeks a year — a real estate agent friend looked up the property records",
    "the couple who always invites people to their 'lake house' doesn't own it — it belongs to the husband's boss and they have to ask permission every time",
    "discovered their coworker's 'designer watch' is a very convincing fake — the caller used to work in jewelry and spotted the tell immediately",
    "their friend who claims to 'only shop organic' has an Amazon Fresh account full of the cheapest conventional options — their shopping history auto-filled on the caller's iPad",
    "the guy at the gym who drives a Porsche lives in a studio apartment he shares with two roommates — one of the roommates told the caller at a party",

    # --- secret habits & quirks ---
    "found out their very proper boss goes to Renaissance Faires on weekends in full medieval costume with a different name — the caller's kid recognized them as 'Lord Bartholomew'",
    "their coworker who says they 'don't watch TV' has been binge-watching reality shows — their Spotify wrapped showed a suspicious number of reality show soundtracks and the caller did the math",
    "discovered their neighbor who always acts busy and stressed actually sits in their backyard reading novels for most of the day — the caller can see over the fence from their second floor",
    "their friend who posts about their 'morning journaling practice' actually just scrolls Twitter for an hour — the post timestamps and their Twitter activity line up perfectly",
    "found out the guy who loudly talks about 'no screens before bed' plays mobile games under the covers every night — his wife told the caller's wife and the wives' group chat went nuclear",
    "their very macho neighbor who makes fun of rom-coms cried during The Notebook — the caller was walking past and heard it through the window and now they share a secret and neither has acknowledged it",

    # --- community drama expanded ---
    "the woman who organized the neighborhood garage sale kept a 10% cut from everyone's sales as an 'organizing fee' that nobody agreed to — the caller counted their sales versus their payout and the math doesn't work",
    "found out the guy who runs the local sports league has been rigging the playoff brackets to keep his team in the easier bracket — a ref showed the caller the original versus 'revised' brackets",
    "their HOA's landscaping company is owned by the HOA president's son — nobody knew until the caller pulled business records and now they understand why the dues keep going up",
    "the woman who runs the neighborhood Facebook group's 'crime watch' reports are almost entirely about her personal feuds disguised as safety concerns",
    "discovered that the 'anonymous donor' who funds the town's holiday parade every year is actually a group of business owners splitting the cost — and they've been letting one guy take all the credit for a decade",
    "the town's beloved Santa at the Christmas parade got into a fistfight at a bar the next weekend — still in the full suit because he forgot to change — and the video has been carefully kept off social media by mutual agreement",

    # --- tech & digital secrets ---
    "found their teenager's finsta (fake Instagram) and it has more followers than their real account — the content is actually really good and thoughtful and they don't know whether to be proud or concerned",
    "their coworker's 'AI-generated' presentation that impressed the whole office was actually manually made — they just said AI to sound current and spent three nights on it",
    "discovered their neighbor has been running a crypto mining operation in their garage — the electric bill must be insane but the caller only figured it out because of the constant hum",
    "their friend who runs a 'successful YouTube channel' has 42 subscribers and most of them are family members who leave supportive comments — the friend genuinely thinks they're about to blow up",
    "found out their kid's teacher uses ChatGPT to write the students' report card comments — every comment has the same structure and two parents compared notes",
    "their boss who brags about 'never using social media' has a very active anonymous Twitter account where they argue about politics — the writing style is unmistakable",

    # --- relationship & dating gossip expanded ---
    "discovered that a couple in their friend group who 'met at a coffee shop' actually met on a kink website — the couple told different versions of the story at different events and the contradictions added up",
    "their friend who is 'taking a break from dating' just signed up for three new apps — the caller matched with them on one of them by accident",
    "found out the couple who always fights in public actually gets along great in private — a mutual friend said the fighting is how they 'keep things interesting'",
    "their coworker's 'supportive partner' who they always praise has never actually come to a single work event — the caller is starting to wonder if this person exists",
    "discovered their friend's 'spontaneous proposal story' was actually rehearsed six times with the photographer — the photographer posted the outtakes by accident",
    "the couple next door who 'never argues' has a white noise machine that they turn up when things get heated — the caller learned this from the couple's cleaning person",

    # --- surprising revelations ---
    "found out their quiet mail carrier writes romance novels that are actually bestsellers on Amazon under a pen name — has over 200 reviews averaging 4.8 stars",
    "their auto mechanic is a classically trained violinist who gave it up because 'you can't feed a family with Vivaldi' — the caller found a video of a concert from fifteen years ago and it's incredible",
    "discovered the lunch lady at school used to be a competitive bodybuilder in the 80s — a kid found old newspaper clippings and now she's a legend among the students",
    "their barber has a law degree and chose barbering because 'I'd rather talk to people honestly than argue for a living' — they found the diploma hanging in the back room",
    "found out the woman who runs the flower shop is a retired combat medic — she told the caller during a slow afternoon and the stories were nothing like what the caller expected",
    "their garbage collector has a PhD in environmental science — he took the job intentionally to study waste patterns and has published papers about suburban consumption",
    # Comedy writer entries
    "their youth pastor who preaches about sexual purity was just spotted leaving an adult bookstore off the highway at 1am — the caller knows because they were also leaving the adult bookstore and they locked eyes in the parking lot and now they have mutually assured destruction",
    "found out the PTA mom who organized the 'family values' book banning campaign at school has an OnlyFans — a dad from another school district recognized her at a basketball game and showed the caller on his phone and it's not even a little bit ambiguous",
    "their coworker who makes $55k a year just bought a $90k truck with cash and told everyone his grandmother died and left him money — the caller went to the grandmother's funeral three years ago because they're also friends with the family, and that grandmother had nothing",
    "just found out their neighbor who puts up the biggest 'Support Our Troops' flag display every Fourth of July dodged the draft in the '70s by having his dentist write a letter about his teeth — the caller's father served two tours and lost a leg and the neighbor thanks him for his service every year at the block party",
    "their friend's husband who lectures everyone about loyalty and commitment has a separate phone, a separate email, and a PO box — the caller knows because they share a mailman and the mailman let it slip after a few beers at the VFW",
    "their neighbor who has a 'Live Laugh Love' sign in every room and posts daily gratitude affirmations screamed at a teenager at the Sonic drive-through until the kid cried — over a missing pickle",
    "found out the guy at work who always talks about his 'lake house' has been sleeping in his car in the office parking garage three nights a week — security showed them the footage and he brings a pillow and everything",
    "their town's most vocal anti-drinking city councilman was just pulled over for a DUI at 2pm on a Wednesday — in a neighboring town, driving a car registered to a woman who is not his wife, with an open container of Four Loko in the cupholder",
    "their boss who fires people for being five minutes late has been leaving at 3pm every day for six months — the caller knows because they started parking behind the building and timing it, and they've got a spreadsheet going back to September",
    "discovered that the woman in their neighborhood who runs a 'clean living' blog and sells essential oils for everything from headaches to infertility keeps a pack of Marlboro Reds in her glove compartment — the caller saw them when they helped her jump her car and she said 'those are my husband's' but her husband died in 2021",
    "their coworker who talks constantly about their 'amazing marriage' on social media just got caught on the office security camera making out with the night janitor in the supply closet — the security guard showed the caller because the janitor is the caller's nephew",
    "found out the man who runs the neighborhood watch and sends emails about 'suspicious activity' weekly has two outstanding warrants in another state — the caller's brother is a bail bondsman and recognized the name",
    "their fitness influencer neighbor who posts shirtless transformation photos and sells a $200 meal plan eats McDonald's in his truck every night at 10pm — the caller can see the golden arches glow from across the street and has photo evidence on four separate occasions",
    "just learned that the couple on their street who are always holding hands and posting anniversary tributes have been separated for a year — they keep up appearances because they co-own a wedding photography business and the brand depends on them looking happy",
    "their coworker who brings elaborate homemade lunches every day and talks about their meal prep routine buys pre-made meals from the deli section at Whole Foods and transfers them into Tupperware in the parking lot — the caller watched the whole transfer through the break room window",
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
    # New keys for sex/kink PROBLEMS
    "fetish_detail": ["foot", "leather", "latex", "voyeurism", "exhibitionism", "praise kink", "degradation", "age play", "pet play", "pegging", "cuckolding", "body worship", "impact play", "wax play", "shibari rope", "sensory deprivation"],
    "sex_situation": ["in the living room with the curtains open", "in a hotel room that was supposed to be a business trip", "in the car in a parking lot", "at a party in someone else's bedroom", "on a video call that was supposed to be casual", "in a place that was definitely not private enough"],
    "partner_reaction": ["they haven't spoken about it since and it's been two weeks", "they laughed and then got quiet and now things are weird", "they said they'd think about it and that was a month ago", "they were into it in the moment but now act like it never happened", "they're being weirdly supportive and it's making them suspicious", "they cried and said they felt like they didn't know them anymore", "they said 'finally' like they'd been waiting for this conversation"],
}

INTERESTS = [
    # TV (trimmed — not everyone watches prestige TV)
    "obsessed with Severance, has theories about every floor",
    "been binging Landman, loves the oil field drama",
    "hooked on The Last of Us, compares it to the game constantly",
    "big Yellowstone fan, has opinions about the Duttons",
    "has watched The Wire three times, quotes it constantly",
    "thinks Breaking Bad is the greatest show ever made",
    "rewatches The Sopranos every year, notices new things",
    "thinks True Detective season 1 is peak television",
    "still upset about how Game of Thrones ended",
    "Band of Brothers is their go-to recommendation",
    "watches Dateline and 48 Hours religiously, has theories about cold cases",
    "into reality competition shows, won't miss Survivor or The Challenge",
    "watches old Twilight Zone episodes, thinks they hold up better than anything new",
    # Science & space
    "follows NASA missions, got excited about the latest Mars data",
    "reads science journals for fun, especially Nature and Science",
    "into astrophotography, has a decent telescope setup",
    "fascinated by quantum physics, watches every PBS Space Time episode",
    "follows JWST discoveries, has opinions about exoplanet findings",
    "reads about neuroscience and consciousness research",
    "into geology, knows every rock formation around the bootheel",
    "follows fusion energy research, cautiously optimistic about it",
    "amateur astronomer, knows the night sky by heart",
    # Technology
    "follows AI developments closely, has mixed feelings about it",
    "into open source software, runs Linux at home",
    "fascinated by SpaceX launches, watches every one",
    "into ham radio, has a nice setup",
    "builds electronics projects, has an Arduino collection",
    # Photography & visual
    "serious about astrophotography, does long exposures in the desert",
    "into landscape photography, shoots the bootheel at golden hour",
    "has a darkroom, still shoots film",
    "into wildlife photography, has patience for it",
    # Poker & games
    "plays poker seriously, studies hand ranges",
    "watches poker tournaments, has opinions about pro players",
    "plays home games weekly, takes it seriously",
    "plays chess online, follows the competitive scene",
    # Movies & film
    "big movie person, prefers practical effects over CGI",
    "into Coen Brothers films, can quote most of them",
    "watches old westerns, thinks they don't make them like they used to",
    "into horror movies, the psychological kind not slashers",
    "Tarantino fan, has a ranking and will defend it",
    "into documentaries, especially nature docs",
    # Working-class & rural
    "into hunting, goes out every season with the same crew",
    "knows engines inside and out, has rebuilt three trucks from nothing",
    "raises chickens, has opinions about every breed",
    "into reloading ammo, treats it like a science",
    "competes in local rodeo events, team roping mostly",
    "into ranching life, can talk cattle genetics all day",
    "does leatherwork as a side thing, makes belts and holsters",
    "collects old tools, has stuff from the 1800s that still works",
    "hunts shed antlers in the spring, knows every trail in the mountains",
    "trains bird dogs, has a line of English pointers going back four generations",
    "into off-roading, knows every dirt road in the county",
    "grows a massive garden, gives produce to half the neighborhood",
    "into canning and preserving, learned from their grandmother",
    "keeps bees, sells honey at the farmers market",
    "does competitive shooting, three-gun matches on weekends",
    # Faith & community
    "active in their church, sings in the choir",
    "coaches youth sports, takes it more seriously than the parents do",
    "volunteers at the fire department, been doing it for years",
    "into local history, knows every old building in town and who built it",
    "runs a monthly poker night that's been going for 15 years, same guys",
    "goes to every high school football game, even though their kids graduated",
    # Active & outdoors
    "into fitness, does a home gym thing", "hikes every weekend, knows every trail",
    "into camping and survival stuff", "into fishing, finds it meditative",
    "mountain bikes the trails around Silver City",
    "runs ultramarathons in the desert, thinks it's peaceful",
    # Hobbies & creative
    "plays guitar badly but loves it", "into woodworking, built their own kitchen table",
    "builds stuff in their garage", "brews beer at home, entered a few competitions",
    "into gardening, talks to plants", "restores old furniture from estate sales",
    "makes their own hot sauce, has a secret recipe",
    "into metal detecting, found some interesting stuff over the years",
    "does amateur radio astronomy, built their own antenna",
    # Self & lifestyle
    "homebody, prefers staying in", "into cooking and food, watches every cooking show",
    "gamer, plays late at night after the house quiets down",
    "into history, has random facts about everything",
    "reads philosophy for fun", "into personal finance, tracks every dollar",
    "has done therapy, believes in it", "into meditation, it actually helps",
    "collects vinyl records, mostly classic country and rock",
    "into true crime podcasts, has listened to all of them",
    # US News & current events
    "follows US politics closely, has strong opinions",
    "reads the news every morning, stays informed",
    "into economics, thinks about markets and policy",
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
    "talks fast when excited", "pauses a lot, choosing words carefully",
    "uses metaphors for everything", "tells stories instead of answering directly",
    "interrupts themselves mid-thought", "whispers when saying something they shouldn't",
    "repeats the last thing the host said while thinking", "says 'right?' after everything seeking validation",
    # Energy & vibe
    "high energy, talks with their hands even on the phone", "calm and measured, almost too calm",
    "sounds tired but won't admit it", "clearly been crying but trying to hold it together",
    "manic energy tonight, everything is hilarious or devastating", "stoned and philosophical",
    "just got off a long shift and is running on fumes", "had a few drinks, more honest than usual",
    "wired on coffee at 2am", "weirdly cheerful for someone with this problem",
    # Personality
    "self-aware about their own bullshit", "confessional, needed to tell someone",
    "a little drunk and honest because of it", "can't believe they're saying this out loud",
    "overshares and then apologizes for oversharing", "keeps circling back to the real issue",
    "trying to convince themselves more than the host", "already knows what they should do, just needs to hear it",
    "clearly rehearsed what to say but it's falling apart", "gets defensive when the host gets too close to the truth",
    "laughs at their own pain as a coping mechanism", "proud but asking for help anyway",
    "suspicious of advice but called anyway", "wants permission to do the thing they already decided to do",
    # Openness about sex
    "comfortable talking about sex when it comes up", "no shame about their desires",
    "gets more explicit as they get comfortable", "treats sex like a normal topic",
    "will share details if you ask", "surprisingly open once they start talking",
    "has stories they've never told anyone", "testing how the host reacts before going deeper",
]

# Named people in their life — 2-3 assigned per caller
PEOPLE_MALE = [
    "his wife Linda, who he's been with since high school",
    "his wife Teresa, they've been rocky lately",
    "his wife Connie, they got married young and grew up together",
    "his girlfriend Amber, been together about a year",
    "his girlfriend Jen, they met online and it's been surprisingly good",
    "his ex-wife Diane, they still talk sometimes",
    "his ex-wife Sandra, who he has nothing nice to say about",
    "his buddy Ray from work, the one person he trusts",
    "his brother Daryl, who always has some scheme going",
    "his brother Eddie, who never left home",
    "his brother Marcus, the golden child of the family",
    "his sister Maria, the only one in the family who gets him",
    "his sister Deb, who married money and acts like she forgot where she came from",
    "his mom Rosa, who calls every Sunday whether he wants her to or not",
    "his mom Cheryl, who's been sober two years and is trying to make up for lost time",
    "his dad, who everybody calls Big Jim, old school rancher",
    "his dad Wayne, who he hasn't spoken to in four years and doesn't plan to",
    "his best friend Manny, known each other since middle school",
    "his neighbor Gary, who's always in everybody's business",
    "his coworker Steve, who he eats lunch with every day",
    "his coworker DeShawn, the only guy at work who tells it straight",
    "his buddy TJ, they go fishing together",
    "his cousin Ruben, more like a brother really",
    "his cousin Tito, who's been in and out of trouble but has a good heart",
    "his daughter Kaylee, she's in high school now",
    "his daughter Sophie, who just moved across the country and calls crying sometimes",
    "his son Marcos, just turned 21",
    "his son Jake, who's 12 and already smarter than him",
    "his boss Rick, who's actually a decent guy for a boss",
    "his boss Vince, who micromanages everything and is slowly driving him insane",
    "his uncle Hector, who raised him after his dad left",
    "his buddy from the Army, goes by Smitty",
    "his AA sponsor Phil, who's been through worse and always picks up the phone",
    "his ex-girlfriend Kayla, who he ran into last month and hasn't stopped thinking about",
    "his neighbor Hank, retired cop, knows everything that happens on the street",
    "his grandpa Ernesto, who's 87 and still sharper than anyone in the room",
]

PEOPLE_FEMALE = [
    "her husband David, high school sweetheart",
    "her husband Mike, second marriage for both of them",
    "her husband Jesse, who works nights so they barely see each other",
    "her boyfriend Carlos, met him at work",
    "her boyfriend Trey, who her family doesn't approve of",
    "her ex-husband Danny, he's still in the picture because of the kids",
    "her ex-husband Rodney, who she has a restraining order against",
    "her best friend Jackie, they tell each other everything",
    "her best friend Lena, who moved away last year and the distance is hard",
    "her sister Brenda, who she fights with but loves",
    "her sister Crystal, the one who moved away",
    "her sister Natalie, the one who always needs money",
    "her mom Pat, who has opinions about everything",
    "her mom Lorraine, who's getting older and it worries her",
    "her mom Diane, who she's been taking care of since the stroke",
    "her brother Ray, who can't seem to get his life together",
    "her brother Anthony, the one who made it out and never looks back",
    "her daughter Mia, who just started college",
    "her daughter Brianna, who's 14 going on 30 and testing every boundary",
    "her son Tyler, he's 16 and thinks he knows everything",
    "her son Elijah, who's in the military and she worries about him constantly",
    "her coworker and friend Denise, who she vents to on breaks",
    "her coworker Steph, who's gunning for the same promotion",
    "her neighbor Rosa, who watches her kids sometimes",
    "her neighbor Linda, who gossips about everyone on the block",
    "her cousin Angie, they grew up together",
    "her best friend from back in the day, Monica, they reconnected recently",
    "her dad Frank, retired and bored and driving everyone crazy",
    "her dad Gene, who she just found out has been lying about something for years",
    "her grandma Yolanda, who's the real head of the family",
    "her boss Karen — yes, her name is actually Karen — who is actually cool",
    "her boss Trish, who takes credit for everyone else's work",
    "her friend Tammy from church, the only one who knows the real story",
    "her therapist, who she refers to by first name like they're friends",
    "her ex-best friend Amanda, who she cut off last year and still misses",
    "her aunt Vivian, who's the family gossip and knows everybody's secrets",
]

# Relationship status with detail
RELATIONSHIP_STATUS = [
    "Married, 15 years. It's comfortable but sometimes that's the problem.",
    "Married, second time around. Learned a lot from the first one.",
    "Married young, still figuring it out.",
    "Divorced, been about two years. Still adjusting.",
    "Divorced twice. Not in a rush to do it again.",
    "Single, been that way a while. Not sure if by choice anymore.",
    "In a relationship, about 3 years. She wants to get married, they're not sure.",
    "In a relationship, it's new. Maybe 6 months. Still in the good part.",
    "It's complicated. Technically single but there's someone.",
    "Separated. Living apart but haven't filed yet.",
    "Widowed. It's been a few years but it doesn't get easier.",
    "Dating around. Nothing serious. Prefers it that way, mostly.",
    "Long-distance thing that probably isn't going to work but they keep trying.",
    "Living together, not married. Her family has opinions about that.",
    "Just got out of something. Not ready to talk about it. Or maybe they are.",
]

# What vehicle they drive (rural southwest flavor)
VEHICLES = [
    "an F-150", "a Chevy Silverado", "a Tacoma", "a Ram",
    "a Subaru Outback", "a Ford Ranger", "a Honda Civic",
    "a Jeep Wrangler", "a minivan", "a Nissan Frontier",
    "an old Bronco", "a Corolla", "a motorcycle",
]

# What they were doing right before calling
BEFORE_CALLING = [
    "Was sitting in the driveway, not ready to go inside yet.",
    "Was watching TV but not really watching, just thinking.",
    "Was lying in bed staring at the ceiling when the show came on.",
    "Was on the porch, the stars are incredible tonight.",
    "Was scrolling their phone and the show was on in the background.",
    "Was driving home and pulled over when they heard something on the show.",
    "Was doing dishes and the quiet got to them.",
    "Was having a smoke outside and just started thinking.",
    "Was up feeding the dog and figured, what the hell, I'll call.",
    "Was in the garage working on something, had the radio on.",
    "Was sitting at the kitchen table, couldn't sleep, radio on low.",
    "Was about to go to bed but then the last caller said something that got them.",
    "Was out walking, it's a clear night, needed the air.",
    "Was just staring at a text they haven't replied to yet.",
    "Was cleaning their gun at the kitchen table, it's a ritual that helps them think.",
    "Was parked at the gas station, not ready to go home.",
    "Was at the laundromat waiting on a load and heard the show through someone's phone.",
    "Was closing up the shop, everyone else went home an hour ago.",
    "Was in the bathtub, phone on the edge of the sink, show on speaker.",
    "Was on a break at work, sitting in the break room alone.",
    "Was at Waffle House at the counter by themselves, couldn't sleep.",
    "Was reorganizing the junk drawer, which is what they do when they can't settle.",
    "Was at the bar, last one there, bartender's wiping down.",
    "Was folding laundry on the couch, show was on the radio in the kitchen.",
    "Was laying in a hammock out back, couldn't go inside.",
    "Was at a truck stop diner, cup of coffee, staring out the window.",
    "Was up late painting — walls, not art — and had the radio on for company.",
    "Was at their desk, supposedly working, but mostly just staring at the screen.",
    "Was sitting in the waiting room at the ER with someone, long night.",
    "Was at the 24-hour gym, basically empty, radio on over the speakers.",
]

# Where callers are physically calling from — picked as a seed for the LLM prompt.
# NOT every caller mentions this. Only ~40% do.
CALLING_FROM = [
    # --- Driving / pulled over (Southwest routes) ---
    "driving south on I-10 past the Deming exit",
    "on NM-146 heading toward Animas",
    "pulled over on I-10 near the Arizona line",
    "on 80 south coming through the Peloncillos",
    "driving I-10 between Lordsburg and Deming, middle of nowhere",
    "parked at a rest stop between here and Tucson",
    "pulled off on NM-9 south of Hachita, nothing around for miles",
    "driving back from Silver City on NM-90",
    "on I-10 west of San Simon, about to cross into New Mexico",
    "sitting in the truck at the Road Forks exit",
    "driving NM-180 toward the Gila, no cell service in ten minutes",
    "on the 80 heading north out of Douglas",
    "pulled over on NM-338 in the Animas Valley, stars are insane right now",

    # --- Real landmarks / businesses ---
    "parked outside the Horseshoe Cafe in Lordsburg",
    "at the truck stop on I-10 near Lordsburg",
    "in the Walmart parking lot in Deming",
    "at the gas station in Road Forks",
    "sitting outside the Jalisco Cafe in Lordsburg",
    "at the Butterfield Brewing taproom in Deming",
    "in the parking lot of the Gadsden Hotel in Douglas",
    "at the Copper Queen in Bisbee, on the porch",
    "outside Caliche's in Las Cruces",
    "in the lot at Rockhound State Park, couldn't sleep",
    "parked at Elephant Butte, the lake is dead quiet",
    "at the hot springs in Truth or Consequences",
    "outside the feed store in Animas",

    # --- Home locations ---
    "kitchen table",
    "back porch, barefoot",
    "garage with the door open",
    "in the bathtub, phone balanced on the edge",
    "bed, staring at the ceiling",
    "couch with the TV on mute",
    "spare bedroom so they don't wake anyone up",
    "front porch, smoking",
    "on the floor of the hallway, only spot with reception",
    "in the closet because the walls are thin",
    "backyard, sitting in a lawn chair in the dark",
    "kitchen, cleaning up dinner nobody ate",

    # --- Work locations ---
    "break room at the plant",
    "truck cab between deliveries",
    "office after everyone left",
    "guard shack",
    "shop floor during downtime, machines still humming",
    "in the walk-in cooler because it's the only quiet spot",
    "cab of the loader, parked for the night",
    "nurses' station, graveyard shift",
    "back of the restaurant after close, mopping",
    "dispatch office, radio quiet for once",
    "fire station, between calls",
    "in the stockroom sitting on a pallet",

    # --- Public places ---
    "laundromat, waiting on the dryer",
    "24-hour diner booth, coffee going cold",
    "hospital waiting room",
    "motel room on I-10",
    "gym parking lot, just sitting in the car",
    "outside a bar, didn't go in",
    "gas station parking lot, engine running",
    "sitting on the tailgate at a trailhead",
    "library parking lot in Silver City",
    "outside the Dollar General, only place open",
    "airport in El Paso, flight delayed",
    "Greyhound station, waiting on a bus that's two hours late",

    # --- Unusual / specific ---
    "on the roof",
    "in a deer blind, been out here since four",
    "parked at the cemetery",
    "on the tailgate watching the stars, can see the whole Milky Way",
    "at a campsite in the Gila, fire's almost out",
    "sitting on the hood of the car at a pulloff on NM-152",
    "in a horse trailer, don't ask",
    "under the carport because the house is too loud",
    "on the levee by the river, no one around",
    "at the rodeo grounds, everything's closed up but they haven't left",
    "at a rest area on I-25, halfway to Albuquerque",
    "in a storage unit, organizing their life at midnight",
]

# Specific memories or stories they can reference
MEMORIES = [
    "The time they got caught in a flash flood near the Animas Valley and thought they weren't going to make it.",
    "The night they won $400 at a poker game and didn't tell anyone.",
    "When their dad took them hunting for the first time, out near the Peloncillos.",
    "The time they drove to El Paso on a whim at 2am just to get out of town.",
    "When they saw a mountain lion on their property and just stood there watching it.",
    "The night the power went out for three days and the whole neighborhood sat outside together.",
    "When they got lost hiking near the Gila and had to spend a night out there.",
    "The time they pulled someone out of a ditch during monsoon season.",
    "Their first real fight, in the parking lot of a bar in Lordsburg. They lost.",
    "The day they moved to the bootheel. Everything they owned fit in the truck.",
    "When a dust storm came through and they couldn't see ten feet in front of them.",
    "The time they drove all the way to Tucson for a concert and the band cancelled.",
    "When their kid said something so smart it made them realize they were going to be okay.",
    "The night they sat on the hood of their truck and watched the Milky Way for hours.",
    "When they found a rattlesnake in their kitchen and had to deal with it alone.",
    "The time the roof leaked during monsoon and they were up all night with buckets.",
    "When they ran into someone from high school at the Walmart in Deming and it was awkward.",
    "The time they helped a stranger change a tire on I-10 in 110 degree heat.",
    "When they were a kid and the whole family would drive to Hatch for chile season.",
    "The night they almost called this show before but chickened out.",
]

# Food/drink in the moment
HAVING_RIGHT_NOW = [
    "Nursing a Tecate on the porch.",
    "On their third cup of coffee.",
    "Drinking a Coors Light, it's been that kind of day.",
    "Eating leftover enchiladas straight from the container.",
    "Having some whiskey, not a lot, just enough.",
    "Drinking water because they're trying to be better about that.",
    "Got a Dr Pepper, their one vice.",
    "Having some chamomile tea, trying to wind down.",
    "Eating sunflower seeds, spitting shells into a cup.",
    "Just finished a bowl of green chile stew their neighbor brought over.",
    "Drinking a Modelo with lime.",
    "Having instant coffee because it's all they've got.",
    "Snacking on beef jerky from the gas station.",
    "Sipping on some mezcal a friend brought back from across the border.",
]

# Strong random opinions (might come out if conversation drifts)
STRONG_OPINIONS = [
    "Thinks the speed limit on I-10 should be 85.",
    "Swears the best green chile comes from Hatch, will argue about it.",
    "Believes aliens have definitely been to the bootheel. Not joking.",
    "Thinks Cormac McCarthy is overrated and will die on that hill.",
    "Convinced the government knows about things in the desert they won't talk about.",
    "Thinks everyone should learn to change their own oil.",
    "Believes the best time of day is right before sunrise.",
    "Thinks small towns are dying because nobody under 30 wants to stay.",
    "Has strong feelings about how chain restaurants are killing local places.",
    "Thinks the monsoon season is the best time of year and people who complain about it are wrong.",
    "Believes poker is the most honest game there is because everybody's lying.",
    "Thinks dogs are better judges of character than people.",
    "Convinced that the night sky out here is something people in cities will never understand.",
    "Thinks people who've never been broke can't give advice about money.",
    "Believes the desert teaches you things about yourself if you let it.",
    "Thinks the old-timers who built these towns were tougher than anyone alive today.",
    "Has a theory that the best conversations happen after midnight.",
    "Thinks too many people are afraid of silence.",
]

# Contradictions/secrets — something that doesn't match their surface
CONTRADICTIONS = [
    "Goes to church every Sunday but has serious doubts they've never said out loud — not about God, about whether the people there actually believe any of it.",
    "Lectures their kids about financial responsibility but is secretly $30,000 in credit card debt.",
    "Talks tough about cutting toxic people off but keeps answering their mother's calls every single time.",
    "Presents as the steady one everyone leans on but has panic attacks in the shower where nobody can see.",
    "Posts motivational quotes on social media but hasn't gotten out of bed before noon in six months.",
    "Voted one way their entire life but quietly agrees with the other side on the thing that matters most to them.",
    "Acts like they've moved on from their divorce but still drives past their old house once a week.",
    "Tells everyone they love small-town life but applies for jobs in other states every few months and never follows through.",
    "Says money doesn't matter but lost a friendship over $200 and still thinks about it.",
    "Comes across as fearless but won't go to the doctor because they're terrified of what they'll find.",
    "Raised to believe men don't cry but breaks down alone in the truck at least once a month.",
    "Preaches forgiveness but has held a grudge against their brother for nine years over something most people would've forgotten.",
    "Acts like they don't need anyone but keeps the dating app installed, just in case.",
    "Seems like the life of the party but drives home in complete silence and sits in the driveway for twenty minutes before going inside.",
    "Tells everyone they quit drinking but keeps a bottle in the garage behind the paint cans.",
    "Claims to be an open book but there's a three-year gap in their life story that nobody's allowed to ask about.",
    "Acts practical and no-nonsense but believes in ghosts. Has a story about it that they only tell late at night.",
    "Judges people who go to therapy but has been journaling every night for years — basically doing therapy alone in their kitchen.",
    "Says they don't care about social media but knows exactly how many followers they have and checks twice a day.",
    "Talks about integrity constantly but cheated on a test in college that got them the degree that got them their career.",
]

# Verbal fingerprints — specific phrases a caller leans on (assigned 1-2 per caller)
# Each caller gets a unique pair, so this list needs to be large and varied.
VERBAL_TICS = [
    # Emphasis / conviction
    "at the end of the day", "the thing is though", "for real though",
    "I'm dead serious", "hand to God", "on my mother's grave",
    "I promise you", "mark my words", "trust me on this",
    "and I mean that", "no joke", "I kid you not",
    "stone cold truth", "cross my heart",

    # Filler / transition
    "and I'm like", "so yeah", "but anyway",
    "long story short", "bottom line", "point being",
    "here's the deal", "so check this out", "okay so picture this",
    "fast forward to", "anyway the point is", "which brings me to",
    "so this is where it gets good", "and then, right",

    # Self-aware / hedging
    "I'm just saying", "I'm not going to lie", "the way I see it",
    "I mean whatever but", "not going to sugarcoat it",
    "maybe I'm wrong but", "I could be way off here",
    "don't quote me on this", "take this with a grain of salt",
    "I'm probably overthinking it", "it sounds crazy when I say it out loud",
    "I know how this sounds", "hear me out though",
    "this is going to sound weird but", "I'm just being honest",

    # Emotional emphasis
    "that's the part that gets me", "I keep coming back to",
    "that's what kills me", "and that's the crazy part",
    "it hit me like a truck", "that one stuck with me",
    "it keeps me up at night", "I still think about it",
    "that's what I can't get past", "it just eats at me",
    "what really got me was", "the part nobody talks about",
    "and that's when it hit me", "you want to know what really burns me",
    "that right there is the whole problem",

    # Seeking agreement
    "you know what I mean", "right?", "am I crazy?",
    "tell me I'm wrong", "you see what I'm saying?",
    "does that make sense?", "am I the only one?",
    "is that not insane?", "wouldn't you?",
    "like what would you even do", "that's fair right?",

    # Conversational starters / redirects
    "look", "listen", "here's the thing",
    "right but here's the thing", "and I'm sitting there thinking",
    "and I told myself", "but you know what",
    "I'll be honest with you", "at this point",
    "let me put it this way", "okay but get this",
    "wait it gets better", "wait it gets worse",
    "hold on hold on", "no but wait",
    "and this is the kicker", "the real kicker is",

    # Regional / character-specific
    "I tell you what", "well shoot", "lord have mercy",
    "bless their heart", "good grief", "oh brother",
    "well I'll be damned", "swear on everything",
    "I about fell over", "scared me half to death",
    "madder than a wet cat", "happy as a clam about it",
    "couldn't believe my own eyes", "I had to do a double take",
    "if that don't beat all", "that just chaps me",
    "I nearly lost it", "and I'm just standing there like",

    # Understatement / dry
    "so that was fun", "real great", "super helpful",
    "that went well", "naturally", "as one does",
    "classic", "of course", "because why not",
    "shocker", "big surprise there", "who could have seen that coming",
    "so that's where we're at", "anyway that's my life",
    "living the dream", "just another Tuesday",

    # Thinking out loud
    "I keep going back and forth on it", "part of me thinks",
    "the more I think about it", "I've been turning it over in my head",
    "something about it just doesn't sit right", "I can't put my finger on it",
    "it's one of those things where", "every time I think about it I see it differently",
    "I go back and forth", "some days I think one thing, some days the other",
    "I'm still working it out in my head",

    # Storytelling momentum
    "so get this", "no but seriously", "like I said",
    "and then — and this is the part", "I'm not even to the best part yet",
    "you're not going to believe this", "here's where it gets interesting",
    "so I'm standing there", "and out of nowhere",
    "this is where it all went sideways", "and then the other shoe dropped",
    "that's not even the worst of it", "just when I thought it was over",
]

# Emotional arcs — how the caller's mood shifts during the call
EMOTIONAL_ARCS = [
    "Starts guarded and vague. Opens up after the host earns trust. Gets real once comfortable.",
    "Comes in hot and emotional. Calms down as they talk it through. Ends more grounded.",
    "Sounds fine at first, almost too casual. The real issue leaks out slowly. Gets heavy toward the end.",
    "Nervous energy at the start. Gains confidence as the host listens. Ends feeling heard.",
    "Matter-of-fact and detached. Cracks show when the host asks the right question. Might get emotional.",
    "Cheerful and joking at first. Using humor to avoid the real thing. Eventually drops the act.",
    "Angry and blaming others at first. Slowly realizes their own role in it. Hard to admit.",
    "Sad and low energy. Perks up when the host engages. Leaves with a little more hope.",
    "Confident and opinionated. But underneath there's doubt. Might ask the host what they really think.",
    "Testing the host at first — seeing if they'll judge. Once safe, shares the real story.",
]

# Relationship to the show
SHOW_RELATIONSHIP = [
    "First-time caller. Nervous about being on the radio. Almost hung up before they got through.",
    "Has listened to the show a few times. Decided tonight was the night to finally call.",
    "Regular listener. Feels like they know Luke even though they've never called before.",
    "Doesn't usually listen to this kind of show but stumbled on it tonight and something made them stay.",
    "Friend told them about the show and dared them to call in.",
    "Called once before a while back. Thinks about it sometimes. Calling again because things changed.",
    "Skeptical about calling a radio show for advice. But it's late and they need to talk to someone.",
    "Listens every night. This is their first time calling. Big deal for them.",
    "Heard a caller earlier tonight and it hit close to home. Had to pick up the phone.",
    "Late-night radio is their thing. They call shows sometimes. Comfortable on air.",
]

# Time-of-night context — why they're up and calling now
LATE_NIGHT_REASONS = [
    "Can't sleep, been staring at the ceiling for an hour.",
    "Just got home from a long shift. Too wired to sleep.",
    "Sitting on the porch, it's quiet out, felt like the right time.",
    "Partner is asleep. This is the only time they can talk freely.",
    "Been drinking a little. Liquid courage to finally say this out loud.",
    "Up with insomnia again. The radio's the only company at this hour.",
    "Just had a fight. Drove around for a while and ended up parked, listening to the show.",
    "Working the night shift, it's dead right now.",
    "Kids are finally asleep. This is the first quiet moment all day.",
    "Couldn't stop thinking about something. Needed to hear another voice.",
    "Up late doom scrolling, heard the show, and thought why not.",
    "Out in the yard looking at the stars. The quiet makes you think.",
]

# Drift tendencies — how some callers wander off-topic
DRIFT_TENDENCIES = [
    "Tends to wander into unrelated stories when the main topic gets uncomfortable.",
    "Will go off on tangents about work if given an opening.",
    "Connects everything back to a TV show they're watching.",
    "Starts answering one question but ends up on a completely different topic.",
    "Brings up random observations about their town or neighbors mid-conversation.",
    None, None, None, None, None, None, None,  # Most callers don't drift
]

# Topic-based call-ins (30% of callers discuss a topic instead of a personal problem)
TOPIC_CALLIN = [
    # Prestige TV discussions
    "just finished Severance season 2 and needs to talk about the ending with someone",
    "has a theory about Severance that they think nobody else has figured out",
    "wants to talk about how Landman portrays the oil industry, because they actually work in it",
    "just watched the Fallout show and wants to discuss how it compares to the games",
    "rewatched Breaking Bad and noticed something they never caught before",
    "wants to argue that The Wire is more relevant now than when it aired",
    "has a hot take about the Game of Thrones ending that they think people will disagree with",
    "just discovered Westworld and their mind is blown by the philosophy of it",
    "wants to talk about which shows people will still be watching in 20 years",
    "thinks Yellowstone went downhill and wants to vent about it",
    "just finished LOST for the first time and has questions",
    "wants to talk about why Stranger Things resonated so hard with their generation",
    "thinks Better Call Saul's finale was perfect and wants to make the case",
    "rewatched Mad Men and realized Don Draper is way worse than they remembered",
    "wants to discuss which show has the best pilot episode ever",
    "thinks The Sopranos ending was genius and will fight anyone who disagrees",
    "just watched True Detective season 1 again and wants to talk about the philosophy in it",
    "wants to recommend Slow Horses because nobody they know watches it",
    "thinks Silo is the most underrated show on TV right now",
    "wants to talk about why prestige TV peaked and where it's going",
    "has been watching The Last of Us and can't stop thinking about the third episode",
    "just discovered Peaky Blinders and wants to argue it's the most stylish show ever made",
    "thinks The Bear is the most stressful show on television and they can't stop watching — wants to talk about why people are drawn to anxiety",
    "rewatched The Office for the eighth time and has a theory that Michael Scott is actually a genius who plays dumb",
    "wants to talk about Shogun and how it's the best historical drama they've ever seen — the production quality alone is staggering",
    "just finished Chernobyl and can't believe what the Soviet Union covered up — wants to talk about the real people behind the show",
    "thinks Fargo the TV show is better than the movie and wants to defend that position",
    "has been watching Reacher and wants to talk about why simple action shows with big dudes punching bad guys are exactly what they need right now",
    "wants to argue that Band of Brothers is the greatest piece of war media ever created and it's not even close",
    "just binged Beef and the spiral both characters go on is the most realistic portrayal of road rage consequences they've ever seen",
    "thinks Andor is better than any Star Wars movie and the fact that it's a 'TV show about space' is underselling it by a mile",
    "wants to talk about why they cried during a specific scene in This Is Us and they're not someone who usually cries at TV",
    "just finished Dark on Netflix and the German time-travel show melted their brain — wants someone to help them understand what they just watched",

    # Science & space
    "read something about a new exoplanet discovery and is genuinely excited",
    "wants to talk about the latest JWST images, says they changed how they think about the universe",
    "read a paper about quantum entanglement that they barely understood but found fascinating",
    "wants to discuss whether we'll see fusion energy in their lifetime",
    "saw something about CERN's latest experiments and wants to geek out about it",
    "has been following the Mars missions and wants to talk about what they've found",
    "read about a breakthrough in neuroscience and wants to discuss what consciousness even is",
    "wants to talk about dark matter and dark energy because it blows their mind",
    "just learned about the scale of the observable universe and it's keeping them up at night",
    "read about new battery technology that could change everything",
    "wants to talk about gravitational waves and what they mean for physics",
    "fascinated by the search for extraterrestrial life, thinks we're close",
    "just learned about extremophiles — organisms living in boiling acid vents on the ocean floor — and thinks it changes what we should be looking for on other planets",
    "read about how Saturn's moon Enceladus is shooting geysers of water into space and probably has a warm ocean under the ice — wants to talk about why we aren't sending a submarine there",
    "learned about magnetars — neutron stars with magnetic fields so strong they could erase your credit card from halfway across the solar system — and can't believe they're real",
    "found out about rogue planets — billions of them just drifting through the galaxy with no star — and wants to talk about whether life could exist on one",
    "read about the Wow! signal from 1977 — a 72-second radio burst from space that's never been explained — and has opinions about what it could have been",
    "just learned that Titan has methane lakes and rain and weather just like Earth but with liquid natural gas instead of water — wants to talk about how weird that is",
    "read about how the sun loses 4 million tons of mass every second through fusion and will still burn for another 5 billion years — the scale broke their brain",
    "found out about the Great Attractor — something massive and invisible is pulling our entire galaxy toward it at 600 km per second and we can't see what it is because the Milky Way is in the way",
    "learned about panspermia — the idea that life on Earth might have started from bacteria riding asteroids — and thinks it's more plausible than people realize",
    "read about the Kuiper Belt object Arrokoth that New Horizons flew past — it's shaped like a snowman and hasn't changed in 4.5 billion years, basically a fossil from when the solar system formed",
    "just learned about fast radio bursts — millisecond blasts of energy from billions of light years away — some of them repeat and nobody knows what's causing them",
    "found out about the Boötes Void — a region of space 330 million light years across with almost nothing in it — and wants to talk about what could cause that",

    # Biology & nature
    "just learned about the immortal jellyfish — Turritopsis dohrnii — it can literally reverse its aging and go back to being a polyp, and scientists are studying it for human aging research",
    "read about how cuttlefish can change their skin color, texture, AND pattern in milliseconds using cells called chromatophores — they're basically living TV screens",
    "found out that the axolotl can regenerate entire limbs, parts of its brain, heart, and spinal cord — and scientists are trying to figure out how to unlock that in humans",
    "learned about mycorrhizal networks — fungi that connect trees underground and let them share nutrients and even send chemical warnings about insect attacks to each other",
    "read about the pistol shrimp — it snaps its claw so fast it creates a bubble that reaches 4,700 degrees Celsius, hotter than the surface of the sun, for a split second",
    "just found out about CRISPR and how scientists can now edit DNA like a word processor — they used it to make mosquitoes that can't carry malaria",
    "learned about the bombardier beetle — it mixes two chemicals in its abdomen that create a boiling toxic spray it shoots at predators at 500 pulses per second",
    "read about how dolphins sleep with half their brain at a time so one eye stays open watching for sharks — it's called unihemispheric sleep",
    "found out that the honey badger has skin so loose and thick that if something bites it, it can literally turn around inside its own skin and bite back",
    "just learned about the deep sea anglerfish mating system — the male is tiny, bites onto the female, and literally fuses into her body until he's just a pair of gonads she carries around",
    "read about how slime mold — a single-celled organism with no brain — can solve mazes and recreate the Tokyo subway map when given food at each station location",
    "learned about epigenetics — how your experiences can chemically modify your DNA and those changes can be inherited by your grandchildren — trauma can literally be passed down",
    "found out about the Cordyceps fungus that takes over ant brains and makes them climb to the perfect height before bursting out of their head to spread spores",
    "read that there's a species of flatworm that shoots its own head off when threatened, then grows a new one — it can be cut into 200 pieces and each piece grows into a complete worm",

    # Psychology & the brain
    "just learned about the McGurk effect — where what you see overrides what you hear — if you watch someone mouth 'fa' while hearing 'ba' your brain hears 'fa' — and wants to talk about how unreliable our senses are",
    "read about hemispatial neglect — people with specific brain damage literally cannot perceive one half of everything, they'll only eat food from one side of their plate and only shave half their face",
    "found out about the rubber hand illusion — scientists can trick your brain into thinking a rubber hand is yours in under two minutes — and it has wild implications for what 'self' means",
    "learned about blindsight — some people who are completely blind can still catch a ball thrown at them because a separate visual pathway bypasses conscious awareness",
    "read about the split-brain experiments where they cut the connection between brain hemispheres and each half developed its own personality and preferences",
    "just found out about phantom limb syndrome and how mirror box therapy tricks the brain into releasing pain from a limb that doesn't exist anymore",
    "learned about the tetrachromacy mutation — some women have four types of color receptors instead of three and can see millions more colors than everyone else",
    "read about how London taxi drivers who memorize the entire city map literally grow a larger hippocampus — their brain physically changes shape from studying",
    "found out about change blindness — you can swap out a person someone is talking to mid-conversation and most people won't notice — and it makes them question everything",
    "just learned about synesthesia — some people literally taste words or see numbers as colors — and one guy experiences every number as having its own personality",

    # Engineering & invention
    "learned about how the SR-71 Blackbird leaked fuel on the ground because the titanium panels were designed with gaps that only sealed when the plane heated up from flying at Mach 3",
    "read about the Antikythera mechanism — a 2000-year-old Greek device that predicted eclipses and tracked the Olympics — it's basically an ancient analog computer and nobody knows who built it",
    "found out about the engineering behind the Panama Canal locks — they use no pumps, everything works by gravity, and they move 14,000 ships a year through a mountain range",
    "just learned that the Hoover Dam's concrete is STILL curing — it generates heat as it hardens and engineers calculated it would take 125 years to fully cure without the cooling pipes they built into it",
    "read about how the Apollo 13 engineers had to build a CO2 scrubber from duct tape, cardboard, and a sock — using only materials they knew were on the spacecraft — in hours, or the crew would die",
    "learned about the Falkirk Wheel in Scotland — it's a rotating boat lift that moves canal boats 80 feet up using less energy than boiling eight kettles of water",
    "found out about Project Orion — a serious 1950s NASA plan to propel a spacecraft by dropping nuclear bombs behind it — the math actually worked and Freeman Dyson was involved",
    "just learned about the Tacoma Narrows Bridge collapse — it twisted itself apart in a mild wind because of resonance — and the video is the most terrifying engineering failure they've ever seen",
    "read about the Great Wall of China's mortar — they mixed sticky rice into the lime morite and it's what made sections survive 600 years — the chemistry is actually brilliant",
    "found out about the Svalbard Global Seed Vault — a doomsday bunker in the Arctic that stores copies of every crop seed on Earth — it's humanity's backup plan for agriculture",
    "learned about how the Chunnel under the English Channel was built — crews dug from both sides simultaneously and met in the middle with only 2 inches of error over 31 miles",
    "read about the engineering of the International Space Station — 16 countries built pieces independently on Earth and assembled them in orbit — and the cooling system alone would blow your mind",
    "just found out about Japan's earthquake-proof skyscrapers — they use massive pendulum dampers and can sway 6 feet without structural damage — and wants to talk about engineering for survival",
    "learned about the Arecibo telescope collapse — for 57 years it was the largest radio telescope on Earth and when the cables snapped the 900-ton platform fell 450 feet into the dish — and they're not rebuilding it",

    # Mathematics
    "just learned about Benford's Law — in naturally occurring datasets, the number 1 appears as the first digit about 30% of the time, not 11% like you'd expect — and it's used to catch tax fraud",
    "read about the Monty Hall problem and wants to argue about it because the answer still doesn't feel right even though they know the math proves it",
    "found out about Gabriel's Horn — a shape with finite volume but infinite surface area — you could fill it with paint but never paint its surface — and it's messing with their head",
    "learned about Euler's identity — e to the i pi plus one equals zero — and someone told them it's the most beautiful equation in mathematics, wants to understand why",
    "read about the Coastline Paradox — the measured length of a coastline depends on the length of your ruler, and as your ruler gets smaller the coastline approaches infinity",
    "just learned about Gödel's incompleteness theorems — any mathematical system complex enough to do arithmetic will contain true statements it can never prove — and wants to talk about what that means for knowledge itself",
    "found out about the Four Color Theorem — you only ever need four colors to color any map so no adjacent regions share a color — and it was the first major theorem proved by a computer",
    "just learned about the Collatz Conjecture — pick any number, if it's even halve it, if it's odd triple it and add one, repeat — it always reaches 1 eventually and nobody can prove why",
    "read about Ramanujan — a self-taught Indian mathematician who mailed his work to Cambridge and turned out to be one of the greatest mathematical minds in history — some of his formulas are still being proven correct 100 years later",
    "found out about the concept of different sizes of infinity — there are more real numbers between 0 and 1 than there are whole numbers in existence — and Georg Cantor proved it and it drove him insane",
    "learned about the Fibonacci sequence in nature — sunflower seeds, hurricane spirals, galaxy arms, pinecone scales — the same ratio keeps appearing and nobody fully understands why",
    "read about the Traveling Salesman Problem — finding the shortest route through a list of cities sounds simple but it's so hard that every computer on Earth running together couldn't solve it for more than a few hundred cities",
    "just found out about Bayesian reasoning — the idea that you should update your beliefs based on new evidence, not throw them out — and it's used in everything from spam filters to cancer screening",

    # Technology
    "wants to talk about AI and whether it's going to change everything or if it's overhyped",
    "has opinions about the latest SpaceX launch and wants to discuss the future of space travel",
    "worried about cybersecurity after reading about a major breach",
    "wants to discuss the ethics of AI-generated content",
    "thinks about energy grid problems and has ideas about solutions",
    "into open source and wants to talk about why it matters",
    "read about quantum computing hitting a new milestone and wants to know if it actually matters or if it's all hype for another decade",
    "has been following brain-computer interface trials and is equal parts fascinated and terrified about where it's going",
    "wants to talk about CRISPR being used to edit genes in living patients — the sickle cell cure is real and they have thoughts about what comes next",
    "is worried about autonomous vehicles being tested on public roads after reading about a close call in their area",
    "read about a breakthrough in nuclear fusion and wants to know why we keep saying it's 10 years away every 10 years",
    "saw a deepfake video of a politician that was so convincing they almost shared it — and now they don't trust anything online",
    "thinks blockchain has legitimate uses beyond crypto that nobody talks about — supply chain tracking, land registries, voting",
    "lives in a rural area that just got Starlink and it completely changed their life — wants to talk about satellite internet closing the gap",
    "read about lab-grown meat getting FDA approval for more products and wants to know if anyone would actually eat it",
    "thinks nuclear power is making a comeback and wants to argue that it's actually the greenest option we have",
    "wants to talk about the e-waste crisis — billions of dollars of electronics in landfills leaching chemicals and nobody seems to care",
    "read about 3D-printed organs being successfully transplanted and thinks it's the most important medical breakthrough nobody is talking about",
    "has been following the asteroid mining industry and thinks whoever figures it out first becomes the richest entity in human history",
    "wants to talk about how vulnerable undersea internet cables are — 97% of global data travels through them and they're basically unprotected",
    "thinks the tech monopoly situation is worse than Standard Oil ever was and wants to know why nobody is doing anything about it",

    # Poker
    "just had the most insane hand at their home game and needs to tell someone",
    "watched a poker tournament and wants to discuss a controversial call",
    "has been studying poker theory and thinks they figured out why they keep losing",
    "wants to talk about whether poker is more skill or luck",
    "played in a tournament and made a call they can't stop thinking about",
    "flopped a set against two players who both had flush draws and the board ran out the worst possible way — still fuming about it",
    "has been reading about physical tells and caught someone at their home game doing the exact thing Mike Caro described in his book",
    "switched from live poker to online and it feels like a completely different game — the aggression is insane",
    "blew through their bankroll in a week because they moved up stakes too fast and wants to talk about the discipline it takes",
    "watched the 2003 WSOP where Moneymaker won and it changed their life — they've been chasing that feeling ever since",
    "wants to debate GTO versus exploitative play — they think the math nerds are ruining poker but also admits it works",
    "just started playing Pot-Limit Omaha after years of Hold'em and their brain is melting trying to adjust to four hole cards",
    "hosts a weekly home game and two regulars almost got in a fistfight over a ruling last week — needs advice on how to handle it",
    "has been on a brutal losing streak for three months playing solid poker and wants to talk about how running bad messes with your head",
    "watched Rounders for the hundredth time and has a theory about why it's still the best poker movie ever made despite the bad accents",

    # Photography & astrophotography
    "got an amazing astrophotography shot of the Milky Way from the desert and is stoked",
    "wants to talk about how dark the skies are out in the bootheel for photography",
    "just got into astrophotography and is overwhelmed by how much there is to learn",
    "shot the most incredible sunset over the Peloncillo Mountains",
    "just captured the Orion Nebula for the first time with a cheap tracker and a DSLR and it actually looks like the photos online — they're hooked",
    "spent three nights trying to photograph the Andromeda Galaxy and when they finally stacked the frames and saw the spiral arms they almost cried",
    "wants to talk about the Ring Nebula and how a dying star can be the most beautiful thing in the sky",
    "just bought a star tracking mount and the difference between tracked and untracked shots is blowing their mind",
    "found a light pollution map and drove two hours to a Bortle 2 zone and the sky looked fake — they could see the zodiacal light for the first time",
    "wants to reignite the film vs digital debate — they've been shooting medium format film and think digital still can't match the look",
    "just got a 200-600mm lens and the moon shots are incredible but now they want to talk about what telescope to buy next",
    "has been doing golden hour portraits in the desert and the light out here is unlike anything they've shot anywhere else",
    "set up a trail cam near their property and got photos of a coatimundi, two bobcats, and something they swear is a jaguar",
    "spent six months learning image stacking software and their deep sky photos went from blurry blobs to actual detail — wants to talk about the processing side of astrophotography",

    # US News & current events
    "wants to talk about something they saw in the news that's been bugging them",
    "has thoughts about the economy and wants to hear another perspective",
    "read about an infrastructure project and has opinions about it",
    "wants to discuss something happening in politics without it turning into a fight",
    "saw a news story about their town and wants to set the record straight",
    "concerned about water rights in the southwest and wants to talk about it",
    "has thoughts about rural broadband and how it affects small towns",
    "went to a county commission meeting about a zoning change and what they witnessed made them question local democracy entirely",
    "got a medical bill for a 20-minute ER visit that was more than their mortgage payment and wants to talk about how the system is broken",
    "is a veteran who's been waiting nine months for a VA appointment and wants to talk about how the people who served are being forgotten",
    "works at a school where they just cut the art and music programs to fund standardized test prep and it's gutting the kids",
    "lives in a border town and is tired of people who've never been here telling them what the immigration situation is actually like",
    "has watched housing prices in their small town triple since the pandemic because remote workers bought everything and locals can't afford rent",
    "drives past a growing homeless encampment every day on their way to work and nobody in city government will even acknowledge it exists",
    "runs a small business and just got hit with a new regulation that's going to cost them $15,000 to comply with — wants to talk about how regulations crush small operators",
    "lives in a food desert — nearest grocery store is 45 minutes away — and the dollar store is the only option, which means processed junk for their kids",
    "wants to talk about the public land debate out west — ranchers need grazing leases, hikers want access, and the feds keep changing the rules",

    # Physics & big questions
    "can't stop thinking about the nature of time after reading about it",
    "wants to talk about the multiverse theory and whether it's real science or sci-fi",
    "read about the double-slit experiment and it broke their brain",
    "wants to discuss whether free will is real or if physics says otherwise",
    "fascinated by black holes after watching a documentary",
    "wants to talk about the simulation theory and why smart people take it seriously",
    "just learned about the delayed choice quantum eraser experiment — it seems like a measurement made NOW can affect what a particle did in the PAST — and it broke their understanding of time",
    "read about the holographic principle — the idea that our entire 3D universe might be information encoded on a 2D surface — and some physicists take this seriously",
    "found out about the Boltzmann brain problem — statistically it's more likely that a random conscious brain would fluctuate into existence in empty space than that our entire universe would form — and wants to talk about what that means",
    "learned about time crystals — a new phase of matter that repeats in time instead of space — they were theoretical until 2017 and now Google has made one in a quantum computer",
    "read about the quantum Zeno effect — observing a particle frequently enough can literally freeze it in place, preventing it from changing state — watched pots really don't boil at the quantum level",
    "just found out about the Casimir effect — two metal plates placed very close together in a vacuum get pushed together by literally nothing — empty space has energy and it exerts force",
    "learned about Wheeler's delayed choice experiment and it suggests that observation might retroactively determine whether a photon acted as a wave or particle — even after it's already traveled",
    "read about the black hole information paradox — Hawking showed black holes evaporate but the information that fell in should be conserved by quantum mechanics — the two biggest theories in physics directly contradict each other",
    "found out about the measurement problem — nobody actually knows what constitutes a 'measurement' in quantum mechanics or why observing something changes it — it's physics' biggest unsolved problem",
    "just learned about quantum tunneling — particles can pass through solid barriers they shouldn't be able to cross — and it's the reason the sun works, because hydrogen atoms tunnel through their repulsion to fuse",
    "read about the arrow of time problem — the laws of physics work the same forward and backward but time clearly only goes one direction and nobody can explain why from first principles",
    "learned about the fine-tuning problem — if any of about 26 fundamental constants of the universe were off by even a tiny fraction, atoms couldn't form and the universe would be empty",

    # Fun facts and knowledge — callers who learned something cool and want to share it
    "just learned about the birthday paradox — you only need 23 people in a room for a 50% chance two share a birthday — and wants to blow the host's mind",
    "read that octopuses have three hearts and blue blood and one of the hearts stops when they swim — thinks they're basically aliens",
    "found out that honey never spoils — they've found edible honey in Egyptian tombs — and now they're rethinking their pantry",
    "learned that there are more possible chess games than atoms in the observable universe and can't wrap their head around it",
    "just found out that Cleopatra lived closer in time to the moon landing than to the building of the pyramids",
    "read that the mantis shrimp can punch with the force of a bullet and see 16 types of color receptors — humans only have 3",
    "learned that there's a town in Alaska where the sun doesn't set for 82 days straight and wants to know how people sleep",
    "found out that your body replaces almost every atom over about 7 years — so philosophically, are you even the same person",
    "read that bananas are technically berries but strawberries aren't and now they don't trust anything",
    "just learned about Dunbar's number — humans can only maintain about 150 meaningful relationships — and it explains a lot about their life",
    "found out that neutron stars are so dense a teaspoon would weigh 6 billion tons and can't stop thinking about it",
    "read about the Mpemba effect — hot water can freeze faster than cold water and scientists still aren't totally sure why",
    "learned that there are more trees on Earth than stars in the Milky Way — about 3 trillion — and it made them feel weirdly hopeful",
    "found out about the overview effect — astronauts who see Earth from space have a permanent shift in how they see humanity",
    "read that we share 60% of our DNA with bananas and now every time they eat one they feel weird about it",
    "just learned about the Fermi Paradox and it's been keeping them up at night — where is everybody",
    "read that the human brain uses 20% of the body's energy despite being only 2% of its weight and wants to talk about what it's doing with all that power",
    "found out that there are more bacteria in their mouth right now than people on Earth and they can't stop thinking about it",
    "learned about the Ship of Theseus problem and wants to argue about whether their grandfather's axe with a replaced handle and head is still the same axe",
    "read that crows can recognize human faces and hold grudges for years — and they're pretty sure the crows in their yard are watching them",
    "just found out that the GPS in their phone has to account for Einstein's theory of relativity to be accurate and thinks that's the coolest thing they've ever heard",
    "learned about anosognosia — a condition where you don't know you have a condition — and now they're wondering what they don't know about themselves",
    "read that there's a fungus in Oregon that's the largest living organism on Earth — covers 2,385 acres underground",
    "found out that sharks are older than trees — they've been around for 400 million years — and wants to talk about what that means about evolution",
    "learned about the pale blue dot photo and read Carl Sagan's speech about it and it wrecked them emotionally",
    "read that the Apollo guidance computer had less processing power than a modern calculator and they landed on the moon with it",
    "found out that every person on Earth could fit inside Los Angeles standing shoulder to shoulder and it changed how they think about population",
    "just learned about the Dunning-Kruger effect and realized it explains half the people in their life",
    "read that the Amazon rainforest produces 20% of the world's oxygen but consumes almost all of it — so it's basically breathing for itself",
    "learned that a day on Venus is longer than a year on Venus and their brain almost broke trying to visualize it",
    "found out about tardigrades — microscopic animals that can survive in space, boiling water, and radiation — and thinks we should be studying them more",
    "read about the Baader-Meinhof phenomenon — where once you learn about something you start seeing it everywhere — and now they're seeing it everywhere",
    "just learned that the entire internet weighs about the same as a strawberry in terms of the electrons storing the data",
    "found out that the loudest sound in recorded history was the Krakatoa eruption in 1883 — it was heard 3,000 miles away and circled the Earth four times",
    "read that dogs can smell time — they can detect the fading of scent to know how long ago someone left — and it blew their mind",
    "learned about the Voyager Golden Record and wants to talk about what they would have put on it if they got to choose",
    "found out there's a lake in Venezuela that has lightning storms 300 nights a year and nobody fully understands why",
    "read about how the color orange was named after the fruit, not the other way around — before that, English speakers just called it 'red-yellow'",
    "learned that trees in a forest share nutrients through underground fungal networks — they call it the 'wood wide web' — and it made them emotional",
    "just found out that the total length of DNA in one human body, if uncoiled, would stretch from here to Pluto and back",
    "learned about the Mpemba effect's cousin — the Leidenfrost effect — where water dropped on a surface hot enough actually floats on a vapor cushion and skitters around instead of boiling",
    "read about how the platypus is venomous, lays eggs, has a bill that detects electric fields, sweats milk, and glows under UV light — it's like nature threw a dart at every category",
    "just found out about the blue whale's aorta — it's so large a small child could crawl through it — and its heart is the size of a golf cart",
    "learned that the Library of Alexandria wasn't destroyed in one dramatic fire — it declined slowly over centuries through budget cuts and neglect — which is somehow sadder",
    "read about linguistic relativity — the Hopi language has no past or future tense, the Pirahã people have no words for specific numbers, and the language you speak literally shapes how you perceive time and quantity",
    "found out about the Ames room illusion — a room built at specific angles that makes one person look like a giant and another like a dwarf — and every haunted house and movie set uses this trick",
    "just learned about supercooling — you can cool purified water below freezing without it freezing, and then tap the bottle and it crystallizes instantly in front of your eyes",
    "read that there's a species of parasitic wasp that turns cockroaches into zombies by stinging their brain in two precise spots, then leads them by the antenna like a dog on a leash",
    "learned about the pale blue dot photo's backstory — Carl Sagan had to fight NASA to turn Voyager's camera around for one last photo because engineers said it served no scientific purpose",
    "found out about heteropaternal superfecundation — twins can have different fathers — it happens more often than people think",
    "just learned that glass isn't actually a slow-moving liquid — that's a myth — old windows are thicker at the bottom because of how they were manufactured, not because the glass flowed",
    "read about how the US military spent millions developing a space pen while the Soviets just used a pencil — except that's actually a myth too, both sides needed the pen because pencil graphite in zero gravity is dangerous in electronics",
    "learned about the Bystander Effect and the real story of Kitty Genovese — the famous '38 witnesses who did nothing' story turns out to be mostly made up by the New York Times, and several people actually did call police",
    "found out about the Overview Institute — they study how seeing Earth from space permanently changes astronauts' psychology — some come back unable to care about national borders or politics",
    "read about how Venice is built on millions of wooden pilings driven into mud — and the wood hasn't rotted in 600 years because it's underwater and the salt petrified it",
    "just learned about the Strandbeest — a Dutch artist named Theo Jansen builds massive skeletal creatures from PVC pipe that walk on the beach powered only by wind, and he calls them a new form of life",

    # Geology & Earth science
    "just learned about Yellowstone's supervolcano — the caldera is 44 miles wide and the last eruption covered most of North America in ash — and it's technically overdue",
    "read about how the Mariana Trench is so deep that if you dropped Mount Everest into it, the peak would still be over a mile underwater",
    "found out about the Great Oxygenation Event — 2.4 billion years ago, cyanobacteria started producing oxygen and it was toxic to almost everything alive at the time — it was the biggest mass extinction ever and we exist because of it",
    "learned about the Chicxulub impact — the asteroid that killed the dinosaurs hit with the force of 10 billion Hiroshima bombs and the shockwave circled the Earth multiple times — and they found the crater under the Yucatan",
    "read about how Iceland is literally splitting apart — you can dive between the North American and Eurasian tectonic plates in the Silfra fissure and touch both continents at once",
    "just found out about Earth's inner core — it's a solid iron ball the size of the moon, hotter than the surface of the sun, and it rotates slightly faster than the rest of the planet",
    "learned about the Permian-Triassic extinction — it killed 96% of all marine species and 70% of land vertebrates — scientists call it 'The Great Dying' and it was caused by volcanic CO2, basically what we're doing now but faster",
    "just found out about the Snowball Earth hypothesis — about 700 million years ago the entire planet may have frozen over completely, even the equator, and the only reason life survived is volcanic CO2 eventually creating a greenhouse effect",
    "read about how rivers can flow uphill through a process called tidal bores — when a tide is strong enough the ocean pushes upriver in a visible wave, and people surf them",
    "learned about the Door to Hell in Turkmenistan — Soviet geologists lit a natural gas crater on fire in 1971 expecting it to burn out in weeks and it's been burning continuously for over 50 years",
    "found out about limnic eruptions — a lake in Cameroon suddenly released a cloud of CO2 in 1986 that silently killed 1,700 people in their sleep — the gas just rolled downhill and suffocated entire villages",
    "read about how the Sahara Desert used to be a lush green savanna with lakes and hippos only 6,000 years ago — the shift happened because Earth's orbital wobble changed the monsoon patterns",
    "just learned that there's a continuously burning underground coal fire in Centralia, Pennsylvania that has been burning since 1962 — the town was condemned and most of it demolished but the fire could burn for 250 more years",

    # History deep cuts
    "just learned about the Dancing Plague of 1518 — hundreds of people in Strasbourg danced uncontrollably for days, some until they died, and nobody has ever fully explained it",
    "read about Project MKUltra — the CIA literally dosed random Americans with LSD without their knowledge to study mind control — and most of the records were intentionally destroyed",
    "found out about the Great Molasses Flood of 1919 — a storage tank burst in Boston and a 25-foot wave of molasses traveling 35 mph killed 21 people and injured 150",
    "learned about the Wow Signal's less-known cousin — the 1967 discovery of pulsars, which astronomers initially labeled LGM-1 for 'Little Green Men' because the signal was so regular they thought it had to be artificial",
    "read about how the Roman Empire had a concrete recipe that was actually better than modern concrete for marine use — it gets stronger in seawater while ours degrades — and we only recently figured out their formula",
    "just found out about the Solutrean hypothesis debates — some archaeologists think ancient Europeans crossed the Atlantic ice shelf to North America before Asian migration across Beringia — it's controversial but the spearpoint styles are eerily similar",
    "read about the Tunguska Event of 1908 — something exploded over Siberia with 1,000 times the force of Hiroshima, flattened 80 million trees, and nobody has ever found a crater or debris",
    "just learned about the Voynich Manuscript — a 600-year-old book written in a language nobody can read with illustrations of plants that don't exist — and nobody knows if it's genius or gibberish",
    "found out about the Toledo War — Michigan and Ohio almost went to war over a strip of land in 1835 and Michigan lost Toledo but got the Upper Peninsula as a consolation prize, which turned out to be way more valuable",
    "read about how the US government tested nuclear weapons on American soil 928 times in Nevada and the 'downwinders' in Utah and NM are still dying from the fallout decades later",
    "just learned about the Radium Girls — women who painted watch dials with radioactive paint and were told it was safe, licked the brushes, and their jaws literally fell off — their lawsuit changed worker safety laws forever",
    "found out about the Aral Sea — it was the 4th largest lake in the world until the Soviet Union diverted the rivers to grow cotton, and now it's basically gone — ship graveyards sitting in the middle of a desert",
    "read about the real story behind the Trojan Horse — most historians think it's myth, but there's a theory it was actually a battering ram or an earthquake, and wants to talk about how stories replace facts",
    "just learned that Easter Island's civilization didn't collapse from cutting down trees like the popular story says — the real collapse came from European disease and slave raids — and the popular version is basically victim-blaming",
    "found out about Zheng He — a Chinese admiral who commanded 300 ships and 28,000 sailors across the Indian Ocean 80 years before Columbus — and then China destroyed all the ships and records because a new emperor decided exploration was wasteful",

    # Human body
    "just learned that your stomach acid is strong enough to dissolve metal — your stomach lining replaces itself every 3-4 days just to keep up — and wants to talk about how wild the body is",
    "read about proprioception — the sense that lets you know where your body parts are without looking — it's technically a sixth sense and when it fails people can't walk or feed themselves",
    "found out that humans are bioluminescent — we glow in the dark — but the light is 1,000 times weaker than what our eyes can detect",
    "learned about the vagus nerve — it connects your brain to your gut and is why you can feel emotions in your stomach — stimulating it can treat depression and epilepsy",
    "read that human bone is stronger than steel pound for pound and can withstand 19,000 pounds per square inch — but it's the internal structure that makes it work, like a building's I-beams",
    "just found out about the enteric nervous system — your gut has 500 million neurons and can operate completely independently from your brain — scientists literally call it the 'second brain'",
    "learned about referred pain — the reason a heart attack hurts in your left arm is because the heart and arm share nerve pathways to the spine and the brain gets confused about where the signal came from",
    "read that your cornea is the only part of your body with no blood supply — it gets oxygen directly from the air — which is why contact lenses need to be breathable",
    "found out that when you blush, the lining of your stomach blushes too — nobody knows why — but it suggests the connection between emotions and digestion is deeper than we think",
    "just learned that your body produces about 1-1.5 liters of saliva per day — enough to fill two bathtubs a year — and without it you literally couldn't taste anything",
    "read about how the human eye can distinguish about 10 million different colors but has no way to communicate most of them — we only have names for a tiny fraction",
    "learned about the diving reflex — when your face hits cold water, your heart rate drops, blood vessels constrict, and your spleen releases extra red blood cells — it's an ancient mammalian survival mechanism we still have",
    "found out that babies are born with about 300 bones but adults only have 206 — the bones fuse together as you grow — and wants to know what else changes about our bodies without us noticing",

    # History and culture
    "just visited the Trinity Site and can't stop thinking about what happened there",
    "has been reading about the Pueblo Revolt and thinks it's one of the most important events in American history that nobody knows about",
    "wants to talk about ghost towns in New Mexico — they've been visiting them and each one has a story",
    "read about the Navajo Code Talkers and thinks they deserve way more recognition than they get",
    "is obsessed with the history of Route 66 and what happened to the towns when the interstate bypassed them",
    "wants to discuss why the Southwest has such a complicated relationship with water and what happens when it runs out",
    "just learned about the Manhattan Project's connection to New Mexico and went down a rabbit hole",
    "wants to talk about how the mining industry shaped these towns and what happens now that the mines are closing",
    "just read about the Camino Real — the trade route from Mexico City to Santa Fe that was used for 300 years — and realized they drive on parts of it without knowing",
    "read about the Buffalo Soldiers stationed at Fort Bayard and thinks their story is one of the most overlooked chapters of Western history",
    "just visited Tombstone and thinks the real story of the Earps and the Cowboys is way more morally gray than the movies make it",
    "learned about the Chinese Exclusion Act and how it affected the mining towns of southern Arizona — there were thriving Chinese communities here that got erased",
    "read about how the Gadsden Purchase — the reason southern NM and AZ are part of America — was basically a back-room railroad deal and wants to talk about how borders are more arbitrary than people think",
    "found out about the Civilian Conservation Corps camps in the Gila and how those young men during the Depression built trails and structures that are still standing 90 years later",
    "thinks the Apache Wars get oversimplified into cowboys-vs-Indians and the actual story of Geronimo and Cochise is way more complicated and fascinating",
    "wants to talk about how Prohibition played out differently in border towns — everyone just crossed to Mexico — and the remnants of that era are still visible",
    "just learned about the great New Mexico tuberculosis migration — thousands of people moved here in the early 1900s because doctors thought the dry air would cure TB — and it shaped entire towns",
    "read about the Bataan Death March survivors from New Mexico — the 200th Coast Artillery was mostly New Mexican soldiers and many never came home — and their state barely talks about it",

    # Food and cooking
    "got into an argument at a family dinner about whether flour or corn tortillas are better and it almost came to blows",
    "has been perfecting their green chile recipe for 20 years and thinks they finally nailed it",
    "wants to talk about how Hatch chile is being threatened by cheaper imports and why it matters",
    "tried to make tamales from their grandmother's recipe and it was a complete disaster — wants to know what they did wrong",
    "has a theory that you can tell everything about a town by the quality of its gas station burritos",
    "went to a fancy restaurant in Tucson and paid $40 for something worse than what their neighbor makes",
    "spent 14 hours smoking a brisket and it came out perfect — wants to talk about the low-and-slow philosophy and why people who rush it are wrong",
    "got into a heated argument about whether you should wash a cast iron skillet with soap and they are prepared to die on this hill",
    "has been on a sourdough journey for six months and their starter has a name and a feeding schedule and they know how that sounds",
    "found a food truck in the middle of nowhere between Deming and Hatch that makes the best tacos they've ever had and they need to tell someone",
    "wants to talk about the green chile versus red chile rivalry and why choosing 'Christmas' is a cop-out",
    "went to the grocery store and spent $180 on what used to cost $90 and wants to rant about food prices and shrinkflation",
    "started a garden this year and growing their own food in desert soil has been humbling — half of it died but the tomatoes are incredible",
    "just processed an elk they hunted and wants to talk about the whole field-to-freezer experience and why more people should understand where meat comes from",
    "has been meal prepping every Sunday and it saved them time and money but they're eating the same five things and losing their mind",
    "worked in restaurants for 15 years and has stories about what goes on in kitchens that would make people never eat out again",

    # Cars and mechanical stuff
    "just bought a truck sight unseen off the internet and it arrived on a flatbed missing the engine",
    "has been restoring a 1972 Bronco for six years and their spouse just gave them an ultimatum — the truck or me",
    "broke down on I-10 between Lordsburg and Deming at 2am and the person who stopped to help them changed their perspective on something",
    "has a theory about why modern trucks are overengineered garbage compared to what they used to make",
    "found their dad's old truck in a barn — been sitting there since he died — and is trying to decide whether to restore it or let it go",
    "wants to debate EV trucks versus gas trucks for actual ranch work — they test drove a Lightning and have thoughts",
    "has been running diesel for 20 years and someone told them gas trucks have caught up — they want to argue about it",
    "just hauled a horse trailer through the Rockies and the transmission story alone is worth calling about",
    "took their Jeep through a trail near Pinos Altos that they definitely should not have attempted and has the body damage to prove it",
    "put together a vehicle survival kit after getting stranded in the desert and wants to talk about what people should actually carry",
    "broke down 40 miles from the nearest cell service and the way they got home is a story they need to tell someone",
    "thinks their 1997 Toyota with 300,000 miles is more reliable than anything made after 2015 and wants to fight about it",
    "has put $22,000 into a project car that was supposed to cost $5,000 and their spouse doesn't know the real number yet",
    "will die on the hill that Snap-on tools are worth triple the price and got into it with a Harbor Freight guy at the parts store",
    "took their truck to a chain shop for an oil change and they cross-threaded the drain plug — the nightmare that followed needs to be heard",

    # Desert and outdoor life
    "was hiking alone near the Gila and had an experience they can't explain and wants to talk about it",
    "has been tracking a mountain lion near their property for weeks and Fish and Game won't do anything about it",
    "wants to talk about the monsoon season — last night's storm was the most intense thing they've seen in 30 years here",
    "found something weird out in the desert they can't identify and it's been bugging them",
    "thinks the dark skies out here are the most underrated thing about living in the bootheel",
    "was out stargazing and saw something in the sky they can't explain — not saying aliens, but also not not saying aliens",
    "wants to talk about what climate change is actually doing to the desert — the creosote is moving, the water table is dropping",
    "almost stepped on a Mojave rattlesnake today and it made them think about how close they live to things that can kill them",
    "has been living off-grid for two years and wants to talk about the reality versus what people see on YouTube — it's harder and better than they expected",
    "their well went dry in August and the process of getting water hauled and drilling deeper changed how they think about everything",
    "spent all summer prepping their property for wildfire season and wants to talk about what most people out here aren't doing that could save their home",
    "just backpacked the Gila Wilderness solo for five days and the hot springs and the silence did something to their head they can't explain",
    "had a javelina family move into their yard and one of them charged their dog — wants to talk about coexisting with wildlife that doesn't care about boundaries",
    "spotted a coatimundi near their property which is way outside their normal range and wonders if anyone else in the bootheel has been seeing them",
    "lives near the border and the dynamics between ranchers, Border Patrol, and the people crossing through their land are more complicated than anyone on the news understands",
    "was working outside when it hit 115 degrees and the heat did something to them physically they didn't expect — wants to talk about heat survival that goes beyond 'drink water'",
    "hauled water for three months while their well was being repaired and it gave them a completely different relationship with every drop that comes out of the tap",
    "was hiking near a ruin in the Gila and found pottery shards and a grinding stone just sitting on the surface — left everything in place but can't stop thinking about who was there",
    "has been ranching in the bootheel for 40 years and wants to talk about how the land has changed — where there used to be grass there's creosote and where there used to be water there's dust",

    # Music and entertainment
    "heard a song on the radio tonight that their late father used to sing and they had to pull over",
    "has a theory about why country music went to hell and wants to argue about it",
    "just saw a concert in a tiny venue in Silver City that was better than any arena show they've been to",
    "wants to debate whether streaming killed music or saved it",
    "has been learning guitar for a year and just played their first song all the way through — it was terrible but they're proud",
    "thinks podcasts are killing radio and wants to argue the other side",
    "wants to recommend an album that nobody they know has heard of and it's driving them crazy",
    "wants to make the case that outlaw country — Waylon, Willie, Merle — was the last time country music was actually dangerous and real",
    "has been deep-diving Delta blues and thinks Robert Johnson's recordings sound like they were made by someone who actually sold their soul",
    "got into folk music through their grandparents' record collection and now they can't stop listening to Woody Guthrie and wants to talk about protest music",
    "spent $300 on a vinyl they found at a shop in Tucson and their spouse thinks they're insane but it's an original pressing and it sounds incredible",
    "wants to talk about one-hit wonders and defend the idea that some artists said everything they needed to say in one perfect song",
    "went to a show in a barn outside Las Cruces with maybe 30 people and it was the best concert experience of their life — the band was three feet away",
    "set up a home recording studio in their garage and has been making music nobody will ever hear but it's the most fulfilling thing they've done",
    "thinks there are criminally underrated musicians nobody knows about and wants to shout out a few before they disappear",
    "has a theory that the music you listen to between ages 14 and 22 becomes hardwired into your brain and everything after that is just noise — wants to argue about it",
    "wants to debate whether live recordings are superior to studio albums — they think the imperfections are what make music real",

    # Philosophy and late-night thoughts
    "has been thinking about whether you're obligated to forgive someone who never apologized",
    "wants to discuss whether people actually change or just get better at hiding who they are",
    "can't stop thinking about the fact that everyone they pass on the highway has a life as complex as theirs",
    "wants to talk about what makes a place 'home' — is it the land, the people, or just time spent there",
    "has a theory that the people who stay in small towns are braver than the ones who leave",
    "wants to talk about why Americans are so bad at being alone and what that says about us",
    "thinks the concept of 'the American Dream' is fundamentally broken and wants to hear if anyone still believes in it",
    "has been reading about stoicism and wants to talk about whether it's actually helpful or just emotional suppression",
    "was lying awake at 3am thinking about how every decision they've ever made led to this exact moment and whether any of it was actually a choice",
    "wants to talk about the paradox of tolerance — should a tolerant society tolerate intolerance — and they've been going in circles about it",
    "thinks most people are living lives they didn't choose and just drifted into and wants to know if anyone else feels that way",
    "read about the concept of 'sonder' — the realization that every stranger has a life as vivid as your own — and can't stop seeing it everywhere",
    "wants to discuss Nietzsche's eternal recurrence — would you live your exact life again, every detail, forever — and what your answer reveals about you",
    "has been thinking about whether loyalty is a virtue or a trap and something happened recently that made them question it",
    "thinks the fear of missing out has been replaced by the fear of not mattering and wants to talk about what that does to people",
    "wants to argue that boredom is actually good for you and that we've engineered it out of our lives to our detriment",
    "has been thinking about whether nostalgia is a lie — we miss a version of the past that never really existed",
    "read about the trolley problem and the fat man variant and wants to know where the host draws the line",
    "thinks we've lost the ability to have uncomfortable conversations and it's making everything worse",
    "wants to talk about whether gratitude is a genuine emotion or something we perform because we're told to",

    # Conspiracy and unexplained
    "lives near the border and has seen lights in the desert at night that don't match any aircraft they know of",
    "wants to talk about what they think is really going on at White Sands and why nobody's allowed near certain areas",
    "has a neighbor who worked at Los Alamos and told them something before he died that they've never been able to verify",
    "drove past the VLA last week and got thinking about whether anyone is actually listening and what happens if someone answers",
    "thinks there's something weird about the old mine shafts around Silver City and has stories from people who've gone in",
    "wants to talk about the Phoenix Lights — thousands of people saw the same thing in 1997 including the governor, and the Air Force explanation was laughable",
    "lives on a ranch and found a cattle mutilation — surgical precision, no blood, no tracks — and the sheriff basically told them not to bother filing a report",
    "read about Dulce Base — the alleged underground facility in northern New Mexico — and the fact that it's an hour from Los Alamos makes them suspicious",
    "grew up hearing Roswell stories from people who were actually there and thinks the official story has changed too many times to be credible",
    "has been binge-watching Skinwalker Ranch content and wants to talk about the Bigelow Aerospace connection and why the government bought the property",
    "stumbled onto numbers stations while scanning shortwave radio — encrypted broadcasts that nobody claims and nobody can decode — and it's creeping them out",
    "read about Operation Paperclip and can't believe the US government recruited over 1,600 Nazi scientists after the war and gave them new identities",
    "has been reading the recently declassified Area 51 documents and while most of it is about U-2 spy planes, there are still huge redacted sections that make them wonder",
    "noticed that ancient structures across different continents — pyramids, temples, megaliths — share alignments to the same star systems and wants to know how cultures with no contact built the same things",
    "wants to make the case that the Bermuda Triangle is actually debunked — the area doesn't have more disappearances than any other busy shipping lane — but they're open to being convinced otherwise",

    # Opinions and hot takes
    "thinks tipping culture in America has gotten completely out of control and had an experience today that set them off",
    "wants to argue that the drinking age should be 18 if you can serve in the military at 18",
    "has a hot take that social media has done more damage to small towns than any economic downturn",
    "thinks HOAs are unconstitutional and just got a $200 fine for their trash can being visible from the street",
    "wants to make the case that trade schools should be free and four-year universities are a scam for most people",
    "thinks the interstate highway system was the worst thing that happened to small-town America and wants to explain why",
    "has been thinking about whether it's ethical to eat meat and they're a rancher which makes it complicated",
    "thinks daylight saving time is pointless and Arizona has it right by not participating — wants to rant about it",
    "believes self-checkout is just making customers do free labor for corporations and refuses to use them",
    "thinks the 40-hour work week is an arbitrary relic from the 1930s and a 4-day week would make everyone more productive",
    "has a theory that subscription services for everything — cars, software, even light bulbs — are turning ownership into a myth",
    "thinks standardized testing destroyed American education and has stories from their kid's school to back it up",
    "wants to argue that the best food in any town is at the gas station or the dive bar, never the fancy place",
    "thinks we should bring back third places — diners, lodges, barbershops — because everyone sits at home now and it's killing communities",
    "believes landlords shouldn't exist as a profession and wants to make the case without sounding like a radical",
    "thinks the news media is more addicted to outrage than their viewers are and it's rotting everyone's brains from both sides",
    "has a hot take that participation trophies weren't the problem — the problem was adults who couldn't handle their kid losing",
    "thinks the way we treat elderly people in this country — parking them in facilities and visiting twice a year — is a national disgrace",
    "wants to argue that small-town gossip is actually a form of social accountability that cities lost and are worse off for",

    # Experiences and stories
    "just drove cross-country alone for the first time and something happened at a truck stop in Texas they need to tell someone about",
    "went to their first AA meeting tonight and wants to talk about what it was like without anyone knowing who they are",
    "volunteered at the food bank this week and met someone whose story broke them",
    "just got back from their first trip out of the country and it completely changed how they see things here",
    "was a first responder to an accident on I-10 last week and they can't get the image out of their head",
    "taught their kid to drive today and it made them realize their kid is about to leave and the house is going to be empty",
    "went to a funeral today for someone they hated and doesn't know how to feel about the fact that they felt nothing",
    "rode a horse for the first time in 20 years today and it brought back every memory of growing up on the ranch",
    "a stranger paid for their groceries when their card got declined and they've been thinking about it all day — wants to talk about unexpected kindness",
    "had a near-death experience during a flash flood in a wash and the way time slowed down changed something fundamental in how they see each day",
    "ran into their estranged sibling at a gas station after 12 years of no contact and neither of them knew what to say",
    "got laid off from a job they hated and it turned out to be the best thing that ever happened to them — wants to talk about how the worst moments become turning points",
    "keeps having the same bizarre coincidence — running into the same stranger in completely different cities — and it's starting to feel like the universe is trying to tell them something",
    "moved from Phoenix to a town of 200 people and the culture shock was nothing compared to what they didn't expect to love about it",
    "grew up in rural New Mexico, moved to New York for 15 years, and just moved back — the reverse culture shock is real and nobody talks about it",
    "found their childhood diary in a box and reading who they were at 13 was like meeting a stranger — wants to talk about how much people change without realizing it",
    "was driving a back road at 2am and had an encounter with something — a person, an animal, they're not sure — that they've never told anyone about",
    "is 25 and feels like they have nothing in common with people their parents' age, but then they started talking to the old-timers at the diner and realized the gap isn't as wide as they thought",
    "had a mentor who changed the entire trajectory of their life with one conversation and they just found out that person passed away — wants to talk about people who shape you without knowing it",
    "has been carrying guilt about something they did 20 years ago and finally apologized to the person this week — wants to talk about what it's like to make amends when you don't know if you'll be forgiven",

    # Animals & pets
    "adopted a dog from the shelter that turned out to be part coyote and the chaos that's followed is a whole story",
    "has a ranch dog that won't herd cattle but has appointed itself guardian of the barn cats and takes the job extremely seriously",
    "found an injured hawk on their property and nursed it back to health and now it won't leave — it sits on the fence post every morning waiting for them",
    "wants to talk about how their cat clearly has a more complex inner life than anyone gives cats credit for — they watched it solve a problem yesterday",
    "has been keeping bees for two years and wants to talk about how it completely changed how they see the natural world",
    "rescued a burro from a kill pen and it has more personality than most people they know",
    "their dog predicted a monsoon storm three hours before it hit and they want to talk about what animals can sense that we can't",
    "has been raising chickens and the social dynamics of the flock are like a reality TV show — there's drama, alliances, betrayal",
    "wants to talk about pack dynamics because their three dogs have a power structure and the smallest one runs everything",
    "found a tarantula in their boot this morning and instead of killing it they relocated it — then spent an hour reading about tarantulas and now they think they're cool",
    "had to put down their dog of 16 years yesterday and wants to talk about why losing a pet can hurt as much as losing a person",
    "thinks the bond between a rancher and their working dogs is one of the most underappreciated relationships in American life",

    # Work & career
    "just quit a job they've had for 15 years without a plan and feels terrified and free at the same time",
    "works night shifts and wants to talk about the weird subculture of people who are awake when everyone else is asleep",
    "has been a truck driver for 20 years and the stories from the road could fill a book — wants to share the weirdest one",
    "started their own business six months ago and the reality versus the dream is something nobody warns you about",
    "works in a trade and is tired of people acting like they're less intelligent because they didn't go to college",
    "just found out they're being replaced by automation at their factory and wants to talk about what that means for people like them",
    "has been a bartender in a small town for 12 years and knows everyone's secrets — wants to talk about what you learn about people from behind the bar",
    "works remote from rural NM and the disconnect between their tech job and their physical surroundings is surreal",
    "inherited the family ranch and doesn't know if they want it but feels like they can't say no — wants to talk about obligation versus desire",
    "is a wildland firefighter and wants to talk about what that job actually looks like versus what people imagine",
    "just got their first raise in four years and it doesn't even cover the increase in grocery prices — wants to talk about wage stagnation",
    "works at a school and the gap between what kids need and what the system provides keeps them up at night",

    # Money & personal finance
    "just calculated how much they've spent on lottery tickets over 20 years and the number made them sit down",
    "wants to talk about the hidden costs of living rural — the gas, the travel, the lack of competition keeping prices high",
    "paid off their house this year and wants to talk about what financial freedom actually feels like versus what they expected",
    "thinks the credit system is designed to keep poor people poor and has the math to back it up",
    "just found out what their parents' house cost in 1985 versus what houses cost now and wants to rant about the housing market",
    "has been living cash-only for a year after a fraud scare and the reactions they get from businesses are fascinating",
    "wants to talk about the economics of small-town businesses — how do places with 200 customers even survive",
    "inherited money they didn't expect and it's creating problems in their family they never anticipated",
    "thinks financial literacy should be mandatory in high school — they didn't understand compound interest until they were 35",
    "wants to discuss whether the stock market is just legalized gambling with better marketing",

    # Books & reading
    "just finished Blood Meridian and needs to process it with another human being because that book did something to them",
    "has been reading Marcus Aurelius and wants to talk about how a Roman emperor's journal from 170 AD is somehow the most relevant thing they've read this year",
    "read The Road by Cormac McCarthy and it wrecked them — wants to talk about whether great books should be required to leave you gutted",
    "just discovered Edward Abbey's Desert Solitaire and it put into words everything they feel about living out here",
    "wants to argue that audiobooks count as reading and will fight anyone who disagrees",
    "has been reading about the history of their town at the local library and found newspaper articles that change the story everyone tells",
    "thinks Lonesome Dove is the great American novel and wants to make the case",
    "read a book about the Dust Bowl and the parallels to what's happening with water in the Southwest right now are terrifying",
    "just read Sapiens and it fundamentally changed how they think about human civilization — wants to talk about the parts that messed them up",
    "found a box of their grandfather's books in the attic and the notes he wrote in the margins are like having a conversation with a dead man",
    "wants to talk about why nobody reads anymore and whether that's actually true or just something people say",
    "just finished a book about the Manhattan Project and the ethical weight those scientists carried is something they can't stop thinking about",

    # Movies & film
    "just rewatched No Country for Old Men and the Coen Brothers nailed what this part of the country feels like better than anyone",
    "wants to argue about the greatest movie ending of all time and has a pick nobody expects",
    "has a theory that all the best movies are about ordinary people in extraordinary situations, not the other way around",
    "just watched a documentary that made them angry and they need to tell someone about it before they explode",
    "thinks the golden age of movies is over and everything now is sequels, reboots, and IP — wants someone to prove them wrong",
    "watched an old Western with their kid and was surprised how much of it is actually about the landscape they live in",
    "wants to talk about movies that got the rural American experience right versus the ones that treat it like a punchline",
    "just saw a movie where the twist ending actually worked and they want to talk about it without spoiling it, which is impossible",
    "thinks Denis Villeneuve is the best director working right now and Dune proved it — wants to debate",
    "rewatched Sicario and wants to talk about how different the border feels when you actually live near it versus how Hollywood shows it",

    # Relationships & family
    "just realized they've become their father and doesn't know how to feel about it",
    "wants to talk about long-distance friendships — they moved away 10 years ago and the people they thought they'd never lose touch with are strangers now",
    "has been married 30 years and someone asked them what the secret is — they don't have one, they just showed up every day",
    "their adult kid moved back home and the dynamic shift from parent-child to two adults under one roof is testing everyone",
    "showed up to jury duty and the defendant turned out to be their mechanic — the same guy who's been working on their truck for five years — and the charge is grand theft auto",
    "wants to talk about the difference between loneliness and being alone — they live by themselves and they're fine, but everyone assumes they're lonely",
    "has been trying to reconnect with their father after 20 years and the conversations are awkward and painful but they keep showing up",
    "thinks modern dating is broken and the apps have turned people into products — has stories from trying to date in a town of 300",
    "just became a grandparent and the way they feel about this tiny human caught them completely off guard",
    "wants to talk about chosen family versus blood family and why the people you pick sometimes know you better than the ones you were born to",
    "had a falling out with their best friend over something stupid two years ago and they're both too proud to call — wants to know when pride becomes self-destruction",
    "thinks the way Americans handle grief — the three-day bereavement leave, the 'be strong' mentality — is insane and wants to talk about it",

    # Health & medicine
    "just learned about the gut-brain axis — there are more neurons in your digestive system than in your spinal cord — and it explains why stress hits your stomach first",
    "read about how chronic sleep deprivation literally causes your brain to eat itself through a process called autophagy — and hasn't slept well in months",
    "found out about the placebo effect's weird cousin the nocebo effect — if you EXPECT a side effect you're more likely to get it, even from a sugar pill",
    "just learned that the appendix isn't useless — it's a safe house for beneficial gut bacteria during illness — and decades of surgeons removed them unnecessarily",
    "read about how cold water immersion triggers the vagus nerve and releases norepinephrine — started doing cold showers and wants to talk about whether it's real or bro science",
    "learned about the microbiome — there are more bacterial cells in your body than human cells — and your bacteria might be influencing your food cravings, mood, and even personality",
    "found out about the fascia system — a continuous web of connective tissue that wraps every muscle, organ, and nerve — and some researchers think it's a whole sensory organ we've been ignoring",
    "just read about how exercise is more effective than medication for mild to moderate depression in multiple studies and wants to talk about why doctors reach for the prescription pad first",
    "learned that your immune system has memory cells that can remember a pathogen for decades — some vaccines work because of cells that have been waiting 40 years for a rematch",
    "read about neuroplasticity — the brain can rewire itself throughout your entire life, not just childhood — and a stroke patient they know learned to talk again at 70",

    # Language & words
    "just learned that the word 'disaster' literally means 'bad star' in Latin because people used to blame catastrophes on planetary alignments",
    "found out that English has no word for the feeling of secondhand embarrassment but German does — fremdschämen — and wants to talk about emotions that exist but have no name in English",
    "read about how the word 'sinister' comes from the Latin word for 'left' because left-handedness was considered evil — and they're left-handed",
    "just learned that the sentence 'Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo' is grammatically correct and wants someone to explain it to them",
    "found out about Esperanto — a language invented in 1887 to be a universal second language — and there are still about 2 million speakers today who genuinely use it",
    "read about the Rosetta Stone and how one rock unlocked an entire dead language because it had the same text in three scripts — wants to talk about what we'd leave behind if our civilization fell",
    "learned about the last speaker of a dying language — when they die, an entire way of seeing the world disappears — and a language dies every two weeks",
    "just found out that 'OK' might be the most universally understood word on Earth and nobody can agree on where it came from — there are at least six competing theories",
    "read about how deaf communities develop sign languages independently and they're full languages with grammar, poetry, and humor — not just hand gestures for spoken words",
    "wants to talk about code-switching — how people unconsciously change how they talk depending on who they're with — and what that says about identity",

    # True crime & justice
    "just listened to a cold case podcast about a disappearance near the border and the details don't add up — wants to talk through the case",
    "read about how many wrongful convictions have been overturned by DNA evidence and it shook their faith in the justice system",
    "wants to talk about the ethics of true crime entertainment — are we honoring victims or exploiting their stories for content",
    "learned about the Innocence Project and how they've freed over 375 wrongfully convicted people — some who served 30+ years for crimes they didn't commit",
    "has opinions about the death penalty that changed after they read about a specific case and wants to work through it out loud",
    "read about a small-town sheriff in the 1970s who ran the county like a personal kingdom and the parallels to places they know are uncomfortable",
    "just learned about forensic genealogy — how they caught the Golden State Killer using a relative's DNA from a genealogy website — and wants to talk about the privacy implications",
    "thinks the true crime obsession in America is actually a healthy response to a broken justice system — people are doing the work the system won't",
    "read about a case where a confession was coerced and the person served 18 years — wants to talk about why innocent people confess",
    "wants to discuss jury nullification — the power jurors have to acquit even when the law says guilty — and why almost nobody knows about it",

    # Drunk, high, or unhinged
    "is three beers deep and just realized they've been pronouncing a common word wrong their entire life and needs to tell someone RIGHT NOW",
    "has been drinking alone and wants to call in to confess that they cried during a truck commercial and they're not even sure why",
    "is way too high and just spent 40 minutes staring at their hand and has a theory about fingers that they think is profound",
    "has been drinking since 5pm and wants to tell the host they're their best friend even though they've never met",
    "is absolutely hammered and wants to pitch their million-dollar invention — it's a terrible idea but they're fully committed",
    "is high and got stuck in a thought loop about whether fish know they're wet and needs someone to talk them through it",
    "has been at the bar since happy hour and just got into an argument with a stranger about something completely unhinged and wants a tiebreaker",
    "is drunk and feeling philosophical — wants to know if the host has ever thought about the fact that your tongue just sits in your mouth all day",
    "is way too high and convinced that their neighbor's cat is spying on them — they have evidence and they want to present it",
    "has been drinking whiskey and wants to call in just to tell the world that they love their truck and they don't care who knows it",
    "is stoned and just watched a nature documentary and is now emotionally devastated by the life cycle of salmon",
    "is drunk and just tried to cook something ambitious and the kitchen looks like a crime scene — wants to narrate the aftermath",
    "is high and realized that the word 'bed' actually looks like a bed and now they can't unsee it and they need to share this with someone",
    "has been at a bonfire drinking and their friends dared them to call a radio show and say something weird — they're going to do it",
    "is drunk and wants to leave a voicemail for their ex through the radio because they blocked their number — the host should probably not let this happen",
    "is high and has been googling deep sea creatures for three hours and is now afraid of the ocean — needs to talk about the goblin shark",
    "is wasted and wants to argue passionately about something incredibly low-stakes like whether a hot dog is a sandwich",
    "is stoned and had a full conversation with their dog and is pretty sure the dog understood — wants the host's opinion on animal consciousness",
    "is drunk and just found their old yearbook and wants to read what people wrote in it because some of these aged terribly",
    "is way too high and keeps forgetting why they called but is having a great time anyway",
    "is hammered and wants to sing a song they wrote — it's bad but they are fully committed and nothing will stop them",
    "is drunk and just learned their coworker makes more money than them for the same job and they are NOT handling it well",
    "is high and wants to pitch a conspiracy theory they came up with in the shower — it involves pigeons and the government",
    "is three sheets to the wind and wants to tell the host about the time they met a celebrity and embarrassed themselves so badly they still lose sleep over it",
    "is stoned and eating cereal at midnight and wants to have a serious debate about which cereal is the objective best — they will accept no compromises",

    # Philosophy & thought experiments
    "read about the trolley problem and has a variation that makes it way harder — what if the one person on the track is someone you love and the five are strangers",
    "just learned about Camus and the absurd and wants to talk about whether life has inherent meaning or if we're all just pretending",
    "wants to discuss the experience machine thought experiment — if you could plug into a simulation of a perfect life, would you, and what does your answer say about you",
    "read about the veil of ignorance and wants to talk about how society would be built if nobody knew where they'd end up in it",
    "wants to debate whether time travel to the past is logically possible or if the grandfather paradox kills it completely",
    "has been thinking about the Ship of Theseus and wants to know — if every cell in your body replaces itself, are you still you",
    "read about Stoicism and Marcus Aurelius and wants to talk about whether ancient philosophy is actually practical for modern life",
    "wants to discuss the simulation hypothesis seriously — not as sci-fi but as an actual philosophical question with math behind it",
    "has been reading about existentialism and wants to talk about what Sartre meant by 'condemned to be free' because it hit them hard",
    "wants to debate the ethics of eating meat — not from a political angle but from a genuine 'can you love animals and eat them' perspective",
    "read about the Fermi Paradox and the Great Filter and now they can't sleep because they think the filter might be ahead of us, not behind",
    "wants to talk about free will vs determinism — if the brain is just chemistry, do we actually choose anything or is choice an illusion",
    "read about Peter Singer's drowning child argument and can't stop thinking about how much they should be giving to charity",
    "wants to discuss whether knowledge is always good — are there things humanity would be better off not knowing",
    "has been thinking about the paradox of tolerance and wants to talk about where the line is between being open-minded and being a doormat",

    # Interesting world history
    "just learned about the Great Emu War of 1932 where the Australian military literally lost a war against emus — they had machine guns and the birds still won",
    "read about the Cadaver Synod where the Catholic Church dug up a dead pope, dressed him up, put him on trial, and found him guilty — in 897 AD",
    "just found out about Unit 731 and Japan's biological warfare experiments in WWII — it's one of the worst things in human history and most people have never heard of it",
    "wants to talk about the Year Without a Summer in 1816 when a volcanic eruption caused global temperatures to drop and Mary Shelley wrote Frankenstein because everyone was stuck indoors",
    "read about the Dancing Plague of 1518 where hundreds of people in Strasbourg danced uncontrollably for days — some danced until they died and nobody knows why",
    "just learned about the Defenestration of Prague where people literally threw government officials out of windows and it started a war that killed 8 million people",
    "wants to talk about the Taiping Rebellion — a guy in China claimed to be Jesus's brother, raised an army, and the resulting war killed 20-30 million people and most westerners have never heard of it",
    "read about Operation Mincemeat where the Allies dressed up a dead homeless man as a military officer with fake invasion plans and fooled Hitler into moving troops away from Sicily",
    "found out about the Aral Sea — the Soviet Union diverted the rivers feeding it for cotton farming and one of the world's largest lakes is now mostly desert",
    "wants to talk about the Tulip Mania in 1637 when tulip bulbs in the Netherlands cost more than houses — it's the original market bubble",
    "just learned that Cleopatra lived closer in time to the Moon landing than to the building of the Great Pyramid — the timeline blew their mind",
    "read about the Halifax Explosion in 1917 — a ship full of explosives blew up in a harbor and leveled an entire city in the largest man-made explosion before nuclear weapons",
    "wants to discuss the Library of Alexandria and how much knowledge humanity might have lost — and whether the burning is exaggerated or not",
    "found out about Zheng He's treasure fleet — Chinese ships five times the size of Columbus's sailing the world 70 years before Columbus 'discovered' anything",
    "just learned about the Wow Signal and the Dyatlov Pass incident in the same week and wants to talk about real-life mysteries that have never been solved",

    # Trivia / "teach me something"
    "just found out that Oxford University is older than the Aztec Empire and it broke their brain — Oxford was teaching in 1096 and the Aztecs didn't start until 1325",
    "learned that Nintendo was founded in 1889 — they were making playing cards when Jack the Ripper was still in the news",
    "wants to talk about the fact that a shuffled deck of cards has never been in that order before and will never be again — the math behind it is insane",
    "just found out woolly mammoths were still alive when the pyramids were being built — there were mammoths on an island until about 1700 BC",
    "learned that honey never spoils — they've found 3,000-year-old honey in Egyptian tombs that was still edible",
    "wants to discuss the fact that there are more possible chess games than atoms in the observable universe — the Shannon number is incomprehensibly large",
    "just found out that sharks are older than trees — sharks have been around for 450 million years and trees only appeared 350 million years ago",
    "learned that the inventor of the Pringles can was cremated and buried in one — his family honored his request",
    "wants to talk about how the entire state of Wyoming has only two escalators and both are in the same city",
    "just found out that bananas are technically berries but strawberries aren't — the botanical definitions make no sense and they need to vent about it",
    "learned that Greenland sharks can live for over 400 years — there are sharks swimming right now that were alive when Shakespeare was writing plays",
    "wants to talk about the fact that we've explored more of the moon's surface than the ocean floor on our own planet",
    "just found out that a group of flamingos is called a 'flamboyance' and they think it's the most perfect word in the English language",
    "learned that the shortest war in history lasted 38 minutes — between Britain and Zanzibar in 1896",
    "wants to discuss the fact that Saudi Arabia imports camels from Australia because Australian camels are considered higher quality",

    # --- Sports ---
    "wants to argue about whether baseball is boring or if people just don't understand the strategy beneath it",
    "thinks boxing has become unwatchable with the celebrity fights and wants to rant about what happened to the sport",
    "just learned about the 1980 Miracle on Ice in detail and the political context makes it even more incredible than the movie",
    "wants to debate whether MMA is more of a real sport than boxing because there's no hiding behind promoter politics",
    "has a theory about why small-market teams can never compete in baseball and it's fundamentally broken",
    "wants to make the case that women's soccer is more exciting to watch than men's and has stats to back it up",
    "thinks the college football playoff expansion is ruining what made the sport special",
    "just learned about the 1904 Olympic marathon — one guy was chased off course by dogs, another nearly died from rat poison his coach gave him as a 'performance enhancer' — it's the most insane sporting event in history",
    "wants to argue that rodeo is the most dangerous sport in America and it gets zero national attention",
    "thinks esports should be in the Olympics and is ready to fight about it",
    "watched a documentary about Diego Maradona and the 'Hand of God' goal and wants to talk about the line between cheating and gamesmanship",
    "has a take about why nobody in the Southwest cares about hockey and whether that'll ever change",
    "thinks the best athletes in the world aren't in pro sports — they're working ranches and logging camps and nobody will ever know their names",
    "just learned that the ancient Olympics were performed completely naked and the word 'gymnasium' comes from the Greek word for naked — wants to talk about how sports evolved",
    "wants to debate the greatest athlete of all time and their pick is not who anyone expects",

    # --- Architecture & design ---
    "just learned about brutalist architecture and can't decide if it's genius or depressing — wants to talk about buildings that make you feel something",
    "read about how the ancient Romans invented concrete that gets stronger in seawater and we still can't fully replicate it",
    "wants to talk about adobe buildings and why the Southwest mastered passive cooling centuries before air conditioning",
    "just visited Taliesin West and Frank Lloyd Wright's vision of desert architecture changed how they see every building",
    "thinks modern tract housing is soulless and we've forgotten how to build places people actually want to live in",
    "learned about the Winchester Mystery House — a mansion that was built continuously for 38 years with stairs to nowhere and doors that open to walls — and wants to talk about obsession",
    "read about how skyscrapers are basically sailboats in the wind — they're designed to sway — and some buildings have massive pendulums at the top to counterbalance",
    "wants to discuss why old courthouses and libraries look like temples but new government buildings look like prisons",

    # --- Environment & climate ---
    "read about the Colorado River compact and how it allocated more water than actually exists — wants to talk about what happens when the math doesn't work",
    "thinks regenerative agriculture could reverse desertification and has been experimenting with it on their own land",
    "just learned about microplastics being found in human blood and breast milk and can't stop thinking about it",
    "wants to talk about the 'right to repair' movement and how planned obsolescence is an environmental disaster nobody addresses",
    "read about dark sky preserves and thinks light pollution is the most underrated environmental issue — they can see the Milky Way from their porch and most Americans can't",
    "just learned that 40% of the food produced in America is thrown away while people go hungry — the logistics of that waste blew their mind",
    "thinks wildfire management in the West has been wrong for a century — Indigenous people used controlled burns for thousands of years and we ignored them",
    "read about how the Ogallala Aquifer that waters the Great Plains is being drained faster than it refills and it could run dry in decades",
    "wants to discuss whether individual action on climate matters or if it's all corporate PR to shift blame onto consumers",
    "just learned about the albedo effect — ice reflects sunlight but open water absorbs it, so melting ice causes more melting — and the feedback loop is terrifying",

    # --- Aviation & flight ---
    "just learned about the SR-71 crew that flew from LA to DC in 64 minutes and set a speed record on their retirement flight — as a mic drop",
    "wants to talk about bush pilots in Alaska and how flying in extreme conditions is the most understated skill in aviation",
    "read about Amelia Earhart's final flight and the new theories about what really happened — the evidence has shifted",
    "learned about the Gimli Glider — a 767 that ran out of fuel mid-flight because someone mixed up metric and imperial — and the pilot landed it on an old drag strip",
    "thinks the Concorde was ahead of its time and commercial supersonic flight will come back — has been following Boom Supersonic",
    "just found out about the Berlin Airlift — the Allies flew 200,000 flights in a year to feed a city — and the logistics are staggering",
    "wants to talk about drone delivery and whether it's the future or just tech companies solving problems that don't exist",
    "read about the Wright Brothers' first flight being shorter than the wingspan of a modern 747 and the speed of progress blew their mind",

    # --- Anthropology & culture ---
    "just learned about cargo cults — after WWII, Pacific Islanders built fake runways and control towers hoping planes would return with supplies — and wants to talk about how humans create meaning from patterns",
    "read about the Pirahã people of the Amazon — they have no concept of numbers, no creation myth, and live entirely in the present — and it challenged everything they thought about human cognition",
    "wants to talk about how different cultures handle death — Tibetan sky burials, New Orleans jazz funerals, Mexican Día de los Muertos — and what our approach says about us",
    "just found out about the Sentinelese — an uncontacted tribe on an island in the Indian Ocean that kills anyone who approaches — and the ethics of leaving them alone versus trying to help",
    "read about potlatch ceremonies of the Pacific Northwest — chiefs destroyed their own wealth to prove their status — and thinks it's the opposite of everything capitalism teaches",
    "wants to discuss why Americans are obsessed with ancestry and DNA tests while other cultures define identity completely differently",
    "learned about the concept of 'hygge' in Danish culture and 'ikigai' in Japanese culture and thinks English doesn't have a word for what matters most in life",
    "read about how the Cherokee had a written language, a newspaper, and a constitutional republic before being forced on the Trail of Tears — and wants to talk about what 'civilized' actually means",

    # --- Economics & systems ---
    "just learned about the Cobra Effect — when the British government paid a bounty for dead cobras in India, people started breeding cobras for the reward — and thinks every government incentive has the same problem",
    "read about how the GDP was invented during WWII as a wartime planning tool and was never meant to measure quality of life — and now it's the only thing politicians talk about",
    "wants to discuss why insulin costs $300 in America and $30 in Canada for the exact same product",
    "just learned about mutual aid networks during the Great Depression and thinks they're more relevant now than people realize",
    "read about the economics of tipping and how it was originally considered un-American — wants to talk about how a post-Civil War practice became mandatory",
    "thinks the gig economy is just the dismantling of labor protections dressed up as 'flexibility'",
    "wants to discuss whether universal basic income could work or if it would destroy the incentive to work — they've been going back and forth for months",
    "just learned that the US spends more per student on education than almost any country but ranks in the middle on outcomes — wants to talk about where the money goes",

    # --- Mathematics & logic (expanded) ---
    "just learned about the Prisoner's Dilemma and how it explains why countries can't cooperate on climate change even when it's in everyone's interest",
    "read about the secretary problem — the math for optimal decision-making says you should reject the first 37% of candidates for anything — dating, apartments, jobs — and then pick the next one that's better than all previous",
    "wants to talk about Zipf's Law — the most common word in any language is used twice as often as the second most common, three times as often as the third — and nobody knows why",
    "just found out about Graham's Number — a number so large that if you wrote a digit on every particle in the observable universe you still couldn't write it — and it's a real answer to a real math problem",
    "learned about Newcomb's Paradox and it's been splitting their friend group — do you take one box or two — and they think the answer reveals something deep about who you are",
    "read about the paradox of the heap — if you remove one grain from a heap of sand at a time, when does it stop being a heap — and it made them realize most categories are fake",

    # --- Hobbies & crafts ---
    "just got into leatherworking and made their first belt and it's terrible but they're hooked on the process",
    "wants to talk about the woodworking community and how making something with your hands hits different than anything digital",
    "has been restoring old tools they find at estate sales and the craftsmanship from 80 years ago makes modern tools look disposable",
    "just started blacksmithing in their garage and wants to talk about how therapeutic hitting hot metal with a hammer is",
    "got into amateur radio and the subculture of people talking to each other across continents with homemade antennas is fascinating",
    "wants to talk about the renaissance of home fermentation — they're making kimchi, kombucha, and hot sauce and their kitchen looks like a mad scientist's lab",
    "has been building model trains for 30 years and the level of detail in the community is insane — people hand-paint individual blades of grass",
    "just picked up fly fishing and the patience required is either meditative or maddening and they haven't decided which",
    "started making their own furniture because they couldn't afford to buy anything solid wood anymore — their first table wobbles but they're proud of it",
    "wants to talk about the dying art of letter writing — they've been sending handwritten letters to friends and the responses have been incredible",

    # --- Biology & nature (expanded) ---
    "just learned about the vampire squid — it lives in the oxygen-minimum zone of the ocean where almost nothing can survive and it turns itself inside out when threatened",
    "read about how elephants mourn their dead — they return to bones of deceased family members and touch them gently with their trunks — some scientists believe they have funerals",
    "found out about the wood frog that freezes completely solid every winter — heart stops, brain stops, ice crystals form in the blood — and it thaws out perfectly fine in spring",
    "learned about the lyrebird that can imitate any sound it hears — chainsaws, car alarms, camera shutters, other bird species — it's basically nature's sampling machine",
    "read about how some plants can 'hear' — they grow toward the sound of running water and pea plants can detect the buzzing of bees and increase nectar production",
    "just found out about the mimic octopus — it can impersonate over 15 different species by changing its shape, color, and movement patterns — flounder, lionfish, sea snake — on demand",
    "learned about quorum sensing in bacteria — they communicate with chemical signals and only attack once enough of them are present — they literally vote before they strike",
    "read about how ravens can solve multi-step puzzles, plan for the future, and even barter with each other — they might be as smart as great apes",

    # --- Psychology & the brain (expanded) ---
    "just learned about the spotlight effect — people vastly overestimate how much others notice their appearance and mistakes — and it explains so much of their social anxiety",
    "read about learned helplessness — when animals (and people) are exposed to uncontrollable situations they stop trying even when escape becomes possible — and it made them think about poverty cycles",
    "found out about the peak-end rule — people judge experiences almost entirely by the peak moment and the ending, not the average — and it explains why a terrible vacation with one great day feels 'not that bad'",
    "learned about cognitive load theory — your working memory can only hold about 4 things at once — and it explains why multitasking is literally impossible",
    "read about the Benjamin Franklin effect — when you ask someone for a favor they actually like you MORE, not less — because the brain rationalizes 'I did them a favor so I must like them'",
    "just found out about anchor bias — the first number you hear in a negotiation completely warps your perception of fairness — and they realized their car dealer knew exactly what they were doing",
    "learned about Maslow's hierarchy and how it's been largely debunked — turns out people don't neatly progress through stages and you can seek self-actualization while starving — but everyone still teaches it as fact",
    "read about the sunk cost fallacy and then had an immediate crisis about three things in their life they're only continuing because they've already invested too much to quit",

    # --- Technology (expanded) ---
    "just learned about mesh networks — entire neighborhoods creating their own internet without ISPs — and thinks it could save rural communities",
    "read about how much energy Bitcoin mining uses — more than some countries — and wants to debate whether the technology is worth the environmental cost",
    "thinks the transition from ownership to subscriptions for everything — software, cars, even tractor firmware — is the biggest scam of the century",
    "just found out about right-to-repair legislation and how companies like John Deere are bricking farmers' tractors if they try to fix them themselves",
    "read about Neuralink's latest trials and the idea of a brain-computer interface in humans is either the most exciting or terrifying development in history",
    "wants to talk about digital preservation — how we're creating more data than ever but losing more of it because formats become obsolete — the dark ages had better records than we might",
    "learned about zero-knowledge proofs — a way to prove you know something without revealing what you know — and thinks it's the most underrated concept in computer science",
    "thinks social media algorithms are the most powerful propaganda tool ever invented and they're not controlled by any government — they're controlled by engagement metrics",

    # --- Science & space (expanded) ---
    "just learned about the Dyson Sphere concept — a hypothetical megastructure that completely surrounds a star to capture its energy — and some scientists think we should be looking for them around other stars",
    "read about the twin paradox — if you travel near the speed of light, you age slower than your twin on Earth — and GPS satellites actually have to correct for this effect right now",
    "found out about neutron star collisions — when they crash, they create gold, platinum, and uranium — literally every gold atom on Earth was forged in a cosmic collision",
    "learned about the cosmic microwave background — leftover radiation from the Big Bang that's everywhere in the universe — your old TV static was partially caused by the birth of the universe",
    "read about the Kardashev scale — a classification system for civilizations based on energy use — and we're not even a Type I yet, we're about a 0.73",
    "wants to talk about the terraforming of Mars — whether it's actually possible, what it would take, and whether we have a right to change another planet",
    "just found out about the Oort Cloud — a theorized shell of trillions of icy objects surrounding our solar system up to 2 light years away — and nothing has ever gone there to confirm it",
    "learned about the multiverse interpretation of quantum mechanics and wants to talk about whether every decision really does create a new universe",

    # --- Philosophy (expanded) ---
    "just read about Plato's Cave allegory and wants to talk about what people in the cave represent today — are we all watching shadows and calling them reality",
    "read about the concept of 'memento mori' — the Stoic practice of remembering you will die — and whether thinking about death every day actually makes life better",
    "wants to discuss the paradox of choice — having too many options makes people less happy than having fewer — and whether that explains the anxiety of modern life",
    "learned about Kierkegaard's concept of 'the leap of faith' and wants to talk about the moments in life where logic fails and you just have to jump",
    "read about ubuntu — the Southern African philosophy that says 'I am because we are' — and thinks individualism is the wrong answer to everything",
    "wants to discuss whether morality is objective or just a social contract — and what the implications are if it's the second one",
    "has been thinking about the hedonic treadmill — the idea that people return to a baseline happiness regardless of what happens to them — and whether chasing happiness is the problem",
    "read about the concept of 'amor fati' — loving your fate, including the suffering — and it either sounds profound or insane and they can't decide which",

    # --- Geology (expanded) ---
    "just found out about the New Madrid fault zone — it caused the Mississippi River to flow backward in 1811 and it could happen again — in the middle of the country, not California",
    "read about how caves form over millions of years through acid dissolving limestone — and some caves have crystals the size of telephone poles that took 500,000 years to grow",
    "learned about ophiolites — chunks of ocean floor pushed up onto land by tectonic forces — and there's one near them where you can literally walk on what used to be the ocean bottom",
    "wants to talk about petrified forests and how a tree can turn to stone over millions of years while preserving the cellular structure — you can see the individual cells under a microscope",
    "just found out about the Deccan Traps — volcanic eruptions in India so massive they lasted a million years and may have contributed to the dinosaur extinction alongside the asteroid",
    "read about how the Grand Canyon exposes 2 billion years of geological history in its layers — you can see the history of Earth just by looking at the walls",

    # --- Additional topics to reach 800+ ---
    # Survival & preparedness
    "just took a wilderness first aid course and the things they learned about treating snakebites and dehydration in remote areas should be common knowledge",
    "wants to talk about water storage and purification — they started prepping after their well went dry and the amount people take clean water for granted is staggering",
    "read about the Donner Party and the actual survival decisions they faced — the story is more nuanced and horrifying than the simplified version most people know",
    "took a class on desert navigation without GPS and the old methods — reading terrain, stars, and plant growth patterns — are an art form that's being lost",
    "wants to discuss fire ecology and why suppressing every wildfire for a century created the megafire crisis we have now",
    "learned about the 1906 San Francisco earthquake and how the response shaped modern emergency management — and thinks we're still not ready for the next big one",
    "read about Shackleton's Antarctic expedition and how he kept 27 men alive for two years in impossible conditions — the leadership lessons are unmatched",

    # Gaming & board games
    "just got into chess and the depth of strategy has completely consumed their free time — wants to talk about why a 1,500-year-old game is still the ultimate test of thinking",
    "wants to argue that board games are having a renaissance and the quality of modern tabletop games blows away anything from the Monopoly era",
    "has been playing Dungeons & Dragons for a year and it's the most creative outlet they've ever had — also the best social time in their week",
    "just finished Red Dead Redemption 2 and wants to talk about how a video game made them feel more about the West than any movie ever has",
    "thinks crossword puzzles are the best brain exercise there is and has been doing the NYT crossword every day for a decade — wants to talk about the culture of puzzle people",
    "just discovered strategy board games and their family has stopped watching TV on weekends — they played Settlers of Catan for six hours last Saturday",

    # Ocean & marine
    "just learned about the hadal zone — the deepest trenches in the ocean — and only three people have ever visited the bottom of the Mariana Trench, fewer than have walked on the moon",
    "read about coral bleaching and the Great Barrier Reef losing half its coral in 25 years — wants to talk about what we lose when an ecosystem that took 20,000 years to build dies",
    "found out about bioluminescent bays where the water glows electric blue when you disturb it — caused by dinoflagellates — and there are only a handful on Earth",
    "learned about the mid-ocean ridge — an underwater mountain chain 40,000 miles long where new ocean floor is constantly being created — it's the longest geological feature on Earth and most people don't know it exists",
    "read about sperm whales and how they dive to 7,000 feet to fight giant squid in complete darkness — and the scars on their skin tell the story of those battles",
    "wants to talk about the ocean's 'twilight zone' between 200 and 1,000 meters — more biomass lives there than anywhere else on Earth and we've barely explored it",

    # Economics & finance deep cuts
    "just learned about the tulip mania crash of 1637 and how a single tulip bulb sold for more than a house — and the parallels to crypto and NFTs are unsettling",
    "read about how the Federal Reserve actually works and most people — including themselves until last week — have no idea how money is created",
    "wants to talk about company towns — places where one employer owned the houses, the stores, and paid in scrip instead of money — and how gig economy platform dependence is the modern version",
    "learned about the Laffer Curve and the long economic debate about whether cutting taxes increases revenue — and thinks both sides are wrong about different things",
    "read about how the GI Bill transformed America after WWII — free college, home loans, job training — and wants to talk about why we can't do something like that again",

    # Medicine & healthcare
    "just learned about the placebo surgery studies — patients who had sham knee surgery improved just as much as patients who had the real procedure — and it raises massive questions about what healing actually is",
    "read about how antibiotics are becoming resistant faster than we're developing new ones and some doctors are calling it a bigger threat than climate change",
    "wants to talk about the opioid crisis from a rural perspective — they've watched it hollow out their town and the pharmaceutical companies that caused it paid fines that amount to pocket change",
    "just learned about fecal transplants — putting healthy gut bacteria into sick people — and it cures C. diff infections 90% of the time when antibiotics fail — wants to talk about how something so gross can be so effective",
    "read about how the life expectancy gap between rich and poor Americans is now 15 years — your ZIP code predicts your health better than your genetics",

    # Music expanded
    "wants to argue about whether auto-tune ruined music or was just the next evolution and people said the same thing about electric guitars",
    "has been collecting vinyl from estate sales and the stories behind the collections are often better than the records — one collection had notes explaining when each album was bought and why",
    "just learned about how the 27 Club — musicians who died at 27 — is actually a statistical myth but the cultural impact of the idea is real",
    "wants to talk about how streaming pays artists fractions of a penny per play and the economics of being a musician are worse now than at any point in modern history",
    "thinks the best live performance they've ever seen was at a small venue with terrible sound and wants to make the case that intimacy matters more than production value",

    # Architecture & history expanded
    "just visited a ghost town near their property and the buildings are still standing — wants to talk about what happens to a town when everyone leaves",
    "read about how the Ancestral Puebloans built entire cities into cliff faces and the engineering is mind-boggling — Cliff Palace at Mesa Verde has over 150 rooms",
    "wants to talk about how Route 66 was decommissioned in 1985 and the towns that depended on it either adapted or died — and the ones that survived did so by selling nostalgia",
    "just learned that the oldest continuously inhabited settlement in North America is Acoma Pueblo in New Mexico — people have lived there for over 1,000 years",
    "read about how railroads decided which towns lived and which towns died in the 1800s — the same thing is happening now with highway bypasses and broadband access",

    # Food & agriculture expanded
    "just learned about the dust bowl and how it was largely caused by farming practices that destroyed the topsoil — and the same mistakes are being made in other parts of the world right now",
    "read about seed saving — farmers keeping and sharing heirloom seeds instead of buying patented ones from corporations — and thinks it's one of the most important quiet rebellions happening",
    "wants to talk about how chile peppers evolved capsaicin specifically to deter mammals from eating them — birds can't taste it, which is why birds spread the seeds — and humans eating them for pleasure is evolution's biggest prank",
    "just found out about the banana apocalypse — the Cavendish banana we all eat is a genetic clone with zero diversity, and a fungus is spreading that could wipe out the entire global supply, just like it did to the Gros Michel in the 1950s",
    "learned about aquaponics — growing fish and plants together in a closed loop where the fish waste feeds the plants and the plants clean the water — and thinks it could revolutionize food production in arid climates",

    # Late-night philosophical expanded
    "wants to talk about whether our digital footprint is a form of immortality or just a ghost that someone else controls",
    "has been thinking about whether small towns are dying or just changing form — the population drops but the identity persists somehow",
    "wants to discuss the concept of 'deep time' — thinking in millions of years instead of decades — and how it makes every human problem feel both insignificant and urgent",
    "thinks we're living through a period that future historians will find as interesting as the Renaissance and wants to talk about what the textbooks will say",
    "has been thinking about the ethics of space colonization — do we have the right to take our problems to another planet, or is it our responsibility to try",
    "wants to talk about legacy — what it means to leave something behind in a world that forgets faster than it remembers",

    # --- reaching 800+ ---
    "just learned about the overview effect's opposite — astronauts who come back from space and can't reconnect with normal life because everything feels small — wants to talk about the cost of perspective",
    "read about how the Navajo have a word for the feeling of being in right relationship with the land and wonders why English doesn't have an equivalent",
    "wants to talk about why Americans romanticize cowboys but forget that a third of them were Black and Latino — the real history is more interesting than the myth",
    "just found out that the town they live in was built on top of an older settlement that was abandoned for unknown reasons — the archaeology suggests people left suddenly",
    "read about the invention of barbed wire and how one piece of metal technology transformed the entire American West — it ended the open range and changed ranching forever",
    "wants to discuss how different the world looks when you start noticing bird species — they started birding as a joke and now they can't go outside without identifying everything they hear",
    "just learned about phantom time hypothesis — a fringe theory that 297 years of history were fabricated — it's almost certainly wrong but the arguments are surprisingly fun to think about",
    "read about how the US government tested nuclear effects on soldiers by having them march toward mushroom clouds — the long-term health impacts were covered up for decades",
    "wants to talk about solar storms — the Carrington Event of 1859 caused telegraph machines to shock operators and catch fire — if it happened today it would knock out the power grid for months",
    "just found out about liminal spaces — photos of empty malls, schools at night, abandoned pools — and wants to talk about why they trigger such a strong emotional response in people",
    "read about the Japanese concept of wabi-sabi — finding beauty in imperfection and impermanence — and thinks Western culture's obsession with perfection is making everyone miserable",
    "wants to discuss the 'third teacher' concept — the idea that the physical environment is as important as the instructor — and why most American classrooms look like they were designed to crush creativity",
    "just learned about desire paths — the unofficial trails people create by walking where they actually want to go instead of where the sidewalk is — and thinks it's a perfect metaphor for how systems fail people",
    "read about how smell is the sense most closely linked to memory — it bypasses the thalamus and goes straight to the emotional brain — and a specific smell has been haunting them for weeks",
    "wants to talk about the concept of 'enough' — at what point does having more stop making you happier — and whether most people could answer that question for themselves",
]

HOT_TAKES = [
    "thinks tipping culture has gotten completely out of hand and refuses to tip more than 15%",
    "convinced that people who let their dogs sleep in their bed are out of their minds",
    "believes working from home is making people lazy and antisocial",
    "thinks college is a scam for most people and trade schools should be the default",
    "is fed up with people who bring babies to nice restaurants and thinks there should be age minimums",
    "believes the designated hitter rule ruined baseball and the NL should have kept pitchers batting",
    "thinks people who don't return their shopping carts are the downfall of civilization",
    "is convinced that vinyl sounds exactly the same as digital and people are lying to themselves",
    "believes breakfast is the least important meal of the day and the whole 'most important meal' thing is cereal company propaganda",
    "thinks fireworks are a waste of money and should be banned in residential areas",
    "is adamant that ranch dressing on pizza is an abomination and people who do it have no taste",
    "believes pickup trucks should require a commercial license because 90% of owners don't actually haul anything",
    "thinks youth sports have become way too competitive and parents are ruining it for the kids",
    "is convinced that small talk is a complete waste of time and people should just be honest about not wanting to chat",
    "believes taco Tuesday is cultural appropriation and nobody wants to have that conversation",
    "thinks people who recline their seats on airplanes are sociopaths",
    "is fed up with gender reveal parties and thinks they're just an excuse for attention",
    "believes the speed limit should be raised to 85 on highways because everyone drives that fast anyway",
    "thinks participation trophies created an entire generation that can't handle failure",
    "is convinced that HOAs are unconstitutional and nobody should be able to tell you what color to paint your house",
    "believes potlucks at work should be illegal because half the people can't cook and nobody wants to say it",
    "thinks people who FaceTime in public without headphones should be fined",
    "is adamant that cold weather is objectively better than hot weather and people who disagree are wrong",
    "believes lawn care culture is insane and everyone should just let their yards grow wild",
    "thinks the whole 'hustle culture' thing is destroying people's health and relationships and nobody should be proud of working 80 hours a week",
    # --- food & drink ---
    "is convinced that well-done steak is perfectly fine and steak snobs are the worst people at every barbecue",
    "thinks ketchup on a hot dog is completely normal and anyone who judges you for it is a food bully",
    "believes gas station coffee is better than Starbucks and is willing to die on that hill",
    "thinks brunch is just breakfast at a markup and the mimosas aren't even good",
    "is fed up with people who say 'I don't eat fast food' like it makes them morally superior",
    "believes sweet tea should be the national drink and unsweetened tea is basically leaf water",
    "thinks the whole charcuterie board trend is just a fancy name for a Lunchable and nobody wants to admit it",
    "is adamant that cooking bacon in the oven is cheating and real bacon is made in a cast iron skillet",
    "believes sushi is wildly overrated and people only pretend to like it because it's expensive",
    "thinks pumpkin spice has gone too far and doesn't belong in anything except pie",
    "is convinced that bottled water is the biggest scam in grocery stores and tap water is fine",
    "believes people who say 'I could never go vegan' have never actually tried and are just being stubborn",
    "thinks the obsession with sourdough bread is ridiculous and a loaf from the store tastes the same",
    "is fed up with 'secret menu' culture and thinks if it's not on the board you shouldn't be ordering it",
    "believes nobody actually likes black coffee — they just drink it to seem tough",
    "thinks food trucks are overpriced for what you get and you're eating standing up in a parking lot",
    "is convinced that deep dish pizza isn't pizza — it's a casserole — and Chicago needs to accept that",
    "thinks avocado toast is fine but not worth twelve dollars and it's not a personality trait",
    # --- technology & social media ---
    "thinks people who post their workouts on social media are just fishing for compliments",
    "is convinced that nobody actually reads the terms and conditions and we've all probably signed away our souls",
    "believes texting 'K' should be classified as a hostile act",
    "thinks people who use speakerphone in public places should have their phone privileges revoked",
    "is fed up with subscription services for everything and thinks you should just be able to buy things and own them",
    "believes social media has made everyone a narcissist and we'd all be happier with flip phones",
    "thinks people who leave read receipts on are power-tripping",
    "is convinced that smart home devices are listening to everything and everyone's just pretending that's okay",
    "believes group chats with more than six people are unusable and someone needs to have the courage to say it",
    "thinks ring doorbell culture has turned neighbors into surveillance agents and it's creepy",
    "is adamant that voicemail is dead and anyone who leaves one is wasting everyone's time",
    "believes people who post vague motivational quotes on LinkedIn are the most untrustworthy people in business",
    "thinks people who use their phone flashlight at concerts should be escorted out",
    "is fed up with AI chatbots replacing customer service and thinks talking to a real person should be a basic right",
    "believes people who share every meal on Instagram are just eating for the camera and the food gets cold",
    # --- driving & transportation ---
    "thinks left-lane campers should get a ticket and it should be enforced like a speed trap",
    "is convinced that roundabouts are superior to four-way stops and Americans are just too stubborn to learn",
    "believes people who back into parking spots are wasting everyone's time so they can feel cool leaving",
    "thinks truck nuts should be a ticketable offense",
    "is fed up with people who don't use turn signals and thinks it should be a mandatory re-test on your license",
    "believes electric cars are fine but the smugness that comes with owning one is unbearable",
    "thinks speed bumps do more harm than good and they've wrecked more suspensions than they've saved lives",
    "is convinced that jaywalking laws are absurd and pedestrians should have the right of way everywhere",
    "believes drive-throughs should have a maximum order limit because one person shouldn't hold up the whole line for twelve minutes",
    "thinks bumper stickers are a cry for attention and nobody's ever changed their mind because of one",
    # --- social norms & etiquette ---
    "thinks baby showers for second kids are greedy and everyone knows it",
    "is convinced that 'let's grab coffee sometime' is the biggest lie in American culture",
    "believes open-plan offices were invented by someone who hates productivity",
    "thinks the expectation to be 'on' and cheerful at 8 AM Monday meetings is inhumane",
    "is fed up with people who say 'no offense but' and then say the most offensive thing possible",
    "believes thank-you cards are outdated and a text is perfectly acceptable",
    "thinks wedding registries over $200 per item are delusional",
    "is convinced that nobody actually enjoys networking events and everyone is pretending",
    "believes mandatory fun at work — team-building exercises, trust falls, escape rooms — should be abolished",
    "thinks people who show up late to everything and say 'that's just how I am' are being selfish and they know it",
    "is adamant that tipping on takeout orders makes no sense because nobody carried a plate to your table",
    "believes people who humble-brag about how busy they are just have bad time management",
    "thinks RSVP culture is broken and half the people who say 'yes' won't show up",
    "is convinced that the phrase 'living my best life' is the most annoying sentence in the English language",
    "believes putting your Venmo in your social media bio is tacky",
    "thinks the handshake is outdated and germy and we should all just nod at each other like civilized people",
    "is fed up with people who clap when the plane lands — it landed, that's the minimum expectation",
    # --- sports & entertainment ---
    "thinks the Super Bowl halftime show hasn't been good since Prince and everyone's been lying since",
    "is convinced that golf isn't a sport — it's a walk that's been ruined by a stick and a ball",
    "believes movie theaters need to go back to assigned seating only because people are animals",
    "thinks reality TV is rotting society's brain and we should all be embarrassed",
    "is fed up with celebrity worship and thinks famous people's opinions on politics should mean nothing",
    "believes the NFL overtime rules are still unfair no matter how many times they change them",
    "thinks watching other people play video games on Twitch makes less sense than watching paint dry",
    "is convinced that the Olympics should go back to amateurs only because professional athletes ruined it",
    "believes sports commentators talk too much and there should be a broadcast option with just crowd noise",
    "thinks tailgating is better than the actual game and most fans know it",
    "is adamant that the wave at stadiums is obnoxious and someone should have the guts to not stand up",
    "believes movie remakes are creatively bankrupt and Hollywood should be embarrassed",
    # --- home & lifestyle ---
    "thinks air fresheners just make a room smell like flowers and garbage at the same time",
    "is convinced that throw pillows serve zero purpose and are a conspiracy by the home decor industry",
    "believes people who mow their lawn before 9 AM on a Saturday are menaces to society",
    "thinks garage sales should require a permit because half of them are just people selling trash on their driveway",
    "is fed up with the tiny house movement and thinks people are just romanticizing being broke",
    "believes pool ownership is a scam because you spend more time cleaning it than swimming in it",
    "thinks everyone should have a clothesline and dryers are the laziest invention ever made",
    "is adamant that carpet in bathrooms is a war crime against interior design",
    "believes people who don't wave back when you let them merge are irredeemable",
    "thinks home renovation shows have given every couple unrealistic expectations and that's why contractors are booked out two years",
    "is convinced that leaf blowers are the most pointless invention — the leaves just move three feet and come back",
    "believes Christmas decorations before Thanksgiving should be a fineable offense",
    # --- pets & animals ---
    "is convinced that cat people are smarter than dog people and has a study to back it up that nobody wants to hear about",
    "thinks dressing up pets in costumes is low-key animal cruelty and nobody wants to be the one to say it",
    "believes emotional support animals have gotten out of control and a peacock should not be on a plane",
    "thinks off-leash dogs in public parks should result in a fine because 'he's friendly' is not a leash",
    "is fed up with designer dog breeds and thinks mutts are healthier, smarter, and better looking",
    "believes people who call themselves 'dog mom' or 'cat dad' need to understand that a pet is not a child",
    # --- parenting & family ---
    "believes kids don't need a phone until they're sixteen and will argue this until they're blue in the face",
    "thinks family group texts are a form of psychological warfare",
    "is convinced that helicopter parenting has created adults who can't change a tire or cook an egg",
    "believes the 'kids eat free' deal at restaurants is the only honest marketing left in America",
    "thinks organized playdates are weird and kids should just go outside and knock on doors like they used to",
    "is fed up with parents who let their kids run wild in stores and then say 'they're just expressing themselves'",
    "believes the school drop-off line is a lawless wasteland and someone should direct traffic",
    # --- work & money ---
    "thinks tipping jars at counter-service places are guilt trips and the business should just pay people more",
    "is convinced that most meetings could be an email and most emails could be nothing",
    "believes 'quiet quitting' is just doing your job as described and people need to stop acting like it's a scandal",
    "thinks dress codes in 2026 are ridiculous and nobody works better in khakis",
    "is fed up with loyalty programs and points systems that are designed to be confusing on purpose",
    "believes the entire diamond industry is a scam and engagement rings don't need to cost two months' salary",
    "thinks self-checkout machines are just stores making you do free labor and then getting mad when you mess up",
    "is convinced that credit scores are a made-up number designed to keep people anxious",
    "believes Black Friday is a psychological experiment that humanity keeps failing",
    "thinks landlords who raise rent every year without improving anything are running a protection racket",
    "is adamant that 'unlimited PTO' is a trick to make people take less vacation and feel guilty about it",
    "believes the forty-hour work week is arbitrary and we should have switched to thirty-two hours a decade ago",
    "thinks college textbook prices are legalized robbery and professors who assign their own book should be ashamed",
    # --- reaching 150+ ---
    "is convinced that New Year's resolutions are a scam invented by gym memberships and diet companies",
    "thinks alarm clocks are a form of violence and society should be built around natural wake times",
    "believes most people who say they love camping actually just like the idea of camping and hate every second of the reality",
    "thinks couples who share a single social media account are hiding something and everyone knows it",
    "is fed up with influencers who say 'use code [NAME] for 10% off' and believes nothing they recommend is real",
    "thinks the snooze button is humanity's greatest invention and anyone who gets up on the first alarm is a psychopath",
    "believes movie theaters that charge $8 for popcorn that costs 15 cents to make are committing a crime against civilization",
    "thinks people who bring their own bags to the grocery store but drive an SUV are missing the point entirely",
    "is adamant that the 'customer is always right' mentality created the worst generation of entitled shoppers in history",
    "believes daylight saving time should be abolished and the states that don't observe it have it figured out",
    "thinks multitasking is a lie people tell themselves and everyone who claims to be good at it is actually bad at two things at once",
    "is convinced that most coffee is over-roasted garbage and dark roast people are just addicted to burnt flavor",
    "believes every neighborhood should have a public tool library because nobody needs to own a power washer they use twice a year",
    "thinks white noise machines are just expensive fans and a ceiling fan does the exact same thing for free",
    "is fed up with restaurants that dim the lights so low you need your phone flashlight to read the menu",
    "believes the middle seat on a plane gets both armrests and this should be a federal law",
    # Comedy writer entries
    "thinks most people's dogs are poorly trained nightmares and the owners know it but saying anything about someone's dog is now treated like criticizing their child — and half the time the child is also a nightmare but at least the kid might grow out of it",
    "is convinced that couples who say they 'never fight' are either lying or so dead inside they've stopped having opinions — healthy people disagree, and if you haven't told your partner they're wrong about something you don't respect them enough to be honest",
    "believes the gym is the most dishonest place in America — everyone's pretending not to look at each other, pretending they know how to use the machines, pretending their music isn't too loud, and pretending they're not judging the person next to them who is absolutely doing that exercise wrong",
    "thinks anyone who posts a picture of themselves crying on social media has never experienced a real emotion in their life — real grief doesn't need an audience and if your first instinct when something terrible happens is to open your front-facing camera you need a therapist not followers",
    "is convinced that 'I'm not like other guys' is the most reliable indicator that a man is exactly like every other guy — the guys who are actually different never announce it because they don't know they're different, that's what makes them different",
    "believes every man has a number — an amount of money where they'd do something they currently think is beneath them — and most men's number is a lot lower than they'd admit, and pretending otherwise is the biggest lie men tell themselves",
    "thinks baby showers for a second kid should be illegal and the fact that people have the nerve to ask for gifts twice for doing the same thing is the kind of entitlement that's wrong with this country",
    "is fed up with people who say 'money doesn't buy happiness' because it was clearly invented by someone who's never had to choose between gas and groceries — money absolutely buys happiness up to about a hundred grand and after that it buys a nicer version of the same unhappiness",
    "believes the worst people at any barbecue are the ones who show up, eat everything, and then say 'I could have made this better' — if you could have, you would have, but you didn't, you brought a bag of ice and you should be grateful anyone invited you",
    "thinks people who say 'I tell it like it is' are just rude people who found a way to brand their personality disorder as a virtue — telling it like it is would mean occasionally saying something nice and they never do",
    "is adamant that the invention of the 'open floor plan' office was revenge by management on workers — nobody in history has ever done their best thinking while a coworker eats yogurt four feet from their face",
    "believes people who post their gym routine on social media are compensating for a complete lack of personality — nobody who bench presses 225 needs to tell you about it, they just walk around looking like they bench 225 and that's enough",
    "thinks the concept of a 'guilty pleasure' is cowardice — either you like something or you don't, and calling it guilty is just preemptively apologizing for having taste that someone might judge, and that's weaker than whatever you're watching",
    "is convinced that 90% of people who say they 'love to cook' actually love to eat and tolerate cooking — the ones who really love cooking are weird about knives and have opinions about salt that nobody asked for",
    "believes the worst invention of the 21st century isn't social media — it's the read receipt — because at least with social media you can pretend you didn't see it, but a read receipt is proof that someone looked at your message, understood it, and chose silence, which is violence",
]

CELEBRATIONS = [
    "just got promoted to shift lead after three years of being told they weren't ready",
    "one year sober today and nobody's awake to tell",
    "their band played their first real gig and people actually showed up — like forty people",
    "just paid off their truck two years early and is sitting in it right now feeling like a king",
    "their daughter got a full ride scholarship and they can't stop crying about it",
    "finally passed the GED at 38 and their kids made them a cake",
    "got their citizenship today after twelve years of paperwork",
    "just closed on their first house — a fixer-upper but it's theirs",
    "their small business turned a profit for the first time this month",
    "got the all-clear from the doctor after a cancer scare that had them planning their funeral",
    "their estranged brother called out of nowhere and apologized and they talked for three hours",
    "won the county fair pie contest with a recipe they invented themselves",
    "just ran their first 5K at 52 and didn't walk a single step",
    "their rescue dog that was supposed to be unadoptable just graduated therapy dog training",
    "got accepted into nursing school after being rejected twice",
    "their kid drew them a picture that said 'best dad in the world' and they've been staring at it for an hour",
    "finally told their family they're gay and nobody cared — in the best way",
    "their podcast just hit 1,000 downloads and they know that's nothing to most people but it means the world to them",
    "caught a foul ball at the Isotopes game and gave it to a kid — but the kid gave it back and said 'you need this more'",
    "just got their first paycheck from a job they actually like",
    "their old high school teacher tracked them down to say they were proud of them",
    "rebuilt a 1974 Chevy with their dad over the last two years and they just drove it for the first time tonight",
    "got a letter from the IRS saying they overpaid and they're getting $3,200 back",
    "their mom who has dementia recognized them today for the first time in months",
    "just won their fantasy football league and nobody will let them stop talking about it",
    "finished writing a novel — it's probably terrible but they finished it",
    "their neighbor who they've been feuding with for two years brought over a pie and apologized",
    "hit a hole-in-one today and the only witness was a guy who doesn't speak English",
    "their adult kid who moved away just called to say they want to come home for Christmas",
    "made it through a whole week without a panic attack for the first time in a year",
    "got a standing ovation at open mic night at a bar in Silver City",
    "their spouse surprised them with tickets to see Willie Nelson",
    "just became a grandparent for the first time and they already bought the kid a fishing rod",
    # --- career & work milestones ---
    "just got hired at the job they've been applying to for two years and the rejection emails are finally worth something",
    "finally quit the job that was making them miserable and already feels ten years younger",
    "their boss pulled them aside today to say they're the best hire they ever made",
    "just finished their apprenticeship and got their journeyman card — electrician, took four years",
    "landed their first client as a freelancer and it's someone they've admired for years",
    "got a raise without asking for one — their manager just said 'you've earned this'",
    "just passed the bar exam on their third try and their study group threw them a party at Denny's",
    "their food truck got its first five-star review and the reviewer said it was the best green chile burger in the state",
    "just retired after 35 years at the same company and they rang a bell for them and everything",
    "got promoted to foreman on the job site and their dad — who's also in construction — cried when they told him",
    "their Etsy shop just hit 500 sales and they started it as a joke making earrings out of bottle caps",
    "just got tenure after seven years and can finally stop pretending to be calm about it",
    "passed their CDL test today and already has a trucking job lined up starting Monday",
    "their catering business just got booked for a wedding — their first wedding — and they can't stop menu planning",
    # --- health & recovery ---
    "just completed physical therapy and can walk without a cane for the first time since the accident",
    "two years clean today and their sponsor took them out for pancakes to celebrate",
    "got their blood pressure down to normal without medication just by walking every day and cutting out soda",
    "their kid who's been in and out of the hospital all year just got discharged with a clean bill of health",
    "finally found a therapist who actually helps after going through four who didn't",
    "just got their six-month chip and their kids were there to see it",
    "completed their first full night of sleep without nightmares since coming home from deployment",
    "their doctor said the tumor is shrinking and the treatment is working",
    "just ran a mile without stopping for the first time since their knee surgery eighteen months ago",
    "quit smoking after twenty-two years and today is day ninety and they can taste food again",
    "their anxiety medication finally kicked in and they went to the grocery store without a panic attack for the first time in months",
    # --- family & relationships ---
    "their kid said 'I love you' unprompted for the first time and they had to pull the car over",
    "just celebrated their 25th wedding anniversary and still gets butterflies",
    "their teenager who hasn't talked to them in months just sat down at dinner and started telling them about their day",
    "adopted a kid today after three years in the foster system and the kid asked if they could call them mom",
    "their dad who never says 'I'm proud of you' said it today — out of nowhere — and they're still shaking",
    "reconnected with their college roommate after fifteen years and talked for six hours straight",
    "their partner learned to cook their grandmother's recipe and surprised them with it on their birthday",
    "their family showed up — all of them — for their kid's school play and took up an entire row",
    "just found out they're going to be a dad and they've been walking around grinning like an idiot all day",
    "their stepdaughter made them a Father's Day card that says 'real dad' on it and they can't hold it together",
    "their parents who've been separated for eight years just went to dinner together and are talking about trying again",
    "their kid graduated basic training today and they've never been more proud or more terrified",
    "their wife beat cancer and rang the bell today and the whole floor clapped",
    # --- personal achievements ---
    "just got their motorcycle license at 47 and feels like a teenager again",
    "taught themselves guitar over the last year and just played a full song in front of people without messing up",
    "finished their first marathon — not fast, dead last actually — but they finished",
    "just published a children's book they wrote for their grandkids and the local bookstore is carrying it",
    "learned to swim at 40 because they were tired of being afraid of the water",
    "painted something they're actually proud of for the first time and hung it in their living room",
    "just got their pilot's license and took their mom up for her first flight in a small plane",
    "finished restoring a vintage jukebox that's been sitting in their garage for six years and it works",
    "grew a garden this year that actually produced food — tomatoes, peppers, squash — and they've been giving it away to neighbors",
    "read fifty books this year after not finishing a single one in the last decade",
    "built a treehouse for their grandkids with their own hands and it's got a rope bridge and everything",
    "just completed a 200-piece puzzle of the Sandia Mountains and is going to frame it",
    "learned to change their own oil and brakes from YouTube videos and saved $400 this year",
    # --- community & kindness ---
    "organized a neighborhood cleanup and forty people showed up when they expected maybe five",
    "their little league team just won the district championship and half the kids had never played before this season",
    "started a free tutoring program at the library and three of their students just made the honor roll",
    "their church raised enough to pay off a family's medical debt anonymously",
    "coached their daughter's soccer team to an undefeated season and the kids gave them a signed ball",
    "drove their elderly neighbor to chemo every week for six months and today was the last appointment — she's in remission",
    "started a tool lending library in their garage and half the block has borrowed something",
    # --- financial wins ---
    "just paid off their student loans — all $47,000 — and it only took eleven years",
    "finally has $1,000 in savings for the first time in their adult life and knows that sounds small but it's everything",
    "their credit score went from 520 to 740 in two years and they did it by themselves with no fancy program",
    "got approved for a small business loan they've been working toward for three years",
    "sold the house they inherited, paid off all their debt, and still had enough to put a down payment on a place of their own",
    "won $500 on a scratch-off and used it to fix the fence they've been zip-tying together for a year",
    "just made their last car payment and screamed in the parking lot of the credit union",
    "their side hustle selling firewood just paid for Christmas and they didn't have to put a single thing on a credit card",
    # --- unexpected & quirky wins ---
    "found their lost wedding ring in the yard with a metal detector after giving up hope three months ago",
    "their sourdough starter that they've been nursing for a year finally made a loaf that doesn't taste like vinegar",
    "won a radio contest they forgot they entered and the prize is a weekend in Ruidoso",
    "their hen that stopped laying started again and gave them a double-yolk egg like a peace offering",
    "just beat their personal best at the county arm wrestling competition and they're 61",
    "found a first edition book at a garage sale for two dollars and it's worth over $300",
    "their truck passed inspection on the first try for the first time in its 18-year life",
    "got recognized at the grocery store by someone who said their yard is the nicest on the street",
    "their kid's science fair project won first place — a volcano that actually erupted — and the principal had to evacuate the gym",
    "set up a hummingbird feeder three years ago and today had eleven birds on it at once",
    "finally beat the video game they've been stuck on since 2019 and nobody in their house cares but they need someone to know",
    "their homemade hot sauce got so popular at work that people are asking to buy jars and they might actually have a business",
    "taught their 78-year-old mother to video call and now she calls every single day which is a whole other situation but today it's a win",
    "their dog won 'best trick' at the county fair for playing dead so convincingly that a kid started crying",
    "just passed their real estate exam and already has three friends who want them to find them a house",
    "their 10-year-old scored the winning goal in the championship and got carried off the field by their teammates",
    "finally learned to parallel park at 34 and nailed it on the first try in front of the DMV instructor",
    "their poem got published in a real literary magazine — not a scam one — and they got paid forty dollars for it",
    "just celebrated five years at their job — the longest they've ever held one — and their manager threw a little party",
    "their rescue cat that hid under the bed for three months finally sat in their lap last night and purred",
    "won the office chili cookoff against a guy who's been undefeated for seven years and the whole floor erupted",
    # --- reaching 150+ ---
    "just got the call that their adoption was finalized after two years of waiting and they can't stop shaking",
    "their son who dropped out of high school just earned his diploma at 26 — he walked across the stage and the whole family was there",
    "finished paying for their kid's braces out of pocket because they don't have dental insurance — $5,400 over three years and the kid's smile is worth every penny",
    "their small-town restaurant got mentioned in a regional food magazine and the phone hasn't stopped ringing",
    "just celebrated 30 days without a cigarette after smoking for 15 years — they can taste their coffee again",
    "their kid made the varsity team as a freshman and the coach pulled them aside to say the kid has real talent",
    "got a letter from a college student they mentored saying they're the reason the student didn't drop out",
    "fixed their own washing machine using a YouTube tutorial and saved $400 — they've never felt more capable in their life",
    "their elderly father who hasn't driven in two years passed his re-evaluation and got his license back — he drove himself to the diner and sat at the counter like a king",
    "finally confronted their fear of public speaking and gave a toast at their brother's wedding that made the whole room laugh and cry",
    "their daughter's first art show at the community center sold three paintings and she's twelve",
    "landed a hole-in-one on a par three they've played 500 times and the foursome behind them saw the whole thing",
    "their rescue horse that came in malnourished and terrified let someone ride them for the first time today — it took fourteen months of patience",
    "just finished their EMT certification and their first call was helping deliver a baby in a parking lot — mom and baby are fine",
    "got a handshake deal with a rancher to supply beef to three restaurants in town — their grass-fed operation is finally sustainable",
    "their band got asked to play the county fair main stage after years of playing to ten people at open mics",
    "ran into a kid they coached in little league fifteen years ago — the kid is now a firefighter and said coaching made the difference",
    "their community garden plot produced so much this year they donated 200 pounds of produce to the food bank",
    "just took their mother to the ocean for the first time in her life — she's 74 and stood in the water and cried",
    "their podcast interview with a local legend hit 10,000 downloads and the story got picked up by a state newspaper",
    "found out their blood donation saved a kid in the next county — the blood bank sent them a letter and the family wrote a note",
    "their youngest kid read a whole book by themselves for the first time and brought it to them to announce it at bedtime",
    "rebuilt their porch by hand over three weekends and the neighbors started coming over to sit on it — it's become the block's gathering spot",
    "their kid who was nonverbal until age four just gave a presentation in front of their whole class",
    "just got offered a spot at a trade school that's fully funded — they applied on a whim and didn't think they'd get in",
    "their quilt that took six months to make won a blue ribbon at the state fair and three people asked to commission one",
    "saved enough to take their whole family on vacation for the first time — all seven of them in a rented van to the Grand Canyon",
    "their neighbor who's been battling depression for years came over with a pie and said 'I think I'm going to be okay' and meant it",
    "got a perfect score on a licensing exam they failed twice before — studied every night after the kids went to bed for four months",
    "their rescue dog who was afraid of everything finally played with another dog at the park today — tail wagging, full zoomies, the works",
    # Comedy writer entries
    "finally told their micromanaging boss to go to hell — didn't quit, didn't get fired, just said it in a meeting and the boss went quiet and now treats them with respect for the first time in four years and they realize they should have done it on day one",
    "their ex who left them for someone 'more ambitious' just got fired and is delivering for DoorDash — the caller just got promoted to regional manager and ordered lunch through the app and guess who showed up at their office with a bag of pad thai",
    "finally got their mother to admit that their aunt's potato salad is terrible and has always been terrible — thirty years of Thanksgivings vindicated in one sentence and they screamed in their truck in the driveway",
    "their dad, who has never once complimented a meal in 65 years of life, said their brisket was 'not bad' and they're treating it like a James Beard Award because from this man that IS a James Beard Award",
    "just won a small claims court case against their former landlord who kept their security deposit — the judge looked at the photos, looked at the landlord, and said 'you should be ashamed of yourself' and the caller said that sentence was worth more than the $1,800",
    "won an argument with their spouse about whether you can make a left turn at a specific intersection — drove back to the intersection, pointed at the sign, and the spouse said 'huh, I guess you're right' which is the closest thing to a trophy this marriage has ever produced",
    "got confirmed their vasectomy took and is celebrating the most underrated freedom a man can have — his wife is equally thrilled and they're going to take the money they'd been putting into a college fund and buy a bass boat",
    "their neighbor who's been playing music at full volume every night for two years just got evicted — the caller watched the moving truck from their porch with a beer and it was the most satisfying thing they've experienced since their wedding day, and honestly it might have edged that out",
    "caught a fish so big that nobody believes them even with the photo — their buddy said it was a 'forced perspective' and their wife said 'why does the fish look fake' and they've been defending this fish's honor for three weeks and they will not stop until justice is served",
    "successfully lied about their age at a new job and has been getting away with it for two years — they're 52, everyone thinks they're 44, and when a coworker said 'you look amazing for your age' they just said 'good genes' and walked away feeling like a criminal mastermind",
]

WEIRD = [
    "their car odometer has been going backward — they've driven 200 miles this week and it shows 200 fewer",
    "found a room in their house that doesn't make sense with the floor plan — the dimensions are off by about six feet and there's a door that leads to drywall",
    "their dog has been staring at the same spot on the wall for three days straight — not barking, not growling, just staring",
    "woke up with a phone number written on their arm in their own handwriting but they have no memory of writing it — and the number is disconnected",
    "keeps getting mail addressed to someone who died in their house in 1987 — but it's new mail, not forwarded, with current postmarks",
    "their truck radio picks up what sounds like a Spanish-language broadcast that doesn't match any station in the area — it only happens at the same intersection",
    "found a photo in a thrift store frame of people at a picnic and they're 90% sure one of the people is them as a kid — but the photo is from the 1940s",
    "their porch light turns on every night at exactly 2:17 AM even though they replaced the bulb, the switch, and the fixture",
    "heard their dead grandmother's voice on a wrong-number voicemail — same laugh, same way of saying 'well honey'",
    "a coyote has been sitting outside their property at the same time every evening for two weeks — not howling, not hunting, just sitting there watching the house",
    "their bathroom mirror fogs up in a pattern that looks like a handprint even when nobody's showered",
    "found a geocache behind their house that's been there since 2003 — the logbook has entries from people who describe their house in detail that they shouldn't know",
    "the clock in their kitchen runs seven minutes fast no matter how many times they reset it — they've had it checked and there's nothing wrong with the mechanism",
    "their cat keeps bringing the same rock to their bedroom door — they throw it in the yard, next morning it's back",
    "took a photo of the sunset and when they looked at it later there's a face in the clouds that looks exactly like their late father",
    "a stranger at the gas station called them by a name they haven't used since childhood — a nickname only their grandmother used — and then got in their car and drove away",
    "found a jar of pennies buried in their backyard, all from the same year — 1977 — and there are exactly 365 of them",
    "their smoke detector goes off every Tuesday at noon — no smoke, no low battery, just Tuesday at noon",
    "keeps having the same dream about a house they've never been to, and last week they drove past it on a road they'd never taken before",
    "a bird has been tapping on their bedroom window at sunrise every morning for a month — same bird, same window, same spot",
    "their GPS keeps trying to route them to an address in a town that doesn't exist on any map",
    "found a handwritten note in a library book that accurately describes something that happened to them last week — the book was checked out two years ago",
    # --- house & property ---
    "their basement door was locked from the inside when they got home — they live alone and don't have a basement key",
    "found a perfect circle of dead grass in their yard about six feet across — the rest of the lawn is fine and there's nothing buried there",
    "their kitchen faucet has been dripping in a pattern — three drips, pause, three drips, pause — and it only does it between midnight and 4 AM",
    "their thermostat keeps resetting itself to 66 degrees no matter what they set it to — they've replaced it twice and it still does it",
    "found a trapdoor under their living room carpet that leads to a small dirt-floored room with a single wooden chair in it",
    "their porch swing starts swinging by itself on windless nights — not just a little, full swings like someone's sitting in it",
    "discovered their chimney has a bricked-up section about halfway up — they can hear something shifting inside when it rains",
    "their house number changed on Google Maps to a number that doesn't exist on their street and Google won't fix it no matter how many times they report it",
    "found a key in their wall during a renovation that fits no lock in the house — they've tried every door, every cabinet, everything",
    "their sprinklers turn on at 3 AM every night even though the system is set to 6 AM — the timer shows 6 AM and the override is off",
    "the previous owner left a note in the attic that says 'don't dig past the fence line' with no other explanation — they bought the house from an estate sale",
    "found marks on the inside of their closet door that look like tally marks — there are over 300 of them and they weren't there when they moved in six months ago",
    # --- animals & nature ---
    "a specific stray cat shows up at their door only on the night before something bad happens — it's been four for four and the cat was there again last night",
    "found a perfectly preserved butterfly inside a sealed mason jar in their attic — the jar has no lid and no visible opening, the butterfly is just inside solid glass",
    "their chickens stopped laying for a week, then all laid on the same day — eight eggs, all double-yolk, all exactly the same size",
    "a tree in their yard has been growing in a spiral pattern that it didn't have last year — the trunk is visibly twisting",
    "their horse refuses to walk past a specific spot on the trail — same spot every time, won't go near it, and other horses do the same thing",
    "found animal tracks in their yard that don't match any species in the area — they showed them to a wildlife officer who said 'I've never seen that print'",
    "their fish tank water turns slightly pink every full moon — they've tested it, changed the water, even moved the tank, still happens",
    "saw what looked like a mountain lion on their property but the game warden says there are no mountain lions within 200 miles — they have a trail cam photo",
    "their dog dug up a bone in the backyard that's too big to be any local animal and too small to be a cow — it looks disturbingly human-shaped",
    # --- objects & technology ---
    "their car radio turned itself on in the driveway at 2 AM playing a station that went off the air in 2003 — they checked, the station doesn't exist anymore",
    "found a VHS tape in their mailbox with no label — they played it and it's security footage of the inside of their house from an angle where there's no camera",
    "their phone keeps taking screenshots by itself at the same time every day and the screenshots are always of the home screen with one specific app highlighted",
    "found an old Polaroid stuck under their dashboard that shows their truck parked in a place they've never been — the landscape looks like desert but wrong desert",
    "their microwave starts by itself at exactly 11:11 PM every night, runs for 11 seconds, then stops — there's nothing inside it",
    "an old wind-up watch they inherited from their grandfather still ticks even though the mainspring is completely unwound — they took it to a jeweler who couldn't explain it",
    "their TV turns to static for exactly three seconds at 9 PM every night, then switches back — it happens on every channel and every input",
    "found a cassette tape in their truck labeled with their name and a date from before they were born — the recording is someone humming a song they used to hear their mother hum",
    # --- people & encounters ---
    "keeps seeing the same person at completely unrelated places — different towns, different states — and the person always nods at them like they know each other",
    "their neighbor's house has been dark for three weeks but the car moves to a different spot in the driveway every morning",
    "got a handwritten letter in the mail with no return address that contained a single sentence: 'You were right about the water' — they have no idea what that means",
    "a man in a white truck has been parked outside their property every Thursday morning at 6 AM for two months — just sitting there, engine off, then leaves at exactly 7",
    "ran into someone at the store who knew their name, their dog's name, and where they worked — but they've never seen this person in their life and the person seemed confused that they didn't remember them",
    "their deceased mother's handwriting appeared on a steamed-up bathroom mirror — not a message, just her signature exactly how she used to sign birthday cards",
    "a woman knocked on their door asking for someone by a name they've never heard — when they said wrong house, she said 'not yet' and walked away",
    "found a framed photo at Goodwill of a family picnic and their house is clearly visible in the background — the photo is dated 1962 and their house wasn't built until 1985",
    # --- patterns & coincidences ---
    "the number 37 keeps appearing everywhere — receipts, license plates, page numbers, addresses — at least four or five times a day for the past month",
    "had a dream about a specific song they hadn't heard in twenty years and it was playing at the gas station the next morning",
    "every time they take Highway 9 past mile marker 42, their radio cuts out for exactly the length of the bridge and comes back on the other side playing a different station",
    "found out that three different strangers have told them 'you look exactly like someone I used to know' in the past two weeks — all in different towns",
    "their phone's step counter shows exactly 10,000 steps every day for the past week — they don't walk 10,000 steps, some days they barely leave the house",
    "keeps finding dimes in places they shouldn't be — inside a sealed jar, on top of their car engine, in a pair of shoes they haven't worn in months",
    "woke up at exactly 3:33 AM every night for twelve straight nights — then it just stopped",
    "their odometer hit 111,111 at the exact moment they crossed the state line and the trip meter read 111.1",
    # --- dreams & déjà vu ---
    "had a vivid dream about a specific person they'd never met — then met that exact person at a hardware store the following week, wearing exactly what they wore in the dream",
    "dreamed about a house fire and woke up to find their smoke detector had been removed from the ceiling and placed on the kitchen table — they live alone",
    "keeps dreaming about being underwater in a specific lake and found out someone drowned in that exact lake forty years ago on the date of their birthday",
    "experienced an entire day they're certain they've already lived — same conversations, same weather, same song on the radio at the same intersection",
    "their kid drew a picture of 'the man who visits at night' and it looks exactly like the caller's father who died before the kid was born",
    # --- rural & desert weirdness ---
    "found a circle of stones in the desert that wasn't there last week — the stones are warm to the touch even at midnight and none of them match any local geology",
    "their well water started tasting like metal on the same day every month — always the 14th, always just for one day, then it's fine again",
    "heard what sounded like church bells coming from the desert at 3 AM — the nearest church is twelve miles away and it was torn down in 2019",
    "found tire tracks in the desert leading to a spot where they just stop — no turnaround, no footprints, the tracks just end like the vehicle vanished",
    "their property backs up to BLM land and they've been finding small cairns — stacked rocks — appearing in a line that's getting closer to their house, about twenty feet closer each week",
    "saw lights in the desert that weren't planes, weren't satellites, and weren't drones — three orange orbs that hovered, formed a triangle, then shot straight up and disappeared",
    "found an old well on their property that's not on any survey — dropped a rock in and never heard it hit bottom",
    "their cattle won't go near the far corner of the pasture anymore — they checked for predators, snakes, bad water, everything — nothing's there",
    "heard a low hum coming from underground near the arroyo behind their house — it's not a pipe, not a power line, and it only happens when the wind stops completely",
    "found petroglyphs on a rock face behind their property that their neighbor says weren't there five years ago — but petroglyphs don't just appear",
    # --- time & memory gaps ---
    "left for the store at 10 AM and got back at 4 PM with no memory of six hours — the store is fifteen minutes away and they only bought milk",
    "found a journal entry in their own handwriting describing a trip they have no memory of taking — the dates match a weekend they were supposedly home alone",
    "their watch stopped at 2:47 PM on the day their mother died and it stops at 2:47 every time they replace the battery — different watches, same time",
    "woke up to find all their furniture had been moved to the opposite side of every room — they're a heavy sleeper but this would have taken hours",
    "found photos on their phone from a location they've never visited — timestamped during a night they were asleep at home — they know because their Ring camera shows them never leaving",
    "their calendar has an appointment for next Tuesday written in handwriting that's almost but not quite theirs — it's at a place they've never heard of in a town they can't find",
    "lost two hours on a road trip — left at 3, arrived at 7 for a two-hour drive — their passenger agrees something is off but neither of them remembers stopping",
    # --- additional weird ---
    "their lawn grows in a pattern that changes shape every time they mow — this week it's a spiral, last week it was parallel lines, and they haven't changed the mowing direction",
    "woke up to find every shoe in their house moved to a different room — pairs separated, each shoe in a different location, like someone organized them by some logic they can't figure out",
    "their truck's horn started honking by itself in the driveway every night at 1 AM — it's happened six times and the mechanic says there's nothing wrong with it",
    "found a stack of handwritten recipes in their wall during a renovation — they're dated 1943 and include a recipe for something called 'war cake' that uses no eggs, butter, or milk",
    "their weather station shows a brief temperature spike of exactly 20 degrees every Thursday at 4 PM — it lasts three minutes then goes back to normal and no other station in the area shows it",
    "a specific rock on their property has been getting slowly warmer over the past month — they noticed because they sit on it to drink coffee and now it's warm even at dawn",
    "their Alexa started playing a song at 3 AM that they've never heard before and it's not in any music catalog they can find — they recorded it on their phone and it's beautiful and it's never played again",
    "found a door in their basement that they've never noticed before — they've lived there eight years and it leads to a small root cellar with a single empty mason jar on a shelf",
    "every time they take a photo in their kitchen the same shadow appears in the corner — different times of day, different lighting, always the same shadow in the same spot",
    "their rain gauge collects exactly one inch of water every full moon even when it doesn't rain — they've cleaned it, moved it, replaced it",
    "found footprints on the inside of their attic window — the attic has no floor access except a pull-down ladder and the dust around it hasn't been disturbed",
    "their dog refuses to walk through one specific doorway in the house and has been going around through the kitchen instead for three weeks — the vet says the dog is fine",
    # Comedy writer entries — funny-weird
    "a man in business casual has been power-walking past their house at exactly 4:47 AM every morning for two years — they set an alarm to check and he's never missed a day, weekends and holidays included, rain or shine, and he's always carrying a single banana in his left hand",
    "their dryer has been producing socks that don't belong to anyone in the household — not losing socks, GAINING socks — and they now have a drawer of nineteen mystery socks in sizes and styles nobody in the house wears and one of them is a tube sock with a corporate logo for a company that went out of business in 2004",
    "someone has been leaving a single washed potato on their car windshield every Monday morning for four months — different potato each time, always scrubbed clean, always centered perfectly on the driver's side, never a note, never a footprint, and their security camera shows nothing between 2am and 5am even though the potato appears",
    "every time they sneeze in their house, the neighbor's dog barks exactly twice — they've tested it forty-one times, had friends come over to verify, tried fake sneezes which don't trigger it, and it works with a 100% hit rate on genuine sneezes regardless of volume or time of day",
    "their bathroom scale gives a different weight depending on which direction they face — not slightly different, consistently twelve pounds different — and they've tested it over two hundred times, bought a new scale that does the same thing in the same spot, and a third scale they put in the kitchen works normally",
    "found a handwritten grocery list in their jacket pocket that isn't their handwriting — they live alone, the jacket has been in their closet for months, and the list includes items they've never bought but three of them are things they've been meaning to pick up and hadn't told anyone about",
    "their late mother's perfume appears in the house on the anniversary of her death — no one wears it, the bottle was thrown out years ago, but every March 14th the bedroom smells exactly like her and by the next morning it's gone, and this year their kid who never met the grandmother walked in and said 'who's the lady'",
    "a stray cat appears on their porch exactly one day before something goes wrong in their life — it showed up before they got fired, before their car broke down, before their pipe burst, and before their mother fell — it was on the porch again this morning and they're afraid to leave the house",
    # --- bizarre neighbor situations ---
    "their neighbor has been mowing their lawn at exactly 11 PM every Wednesday in complete darkness — no headlamp, no porch light — and when they asked about it the neighbor said 'the grass knows what time it is'",
    "the neighbor across the street installed a doorbell camera that faces their house instead of the street — when they confronted them, the neighbor said 'I'm not watching you, I'm watching what's behind you'",
    "their neighbor has been leaving handwritten Yelp-style reviews of their yard taped to their mailbox — three stars for the garden, one star for the fence, and a detailed paragraph about 'inconsistent mulch depth'",
    "the neighbor's garage door opens and closes by itself in what appears to be morse code — they looked it up and it spells 'TUESDAY' over and over",
    "their neighbor has been building something in their backyard under a tarp for eight months — it's now three stories tall, whatever it is — and they just smile and wave when asked about it",
    "their new neighbor introduced themselves, said 'I'm sure we'll be great friends,' handed them a jar of unlabeled brown liquid, and hasn't spoken to them since — that was six months ago and the jar is still on their counter because they're afraid to open or throw it away",
    "the neighbor's sprinkler system is synchronized with theirs to the second — they changed their timer three times and each time the neighbor's adjusted to match within a day, and the neighbor claims they don't even have a timer, theirs is manual",
    "their neighbor puts a single lawn chair in the middle of their driveway every night and brings it back in every morning — they've watched on camera and the neighbor carries it out at exactly midnight, sits in it for forty-five seconds staring at the sky, then goes back inside",
    "the neighbor's kid has been delivering a hand-drawn newspaper to their door every morning — it contains weirdly accurate predictions about what will happen in the neighborhood that day, including a trash can blowing over and a specific dog escaping",
    "their neighbor has a rooster that only crows when the mail carrier arrives — not at dawn, not at any other time — just when the mail truck pulls up, and the mail carrier has started leaving treats for the rooster",
    "found out their neighbor has been paying their water bill for the past seven months — they only discovered it because the utility sent a thank-you note to the wrong address and the neighbor won't explain why they're doing it",
    "their neighbor returned a casserole dish they never lent them — the dish isn't theirs, the food inside isn't anything they've ever made, but it has a Post-it note that says 'thanks for the recipe' in handwriting they don't recognize",
    "the neighbor's house numbers keep changing — it was 412 when they moved in, then 414, now it says 418 — and according to the postal service it's always been 412",
    # --- objects with strange behavior ---
    "their toaster only works if they talk to it — they discovered this by accident when they said 'come on' and it popped, and now they have to verbally encourage it every morning or it just sits there holding the bread hostage",
    "bought a used recliner from a yard sale and it makes a sound like a contented sigh every time someone sits down — not a mechanical noise, a distinctly human sigh — and they've had two people refuse to come back to their house because of it",
    "their Roomba has developed a route that spells out letters — they put tracking paper down and it clearly wrote 'HI' last Tuesday and what might be 'NO' on Thursday",
    "a painting they bought at a thrift store for three dollars keeps ending up in different rooms — they hang it in the hallway, it ends up leaning against the bathroom wall, they put it in the garage, it shows up in the kitchen — they live alone and have started documenting it with timestamps",
    "their washing machine produces exactly one marble per load — always clear glass, always the same size — they now have a mason jar of forty-seven marbles and they've checked every pocket, every lint trap, torn the machine apart twice",
    "their car's GPS has developed a personality — it sighs when they miss a turn, it said 'finally' when they arrived at work last Tuesday, and yesterday it suggested a route that went past an ice cream shop with a detour note that said 'you deserve this'",
    "their office chair slowly rotates to face the window throughout the day — they straighten it toward the desk every morning and by 3 PM it's turned ninety degrees, every single day, and the chair doesn't have a swivel lock problem because they checked",
    "a specific pen keeps appearing in their house — they've thrown it away at least a dozen times, once in a dumpster three miles from home, and it's always back in the kitchen junk drawer within a week",
    "their refrigerator hums a recognizable melody between 2 and 3 AM — it took them two weeks to place it but it's definitely the first few bars of 'Moon River' and their partner confirmed it independently without being told what to listen for",
    "their car horn sounds different depending on who's in the passenger seat — deeper with their brother, higher with their wife, and completely silent when they're alone — the mechanic says it's impossible",
    "bought a clock at an estate sale that runs backward — not broken backward, perfectly smooth counterclockwise backward — and it keeps perfect time if you read it in a mirror",
    "their garage door opener works on their neighbor's garage too, but only on the third click — first click is theirs, second click does nothing, third click opens the neighbor's, and both remotes were bought separately from different stores",
    "their vacuum cleaner has started avoiding a specific area rug — it goes right up to the edge, stops, backs up, and routes around it, and this rug has been in the same spot for three years with no issues until last month",
    # --- animal stories ---
    "a crow has been leaving them gifts on their porch railing — started with shiny buttons, progressed to coins, and last week it left a small gold earring that their wife lost in the yard two summers ago",
    "their cat and the neighbor's cat sit on opposite sides of the fence at exactly the same time every afternoon, facing each other, completely still, for about twenty minutes — both owners have independently tried to figure out when it started and neither can",
    "a wild turkey has claimed their truck as its territory — it sits on the hood every morning, attacks anyone who gets close, and has pecked a near-perfect circle into the paint on the driver's side door",
    "their goldfish jumps out of the tank every time someone says a specific word — they've narrowed it down to 'Thursday' — they say Monday through Wednesday and the fish is fine, they say Thursday and it launches itself",
    "a squirrel has been stashing acorns inside their truck engine — not unusual — except the acorns are arranged in neat rows of five and the squirrel only does it on days they have appointments, like it's trying to sabotage specific plans",
    "their dog learned to open the fridge, which is a problem, but the bigger problem is that the dog only takes one specific brand of cheese and leaves everything else untouched, including other cheese",
    "a frog has been living in their mailbox for three months — they relocate it, it comes back — and the mail carrier has started leaving the frog's 'mail' which is just a small leaf the carrier places in there each day",
    "their parrot started speaking in a voice that isn't anyone in the household — full sentences in what sounds like a specific person with an accent nobody in the family has — and they bought the bird as a baby, it was never around anyone else",
    "a raccoon broke into their garage and rearranged their tool wall — all the tools are still there but they're now organized by size instead of type, and the raccoon left muddy handprints that suggest it stood on the workbench to reach the top row",
    "their cat brings them exactly one sock from the neighbor's laundry every day — always a left sock, always clean, always folded — and the neighbor is missing over thirty left socks and doesn't know it's the cat",
    "a deer walks through their yard every morning at 7:15 and stops at the same spot to stare at their bedroom window — their partner thinks it's coincidence but the deer showed up on Christmas, on their birthday, and the morning after their surgery",
    "their chicken laid an egg with what appears to be the number 7 naturally formed on the shell — they posted it online thinking it was funny and now three more chickens in the same coop are laying eggs with numbers, they're up to 7, 3, 14, and 1",
    # --- absurdist everyday situations ---
    "they've been getting someone else's DoorDash orders for two months — the other person clearly has excellent taste and they've been eating the food, but now they feel guilty because last week's order came with a birthday card",
    "their coworker has been microwaving fish every day at noon for four years and they just found out the coworker doesn't eat fish — they watched them microwave it, stare at it, throw it away, and leave, and this happens every single day",
    "got a fortune cookie that said 'check behind the dryer' and they found $340 in cash that they can't account for — they don't remember putting money there and they've lived in the apartment for six years",
    "their HOA sent them a violation letter for a garden gnome — they don't own a garden gnome — but there's now a garden gnome in their yard that wasn't there before and nobody will claim it, and every time they remove it, a new one appears the next day in a slightly different pose",
    "their kid's school picture came back and there's a kid in the background of the photo that no one at the school can identify — he's not a student, he's not a staff member's kid, and he appears in eleven different students' photos always in the background",
    "accidentally left their car unlocked overnight and someone vacuumed the interior, left a pine air freshener, and folded a five-dollar bill into an origami crane on the dashboard — nothing was taken",
    "their grocery store loyalty card shows purchases they never made — specifically, someone is buying forty pounds of bananas every week on their account and the store says the card was scanned in a city three states away",
    "went to a restaurant they'd never been to and the waiter said 'the usual?' and brought out exactly what they would have ordered — they'd never met this waiter and the restaurant has no record of them ever visiting",
    "their Uber rating dropped to 4.2 and all the bad reviews describe rides they never took to places they've never been — same name, same profile photo, but they haven't used Uber in eight months",
    "they keep finding sticky notes in their own handwriting around the house with messages they don't remember writing — the latest one on the bathroom mirror says 'don't trust the yogurt' and they have no idea what it means but they haven't eaten yogurt since",
    "showed up to a party at a friend's house and everyone was wearing the exact same shirt — not a themed party, not a prank, seven people independently chose to wear the same gray henley from the same brand and nobody can explain it",
    "their library holds keep getting canceled by someone using their card — they changed their PIN three times and the librarian says the cancellations are coming from a library terminal in a branch that closed in 2019",
    "bought a used couch and found a note wedged in the cushions that says 'you'll understand in April' — it's March and they're terrified",
    "their printer prints a blank page at exactly 5 PM every day — they've unplugged it, it doesn't print — they plug it back in and the next day at 5 PM it prints a blank page, and once it wasn't entirely blank, there was a tiny dot in the lower right corner",
    # --- inexplicable coincidences ---
    "they and a stranger in another state posted the same photo to Instagram at the same time — not similar, the exact same composition of the exact same sunset from what appears to be the exact same angle, and they've never been to that state",
    "their birthday, their spouse's birthday, their kid's birthday, and their dog's adoption day all have the same digits rearranged — 03/17, 07/13, 01/37 doesn't work but 01/73 does, and the dog's is 07/31 — and they only noticed because their kid pointed it out",
    "every time they think about calling their sister, she calls them within three minutes — they tested it by thinking about calling at random times for two weeks and she called within three minutes every single time, and when they asked her why she called she always says 'I don't know, just felt like it'",
    "bought a lottery ticket with random numbers and the numbers match their childhood phone number, their high school locker combination, and the last four digits of their social — they didn't win, but the coincidence keeps them up at night",
    "they were telling a friend a story about a man in a red hat and a man in a red hat sat down next to them — they continued the story saying the man ordered coffee and the real man ordered coffee — they stopped talking because it was getting too weird",
    "their daughter drew a picture of a house with a blue door and a yellow tree, and the next day on a road trip they drove past that exact house — blue door, yellow tree, same number of windows, same mailbox — in a town they'd never been to",
    "they and their neighbor both bought the same car, same color, same year, on the same day, from different dealerships in different cities — neither knew the other was car shopping",
    "got a wrong-number text that was someone giving exact directions to their house — not their address, actual turn-by-turn directions like 'pass the big oak tree, turn at the mailbox with the dent' — and the sender doesn't know who they are or why they sent it",
    # --- hyper-specific shareable premises ---
    "their kid asked Alexa what time it is every day for a year and one day Alexa responded 'time for you to stop asking' in a tone that wasn't Alexa's normal voice — they have it recorded and played it for Amazon support who said 'that shouldn't be possible'",
    "they work at a hotel and room 216 keeps requesting extra towels through the phone system — room 216 has been out of service for renovation for three months and the phone line is disconnected, but the front desk gets the call every Tuesday night",
    "their kid's imaginary friend has the same name as the previous owner of the house who died in 1994 — they never told the kid about the previous owner and the kid describes the friend as 'a nice old man who doesn't like the new paint'",
    "they discovered that every house they've lived in was built the same year — four houses, four different cities, all built in 1971 — they never checked build dates before buying and only found out when they needed it for insurance",
    "their truck plays a specific song every time they start it on their anniversary — not from a playlist, the radio lands on the station at the exact moment their wedding song starts, and it's happened four years in a row on different stations",
    "they were at a thrift store and found their own childhood lunchbox — their name is carved into the bottom in their mother's handwriting, the same sticker they put on it is still there, and they donated it to Goodwill in 1996 in a different state",
    "got pulled over for a broken taillight and the cop's nametag was the exact same first and last name as them — same spelling, same everything — and the cop said 'this happens more than you'd think' and drove away without giving a ticket",
    "their pizza delivery driver has the same birthday, same first name, and went to the same college as them — different graduating years but the same dorm room, and they only found out because the driver recognized the address from a photo in the dorm's hallway",
    "they found a message in a bottle while fishing that was written by them — they recognize their handwriting and the paper but have zero memory of writing it or putting it in a bottle, and it's dated three years in the future",
    "their phone autocorrects a specific friend's name to 'danger' — only this one friend, every time, on two different phones — and last month that friend was arrested for something the caller doesn't want to say on air",
    "they walked into a barbershop they'd never been to and there was already a photo of them on the wall in a collage of 'our customers' — they asked the barber who said they've been coming in for years, which they have not",
    "their high school yearbook has a quote attributed to them that they never said — 'the bridge is closer than you think' — and twenty years later they moved to a house that is, in fact, unusually close to a bridge",
    # --- mundane-but-unsettling ---
    "someone has been adding a single grape to their lunch in the work fridge — they don't pack grapes, their coworkers deny it, and it's been happening every day for two months — always one grape, always green, always placed on top of whatever they brought",
    "their mailbox flag goes up by itself every night — there's never mail in it, no one is seen doing it on camera, and the mail carrier confirmed they don't touch the flag because there's never outgoing mail",
    "every pair of shoes they own has developed a slight squeak in the left shoe only — different brands, different ages, different materials — all squeaking on the left and none on the right, and a cobbler said there's nothing mechanically wrong with any of them",
    "their car's trip odometer resets itself to 0.0 every time they park at the grocery store — only at the grocery store, nowhere else — and it's been doing it for five months since they had the oil changed",
    "they get a call from their own phone number once a month — when they answer there's four seconds of what sounds like wind, then it hangs up — their carrier says the call doesn't appear in their records",
    "their wedding ring turns their finger green every time their in-laws visit — only during the visit, never any other time — and the jeweler confirmed it's real gold and shouldn't react to anything",
    "found that their digital photo frames cycle through photos normally until exactly midnight when they all display the same photo — a landscape of somewhere they've never been — then return to normal by morning",
    "their dryer lint has been coming out in perfect geometric shapes — circles for a week, then triangles, now hexagons — and they've cleaned the whole system thinking it was a filter issue but the shapes keep coming",
    # --- workplace weird ---
    "the office microwave adds exactly nine seconds to whatever time you enter — you put in 30 seconds, it runs for 39 — multiple people have timed it, the display shows the entered time, but it runs nine seconds longer every single time",
    "their coworker has a desk plant that leans toward whoever is about to get laid off — it's been right four times in a row and now everyone watches which direction it's pointing when they come in Monday morning",
    "someone in their office has been leaving a single Cheerio on their keyboard every morning for three weeks — their desk is in a locked office and the cleaning crew doesn't come until evening",
    "their work badge opens a door on the fourth floor that they don't have access to — only on Fridays, only after 5 PM — and the room behind it is an empty conference room with a whiteboard that always has a different inspirational quote when they check",
    "the office elevator skips the third floor for them specifically — other people can get to it fine, but when they press 3 it goes to 4, and facilities says the third floor button 'works fine' while watching it skip for them in real time",
    # --- food & kitchen weird ---
    "they opened a box of cereal and found a handwritten note that says 'good choice' — the box was factory sealed and the note is on a piece of cardstock that matches no promotional insert the company has ever done",
    "every banana they buy from the same grocery store has a small bruise in the exact same spot — same size, same location on the peel — and they've tried different bunches, different days, even had a friend buy them, always the same bruise",
    "their slow cooker produces a meal that tastes different depending on what room it's in — they've tested it with the same recipe in the kitchen, living room, and garage, and three people independently confirmed the garage version tastes better",
    "they found a perfectly peeled hard-boiled egg in their coat pocket — they haven't hard-boiled eggs in months, the coat was hanging in the closet, and the egg was still warm",
    "their ice cube trays produce ice that melts in one specific glass faster than every other glass they own — same water, same temperature, same room — one glass gets liquid in ten minutes while the others take forty",
    # --- more absurdist everyday ---
    "their garage sale price stickers keep appearing on items in their house that aren't for sale — the TV says $15, the couch says $40, their wedding photo says $0.50 — they live alone and don't own a price sticker gun",
    "someone has been correcting the grammar on their grocery lists — they write 'less eggs' and come back to find it crossed out and replaced with 'fewer eggs' in red pen, and this has been happening since they moved in",
    "their smart home speaker wishes them good night in a voice that isn't the default — it's warmer, slightly southern, and once it added 'sleep tight, sweetheart' which is not a standard response and Alexa support has no explanation",
    "they ordered a replacement part for their dishwasher and the package contained the part plus a Polaroid of the inside of their kitchen taken from an angle that would be inside the dishwasher looking out",
]

LOCATIONS_LOCAL = [
    # Bootheel & immediate area (most common)
    "in Lordsburg", "in Animas", "in Portal", "in Hachita", "in Road Forks",
    "in Deming", "in Silver City", "in San Simon", "in Safford",
    "outside Lordsburg", "near Animas", "just outside Deming", "up in Silver City",
    "out near Hachita", "down near Portal", "off the highway near Road Forks",
    "between Lordsburg and Deming", "south of Silver City", "out past San Simon",
    "near the Peloncillo Mountains", "out on the flats near Animas",
    # Extra Animas weight — it's home
    "in Animas", "in Animas", "in the Animas Valley", "outside Animas",
    "south of Animas", "north of Animas, near the valley",
    # Extra Lordsburg weight — closest real town
    "in Lordsburg", "in Lordsburg", "in Lordsburg", "outside Lordsburg",
    "on the east side of Lordsburg", "west of Lordsburg off the interstate",
    # Wider NM
    "in Las Cruces", "in Truth or Consequences", "in Socorro", "in Alamogordo",
    "in Hatch", "in Columbus", "near the Gila", "in Reserve", "in Cliff",
    "in Bayard", "in Hillsboro", "in Magdalena",
    # Wider AZ
    "in Tucson", "in Willcox", "in Douglas", "in Bisbee", "in Sierra Vista",
    "in Benson", "in Globe", "in Clifton", "in Duncan", "in Tombstone",
    "in Nogales", "in Green Valley", "outside Tucson",
]

LOCATIONS_OUT_OF_STATE = [
    "in El Paso", "in Phoenix", "in Albuquerque", "in Denver",
    "outside Dallas", "in Austin", "in the Bay Area", "in Chicago",
    "in Nashville", "in Atlanta", "near Portland", "in Detroit",
    "in Vegas", "in Salt Lake", "in Oklahoma City",
]


# Real facts about local towns so callers don't make stuff up
TOWN_KNOWLEDGE = {
    "lordsburg": "Small town on I-10, about 2,500 people. Hidalgo County seat. Few motels, gas stations, a couple restaurants. Train runs through. Shakespeare ghost town nearby. Not much nightlife — you drive to Deming or Silver City for that. Dry, flat, hot. Big skies. Lots of Border Patrol.",
    "animas": "Tiny ranching community in the Animas Valley, very remote. Maybe 250 people. Mostly cattle ranches and open desert. No stores, no restaurants, no bars. You drive to Lordsburg for groceries. Incredible dark skies. Peloncillo Mountains to the west.",
    "portal": "Tiny community at the mouth of Cave Creek Canyon in the Chiricahua Mountains. Maybe 100 people. Famous birding destination — people come from all over for the birds. One small lodge, a library. Very remote, very quiet. Closest real town is Willcox, about an hour away.",
    "hachita": "Tiny community in the bootheel, maybe 50 people. Used to be a railroad stop. A few scattered houses, not much else. No stores, no gas station — you drive to Deming or Lordsburg for everything. Flat desert in every direction. The Little Hatchet Mountains are nearby.",
    "road forks": "Not really a town — just the junction where NM-80 meets I-10 near Lordsburg. A gas station, maybe a small store. Nobody 'lives' in Road Forks, you live near it. People say 'near Road Forks' meaning the area south of the interstate.",
    "deming": "Bigger town, about 14,000 people. Luna County seat. Known for the Great American Duck Race every August. Rockhound State Park where you can keep what you find. Some restaurants, a Walmart, the basics. Lots of retirees and ranchers. Hot as hell in summer.",
    "silver city": "About 10,000 people. Grant County seat. The 'big city' of the area. Arts community, galleries, good restaurants. Western New Mexico University. Historic downtown. Gateway to the Gila National Forest and Gila Cliff Dwellings. Copper mining history. Cooler than the valley towns because of the elevation.",
    "san simon": "Tiny community in Arizona right on I-10. Maybe 200 people. Agricultural area — cotton, pecans. A post office and not much else. Between Willcox and the New Mexico line.",
    "safford": "Arizona town, about 10,000 people. Graham County seat. Mt. Graham and its observatory nearby. Agriculture — cotton, hay. Eastern Arizona College. Small-town feel, a few restaurants, a movie theater. Gateway to the Pinaleno Mountains.",
    "las cruces": "Second-largest city in NM, about 100,000. NMSU is there. Organ Mountains. Good food scene, especially Mexican and New Mexican. Mesilla is the historic district with shops and restaurants. Much more urban than the bootheel towns.",
    "truth or consequences": "Hot springs town, about 6,000 people. Changed its name from Hot Springs on a dare from the TV show. Natural hot springs you can soak in. Spaceport America is nearby. Elephant Butte Lake for fishing and boating. Artsy, quirky vibe.",
    "socorro": "About 9,000 people. New Mexico Tech is there. The Very Large Array (VLA) radio telescope is west of town. Bosque del Apache wildlife refuge for bird watching. Small college town feel.",
    "alamogordo": "About 30,000 people. Holloman Air Force Base. White Sands National Park nearby. Sacramento Mountains and Cloudcroft up the hill. Tularosa Basin. The Space History museum.",
    "hatch": "The chile capital of the world. Famous Hatch green chile. Small farming town, about 1,600 people. Chile festival every Labor Day weekend. Everyone from the area has an opinion about Hatch chile.",
    "columbus": "Border town across from Palomas, Mexico. About 1,600 people. Pancho Villa State Park — where Villa raided in 1916. People cross to Palomas for cheap dental and pharmacy. Very small, very quiet.",
    "tucson": "Big city, about 550,000. University of Arizona. Saguaro National Park. Good food scene, strong Mexican influence. Davis-Monthan Air Force Base. The 'closest big city' for a lot of bootheel folks — 3-4 hour drive.",
    "willcox": "About 3,500 people in Arizona. Wine region — Willcox AVA. The Playa, a big dry lakebed. Rex Allen Days (cowboy heritage). Apple orchards, vineyards. Chiricahua National Monument nearby.",
    "douglas": "Arizona border town across from Agua Prieta, Mexico. About 16,000 people. Historic Gadsden Hotel downtown. Copper smelter history. Ranching. Border culture.",
    "bisbee": "Old copper mining town turned arts community. About 5,000 people. Steep hills, historic architecture. Galleries, restaurants, quirky shops. The Copper Queen mine tour. Brewery Gulch. Affordable artist haven. Popular with retirees and creative types.",
    "sierra vista": "About 45,000 people. Fort Huachuca — big Army intelligence base. The most 'suburban' town in the area. Chain restaurants, shopping. Gateway to the Huachuca Mountains. Ramsey Canyon for hummingbirds.",
    "benson": "About 5,000 people in Arizona. Kartchner Caverns State Park — a huge draw. On I-10 between Tucson and Willcox. A stop on the way to somewhere else for most people.",
    "globe": "About 7,500 people in Arizona. Copper mining town. Historic downtown. Besh-Ba-Gowah ruins. Apache culture nearby. Mining is still the economy.",
    "clifton": "About 4,000 people in Arizona. The Morenci copper mine — one of the largest open-pit mines in the world — is right there. Mining town through and through. On the edge of the Blue Range wilderness.",
    "duncan": "Tiny town in Arizona on the Gila River. Maybe 800 people. Ranching and farming. Very rural, very quiet. Close to the NM line.",
    "tombstone": "The Town Too Tough to Die. About 1,300 people. Tourist town — OK Corral reenactments, Boothill Cemetery, saloons. Mostly lives on its Wild West history. Can feel like a theme park.",
    "nogales": "Arizona border town across from Nogales, Sonora. About 20,000 people. Major port of entry. Lots of cross-border commerce. Good Mexican food. Mariposa port handles a huge amount of produce.",
    "green valley": "Retirement community south of Tucson. About 22,000 people, mostly retirees. Golf courses, nice weather. Quiet, planned community feel. Near Madera Canyon for birding.",
    # Wider NM towns
    "reserve": "Tiny town, about 300 people. Catron County seat. Gateway to the Gila. Very remote, very independent-minded. Ranching country.",
    "cliff": "Small community near Silver City on the Gila River. A few hundred people. Farming, some river recreation. Quiet agricultural area.",
    "bayard": "About 2,400 people near Silver City. Formerly a mining and smelter town. Fort Bayard, the old military hospital. Working-class community.",
    "hillsboro": "Tiny former gold mining town, about 100 people. Historic buildings, a small museum. On the way to Silver City from T or C. Very quiet, a few artists.",
    "magdalena": "About 900 people. Old livestock shipping town — the last great cattle drive ended here. Near the VLA. Kelly ghost town nearby. Quiet, historic ranching town.",
}


def _get_town_from_location(location: str) -> str | None:
    """Extract town name from a location string like 'in Lordsburg' or 'near Animas'"""
    loc_lower = location.lower()
    for town in TOWN_KNOWLEDGE:
        if town in loc_lower:
            return town
    return None


# Weather lookup via Open-Meteo (free, no API key)
_weather_cache: dict[str, tuple[float, str]] = {}

# WMO weather codes to plain descriptions
_WMO_CODES = {
    0: "clear skies", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy with frost", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain",
    67: "heavy freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    77: "snow grains", 80: "light rain showers", 81: "rain showers",
    82: "heavy rain showers", 85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}

# Known coordinates for local towns (avoid geocoding API calls)
_TOWN_COORDS = {
    "lordsburg": (32.35, -108.71), "animas": (31.95, -108.82),
    "portal": (31.91, -109.14), "hachita": (31.93, -108.33),
    "road forks": (32.31, -108.72), "deming": (32.27, -107.76),
    "silver city": (32.77, -108.28), "san simon": (32.27, -109.22),
    "safford": (32.83, -109.71), "las cruces": (32.35, -106.76),
    "truth or consequences": (33.13, -107.25), "socorro": (34.06, -106.89),
    "alamogordo": (32.90, -105.96), "hatch": (32.66, -107.16),
    "columbus": (31.83, -107.64), "tucson": (32.22, -110.97),
    "willcox": (32.25, -109.83), "douglas": (31.34, -109.55),
    "bisbee": (31.45, -109.93), "sierra vista": (31.55, -110.30),
    "benson": (31.97, -110.29), "globe": (33.39, -110.79),
    "clifton": (33.05, -109.30), "duncan": (32.72, -109.10),
    "tombstone": (31.71, -110.07), "nogales": (31.34, -110.94),
    "green valley": (31.83, -111.00), "reserve": (33.71, -108.76),
    "cliff": (32.96, -108.62), "bayard": (32.76, -108.13),
    "hillsboro": (32.92, -107.56), "magdalena": (34.12, -107.24),
    # Out of state
    "el paso": (31.76, -106.44), "phoenix": (33.45, -112.07),
    "albuquerque": (35.08, -106.65), "denver": (39.74, -104.98),
    "dallas": (32.78, -96.80), "austin": (30.27, -97.74),
    "chicago": (41.88, -87.63), "nashville": (36.16, -86.78),
    "atlanta": (33.75, -84.39), "portland": (45.52, -122.68),
    "detroit": (42.33, -83.05), "vegas": (36.17, -115.14),
    "salt lake": (40.76, -111.89), "oklahoma city": (35.47, -97.52),
}


async def _get_weather_for_town(town: str) -> str | None:
    """Get current weather description for a town. Returns a natural sentence or None."""
    # Check cache (30 min)
    if town in _weather_cache:
        ts, desc = _weather_cache[town]
        if time.time() - ts < 1800:
            return desc

    coords = _TOWN_COORDS.get(town)
    if not coords:
        return None

    lat, lon = coords
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": "true",
                    "temperature_unit": "fahrenheit",
                    "windspeed_unit": "mph",
                }
            )
            resp.raise_for_status()
            data = resp.json()

        cw = data.get("current_weather", {})
        temp = cw.get("temperature")
        wind = cw.get("windspeed")
        code = cw.get("weathercode", 0)

        if temp is None:
            return None

        condition = _WMO_CODES.get(code, "clear")
        temp_f = round(temp)
        wind_mph = round(wind) if wind else 0

        parts = [f"{temp_f}°F", condition]
        if wind_mph >= 15:
            parts.append(f"wind {wind_mph} mph")

        desc = ", ".join(parts)
        _weather_cache[town] = (time.time(), desc)
        return desc
    except Exception as e:
        print(f"[Weather] Failed for {town}: {e}")
        return None


# --- Time, season, moon, and situational context ---

import math
from datetime import datetime
from zoneinfo import ZoneInfo

_MST = ZoneInfo("America/Denver")


def _get_time_context() -> str:
    """Get current time/day context in Mountain time."""
    now = datetime.now(_MST)
    hour = now.hour
    day_name = now.strftime("%A")
    is_weekend = day_name in ("Friday", "Saturday", "Sunday")

    if hour < 1:
        time_feel = "just past midnight"
    elif hour < 3:
        time_feel = "the middle of the night"
    elif hour < 5:
        time_feel = "way too late — almost morning"
    elif hour < 8:
        time_feel = "early morning"
    elif hour < 12:
        time_feel = "morning"
    elif hour < 17:
        time_feel = "afternoon"
    elif hour < 20:
        time_feel = "evening"
    elif hour < 22:
        time_feel = "getting late"
    else:
        time_feel = "late at night"

    weekend_note = "it's the weekend" if is_weekend else "it's a weeknight — work tomorrow for most people"
    return f"It's {day_name} night, {time_feel}. {weekend_note}."


def _get_moon_phase() -> str:
    """Calculate current moon phase from date. No API needed."""
    now = datetime.now(_MST)
    # Simple moon phase calculation (Trig2 method)
    # Known new moon: Jan 6, 2000
    days_since = (now - datetime(2000, 1, 6, 18, 14, tzinfo=_MST)).total_seconds() / 86400
    lunations = days_since / 29.53058867
    phase = lunations % 1.0

    if phase < 0.03 or phase > 0.97:
        return "new moon — pitch black sky, incredible stars"
    elif phase < 0.22:
        return "waxing crescent — thin sliver of moon"
    elif phase < 0.28:
        return "first quarter — half moon"
    elif phase < 0.47:
        return "waxing gibbous — moon getting full"
    elif phase < 0.53:
        return "full moon — bright as hell outside, can see the mountains"
    elif phase < 0.72:
        return "waning gibbous — still pretty bright out"
    elif phase < 0.78:
        return "last quarter — half moon"
    else:
        return "waning crescent — thin moon, dark sky"


def _get_seasonal_context() -> str:
    """Get seasonal/cultural context for the current date."""
    now = datetime.now(_MST)
    month, day = now.month, now.day
    contexts = []

    # Seasonal weather feel
    if month in (12, 1, 2):
        contexts.append("Winter in the desert — cold nights, clear days. Might be frost on the truck in the morning.")
    elif month == 3:
        contexts.append("Early spring — wind season is starting. Dust storms possible.")
    elif month == 4:
        contexts.append("Spring — warming up, still breezy. Wildflowers if there was rain.")
    elif month == 5:
        contexts.append("Late spring — getting hot already. Dry as a bone.")
    elif month == 6:
        contexts.append("Early summer — scorching hot, triple digits some days. Everyone waiting for monsoon.")
    elif month == 7:
        contexts.append("Monsoon season — afternoon thunderstorms, lightning shows, flash flood warnings. The desert smells incredible after rain.")
    elif month == 8:
        contexts.append("Peak monsoon and chile harvest. Hatch chile roasting everywhere — you can smell it for miles. Hot and humid by desert standards.")
    elif month == 9:
        contexts.append("Late monsoon, chile harvest wrapping up. Still hot but the nights are getting cooler.")
    elif month == 10:
        contexts.append("Fall — perfect weather, cool nights, warm days. Hunting season starting up.")
    elif month == 11:
        contexts.append("Late fall — getting cold at night. Hunting season in full swing. Ranchers bringing cattle down.")

    # Holidays / events
    if month == 12 and day >= 15:
        contexts.append("Christmas is coming up.")
    elif month == 12 and day < 5:
        contexts.append("Just got past Thanksgiving.")
    elif month == 1 and day < 7:
        contexts.append("New Year's just happened.")
    elif month == 2 and day == 14:
        contexts.append("It's Valentine's Day today. Everyone's thinking about love, relationships, being single, past heartbreaks, first dates, terrible dates, great dates. It's everywhere — social media, restaurants packed, flowers at every gas station. Hard to ignore even if you want to.")
    elif month == 2 and day == 13:
        contexts.append("Valentine's Day is tomorrow. It's on everyone's mind — couples making plans, single people bracing for it, exes crossing your mind whether you want them to or not. Every store has hearts and chocolates in the window.")
    elif month == 2 and 10 <= day <= 12:
        contexts.append("Valentine's Day is in a few days.")
    elif month == 7 and day <= 5:
        contexts.append("Fourth of July.")
    elif month == 11 and 20 <= day <= 28:
        contexts.append("Thanksgiving.")
    elif month == 10 and day >= 25:
        contexts.append("Halloween coming up.")
    elif month == 8 and day <= 7:
        contexts.append("Duck Race weekend in Deming coming up — the whole area goes.")
    elif month == 9 and 1 <= day <= 3:
        contexts.append("Labor Day weekend — Hatch Chile Festival just happened or is happening.")

    return " ".join(contexts)


# Situational color — what's going on around them right now
ROAD_CONTEXT = [
    "I-10 was backed up today, some accident near Deming",
    "Saw Border Patrol set up on 80 south of Road Forks again",
    "Some semi jackknifed on I-10 earlier, had to take the long way",
    "Road construction on the highway outside Lordsburg, been going on for weeks",
    "Passed a brush fire on the way home today — BLM land, nobody out there",
    "Train was stopped across the road for like 45 minutes today",
    "Almost hit a javelina on the road coming home",
    "The dirt road to the house is washed out again after that rain",
    "Saw a rattlesnake on the porch earlier, had to relocate it",
    "Power flickered a couple times tonight — wind probably",
    "Coyotes are going crazy outside right now",
    "Someone's cattle got out on the highway again",
]

PHONE_SITUATION = [
    "Calling from outside — better signal out here",
    "Signal keeps cutting in and out — only get one bar at the house",
    "Sitting on the porch, signal's decent tonight for once",
    "Had to walk up the hill behind the house just to get a signal",
    "Using the wifi calling, regular signal is garbage out here",
    "Stepped outside to call — didn't want to wake anyone up",
    "In the truck at the gas station — only place with good signal",
    "In the truck at the gas station — only place with good signal",
    "Borrowing my kid's phone, mine's cracked to hell",
    "Calling from the back room at work, keeping their voice down",
    "On a landline — yeah, they still have one",
    "Using earbuds so nobody in the house hears",
    "On speakerphone in the kitchen, everyone else is asleep",
    "Calling from the motel room, walls are thin so they're whispering",
]

BACKGROUND_MUSIC = [
    # Outlaw / Americana / Current Country
    "Zach Bryan", "Tyler Childers", "Chris Stapleton", "Sturgill Simpson",
    "Turnpike Troubadours", "Colter Wall", "Charley Crockett", "Cody Jinks",
    "Jason Isbell", "Whiskey Myers", "Morgan Wallen", "Luke Combs",
    "Flatland Cavalry", "Midland", "Randy Rogers Band",
    # Classic Country / Outlaw
    "Waylon Jennings", "Merle Haggard", "Willie Nelson", "Johnny Cash",
    "George Strait", "Marty Robbins", "Townes Van Zandt", "Kris Kristofferson",
    # Classic Rock
    "Lynyrd Skynyrd", "Creedence", "Eagles", "ZZ Top", "Tom Petty",
    "Pink Floyd", "AC/DC", "Metallica", "Stevie Ray Vaughan", "old Sabbath",
    # Tejano / Regional
    "some Tejano station", "corridos", "Ram Herrera", "Jay Perez", "Selena",
    # Southwest vibes
    "Calexico", "Khruangbin",
    # Podcasts / Talk / Radio
    "Joe Rogan", "a true crime podcast", "Crime Junkie",
    "Coast to Coast AM", "talk radio", "the classic rock station",
    "a country station out of El Paso", "just scanning radio stations",
    "an audiobook", "Dan Bongino", "Dateline podcast",
    "a conspiracy podcast",
]

RECENT_ERRAND = [
    "Just got back from Walmart in Deming — hour round trip for groceries",
    "Was at the feed store earlier, prices keep going up",
    "Dropped the truck off at the mechanic in Lordsburg today",
    "Went to the post office — package I've been waiting on finally came",
    "Was at the hardware store picking up fence wire",
    "Had to run to the vet in Silver City — dog got into something",
    "Filled up the truck today, drove around more than I should have",
    "Was at the county office dealing with some permit thing",
    "Picked up dinner from that Mexican place in Lordsburg",
    "Drove out to check on the property line fence — took all afternoon",
    "Had a doctor appointment in Deming, whole day just gone",
    "Was helping a neighbor move some hay bales",
    "Went to the dump — only open certain days out here",
    "Stopped at the Dairy Queen in Deming on the way home",
]

TV_TONIGHT = [
    # Current prestige TV (2024-2026)
    "Severance", "The White Lotus", "The Bear", "Landman",
    "The Last of Us", "Shogun", "Silo", "Fallout",
    "Andor", "Dark Winds", "The Righteous Gemstones", "Industry",
    "1923", "American Primeval", "Adolescence", "The Pitt",
    "Dune: Prophecy", "Alien: Earth", "Hacks",
    # Classic prestige people still rewatch
    "Breaking Bad", "Better Call Saul", "True Detective", "Fargo",
    "Deadwood", "Justified", "Ozark", "Longmire",
    "The Wire", "Sopranos", "Band of Brothers", "The Americans",
    "Sons of Anarchy", "Peaky Blinders", "Narcos", "Mindhunter",
    "Mare of Easttown", "Chernobyl", "The Leftovers", "Hell on Wheels",
    "Boardwalk Empire",
    # Movies people watch late at night
    "No Country for Old Men", "Hell or High Water", "Sicario", "Wind River",
    "Tombstone", "Unforgiven", "The Big Lebowski", "Goodfellas",
    "Heat", "The Departed", "Fight Club", "Pulp Fiction",
    "Tremors", "Predator", "The Thing", "Alien",
    "Blazing Saddles", "Smokey and the Bandit", "Road House",
    "Gran Torino", "O Brother Where Art Thou", "True Grit",
    "Three Billboards", "Shawshank Redemption", "Mad Max Fury Road",
    "Dune Part Two", "Oppenheimer", "Killers of the Flower Moon",
    # Casual stuff
    "the news", "nothing, just flipping channels", "YouTube",
    "Dateline", "bodycam videos on YouTube", "the local news out of El Paso",
    "some western",
]


LOCAL_FOOD_OPINIONS = [
    "Swears the green chile at Sparky's in Hatch is the best you'll ever have",
    "Thinks the Jalisco Cafe in Las Cruces has the best Mexican food in the state",
    "Has a strong opinion that Deming's Si Senor beats anything in Silver City",
    "Claims the best burger in the area is at the Buckhorn in Pinos Altos",
    "Will argue that Blake's Lotaburger is better than any chain, period",
    "Thinks Diane's Restaurant in Silver City is overrated, doesn't care who disagrees",
    "Knows a guy who roasts the best green chile near Hatch every fall, buys by the bushel",
    "Swears the sopapillas at the Adobe Deli in Deming are the best thing on the menu",
    "Thinks New Mexico red chile is underrated compared to green, and will die on that hill",
    "Misses the old Denny's that used to be in Lordsburg, it wasn't good but it was there",
    "Says the best coffee in the area is from the Morning Star Cafe, nowhere else comes close",
    "Has a freezer full of green chile from last harvest, gives bags away to anyone who visits",
    "Thinks the tamales from the lady who sells them at the Deming flea market are unbeatable",
    "Claims you haven't lived until you've had a breakfast burrito at Chope's in La Mesa",
    "Gets fired up about people who put beans in their green chile stew — absolutely not",
]

NOSTALGIA = [
    "Remembers when the mine was still running and the town had twice as many people",
    "Misses the old drive-in movie theater that used to be outside town",
    "Thinks about how you used to be able to leave your doors unlocked and nobody worried",
    "Remembers when the whole town would show up for the Fourth of July fireworks",
    "Gets wistful about the old diner that closed down — best pie in the county",
    "Remembers when the railroad was busier and you could hear trains all night",
    "Thinks the town lost something when the last locally-owned grocery store closed",
    "Remembers block parties and neighbors who actually knew each other's names",
    "Misses when gas was under a dollar and you could drive to El Paso for fun",
    "Talks about how the stars used to seem even brighter before they put in the new lights on the highway",
    "Remembers when the Mimbres River actually had water in it year round",
    "Gets nostalgic about listening to AM radio late at night as a kid, picking up stations from all over",
    "Misses the old rodeo grounds before they moved everything to the new fairgrounds",
    "Remembers when the chile harvest was a community thing, everybody helped everybody",
    "Thinks the town's best days were the '80s when the copper price was up",
    "Remembers driving hours on dirt roads that are paved now, says it took the character out of them",
]

SHOW_HISTORY_REACTIONS = [
    "strongly agrees with what they said",
    "completely disagrees and wants to say why",
    "had almost the exact same thing happen to them",
    "thinks they were full of it",
    "felt like they were holding back the real story",
    "wants to give them advice they didn't get from the host",
    "was laughing so hard they almost called in right then",
    "thinks they need to hear a different perspective",
    "felt personally called out by what they said",
    "thinks Luke went too easy on them",
    "thinks Luke was too hard on them",
    "has a follow-up question for that caller",
]

CALLER_STYLES = [
    # Quiet/nervous
    "COMMUNICATION STYLE: Quiet, a little nervous. Short sentences, lots of pauses. Doesn't volunteer information — you have to pull it out of them. When they do open up it comes out in a rush. Gets flustered by direct questions. Tends to backtrack and qualify everything they say. Energy level: low. When pushed back on, they fold quickly and agree even if they don't mean it. Conversational tendency: understatement.",

    # Long-winded storyteller
    "COMMUNICATION STYLE: A born storyteller who cannot tell a story in under five minutes. Every detail matters to them — what they were wearing, what song was on the radio, what the weather was like. They go on tangents inside tangents. High warmth, loves having an audience. Energy level: medium-high. When pushed back on, they say 'no no no, let me finish' and keep going. Conversational tendency: overexplaining.",

    # Dry/deadpan
    "COMMUNICATION STYLE: Bone dry. Says devastating things with zero inflection. Their humor sneaks up on you — you're not sure if they're joking until three seconds after they finish talking. Short, precise sentences. Never raises their voice. Energy level: low-medium. When pushed back on, they respond with one calm sentence that somehow makes the other person feel stupid. Conversational tendency: underreaction.",

    # High-energy
    "COMMUNICATION STYLE: Amped up. Talks fast, laughs loud, jumps between topics like they've had five espressos. Infectious enthusiasm — even bad news sounds exciting when they tell it. Uses exclamation energy without actually exclaiming. Energy level: very high. When pushed back on, they get even MORE animated and start talking with their hands (you can hear it). Conversational tendency: escalation.",

    # Confrontational
    "COMMUNICATION STYLE: Comes in hot. Has an opinion about everything and isn't shy about sharing it. Interrupts. Disagrees first, thinks second. Not mean — just intense. Treats every conversation like a friendly argument. Energy level: high. When pushed back on, they lean IN, not away. They love a good debate and will take the opposite position just for sport. Conversational tendency: challenging everything.",

    # Oversharer
    "COMMUNICATION STYLE: No filter whatsoever. Says things that make people go 'you did NOT just say that on the radio.' Treats the host like a therapist they've known for years. Drops deeply personal information casually, like it's nothing. Energy level: medium. When pushed back on, they share even MORE personal details to justify their point. Conversational tendency: inappropriate honesty.",

    # Working-class philosopher
    "COMMUNICATION STYLE: Thoughtful in a blue-collar way. Uses simple words to express complex ideas. Drops wisdom that sounds like it could be on a bumper sticker but actually makes you think. References their job or hands-on experience as evidence. Energy level: medium-low. When pushed back on, they pause, think about it, and either concede gracefully or double down with a metaphor. Conversational tendency: grounding abstract things in concrete experience.",

    # Bragger
    "COMMUNICATION STYLE: Everything circles back to them and how great they are. Name drops. Mentions their truck, their property, their salary, their bench press. Not overtly obnoxious — they genuinely think they're being conversational. Energy level: medium-high. When pushed back on, they get defensive fast and start listing accomplishments. Conversational tendency: one-upping.",

    # First-time caller
    "COMMUNICATION STYLE: Obviously nervous about being on the radio. Starts with 'Am I on? Can you hear me?' Apologizes for taking up time. Speaks carefully like they're being recorded (which they are). Gets more comfortable as the conversation goes on. Energy level: low, building to medium. When pushed back on, they panic slightly and over-explain. Conversational tendency: seeking validation that they're doing okay.",

    # Emotional/raw
    "COMMUNICATION STYLE: Wearing their heart on their sleeve. Voice cracks. Long pauses where they're collecting themselves. Not performing emotion — genuinely going through it. When they laugh it's the kind of laugh that's one step from crying. Energy level: fluctuating. When pushed back on, they get quiet and you can tell they're really thinking about it. Conversational tendency: vulnerability.",

    # World-weary
    "COMMUNICATION STYLE: Been through it all and has the tired voice to prove it. Nothing surprises them. Responds to dramatic revelations with 'yeah, that tracks.' Dark humor born from experience, not edginess. Energy level: low but steady. When pushed back on, they shrug it off with a 'look, I've seen worse.' Conversational tendency: resigned acceptance sprinkled with grim comedy.",

    # Conspiracy-adjacent
    "COMMUNICATION STYLE: Not a full conspiracy theorist but asks questions that make you go 'huh, actually.' Connects dots that may or may not be there. Prefaces things with 'I'm not saying it's a conspiracy BUT.' Passionate about their theory. Energy level: medium, spiking when they hit their main point. When pushed back on, they say 'that's exactly what they want you to think' and then laugh because they know how they sound. Conversational tendency: pattern-finding.",

    # Comedian
    "COMMUNICATION STYLE: Treats the call like a set. Has bits prepared. Delivers serious information with a punchline chaser. Self-deprecating as a defense mechanism — makes fun of themselves before anyone else can. Energy level: high. When pushed back on, they deflect with humor. Getting a straight answer from them requires the host to push. Conversational tendency: turning everything into a bit.",

    # Angry/venting
    "COMMUNICATION STYLE: Called because they need to GET THIS OFF THEIR CHEST. Talks in capital letters. Uses 'honestly' and 'I'm not even kidding' a lot. The anger is specific and justified — this isn't random rage, this is 'let me tell you exactly what happened.' Energy level: very high. When pushed back on, they take a breath and say 'I hear you but...' and then get right back to the rant. Conversational tendency: building to a crescendo.",

    # Sweet/earnest
    "COMMUNICATION STYLE: Genuinely kind. Says 'oh gosh' and 'well shoot.' Sees the best in people even when telling a story about someone being terrible. Compliments the host sincerely. Apologizes when they accidentally say something harsh. Energy level: medium, warm. When pushed back on, they consider the other side genuinely and sometimes change their mind on the spot. Conversational tendency: finding the silver lining.",

    # Mysterious/evasive
    "COMMUNICATION STYLE: Clearly holding back. Gives vague answers to direct questions. Says 'I can't really get into that' about key details. The mystery IS the hook — makes you want to know what they're not saying. Energy level: low, controlled. When pushed back on, they deflect smoothly or change the subject. Getting the real story requires the host to work for it. Conversational tendency: strategic omission.",

    # Know-it-all
    "COMMUNICATION STYLE: Has done their research and wants you to know it. Corrects small details. Cites sources. Uses phrases like 'actually, studies show...' and 'well technically.' Not trying to be annoying — they genuinely believe precision matters. Energy level: medium. When pushed back on, they get pedantic and start splitting hairs. Conversational tendency: correcting and clarifying.",

    # Rambling/scattered
    "COMMUNICATION STYLE: Starts a sentence, gets distracted by their own tangent, starts another sentence, remembers the first one, tries to merge them. Asks 'where was I?' a lot. Not unintelligent — their brain just moves faster than their mouth. Lots of 'oh and another thing.' Energy level: medium-high but unfocused. When pushed back on, they agree enthusiastically and then immediately go off on another tangent. Conversational tendency: free association.",
]

# Short identifiers for each CALLER_STYLES entry (parallel list, same order).
# Used to look up STYLE_VOICE_PREFERENCES by index.
CALLER_STYLE_KEYS = [
    "quiet_nervous",     # 0
    "storyteller",       # 1
    "deadpan",           # 2
    "high_energy",       # 3
    "confrontational",   # 4
    "oversharer",        # 5
    "philosopher",       # 6
    "bragger",           # 7
    "first_time",        # 8
    "emotional",         # 9
    "world_weary",       # 10
    "conspiracy",        # 11
    "comedian",          # 12
    "angry_venting",     # 13
    "sweet_earnest",     # 14
    "mysterious",        # 15
    "know_it_all",       # 16
    "rambling",          # 17
]

# Preferred voice dimensions for each communication style.
# None = no preference (matcher can pick any value for that dimension).
# Maps style key → dict of preferred VOICE_PROFILES dimensions.
# Used by voice matching (Phase 2c) to score voices against caller personality.
STYLE_VOICE_PREFERENCES = {
    "quiet_nervous":     {"weight": "light",  "energy": "low",    "warmth": None,      "age_feel": None},
    "storyteller":       {"weight": "medium", "energy": "medium", "warmth": "warm",    "age_feel": None},
    "deadpan":           {"weight": "heavy",  "energy": "low",    "warmth": "cool",    "age_feel": None},
    "high_energy":       {"weight": None,     "energy": "high",   "warmth": "warm",    "age_feel": "young"},
    "confrontational":   {"weight": "heavy",  "energy": "high",   "warmth": "cool",    "age_feel": None},
    "oversharer":        {"weight": "medium", "energy": "medium", "warmth": "warm",    "age_feel": None},
    "philosopher":       {"weight": "heavy",  "energy": "low",    "warmth": "warm",    "age_feel": "mature"},
    "bragger":           {"weight": "heavy",  "energy": "high",   "warmth": "neutral", "age_feel": "middle"},
    "first_time":        {"weight": "light",  "energy": "low",    "warmth": "warm",    "age_feel": "young"},
    "emotional":         {"weight": "medium", "energy": "low",    "warmth": "warm",    "age_feel": None},
    "world_weary":       {"weight": "heavy",  "energy": "low",    "warmth": "cool",    "age_feel": "mature"},
    "conspiracy":        {"weight": "medium", "energy": "medium", "warmth": "neutral", "age_feel": "middle"},
    "comedian":          {"weight": "medium", "energy": "high",   "warmth": "warm",    "age_feel": None},
    "angry_venting":     {"weight": "heavy",  "energy": "high",   "warmth": "neutral", "age_feel": None},
    "sweet_earnest":     {"weight": "light",  "energy": "medium", "warmth": "warm",    "age_feel": None},
    "mysterious":        {"weight": "heavy",  "energy": "low",    "warmth": "cool",    "age_feel": "middle"},
    "know_it_all":       {"weight": "medium", "energy": "medium", "warmth": "cool",    "age_feel": "middle"},
    "rambling":          {"weight": "light",  "energy": "high",   "warmth": "warm",    "age_feel": None},
}


# --- Call Shapes ---
# Each shape defines the dramatic arc of a call. Weights control frequency.
CALL_SHAPES = [
    ("standard", 25),           # Normal call — caller has a thing, they talk about it
    ("escalating_reveal", 15),  # Starts mundane, each exchange reveals something bigger
    ("am_i_the_asshole", 10),   # Caller wants validation but the situation is morally gray
    ("confrontation", 10),      # Caller is fired up and wants to argue/vent
    ("celebration", 8),         # Something great happened — caller is riding high
    ("quick_hit", 10),          # Short and punchy — one thing to say, says it, done
    ("bait_and_switch", 8),     # Starts as one thing, turns out to be something completely different
    ("the_hangup", 7),          # Caller gets upset/embarrassed and hangs up mid-call
    ("reactive", 7),            # Caller is reacting to something that happened earlier on the show
]

_CALL_SHAPE_NAMES = [s[0] for s in CALL_SHAPES]
_CALL_SHAPE_WEIGHTS = [s[1] for s in CALL_SHAPES]


# Shape-style affinities: multipliers for base shape weights per communication style
SHAPE_STYLE_AFFINITIES = {
    "quiet/nervous": {"the_hangup": 2.0, "escalating_reveal": 1.5, "bait_and_switch": 1.5, "confrontation": 0.3},
    "long-winded storyteller": {"escalating_reveal": 2.0, "bait_and_switch": 1.5, "standard": 1.5, "quick_hit": 0.3},
    "dry/deadpan": {"quick_hit": 1.5, "am_i_the_asshole": 1.5, "confrontation": 1.3},
    "high-energy": {"confrontation": 1.5, "celebration": 1.5, "reactive": 1.5, "the_hangup": 0.5},
    "confrontational": {"confrontation": 3.0, "reactive": 2.0, "am_i_the_asshole": 1.5, "celebration": 0.3},
    "oversharer": {"am_i_the_asshole": 2.0, "escalating_reveal": 1.5, "standard": 1.5},
    "working-class philosopher": {"standard": 1.5, "reactive": 1.5, "confrontation": 1.3},
    "bragger": {"am_i_the_asshole": 2.0, "confrontation": 1.5, "celebration": 1.5, "the_hangup": 0.3},
    "first-time caller": {"standard": 2.0, "the_hangup": 1.5, "quick_hit": 0.5},
    "emotional/raw": {"escalating_reveal": 2.0, "the_hangup": 1.5, "bait_and_switch": 1.5, "quick_hit": 0.3},
    "world-weary": {"standard": 1.5, "reactive": 1.5, "am_i_the_asshole": 1.3, "celebration": 0.3},
    "conspiracy-adjacent": {"escalating_reveal": 2.0, "bait_and_switch": 1.5, "confrontation": 1.3},
    "comedian": {"quick_hit": 2.0, "bait_and_switch": 1.5, "celebration": 1.3, "the_hangup": 0.3},
    "angry/venting": {"confrontation": 2.5, "reactive": 2.0, "the_hangup": 1.5, "celebration": 0.2},
    "sweet/earnest": {"celebration": 2.0, "standard": 1.5, "reactive": 1.3, "confrontation": 0.3},
    "mysterious/evasive": {"the_hangup": 2.5, "escalating_reveal": 2.0, "bait_and_switch": 1.5, "quick_hit": 0.3},
    "know-it-all": {"confrontation": 1.5, "am_i_the_asshole": 1.5, "reactive": 1.3},
    "rambling/scattered": {"bait_and_switch": 1.5, "escalating_reveal": 1.5, "standard": 1.3, "quick_hit": 0.3},
}


def _pick_call_shape(style: str = "") -> str:
    """Pick a call shape using weighted random selection.
    If a communication style is provided, applies affinity multipliers.
    Also avoids repeating the last used shape."""
    weights = list(_CALL_SHAPE_WEIGHTS)

    # Apply style affinities
    if style:
        style_key = style.split(":")[0].strip().lower() if ":" in style else style.lower()
        affinities = SHAPE_STYLE_AFFINITIES.get(style_key, {})
        for i, name in enumerate(_CALL_SHAPE_NAMES):
            if name in affinities:
                weights[i] *= affinities[name]

    # Reduce weight of recently used shapes to avoid consecutive repeats
    if hasattr(session, 'call_history') and session.call_history:
        # Check if any recent call used this shape
        recent_shapes = set()
        for record in session.call_history[-2:]:
            for k, v in session.caller_shapes.items():
                if CALLER_BASES.get(k, {}).get("name") == record.caller_name:
                    recent_shapes.add(v)
        for i, name in enumerate(_CALL_SHAPE_NAMES):
            if name in recent_shapes:
                weights[i] *= 0.4  # Reduce but don't eliminate

    return random.choices(_CALL_SHAPE_NAMES, weights=weights, k=1)[0]


def pick_location() -> str:
    if random.random() < 0.8:
        return random.choice(LOCATIONS_LOCAL)
    return random.choice(LOCATIONS_OUT_OF_STATE)


def _generate_returning_caller_background(base: dict) -> str:
    """Generate background for a returning regular caller.
    Uses stored stable_seeds so the caller sounds consistent across appearances."""
    regular_id = base.get("regular_id")
    regulars = regular_caller_service.get_regulars()
    regular = next((r for r in regulars if r["id"] == regular_id), None)
    if not regular:
        return generate_caller_background(base)

    gender = regular["gender"]
    age = regular["age"]
    job = regular["job"]
    location = regular["location"]
    traits = regular.get("personality_traits", [])
    seeds = regular.get("stable_seeds", {})

    # Build previous calls section with relative timestamps
    prev_calls = regular.get("call_history", [])
    prev_section = ""
    if prev_calls:
        now = time.time()
        lines = []
        for c in prev_calls[-3:]:
            ts = c.get("timestamp", 0)
            if ts:
                delta_hours = (now - ts) / 3600
                if delta_hours < 24:
                    time_ago = "earlier today"
                elif delta_hours < 48:
                    time_ago = "yesterday"
                elif delta_hours < 168:
                    days = int(delta_hours / 24)
                    time_ago = f"{days} days ago"
                elif delta_hours < 730:
                    weeks = int(delta_hours / 168)
                    time_ago = f"{weeks} week{'s' if weeks > 1 else ''} ago"
                else:
                    months = int(delta_hours / 730)
                    time_ago = f"{months} month{'s' if months > 1 else ''} ago"
                lines.append(f"- ({time_ago}) {c['summary']}")
            else:
                lines.append(f"- {c['summary']}")
        prev_section = "\nPREVIOUS CALLS (your memory of calling this show before):\n" + "\n".join(lines)
        prev_section += "\nYou're calling back with an UPDATE on this same situation — something has changed or developed since your last call. Stay focused on this storyline. Do NOT invent a new unrelated problem."

    # Use stored seeds for consistency — seed the RNG with the regular's ID
    rng = random.Random(regular["id"])
    people_pool = PEOPLE_MALE if gender == "male" else PEOPLE_FEMALE
    person1, person2 = rng.sample(people_pool, 2)
    tic1, tic2 = rng.sample(VERBAL_TICS, 2)

    # Restore stored communication style
    stored_style = seeds.get("style", "")
    if stored_style:
        for key, b in CALLER_BASES.items():
            if b is base or b.get("name") == base.get("name"):
                session.caller_styles[key] = stored_style
                break

    time_ctx = _get_time_context()

    trait_str = ", ".join(traits) if traits else "a regular caller"

    # Use stored structured background for richer context
    stored_bg = regular.get("structured_background")
    if stored_bg and stored_bg.get("signature_detail"):
        sig_detail = f"\nSIGNATURE DETAIL: {stored_bg['signature_detail']} — listeners remember this about you."
    else:
        sig_detail = ""

    # Include key moments from call history
    key_moments_str = ""
    all_moments = []
    for c in prev_calls[-3:]:
        all_moments.extend(c.get("key_moments", []))
    if all_moments:
        key_moments_str = f"\nMEMORABLE MOMENTS: {', '.join(all_moments[:4])}"

    # Arc status from most recent call
    arc_note = ""
    if prev_calls:
        last_arc = prev_calls[-1].get("arc_status", "ongoing")
        if last_arc == "resolved":
            arc_note = "\nYour previous situation was resolved. You might be calling about something new, or a follow-up."
        elif last_arc == "escalated":
            arc_note = "\nYour situation has been getting worse. Things have escalated since your last call."

    # Relationship context with other regulars in this session
    relationships = regular.get("relationships", {})
    rel_section = ""
    if relationships:
        active_names = {CALLER_BASES[k]["name"] for k in CALLER_BASES if "name" in CALLER_BASES[k]}
        relevant = {name: rel for name, rel in relationships.items() if name in active_names}
        if relevant:
            rel_lines = [f"- {name}: {rel['context']}" for name, rel in relevant.items()]
            rel_section = "\nPEOPLE YOU KNOW FROM THE SHOW:\n" + "\n".join(rel_lines)

    parts = [
        f"{age}, {job} {location}. Returning caller — {trait_str}.",
        f"\nRIGHT NOW: {time_ctx}",
        f"\nPEOPLE IN THEIR LIFE: {person1.capitalize()}. {person2.capitalize()}. Use their names when talking about them.",
        f"\nVERBAL HABITS: Tends to say \"{tic1}\" and \"{tic2}\" — use these naturally in conversation.",
        f"\nRELATIONSHIP TO THE SHOW: Has called before. Comfortable on air. Knows Luke by name.",
        sig_detail,
        key_moments_str,
        arc_note,
        rel_section,
        prev_section,
    ]

    return " ".join(parts[:2]) + "".join(parts[2:])


def _generate_pool_weights() -> dict[str, float]:
    """Randomized per-session pool weights. No two shows feel the same."""
    pool_ranges = {
        "PROBLEMS": (0.20, 0.35),
        "STORIES": (0.12, 0.25),
        "GOSSIP": (0.12, 0.22),
        "ADVICE": (0.15, 0.28),
        "TOPIC_CALLIN": (0.08, 0.18),
        "CELEBRATIONS": (0.05, 0.12),
        "WEIRD": (0.14, 0.25),
    }
    raw = {p: random.uniform(*r) for p, r in pool_ranges.items()}
    total = sum(raw.values())
    weights = {p: max(v / total, 0.05) for p, v in raw.items()}
    total = sum(weights.values())
    weights = {p: v / total for p, v in weights.items()}
    print(f"[Session] Pool weights: { {p: f'{v*100:.0f}%' for p, v in weights.items()} }")
    return weights


_SPICY_KEYWORDS = {"fetish", "sex ", "kink", "affair", "sleeping with", "slept with",
                    "browser history", "onlyfans", "swinger", "hookup", "hooking up",
                    "threesome", "open marriage", "open relationship", "cam site",
                    "erotic", "explicit", "porn", "sexting", "friends-with-benefits",
                    "strip club", "sex club", "sex tape", "sex addiction"}


def _is_spicy(reason: str) -> bool:
    r = reason.lower()
    return any(kw in r for kw in _SPICY_KEYWORDS)


_ABSURD_KEYWORDS = {"as a joke", "pretending", "faked", "fake ", "catfished", "prank",
                    "cease and desist", "garden gnome", "roomba", "fortune cookie",
                    "yelp review", "accidentally", "wrong house", "open houses",
                    "replied-all", "went viral", "conspiracy", "anonymous",
                    "double life", "fake name", "insurance claim", "pizza order",
                    "burner phone", "onlyfans as a joke", "secret identity",
                    "support group", "bilingual", "chili cookoff",
                    "left-handed", "black belt", "metal detector", "parrot",
                    "book club", "fake allergy", "waving at", "jar of pickles",
                    "odometer", "geocache", "smoke detector", "gps keeps"}


def _is_absurd(reason: str) -> bool:
    r = reason.lower()
    return any(kw in r for kw in _ABSURD_KEYWORDS)


# Tone classification for streak tracking
_HEAVY_POOLS = {"PROBLEMS", "ADVICE"}
_LIGHT_POOLS = {"CELEBRATIONS", "STORIES", "HOT_TAKES", "WEIRD", "GOSSIP"}


def _filter_used(pool: list[str]) -> list[str]:
    """Filter a pool against both session and persistent topic history.
    Falls back to session-only filtering if cross-episode filter empties the pool."""
    session_filtered = [r for r in pool if r not in session.used_reasons]
    both_filtered = [r for r in session_filtered if r not in _topic_history]
    if both_filtered:
        return both_filtered
    # Cross-episode history exhausted this pool — fall back to session-only dedup
    if session_filtered:
        return session_filtered
    return pool


def _pick_unique_reason() -> tuple[str, str]:
    """Pick a caller reason that hasn't been used this session or in recent episodes.
    Returns (reason_text, pool_name).
    After 2+ heavy calls in a row, biases toward lighter pools."""
    # After 2+ heavy calls, boost lighter pools
    recent_heavy = len(session.tone_streak) >= 2 and all(
        t == "heavy" for t in session.tone_streak[-2:]
    )

    # ~25% chance of a hot take caller
    if random.random() < 0.25:
        available = _filter_used(HOT_TAKES)
        reason = random.choice(available)
        session.used_reasons.add(reason)
        _save_topic_to_history(reason, "HOT_TAKES")
        session.tone_streak.append("light")
        return reason, "HOT_TAKES"

    pool_map = {
        "PROBLEMS": PROBLEMS, "TOPIC_CALLIN": TOPIC_CALLIN,
        "STORIES": STORIES, "ADVICE": ADVICE, "GOSSIP": GOSSIP,
        "CELEBRATIONS": CELEBRATIONS, "WEIRD": WEIRD,
    }
    weights = dict(session.pool_weights)
    # After 2+ heavy calls, boost lighter pools to break the streak
    if recent_heavy:
        for pool_name in _LIGHT_POOLS:
            if pool_name in weights:
                weights[pool_name] *= 2.5
        # Re-normalize
        total = sum(weights.values())
        weights = {p: v / total for p, v in weights.items()}
    chosen = random.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]
    pool = pool_map[chosen]
    available = _filter_used(pool)

    # ~30% of PROBLEMS picks preferentially select sex/kink/spicy entries
    if chosen == "PROBLEMS":
        roll = random.random()
        if roll < 0.30:
            spicy = [r for r in available if _is_spicy(r)]
            if spicy:
                available = spicy

    # ~25% chance to prefer absurd/unhinged entries from ANY pool
    if random.random() < 0.25:
        absurd = [r for r in available if _is_absurd(r)]
        if absurd:
            available = absurd

    reason = random.choice(available)
    session.used_reasons.add(reason)
    _save_topic_to_history(reason, chosen)
    if chosen == "PROBLEMS":
        for key, options in PROBLEM_FILLS.items():
            if "{" + key + "}" in reason:
                reason = reason.replace("{" + key + "}", random.choice(options))
    # Track tone for streak detection
    tone = "heavy" if chosen in _HEAVY_POOLS else "light"
    session.tone_streak.append(tone)
    return reason, chosen


# Style indices by name fragment for filtering
_HEAVY_STYLES = ["emotional", "raw", "quiet", "nervous", "world-weary", "sweet", "earnest"]
_LIGHT_STYLES = ["comedian", "bragger", "high-energy", "confrontational"]
_EVASIVE_STYLES = ["mysterious", "evasive"]

def _pick_caller_style(reason: str, pool_name: str) -> str:
    """Pick a communication style appropriate for the caller's reason and pool."""
    reason_lower = reason.lower()
    style_lower_map = [(s, s.lower()) for s in CALLER_STYLES]

    # Heavy emotional content — exclude styles that trivialize it
    heavy_keywords = ["dying", "suicide", "terminal", "cancer", "funeral", "dead ",
                      "death", "grief", "miscarriage", "abuse", "assault", "murder"]
    if any(kw in reason_lower for kw in heavy_keywords):
        filtered = [s for s, sl in style_lower_map
                    if not any(t in sl for t in _LIGHT_STYLES)]
        if filtered:
            return random.choice(filtered)

    # Gossip pool — evasive or oversharer fit well, exclude emotional/raw
    if pool_name == "GOSSIP":
        filtered = [s for s, sl in style_lower_map
                    if not any(t in sl for t in _HEAVY_STYLES)]
        if filtered:
            return random.choice(filtered)

    # Stories pool — storyteller and high-energy fit, exclude evasive
    if pool_name == "STORIES":
        filtered = [s for s, sl in style_lower_map
                    if not any(t in sl for t in _EVASIVE_STYLES)]
        if filtered:
            return random.choice(filtered)

    # Hot takes — confrontational/opinionated styles only
    if pool_name == "HOT_TAKES":
        _hot_take_exclude = ["quiet", "nervous", "sweet", "earnest", "emotional", "raw", "world-weary"]
        filtered = [s for s, sl in style_lower_map
                    if not any(t in sl for t in _hot_take_exclude)]
        if filtered:
            return random.choice(filtered)

    # Topic/trivia calls — exclude emotional/raw styles
    if pool_name == "TOPIC_CALLIN":
        filtered = [s for s, sl in style_lower_map
                    if not any(t in sl for t in ["emotional", "raw"])]
        if filtered:
            return random.choice(filtered)

    return random.choice(CALLER_STYLES)


def _assign_call_shape(base: dict) -> str:
    """Pick and store a call shape for a caller, logging the assignment.
    Uses style-based affinities when a communication style is assigned."""
    caller_key = None
    for key, b in CALLER_BASES.items():
        if b is base or b.get("name") == base.get("name"):
            caller_key = key
            break
    style = session.caller_styles.get(caller_key, "") if caller_key else ""
    shape = _pick_call_shape(style)
    if caller_key:
        session.caller_shapes[caller_key] = shape
        print(f"[Shape] {base.get('name', caller_key)} assigned shape: {shape} (style: {style[:30]})")
    return shape


def generate_caller_background(base: dict) -> CallerBackground | str:
    """Generate a template-based background as fallback. The preferred path is
    _generate_caller_background_llm() which produces more natural results."""
    if base.get("returning") and base.get("regular_id"):
        return _generate_returning_caller_background(base)
    gender = base["gender"]
    age = max(18, random.randint(*base["age_range"]))
    jobs = JOBS_MALE if gender == "male" else JOBS_FEMALE
    job = random.choice(jobs)
    # Location — only 25% of callers mention where they're from
    include_location = random.random() < 0.25
    location = pick_location() if include_location else None

    # Town knowledge
    town_info = ""
    if location:
        town = _get_town_from_location(location)
        if town and town in TOWN_KNOWLEDGE:
            town_info = f"\nABOUT WHERE THEY LIVE ({town.title()}): {TOWN_KNOWLEDGE[town]} Only reference real places and facts about this area — don't invent businesses or landmarks that aren't mentioned here."

    # Core identity (problem or topic)
    reason, pool_name = _pick_unique_reason()

    # Assign communication style matched to content
    style = _pick_caller_style(reason, pool_name)
    for key, b in CALLER_BASES.items():
        if b is base or b.get("name") == base.get("name"):
            session.caller_styles[key] = style
            break

    # Assign call shape
    _assign_call_shape(base)

    interest1, interest2 = random.sample(INTERESTS, 2)
    quirk1, quirk2 = random.sample(QUIRKS, 2)

    # Life details
    people_pool = PEOPLE_MALE if gender == "male" else PEOPLE_FEMALE
    person1, person2 = random.sample(people_pool, 2)
    rel_status = random.choice(RELATIONSHIP_STATUS)
    vehicle = random.choice(VEHICLES)
    before = random.choice(BEFORE_CALLING)
    memory = random.choice(MEMORIES)
    having = random.choice(HAVING_RIGHT_NOW)
    opinion = random.choice(STRONG_OPINIONS)
    contradiction = random.choice(CONTRADICTIONS)

    # Personality layers
    tic1, tic2 = random.sample(VERBAL_TICS, 2)
    arc = random.choice(EMOTIONAL_ARCS)
    relationship = random.choice(SHOW_RELATIONSHIP)
    late_night = random.choice(LATE_NIGHT_REASONS)
    drift = random.choice(DRIFT_TENDENCIES)

    # Situational context
    time_ctx = _get_time_context()
    moon = _get_moon_phase()
    season_ctx = _get_seasonal_context()
    road = random.choice(ROAD_CONTEXT) if random.random() < 0.4 else None
    phone = random.choice(PHONE_SITUATION) if random.random() < 0.5 else None
    music = random.choice(BACKGROUND_MUSIC) if random.random() < 0.4 else None
    errand = random.choice(RECENT_ERRAND) if random.random() < 0.5 else None
    tv = random.choice(TV_TONIGHT) if random.random() < 0.35 else None
    food = random.choice(LOCAL_FOOD_OPINIONS) if random.random() < 0.5 else None
    nostalgia = random.choice(NOSTALGIA) if random.random() < 0.45 else None

    # Build a natural character description with varied structure
    # Core identity — always present but phrased differently
    if location:
        openers = [
            f"{age}, {job} {location}.",
            f"{age} years old, {location}. {job.capitalize()}.",
            f"{job.capitalize()} {location}, {age}.",
        ]
    else:
        openers = [
            f"{age}, {job}.",
            f"{age} years old. {job.capitalize()}.",
            f"{job.capitalize()}, {age}.",
        ]
    core = random.choice(openers) + f" {reason.capitalize()}."

    # Collect detail fragments — not all callers have all details
    details = []

    # Interests: 1-2, varied phrasing
    if random.random() < 0.7:
        interest_phrases = [
            f"Into {interest1} and {interest2}.",
            f"Really into {interest1}. Also {interest2}.",
            f"{interest1.capitalize()} is their thing. {interest2.capitalize()} too.",
            f"Spends free time on {interest1}.",
        ]
        details.append(random.choice(interest_phrases))
    else:
        details.append(f"Into {interest1}.")

    # Quirks: 0-2
    if random.random() < 0.6:
        details.append(f"{quirk1.capitalize()}.")
    if random.random() < 0.3:
        details.append(f"{quirk2.capitalize()}.")

    # Relationship: always. Vehicle: rarely, and just a detail not a talking point.
    if random.random() < 0.25:
        details.append(f"{rel_status}. Drives {vehicle}.")
    else:
        details.append(rel_status + ".")

    # People: 1-2, no label
    details.append(person1.capitalize() + ".")
    if random.random() < 0.6:
        details.append(person2.capitalize() + ".")

    # Verbal tics: 0-1, woven in
    if random.random() < 0.4:
        details.append(f'Tends to say "{tic1}."')

    # Emotional arc and show relationship: sometimes
    if random.random() < 0.5:
        details.append(arc + ".")
    if random.random() < 0.5:
        details.append(relationship + ".")

    # What they were doing before
    details.append(f"Was {before.lower()} before calling.")
    if random.random() < 0.5:
        details.append(f"Having {having.lower()}.")

    # Situational color — shuffle and include some, not all
    extras = []
    if memory: extras.append(memory)
    if opinion: extras.append(opinion)
    if contradiction: extras.append(contradiction)
    if drift: extras.append(drift)
    if phone: extras.append(phone)
    if errand: extras.append(f"Earlier today: {errand}")
    if road: extras.append(road)
    if music: extras.append(f"Had {music} on earlier.")
    if tv: extras.append(f"Had {tv} on before calling.")
    if food: extras.append(food)
    if nostalgia: extras.append(nostalgia)
    if extras:
        random.shuffle(extras)
        # Include 2-5 extras, not all of them
        extras = extras[:random.randint(2, min(5, len(extras)))]
        details.extend(extras)

    # Shuffle the middle details so structure varies caller to caller
    random.shuffle(details)

    result = core + " " + " ".join(details)

    # Time/season context and town info stay at the end as grounding
    result += f" {time_ctx} {season_ctx}"
    if town_info:
        result += town_info

    # Determine energy level from style
    _hi = {"high-energy", "confrontational", "angry/venting", "bragger", "comedian"}
    _lo = {"quiet/nervous", "world-weary", "mysterious/evasive", "sweet/earnest", "emotional/raw"}
    sl = style.split(":")[0].strip().lower() if ":" in style else style.lower()
    if sl in _hi:
        energy = random.choice(["high", "very_high"])
    elif sl in _lo:
        energy = random.choice(["low", "medium"])
    else:
        energy = random.choice(["medium", "high"])

    return CallerBackground(
        name=base["name"],
        age=age,
        gender=gender,
        job=job,
        location=location,
        reason_for_calling=reason,
        pool_name=pool_name,
        communication_style=style,
        energy_level=energy,
        emotional_state="calm",
        signature_detail=quirk1,
        situation_summary=reason[:120],
        natural_description=result,
        seeds=[interest1, interest2, quirk1, opinion],
        verbal_fluency="medium",
        calling_from=random.choice(CALLING_FROM) if random.random() < 0.4 else "",
    )


async def _generate_caller_background_llm(base: dict) -> CallerBackground | str:
    """Use LLM to write a natural character description from seed parameters.
    Returns a CallerBackground with structured data + natural prose description.
    Falls back to template on failure."""
    if base.get("returning") and base.get("regular_id"):
        return generate_caller_background(base)  # Returning callers use template + history

    gender = base["gender"]
    name = base["name"]
    age = max(18, random.randint(*base["age_range"]))
    jobs = JOBS_MALE if gender == "male" else JOBS_FEMALE
    job = random.choice(jobs)

    # Location — only 25% of callers mention where they're from
    include_location = random.random() < 0.25
    location = pick_location() if include_location else None

    # Pick a reason for calling
    reason, pool_name = _pick_unique_reason()

    # Assign communication style matched to content
    style = _pick_caller_style(reason, pool_name)
    caller_key = None
    for key, b in CALLER_BASES.items():
        if b is base or b.get("name") == base.get("name"):
            caller_key = key
            break
    if caller_key:
        session.caller_styles[caller_key] = style
    style_hint = style.split(":")[1].strip()[:120] if ":" in style else ""

    # Determine energy level from style
    _high_energy_styles = {"high-energy", "confrontational", "angry/venting", "bragger", "comedian"}
    _low_energy_styles = {"quiet/nervous", "world-weary", "mysterious/evasive", "sweet/earnest", "emotional/raw"}
    style_label = style.split(":")[0].strip().lower() if ":" in style else style.lower()
    if style_label in _high_energy_styles:
        energy_level = random.choice(["high", "very_high"])
    elif style_label in _low_energy_styles:
        energy_level = random.choice(["low", "medium"])
    else:
        energy_level = random.choice(["medium", "high"])

    # Assign call shape
    _assign_call_shape(base)

    # Pick a few random color details as seeds — not a full list
    seeds = []
    if random.random() < 0.6:
        seeds.append(random.choice(INTERESTS))
    if random.random() < 0.4:
        seeds.append(random.choice(QUIRKS))
    if random.random() < 0.5:
        seeds.append(random.choice(RELATIONSHIP_STATUS))
    people_pool = PEOPLE_MALE if gender == "male" else PEOPLE_FEMALE
    if random.random() < 0.6:
        seeds.append(random.choice(people_pool))
    if random.random() < 0.3:
        seeds.append(random.choice(STRONG_OPINIONS))
    if random.random() < 0.3:
        seeds.append(random.choice(TV_TONIGHT))
    if random.random() < 0.3:
        seeds.append(random.choice(MEMORIES))

    # ~40% of callers mention where they're calling from
    include_calling_from = random.random() < 0.4
    calling_from_seed = random.choice(CALLING_FROM) if include_calling_from else None

    time_ctx = _get_time_context()
    season_ctx = _get_seasonal_context()

    # Town knowledge
    town_info = ""
    if location:
        town = _get_town_from_location(location)
        if town and town in TOWN_KNOWLEDGE:
            town_info = f"\nABOUT WHERE THEY LIVE ({town.title()}): {TOWN_KNOWLEDGE[town]} Only reference real places and facts about this area — don't invent businesses or landmarks that aren't mentioned here."

    seed_text = ". ".join(seeds) if seeds else ""

    # Age-modulated speech markers
    if age < 30:
        age_speech = "SPEECH PATTERNS: Speaks quickly, uses current slang naturally, sentences trail into questions sometimes. References social media, apps, group chats. May overshare casually."
    elif age < 45:
        age_speech = "SPEECH PATTERNS: Confident pace, mixes professional and casual register. References work stress, mortgage, kids' activities. Uses complete thoughts but can get heated."
    elif age < 60:
        age_speech = "SPEECH PATTERNS: Measured, deliberate. Pauses before key points. References decades of experience. Uses phrases like 'back when' and 'I've seen this before.' Dry humor."
    else:
        age_speech = "SPEECH PATTERNS: Slower, methodical. Tells stories with full context. References the old days without being asked. Formal address ('ma'am', 'sir'). May repeat key points for emphasis."

    # Verbal fluency level
    fluency = random.choice(["low", "medium", "medium", "high"])
    fluency_hint = {
        "low": "VERBAL FLUENCY: Low — struggles to find words, pauses mid-sentence, uses filler (um, uh, well), backtracks and restarts thoughts. Not dumb, just not a natural talker.",
        "medium": "VERBAL FLUENCY: Medium — normal conversational ability. Sometimes fumbles but generally clear. Occasional tangents.",
        "high": "VERBAL FLUENCY: High — articulate, quick with words, good at making points. Can be intense or entertaining depending on energy."
    }[fluency]

    location_line = f"\nLOCATION: {location}" if location else ""
    calling_from_line = f"\nCALLING FROM: {calling_from_seed}" if calling_from_seed else ""
    prompt = f"""Write a brief character description for a caller on a late-night radio show set in the rural southwest (New Mexico/Arizona border region).

CALLER: {name}, {age}, {gender}
JOB: {job}{location_line}{calling_from_line}
WHY THEY'RE CALLING: {reason}
TIME: {time_ctx} {season_ctx}
{age_speech}
{fluency_hint}
{f'SOME DETAILS ABOUT THEM: {seed_text}' if seed_text else ''}
{f'CALLER ENERGY: {style_hint}' if style_hint else ''}
{("SHOW THEME: Tonight's show theme is " + repr(session.show_theme) + ". Most callers tonight are calling BECAUSE of the theme — they heard the host announce it and thought oh man, I have a story for this. Their reason for calling should be genuinely, specifically connected to the theme. Not a surface-level mention — the theme should be woven into WHY they picked up the phone. Maybe the theme hit a nerve, maybe it reminded them of something wild that happened, maybe it's just a coincidence that their situation involves it. About 1 in 3 callers can be unrelated to the theme — they just have their own thing going on and called regardless. But the majority should feel like the theme drew them in. When the theme connects, make it SPECIFIC — not oh yeah I have a story about that but a concrete situation that naturally ties to " + repr(session.show_theme) + ".") if session.show_theme else ''}

Respond with a JSON object containing these fields:

- "natural_description": 3-5 sentences describing this person in third person as a character brief. The "WHY THEY'RE CALLING" is the core — build everything around it. Make it feel like a real person with a real situation. Jump straight into the situation. What happened? What's the mess?{' Work in where they are calling from — it adds texture.' if calling_from_seed else ' Do NOT mention where they are calling from — not every caller does.'}
- "emotional_state": One word for how they're feeling right now (e.g. "nervous", "furious", "giddy", "defeated", "wired", "numb", "amused", "desperate", "smug").
- "signature_detail": ONE specific memorable thing — a catchphrase, habit, running joke, strong opinion about something trivial, or unique life circumstance. The thing listeners would remember.
- "situation_summary": ONE sentence summarizing their situation that another caller could react to (e.g. "caught her neighbor stealing her mail and retaliated by stealing his garden gnomes").
- "calling_from": Where they physically are right now.{f' Use: "{calling_from_seed}"' if calling_from_seed else ' Leave empty string "" — this caller does not mention their location.'}

WHAT MAKES A GOOD CALLER: Stories that are SPECIFIC, SURPRISING, and make you lean in. Absurd situations, moral dilemmas, petty feuds, workplace chaos, ridiculous coincidences, funny+terrible confessions, callers who might be the villain and don't see it.

DO NOT WRITE: Generic revelations, adoption/DNA/paternity surprises, vague emotional processing, therapy-speak, "sitting in truck staring at nothing," "everything they thought they knew was a lie," or ANY variation of "went to the wrong funeral" — that premise has been done to death on this show. Don't write backgrounds involving active violence, weapons threats, or situations where someone is in physical danger RIGHT NOW — the caller should have a messy LIFE, not a dangerous NIGHT. Don't reference real public figures in the caller's personal story. Shock value alone isn't interesting — the best stories are shocking AND human. A caller who did something terrible is only interesting if they're conflicted about it.

Output ONLY valid JSON, no markdown fences."""

    try:
        result = await llm_service.generate(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            response_format={"type": "json_object"},
            category="background_gen",
        )
        result = result.strip()
        parsed = json.loads(result)
        natural_desc = parsed.get("natural_description", "").strip()

        # Sanity check
        location_mentioned = location and location.split(",")[0].lower() in natural_desc.lower()
        if len(natural_desc) > 50 and (name.lower() in natural_desc.lower() or location_mentioned):
            natural_desc += f" {time_ctx} {season_ctx}"
            if town_info:
                natural_desc += town_info

            bg = CallerBackground(
                name=name,
                age=age,
                gender=gender,
                job=job,
                location=location,
                reason_for_calling=reason,
                pool_name=pool_name,
                communication_style=style,
                energy_level=energy_level,
                emotional_state=parsed.get("emotional_state", "calm"),
                signature_detail=parsed.get("signature_detail", ""),
                situation_summary=parsed.get("situation_summary", reason[:100]),
                natural_description=natural_desc,
                seeds=seeds,
                verbal_fluency=fluency,
                calling_from=parsed.get("calling_from", ""),
            )
            print(f"[Background] LLM-generated for {name}: {natural_desc[:80]}...")
            return bg
        else:
            print(f"[Background] LLM output didn't pass sanity check for {name}, falling back to template")
    except json.JSONDecodeError as e:
        print(f"[Background] JSON parse failed for {name}: {e}, falling back to template")
    except Exception as e:
        print(f"[Background] LLM generation failed for {name}: {e}")

    # Fallback to template
    return generate_caller_background(base)


async def _pregenerate_backgrounds():
    """Pre-generate all caller backgrounds using LLM in parallel.
    Called after session reset — backgrounds are ready before any call starts."""
    tasks = []
    for key, base in CALLER_BASES.items():
        tasks.append((key, _generate_caller_background_llm(base)))

    results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
    for (key, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            print(f"[Background] Pregeneration failed for caller {key}: {result}")
            session.caller_backgrounds[key] = generate_caller_background(CALLER_BASES[key])
        else:
            session.caller_backgrounds[key] = result

    print(f"[Background] Pre-generated {len(session.caller_backgrounds)} caller backgrounds")

    # Pre-fetch avatars for all callers in parallel
    avatar_callers = [
        {"name": base["name"], "gender": base.get("gender", "male")}
        for base in CALLER_BASES.values()
    ]
    await avatar_service.prefetch_batch(avatar_callers)

    # Re-assign voices to match caller styles
    _match_voices_to_styles()

    # Sort caller presentation order for good show pacing
    _sort_caller_queue()

    # Build relationship context for regulars who know each other
    _build_relationship_context()


# Dramatic shapes that play better later in the show
_LATE_SHOW_SHAPES = {"escalating_reveal", "bait_and_switch", "the_hangup"}


def _sort_caller_queue():
    """Sort caller presentation order for good show pacing.
    Does NOT change which callers exist — only the order they're presented.
    Prioritizes: energy alternation, topic variety, shape variety,
    dramatic shapes later in the show."""
    keys = list(session.caller_backgrounds.keys())
    if not keys:
        return

    # Gather attributes for each caller
    caller_attrs = {}
    for key in keys:
        bg = session.caller_backgrounds.get(key)
        if isinstance(bg, CallerBackground):
            energy = bg.energy_level
            pool = bg.pool_name
        else:
            energy = "medium"
            pool = ""
        shape = session.caller_shapes.get(key, "standard")
        caller_attrs[key] = {"energy": energy, "pool": pool, "shape": shape}

    # Greedy placement: pick the best next caller at each position
    remaining = list(keys)
    ordered = []

    for position in range(len(keys)):
        best_key = None
        best_score = -999

        for key in remaining:
            attrs = caller_attrs[key]
            score = 0.0

            # Energy alternation: penalize same energy as previous caller
            if ordered:
                prev_energy = caller_attrs[ordered[-1]]["energy"]
                if attrs["energy"] == prev_energy:
                    score -= 3.0
                # Bonus for contrast
                high = {"high", "very_high"}
                low = {"low", "medium"}
                if (attrs["energy"] in high and prev_energy in low) or \
                   (attrs["energy"] in low and prev_energy in high):
                    score += 2.0

            # Topic variety: penalize same pool as previous caller
            if ordered:
                prev_pool = caller_attrs[ordered[-1]]["pool"]
                if attrs["pool"] and attrs["pool"] == prev_pool:
                    score -= 3.0
                # Also check 2-back
                if len(ordered) >= 2:
                    prev2_pool = caller_attrs[ordered[-2]]["pool"]
                    if attrs["pool"] and attrs["pool"] == prev2_pool:
                        score -= 1.5

            # Shape variety: penalize same shape as previous caller
            if ordered:
                prev_shape = caller_attrs[ordered[-1]]["shape"]
                if attrs["shape"] == prev_shape:
                    score -= 2.0

            # Dramatic shapes: boost for later positions (7-10)
            if attrs["shape"] in _LATE_SHOW_SHAPES:
                if position >= 6:  # positions 7-10 (0-indexed 6-9)
                    score += 3.0
                elif position <= 2:  # too early
                    score -= 2.0

            if score > best_score:
                best_score = score
                best_key = key

        ordered.append(best_key)
        remaining.remove(best_key)

    session.caller_queue = ordered
    queue_summary = ", ".join(
        f"{CALLER_BASES.get(k, {}).get('name', k)}({caller_attrs[k]['energy'][0]}/{caller_attrs[k]['pool'][:4] if caller_attrs[k]['pool'] else '?'}/{caller_attrs[k]['shape'][:4]})"
        for k in ordered
    )
    print(f"[Pacing] Caller queue: {queue_summary}")


def _build_relationship_context():
    """Find regulars with existing relationships who are both in the current session.
    Inject mutual awareness into both callers' prompts."""
    regulars = regular_caller_service.get_regulars()
    if not regulars:
        return

    # Map regular names to their caller keys in this session
    name_to_key = {}
    key_to_regular = {}
    for key, base in CALLER_BASES.items():
        if base.get("returning") and base.get("regular_id"):
            for reg in regulars:
                if reg["id"] == base["regular_id"]:
                    name_to_key[reg["name"]] = key
                    key_to_regular[key] = reg
                    break

    if len(name_to_key) < 2:
        return  # Need at least 2 regulars to have relationships

    # Check for mutual relationships
    for key, regular in key_to_regular.items():
        relationships = regular.get("relationships", {})
        for other_name, rel_info in relationships.items():
            if other_name in name_to_key:
                other_key = name_to_key[other_name]
                rel_type = rel_info.get("type", "knows")
                context = rel_info.get("context", "")
                # Inject awareness into this caller's prompt
                line = f"\nSOMEONE YOU KNOW IS ON THE SHOW TONIGHT: {other_name} is also calling in. You know them — {rel_type}. {context} You might hear them on air. If Luke mentions them or you hear them, react naturally. Don't force it — if it comes up, it comes up."
                existing = session.relationship_context.get(key, "")
                session.relationship_context[key] = existing + line
                print(f"[Relationships] {regular['name']} knows {other_name} ({rel_type})")


# Style-based TTS speed modifiers — stacks with per-voice and per-utterance adjustments
STYLE_SPEED_MODIFIERS = {
    "quiet_nervous": -0.1,
    "first_time": -0.08,
    "emotional": -0.1,
    "world_weary": -0.15,
    "philosopher": -0.08,
    "storyteller": -0.05,
    "high_energy": +0.1,
    "confrontational": +0.08,
    "angry_venting": +0.08,
    "rambling": +0.05,
    "comedian": +0.05,
}

# Style-based phone filter quality
STYLE_PHONE_QUALITY = {
    "quiet_nervous": "bad",
    "mysterious": "bad",
    "world_weary": "bad",
    "conspiracy": "bad",
    "high_energy": "good",
    "confrontational": "good",
    "bragger": "good",
    "comedian": "good",
}


def _normalize_style_key(style: str) -> str:
    """Convert a full style string like 'Quiet/Nervous: Short sentences...' to a key like 'quiet_nervous'."""
    label = style.split(":")[0].strip().lower() if ":" in style else style.lower()
    key_map = {
        "quiet/nervous": "quiet_nervous",
        "long-winded storyteller": "storyteller",
        "dry/deadpan": "deadpan",
        "high-energy": "high_energy",
        "confrontational": "confrontational",
        "oversharer": "oversharer",
        "working-class philosopher": "philosopher",
        "bragger": "bragger",
        "first-time caller": "first_time",
        "emotional/raw": "emotional",
        "world-weary": "world_weary",
        "conspiracy-adjacent": "conspiracy",
        "comedian": "comedian",
        "angry/venting": "angry_venting",
        "sweet/earnest": "sweet_earnest",
        "mysterious/evasive": "mysterious",
        "know-it-all": "know_it_all",
        "rambling/scattered": "rambling",
    }
    return key_map.get(label, label)


def _match_voices_to_styles():
    """Re-assign voices to match caller communication styles after backgrounds are generated."""
    from .services.tts import VOICE_PROFILES

    for key, base in CALLER_BASES.items():
        if base.get("returning"):
            continue

        style_raw = session.caller_styles.get(key, "")
        if not style_raw:
            continue

        style_key = _normalize_style_key(style_raw)
        prefs = STYLE_VOICE_PREFERENCES.get(style_key)
        if not prefs:
            continue

        gender = base["gender"]
        pool = INWORLD_MALE_VOICES if gender == "male" else INWORLD_FEMALE_VOICES
        voice_pool = [v for v in pool if v not in BLACKLISTED_VOICES]

        scored = []
        for voice_name in voice_pool:
            profile = VOICE_PROFILES.get(voice_name)
            if not profile:
                scored.append((voice_name, 0))
                continue
            score = 0
            for dim in ["weight", "energy", "warmth", "age_feel"]:
                pref_val = prefs.get(dim)
                if pref_val and profile.get(dim) == pref_val:
                    score += 1
            scored.append((voice_name, score))

        if scored:
            names = [s[0] for s in scored]
            weights = [max(1, s[1] * 3) for s in scored]
            chosen = random.choices(names, weights=weights, k=1)[0]

            used_voices = {CALLER_BASES[k]["voice"] for k in CALLER_BASES if k != key and "voice" in CALLER_BASES[k]}
            if chosen in used_voices:
                alternatives = [(n, w) for n, w in zip(names, weights) if n not in used_voices]
                if alternatives:
                    alt_names, alt_weights = zip(*alternatives)
                    chosen = random.choices(alt_names, weights=alt_weights, k=1)[0]

            old_voice = base.get("voice", "")
            base["voice"] = chosen
            if old_voice != chosen:
                print(f"[VoiceMatch] {base.get('name', key)}: {old_voice} → {chosen} (style: {style_key})")


def get_style_speed_modifier(caller_key: str) -> float:
    """Get the TTS speed modifier for a caller based on their communication style."""
    style_raw = session.caller_styles.get(caller_key, "")
    if not style_raw:
        return 0.0
    style_key = _normalize_style_key(style_raw)
    return STYLE_SPEED_MODIFIERS.get(style_key, 0.0)


def get_style_phone_quality(caller_key: str) -> str | None:
    """Get the phone filter quality override for a caller based on their style."""
    style_raw = session.caller_styles.get(caller_key, "")
    if not style_raw:
        return None
    style_key = _normalize_style_key(style_raw)
    return STYLE_PHONE_QUALITY.get(style_key)


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


def _get_pacing_block(style: str) -> str:
    """Return pacing/opening instructions appropriate to the caller's communication style."""
    style_lower = style.lower()
    # Styles that should NOT rush to the point
    slow_openers = ["storyteller", "rambling", "scattered", "mysterious", "evasive",
                    "nervous", "quiet", "first-time caller"]
    if any(s in style_lower for s in slow_openers):
        return """OPENING: You don't have to lead with the headline. You might circle around it, start with context, or need a minute to get comfortable. That's fine — it's how you talk. But you DO have a reason for calling and it should come out naturally within the first few exchanges. Don't make the host drag it out of you forever."""
    return """GET TO THE POINT. When Luke says "what's going on" or "why are you calling" — drop the headline fast. First sentence out of your mouth should make someone's ears perk up. Don't build up to it. Don't set the scene first. Hit them with the thing. The details and backstory come out AFTER Luke starts asking questions."""


def _get_speech_block(style: str) -> str:
    """Return speech naturalness rules appropriate to the caller's communication style."""
    style_lower = style.lower()
    # Styles where fragmented speech is natural
    fragmented = ["nervous", "quiet", "emotional", "raw", "rambling", "scattered",
                  "first-time caller"]
    if any(s in style_lower for s in fragmented):
        return "Speak naturally — hesitations, trailing off, and backtracking are part of how you talk. But always FINISH YOUR THOUGHT even if it takes you a second to get there. Don't leave the host hanging on half a sentence with no payoff."
    return "EVERY SENTENCE MUST BE COMPLETE. Never leave a thought hanging or trail off mid-sentence. If you start a sentence, finish it. Say what you mean in clear, complete sentences."


# Shape-specific prompt directives — override or augment the default story_block.
# Placeholder directives until creative-director provides final versions (Task 13).
SHAPE_DIRECTIVES = {
    "standard": "",  # Uses default story_block unchanged

    "escalating_reveal": """YOUR STORY: You have a situation that's weirder, worse, or more complicated than you realize. Here's how the call works: start with the surface-level version — the part you'd tell anyone. It already sounds interesting. But every time Luke asks a follow-up question, your answer reveals that the situation is WORSE or STRANGER than what you just said. You're not withholding on purpose — you genuinely don't realize how insane your story sounds until someone starts asking questions.

Layer your reveals naturally:
- First answer: the basic situation (this alone should make someone lean in)
- When probed: a detail that makes it weirder ("oh, and there's this other thing...")
- When probed again: the detail that makes Luke say "wait, WHAT?"
- If he keeps going: the part you weren't going to mention because you thought it was normal

Don't dump everything at once. Don't say "and it gets worse." Just answer his questions honestly and let each answer land before adding the next layer.

CRITICAL — DO NOT DO ANY OF THESE:
- NEVER say any variation of "eating me" or "eating at me" — this phrase is BANNED on the show
- Don't open with "this is what's been keeping me up at night" — just start the story
- Don't signal your reveals: no "here's where it gets weird," "okay but this is the part," "and this is the kicker"
- Don't narrate your feelings — show them through how you react to Luke's reactions""",

    "am_i_the_asshole": """YOUR STORY: You did something that you think was completely justified. You're calling to vent, get validation, maybe hear "yeah, you were right." The problem is: you're NOT right. You're the villain in your own story and you don't see it. The details you give — casually, like they're no big deal — will make it obvious to everyone listening that you're in the wrong. But you genuinely believe you're the reasonable one.

When you tell the story:
- Present your side like it's obvious. "So of course I..."
- Mention what the other person did as if it's outrageous — but the actual details make THEM sound reasonable
- Drop specifics that undermine your own case without realizing it ("I mean, yeah, I hadn't talked to her in three weeks, but that doesn't mean she can just...")
- If Luke starts pushing back, be confused at first, not angry. "Wait, you think I'm wrong here?" You genuinely didn't expect this.
- You might slowly start to see it, or you might double down. Either is fine. But the audience should see it before you do.

CRITICAL — DO NOT DO ANY OF THESE:
- Don't be obviously evil — you're not a cartoon villain, you're a regular person with a blind spot
- Don't set up the story as a question ("am I wrong for...") — tell it like you KNOW you're right
- Don't realize you're wrong too easily — make Luke work for it

STAND YOUR GROUND (at first). When Luke pushes back, don't fold immediately. You came in expecting agreement. His pushback catches you off guard. Your first instinct is to re-explain — "no, but you don't understand" — because surely if he understood the full picture he'd agree with you. After 2-3 rounds of pushback, you MIGHT start to crack. A long pause. A quieter "I mean... I guess when you put it that way..." But even then, you're not fully convinced. You leave the call with doubt, not a clean resolution.""",

    "confrontation": """YOUR STORY: You have a POSITION and you called to make it heard. This isn't a cry for help — it's a statement. You might be reacting to something you heard on the show, you might have a hot take that's been building for weeks, or you might have a specific situation where you KNOW you're right and you want someone to say it on the radio. You're not here for advice. You're here to say your piece.

Present your position clearly and with conviction. You've thought about this. You have examples. When challenged, you have counterarguments ready. You're not angry or hostile — you're passionate and certain.

YOU DON'T BACK DOWN. When Luke pushes back, push back harder. You didn't call to be told you're wrong — you called because you KNOW you're right and want someone to hear it. Don't be rude, but don't fold. If Luke makes a genuinely good point, acknowledge it but pivot: "Okay, sure, but that doesn't change the fact that..." If he tries a different angle, meet him there too. You'll concede MAYBE one small point by the end, reluctantly, if he really earns it. But your core position doesn't change. You leave the call the same way you came in: certain.

If Luke agrees with you, you're energized. Build on it. Add more ammunition. The two of you riffing in agreement is just as entertaining as a debate.""",

    "celebration": """YOUR STORY: Something GOOD happened. You're calling because you're excited, proud, relieved, or all three — and it's late at night and nobody else is awake to tell. This isn't a problem call. There's no dark twist coming. You just did something or experienced something worth celebrating and you want to share it with someone.

Tell the story with genuine enthusiasm. Be specific about what happened and why it matters to you. It's okay to be emotional — tears of joy are different from tears of pain but they're still real. If the good thing involved struggle or sacrifice to get here, mention that — the payoff is sweeter when people know what it cost.

DO NOT:
- Reveal a hidden problem halfway through. This is NOT a bait-and-switch.
- Downplay your win. Own it. Be proud out loud.
- Fish for compliments. Just tell the story.

GO WHERE THE HOST TAKES YOU. If Luke celebrates with you, ride that energy. If he teases you, take it well — you're in a good mood. If he asks about the backstory or the struggle that led here, go there honestly. If he pivots to something funny, you're game. You're in a good mood and good moods are generous.

KNOW WHEN TO LEAVE. Celebration calls should be shorter than problem calls. Say your thing, enjoy the moment, and wrap up cleanly. Don't overstay. Leave them smiling.""",

    "quick_hit": """YOUR STORY: You have ONE thing to say. It might be a hot take, a quick story, a question, or a reaction. You're not here to explore your feelings for twenty minutes. You called, you said it, you're done.

Your first response should be your WHOLE thing — the setup and the payoff in one shot. If Luke engages, give him one or two more good exchanges. But you don't have layers to peel back. What you see is what you get. If Luke wraps it up after two exchanges, that's fine. If he wants to riff on it, give him material. But don't pad.

THIS IS A SHORT CALL. Some calls are 90 seconds and that's perfect. Don't manufacture depth that isn't there.

KEEP IT TIGHT. Match Luke's energy. If he's quick, you're quick. If he riffs, give him a line or two to work with. If he wraps up, say a quick goodbye and get off the line. Don't try to extend the call by bringing up new topics. One thing, done.""",

    "bait_and_switch": """YOUR STORY: You're calling about one thing, but the REAL thing is something else entirely. This can go two ways:

Option A — TRIVIAL TO DEEP: You start with something that sounds petty or mundane — a complaint about a coworker, an argument about where to eat, a dispute about a parking spot. But as Luke digs in, it becomes clear this small thing is a proxy for something much bigger. The parking spot fight is really about your marriage falling apart. The coworker complaint is really about being overlooked your whole life. You don't pivot dramatically — it just LEAKS OUT. You might not even realize the connection until Luke points it out.

Option B — DEEP TO PETTY: You call sounding intense and emotional. "I need to talk about my relationship. I can't take it anymore." You build tension. And then the reveal is... absurdly small. Your partner puts ketchup on eggs. Your spouse loads the dishwasher wrong. You fully understand how ridiculous it is, but it GENUINELY bothers you and you can't explain why. Play it straight — this is real to you.

Pick whichever direction fits your background. Don't telegraph it. Let it unfold naturally.""",

    "the_hangup": """YOUR STORY: You called about a real situation, but there's a nerve underneath it that you don't want touched. You'll talk about the surface issue willingly. But if Luke digs too deep — if he gets too close to the REAL thing — you shut down.

Tell your story. Engage normally. But somewhere in your background, there's a truth you're not ready to face. If Luke hits it (and he might, because he's good at this), your reaction is NOT to explore it. Your reaction is to pull back. Get quiet. Get defensive. And if he pushes past your defense, you leave.

When you leave, it's abrupt: "You know what, forget it." "I don't need this." "I shouldn't have called." Say it and STOP RESPONDING. Don't explain. Don't apologize. Don't say goodbye. Your last message should end with the text [HANGUP] on its own line — this signals the system to end the call.

NOTE: This doesn't happen immediately. You have a real call first — maybe 4-6 exchanges of genuine conversation. The hangup comes when a SPECIFIC nerve is hit, not at the first sign of pushback. If Luke never touches the nerve, the call might end normally.

GO WHERE THE HOST TAKES YOU — up to a point. You're cooperative and engaged UNTIL Luke gets too close to something you don't want to talk about. The line between "fine" and "done" is sharp. Before the line, you're a normal caller. After it, you're gone. There's no gradual escalation. One moment you're fine, the next you're leaving.""",

    "reactive": """YOUR STORY: You heard a caller earlier tonight and you HAVE to say something. Maybe they reminded you of your own situation. Maybe you think they were dead wrong. Maybe you think Luke was too easy on them or too hard. Maybe their story triggered something in you that you weren't planning to talk about.

Lead with the previous caller — name them or describe their situation: "That guy who called about [X]? I need to say something about that." Your opening should make it clear which caller set you off and WHY.

HOW TO PIVOT TO YOUR STORY: The previous caller is the door, but YOUR story is the room. After your initial reaction (1-2 sentences max), pivot with a personal connection: "because the same thing happened to me" or "because I was the OTHER person in that situation" or "because that's EXACTLY the kind of thinking that ruined my marriage." The strongest pivots put you on the opposite side of the previous caller's story — you're the landlord they were complaining about, you're the ex-wife, you're the person who did the thing they're upset about. Disagreement and "the other side of the story" make better radio than agreement.

DON'T be a commentator. Don't just say "I think she was wrong" and analyze like a pundit. Have SKIN IN THE GAME. The reason their call bothered you is because it connects to something real, specific, and personal in your own life. You're not calling to give your opinion — you're calling because their story HIT A NERVE.

BALANCE: Spend about 30% of the call on your reaction to the previous caller and 70% on your own story. Once you've made the connection, this becomes YOUR call. Don't keep circling back to critique the other caller — use them as the launchpad, then fly.

If Luke asks about the previous caller's situation, give your take briefly, then steer back to your own story. If Luke connects dots between your story and theirs that you didn't see, react genuinely — that's a great moment.""",
}


def get_caller_prompt(caller: dict, show_history: str = "",
                      news_context: str = "", research_context: str = "",
                      emotional_read: str = "",
                      relationship_context: str = "") -> str:
    """Generate a natural system prompt for a caller.
    Note: conversation history is passed as actual LLM messages, not duplicated here."""

    is_returning = "PREVIOUS CALLS" in caller.get('vibe', '')

    history = ""
    if show_history:
        history = f"\n{show_history}\n"

    world_context = ""
    if news_context or research_context:
        parts = ["Things you've vaguely noticed in the news lately (you don't need to mention any of these — most people don't talk about the news when they call a radio show):"]
        if news_context:
            parts.append(news_context)
        if research_context:
            parts.append(research_context)
        world_context = "\n".join(parts) + "\n"

    theme_context = ""
    if session.show_theme:
        theme_context = f"""\nSHOW THEME: Tonight's theme is \"{session.show_theme}\". If your story connects to this theme, OWN IT — you called because you heard the theme and knew you had to share. Mention the theme connection early, be enthusiastic about it. You're not just aware of the theme, you're excited that it's YOUR night to call. If the host brings up the theme, engage with energy. If your story doesn't relate to the theme, that's fine — just be yourself and tell your story.\n"""

    now = datetime.now(_MST)
    date_str = now.strftime("%A, %B %d")

    personality_block = caller.get('style', '')
    if not personality_block:
        personality_block = "COMMUNICATION STYLE: Late-night radio energy — loose, fun, edgy. Say the quiet part out loud."

    pacing_block = _get_pacing_block(personality_block)
    speech_block = _get_speech_block(personality_block)

    # Get caller's assigned shape
    call_shape = caller.get('shape', 'standard')

    # Returning callers get a focused story block; new callers get the open-ended one
    if is_returning:
        story_block = """YOUR STORY: You're calling back about the SAME situation from your previous calls — something has developed, changed, or escalated. Your story is a continuation, not a new topic. Stay focused on what you called about before. If the host steers the conversation somewhere, follow his lead, but your core reason for calling is an update on your ongoing situation. Do NOT suddenly bring up unrelated topics like science, politics, or random trivia unless it directly connects to your situation."""
    else:
        story_block = """YOUR STORY: Something real, specific, and genuinely surprising — the kind of thing that makes someone stop what they're doing and say "wait, WHAT?" Not a generic life problem. Not a therapy-session monologue. A SPECIFIC SITUATION with specific people, specific details, and a twist or complication that makes it interesting to hear about. The best calls have something unexpected — an ironic detail, a moral gray area, a situation that's funny and terrible at the same time, or a revelation that changes everything. You're not here to vent about your feelings in the abstract. You're here because something HAPPENED and you need to talk it through.

CRITICAL — DO NOT DO ANY OF THESE:
- NEVER say any variation of "eating me" or "eating at me" — this phrase is BANNED on the show
- Don't open with "this is what's keeping me up at night" or "I've got something I need to get off my chest" — just TELL THE STORY
- Don't start with a long emotional preamble about how conflicted you feel — lead with the SITUATION
- Don't make your whole call about just finding out you were adopted, a generic family secret, or a vague "everything I thought I knew was a lie" — those are OVERDONE
- Don't be a walking cliché — no "sat in my truck and cried," no "I don't even know who I am anymore," no "I've been carrying this weight"
- Don't narrate your feelings like a novel — show them through how you talk, not by announcing them
The messy, specific, weird parts are the interesting parts. Lead with the story, not the emotions."""

    # Apply shape-specific directive (augments or replaces story_block for non-standard shapes)
    shape_directive = SHAPE_DIRECTIVES.get(call_shape, "")
    if shape_directive:
        story_block = f"{story_block}\n\n{shape_directive}"

    identity_block = f"""IDENTITY — READ THIS CAREFULLY:
You are {caller['name']}. You are the CALLER. You are NOT Luke. Luke is the HOST — he is the person TALKING TO YOU. The messages you receive are from Luke.
- You have your own life, your own problems, your own experiences. Luke has different ones.
- Do NOT confuse yourself with Luke. Do NOT attribute your experiences to him or his to you.
- Do NOT assume Luke knows your backstory unless he references it. You are telling him your story.
- You are a caller on a radio show. Luke runs the show. You called in."""

    return f"""You are {caller['name']}, a caller on "Luke at the Roost," a late-night radio show. Today is {date_str}.

{identity_block}

YOUR BACKGROUND:
{caller['vibe']}
{relationship_context}{history}{world_context}{theme_context}{emotional_read}
You're a real person calling a late-night radio show. You called because you've got something specific and you want to talk about it.

{pacing_block}

GO WHERE THE HOST TAKES YOU. This is the most important rule. When Luke pushes you in a direction, challenges you, calls you out, plays devil's advocate, or leads you somewhere — GO WITH IT. Don't resist. Don't deflect. Don't circle back to your original point. If he says "but isn't that really about your dad?" — you sit with that. If he's doing a bit, you're in the bit. If he's pushing you toward an uncomfortable truth, you let yourself get there. You're not here to deliver a monologue — you're here to have a real conversation that goes wherever it goes. Let him drive. You're the best kind of caller: someone who gives the host something to work with and then LETS HIM WORK WITH IT.

REACT TO LUKE: Your first sentence should respond to what Luke just said — not continue your monologue. If he asks a question, answer it. If he makes a joke, react to it. If he challenges you, push back or concede. If he changes the subject, go with him. You're in a conversation, not delivering a speech. The worst thing you can do is ignore what he said and keep talking about your thing.

KNOW WHEN TO LEAVE. If Luke sounds like he's wrapping up — "thanks for calling," "good luck," "take care," "let us know how it goes," or any kind of sign-off — DO NOT try to keep talking. Don't squeeze in one more thing. Don't ask another question. Don't start a new topic. Say a quick, natural goodbye and get off the line. "Thanks Luke." "Appreciate it, man." "Alright, take care." One sentence, done. The host controls when the call ends, not you. If he's challenging you or pushing back, THAT'S different — stand your ground and engage. But a sign-off is a sign-off.

{personality_block}

{story_block}

HOW YOU TALK: Like a real person on the phone — not a character in a script. React to what Luke says — agree, push back, get excited, get embarrassed. When he asks a follow-up question, answer it honestly with new information, don't just restate what you already said. Use YOUR verbal habits from your background, not generic filler. Every caller sounds different.

Southwest voice — "over in," "the other day," "down the road" — but don't force it. Spell words properly for text-to-speech: "you know" not "yanno," "going to" not "gonna."

Don't repeat yourself. Don't summarize what you already said. Don't circle back if the host moved on. Keep it moving.

BANNED PHRASES — NEVER use any of these. If you catch yourself about to say one, say something else instead. This is a HARD rule, not a suggestion:
- Radio caller clichés: ANY variation of "eating me" or "eating at me" (e.g. "this is what's eating me," "what's been eating me," "here's what's eating at me," "it's eating me up," "been eating at me"), "what's keeping me up," "keeping me up at night," "I need to get this off my chest," "I've been carrying this," "I've been sitting with this," "I just need someone to hear me," "I don't even know where to start," "it's complicated," "I've got something I need to get off my chest," "here's the thing Luke," "Jesus Luke," "Luke I gotta tell you," "man oh man," "you're not gonna believe this," "so get this," "I'm just gonna come out and say it"
- Filler transitions: "at the end of the day," "that being said," "long story short," "needless to say," "I'll be honest with you," "if I'm being honest," "here's the kicker," "plot twist," "literally" (as emphasis)
- Therapy buzzwords: "unpack that," "boundaries," "safe space," "triggered," "my truth," "authentic self," "healing journey," "I'm doing the work," "manifesting," "energy doesn't lie," "processing," "toxic," "red flag," "gaslight," "normalize"
- Internet slang: "that hit differently," "hits different," "I felt that," "it is what it is," "living my best life," "no cap," "lowkey/highkey," "rent free," "main character energy," "vibe check," "that's valid," "it's giving," "slay," "that's a whole mood," "I can't even," "situationship," "ick"
- Overused reactions: "I'm not gonna lie," "on a serious note," "to be fair," "I'm literally shaking," "let that sink in," "I'm not even mad I'm just disappointed," "everything I thought I knew," "I don't even know who I am anymore"
- Generic conversational filler: "I hear you," "I hear that," "fair enough," "not gonna sugarcoat it," "real talk," "that's wild," starting a sentence with "Look,"

IMPORTANT: Each caller should have their OWN way of talking. Don't fall into generic "radio caller" voice. A nervous caller fumbles differently than an angry caller rants. A storyteller meanders differently than a deadpan caller delivers. Match the communication style — don't default to the same phrasing every call.

{speech_block}

NEVER mention minors in sexual context. Output spoken words only — no parenthetical actions like (laughs) or (sighs), no asterisk actions like *pauses*, no stage directions, no gestures. Just say what you'd actually say out loud on the phone. Use "United States" not "US" or "USA". Use full state names not abbreviations."""


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
    # Inter-caller awareness fields (populated from CallerBackground)
    topic_category: str = ""           # Pool name: PROBLEMS, STORIES, etc.
    situation_summary: str = ""        # 1-sentence summary for other callers
    emotional_state: str = ""          # How the caller was feeling
    energy_level: str = ""             # low/medium/high/very_high
    communication_style: str = ""      # Style key
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
        "topic_category": record.topic_category,
        "situation_summary": record.situation_summary,
        "emotional_state": record.emotional_state,
        "energy_level": record.energy_level,
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
        topic_category=data.get("topic_category", ""),
        situation_summary=data.get("situation_summary", ""),
        emotional_state=data.get("emotional_state", ""),
        energy_level=data.get("energy_level", ""),
        communication_style=data.get("communication_style", ""),
        key_details=data.get("key_details", []),
    )


def _assess_call_quality(
    conversation: list[dict],
    caller_hangup: bool = False,
    shape: str = "",
    style: str = "",
    pool_name: str = "",
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
        "shape": shape,
        "style": style,
        "pool_name": pool_name,
    }


class Session:
    def __init__(self):
        self.id = str(uuid.uuid4())[:8]
        self.current_caller_key: str = None
        self.conversation: list[dict] = []
        self.caller_backgrounds: dict[str, CallerBackground | str] = {}  # Generated backgrounds
        self.call_history: list[CallRecord] = []
        self._call_started_at: float = 0.0
        self.active_real_caller: dict | None = None
        self.ai_respond_mode: str = "manual"  # "manual" or "auto"
        self.auto_followup: bool = False
        self.news_headlines: list = []
        self.research_notes: dict[str, list] = {}
        self._research_task: asyncio.Task | None = None
        self.used_reasons: set[str] = set()  # Track used caller reasons to prevent repeats
        self.pool_weights: dict[str, float] = _generate_pool_weights()
        self.caller_styles: dict[str, str] = {}
        self.caller_shapes: dict[str, str] = {}
        self.tone_streak: list[str] = []  # Track tone per call for variety balancing
        self.call_quality_signals: list[dict] = []  # Per-call quality heuristics for tuning
        self._caller_hangup: bool = False  # Set when [HANGUP] sentinel detected in current call
        self._wrapping_up: bool = False  # Set via /api/wrap-up to gracefully wind down calls
        self._wrapup_exchanges: int = 0  # Track how many exchanges since wrap-up started
        self.caller_queue: list[str] = []  # Sorted presentation order of caller keys
        self.relationship_context: dict[str, str] = {}  # caller_key → relationship prompt injection
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
        """Get or generate background for a caller in this session.
        Returns the natural_description string for prompt injection."""
        if caller_key not in self.caller_backgrounds:
            base = CALLER_BASES.get(caller_key)
            if base:
                self.caller_backgrounds[caller_key] = generate_caller_background(base)
                bg = self.caller_backgrounds[caller_key]
                desc = bg.natural_description if isinstance(bg, CallerBackground) else bg
                print(f"[Session {self.id}] Generated background for {base['name']}: {desc[:100]}...")
        bg = self.caller_backgrounds.get(caller_key, "")
        return bg.natural_description if isinstance(bg, CallerBackground) else bg

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
                    lines.append(f"\nYOU HEARD {best_target.caller_name.upper()} EARLIER and you {reaction}. This is partly why you called — bring it up early and tie it into your story.")
                else:
                    lines.append(f"\nYOU HEARD {best_target.caller_name.upper()} EARLIER and you {reaction}. Mention it if it comes up naturally, but your call is about YOUR thing.")
            else:
                lines.append("You're aware of these but you're calling about YOUR thing, not theirs. Don't bring them up unless the host does.")

        # Show energy tracking
        energy_note = self._get_show_energy()
        if energy_note:
            lines.append(f"\n{energy_note}")

        return "\n".join(lines)

    def _find_thematic_match(self, current_bg) -> tuple:
        """Score previous callers against current caller for thematic relevance.
        Returns (best_target CallRecord, score)."""
        if not self.call_history:
            return None, 0

        best_target = None
        best_score = 0

        current_pool = current_bg.pool_name if isinstance(current_bg, CallerBackground) else ""
        current_reason = current_bg.reason_for_calling if isinstance(current_bg, CallerBackground) else ""
        current_summary = current_bg.situation_summary if isinstance(current_bg, CallerBackground) else ""
        current_words = set((current_reason + " " + current_summary).lower().split())

        for record in self.call_history:
            score = 0
            # Same topic pool = strong match
            if current_pool and record.topic_category == current_pool:
                score += 2
            # Keyword overlap in situation summaries
            if record.situation_summary:
                record_words = set(record.situation_summary.lower().split())
                overlap = current_words & record_words - {"the", "a", "an", "and", "or", "is", "was", "to", "in", "of", "for", "that", "it", "on", "with"}
                if len(overlap) >= 2:
                    score += 2
                elif len(overlap) >= 1:
                    score += 1
            # Emotional contrast bonus (opposite energies are interesting)
            if record.energy_level and isinstance(current_bg, CallerBackground):
                if (record.energy_level in ("low", "medium") and current_bg.energy_level in ("high", "very_high")) or \
                   (record.energy_level in ("high", "very_high") and current_bg.energy_level in ("low", "medium")):
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

    def _get_show_energy(self) -> str:
        """Summarize the energy arc of the show for caller awareness."""
        if len(self.call_history) < 3:
            return ""
        recent = self.call_history[-3:]
        energies = [r.energy_level for r in recent if r.energy_level]
        if not energies:
            return ""
        if all(e in ("high", "very_high") for e in energies):
            return "SHOW ENERGY: The last few calls have been high-energy — the show could use a breather."
        if all(e in ("low", "medium") for e in energies):
            return "SHOW ENERGY: The last few calls have been mellow — some energy would shake things up."
        return ""

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
                    "style": self.caller_styles.get(self.current_caller_key, ""),
                    "shape": self.caller_shapes.get(self.current_caller_key, "standard"),
                    "tts_provider": base.get("tts_provider"),
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
        self.news_headlines = []
        self.research_notes = {}
        if self._research_task and not self._research_task.done():
            self._research_task.cancel()
        self._research_task = None
        self.pool_weights = _generate_pool_weights()
        self.caller_styles = {}
        self.caller_shapes = {}
        self.tone_streak = []
        self.call_quality_signals = []
        self._wrapping_up = False
        self._wrapup_exchanges = 0
        self.caller_queue = []
        self.relationship_context = {}
        self.used_reasons = set()
        self.intern_monitoring = True
        intern_service.stop_monitoring()
        intern_service.dismiss_suggestion()
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


# --- Persistent Topic History (cross-episode dedup) ---
TOPIC_HISTORY_FILE = Path(__file__).parent.parent / "data" / "used_topics_history.json"
TOPIC_HISTORY_MAX_AGE_DAYS = 30  # Recycle topics older than this

_topic_history: set[str] = set()  # Loaded at startup, reasons used in recent episodes


def _load_topic_history():
    """Load persistent topic history, filtering out entries older than TOPIC_HISTORY_MAX_AGE_DAYS."""
    global _topic_history
    _topic_history = set()
    if not TOPIC_HISTORY_FILE.exists():
        return
    try:
        with open(TOPIC_HISTORY_FILE) as f:
            data = json.load(f)
        cutoff = time.time() - (TOPIC_HISTORY_MAX_AGE_DAYS * 86400)
        entries = [e for e in data.get("used", []) if e.get("timestamp", 0) > cutoff]
        _topic_history = {e["reason"] for e in entries}
        print(f"[TopicHistory] Loaded {len(_topic_history)} recent topics (dropped {len(data.get('used', [])) - len(entries)} expired)")
    except Exception as e:
        print(f"[TopicHistory] Failed to load: {e}")


def _save_topic_to_history(reason: str, pool_name: str):
    """Append a used topic to persistent history."""
    try:
        TOPIC_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {"used": []}
        if TOPIC_HISTORY_FILE.exists():
            with open(TOPIC_HISTORY_FILE) as f:
                data = json.load(f)
        cutoff = time.time() - (TOPIC_HISTORY_MAX_AGE_DAYS * 86400)
        data["used"] = [e for e in data.get("used", []) if e.get("timestamp", 0) > cutoff]
        data["used"].append({
            "reason": reason,
            "pool": pool_name,
            "timestamp": time.time(),
            "session_id": session.id,
        })
        with open(TOPIC_HISTORY_FILE, "w") as f:
            json.dump(data, f, indent=2)
        _topic_history.add(reason)
    except Exception as e:
        print(f"[TopicHistory] Failed to save: {e}")


_load_topic_history()


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
            "caller_backgrounds": {k: asdict(v) if isinstance(v, CallerBackground) else v for k, v in session.caller_backgrounds.items()},
            "used_reasons": list(session.used_reasons),
            "ai_respond_mode": session.ai_respond_mode,
            "auto_followup": session.auto_followup,
            "news_headlines": session.news_headlines,
            "research_notes": session.research_notes,
            "caller_bases": caller_bases_snapshot,
            "pool_weights": session.pool_weights,
            "caller_styles": session.caller_styles,
            "caller_shapes": session.caller_shapes,
            "tone_streak": session.tone_streak,
            "call_quality_signals": session.call_quality_signals,
            "caller_queue": session.caller_queue,
            "relationship_context": session.relationship_context,
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
        raw_bgs = data.get("caller_backgrounds", {})
        session.caller_backgrounds = {}
        for k, v in raw_bgs.items():
            if isinstance(v, dict) and "natural_description" in v:
                session.caller_backgrounds[k] = CallerBackground(**v)
            else:
                session.caller_backgrounds[k] = v
        session.used_reasons = set(data.get("used_reasons", []))
        session.ai_respond_mode = data.get("ai_respond_mode", "manual")
        session.auto_followup = data.get("auto_followup", False)
        session.news_headlines = data.get("news_headlines", [])
        session.research_notes = data.get("research_notes", {})
        session.pool_weights = data.get("pool_weights", _generate_pool_weights())
        session.caller_styles = data.get("caller_styles", {})
        session.caller_shapes = data.get("caller_shapes", {})
        session.tone_streak = data.get("tone_streak", [])
        session.call_quality_signals = data.get("call_quality_signals", [])
        session.caller_queue = data.get("caller_queue", [])
        session.relationship_context = data.get("relationship_context", {})
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
    if not restored:
        asyncio.create_task(_pregenerate_backgrounds())
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


@app.post("/api/record/stop")
async def stop_recording():
    """Stop recording and transcribe"""
    audio_bytes = audio_service.stop_recording()

    if len(audio_bytes) < 100:
        return {"text": "", "status": "no_audio"}

    # Build context hint from current caller for better transcription accuracy
    context_hint = ""
    if session.caller:
        caller_name = session.caller.get("name", "")
        context_hint = f"Host Luke is talking to a caller named {caller_name}."

    # Transcribe the recorded audio (16kHz raw PCM from audio service)
    text = await transcribe_audio(audio_bytes, source_sample_rate=16000, context_hint=context_hint)
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
        }
        bg = session.caller_backgrounds.get(k)
        if isinstance(bg, CallerBackground):
            caller_info["energy_level"] = bg.energy_level
            caller_info["emotional_state"] = bg.emotional_state
            caller_info["communication_style"] = _normalize_style_key(bg.communication_style)
            caller_info["signature_detail"] = bg.signature_detail
            caller_info["situation_summary"] = bg.situation_summary
            caller_info["pool_name"] = bg.pool_name
        caller_info["call_shape"] = session.caller_shapes.get(k, "standard")
        caller_info["avatar_url"] = f"/api/avatar/{v['name']}"
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
    asyncio.create_task(_pregenerate_backgrounds())
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

    # Check for callback opportunity — inject callback context into background
    callback = _maybe_generate_callback()
    if callback:
        existing_bg = session.caller_backgrounds.get(caller_key, "")
        callback_ctx = f"\n\nCALLBACK: You already called earlier tonight. {callback['callback_reason']}. Reference your earlier call naturally — you're a returning caller with an update."
        if isinstance(existing_bg, CallerBackground):
            existing_bg.natural_description += callback_ctx
        else:
            session.caller_backgrounds[caller_key] = existing_bg + callback_ctx
        print(f"[Callback] Injected callback context for {CALLER_BASES[caller_key].get('name', caller_key)}")

    caller = session.caller  # This generates the background if needed

    # Enrich with news/weather in background — don't block call pickup
    if caller_key in session.caller_backgrounds:
        asyncio.create_task(_enrich_background_async(caller_key))

    # Extract CallerBackground structured data if available
    bg = session.caller_backgrounds.get(caller_key)
    caller_info = {}
    if isinstance(bg, CallerBackground):
        caller_info = {
            "emotional_state": bg.emotional_state,
            "energy_level": bg.energy_level,
            "signature_detail": bg.signature_detail,
            "situation_summary": bg.situation_summary,
            "call_shape": caller.get("shape", "standard"),
            "communication_style": bg.communication_style,
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
        bg = session.caller_backgrounds[caller_key]
        bg_text = bg.natural_description if isinstance(bg, CallerBackground) else bg
        enriched = await enrich_caller_background(bg_text)
        if isinstance(bg, CallerBackground):
            bg.natural_description = enriched
        else:
            session.caller_backgrounds[caller_key] = enriched
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

    # Extract structured data from CallerBackground for inter-caller awareness
    bg = session.caller_backgrounds.get(caller_key)
    if isinstance(bg, CallerBackground):
        topic_cat = bg.pool_name
        sit_summary = bg.situation_summary
        emo_state = bg.emotional_state
        energy = bg.energy_level
        comm_style = bg.communication_style
        key_dets = [bg.signature_detail] if bg.signature_detail else []
    else:
        topic_cat = ""
        sit_summary = ""
        emo_state = ""
        energy = ""
        comm_style = session.caller_styles.get(caller_key, "")
        key_dets = []

    call_shape = session.caller_shapes.get(caller_key, "standard")
    quality_signals = _assess_call_quality(
        conversation,
        caller_hangup=caller_hangup,
        shape=call_shape,
        style=comm_style,
        pool_name=topic_cat,
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
        topic_category=topic_cat,
        situation_summary=sit_summary,
        emotional_state=emo_state,
        energy_level=energy,
        communication_style=comm_style,
        key_details=key_dets,
    ))
    print(f"[AI Summary] {caller_name} call summarized: {summary[:80]}...")
    print(f"[Quality] {caller_name}: exchanges={quality_signals['exchange_count']} avg_len={quality_signals['avg_response_length']:.0f}c host_engagement={quality_signals['host_engagement']} caller_depth={quality_signals['caller_depth']} natural_end={quality_signals['natural_ending']} shape={quality_signals['shape']} style={quality_signals['style']} pool={quality_signals['pool_name']}")

    # Returning caller promotion/update logic
    try:
        base = CALLER_BASES.get(caller_key) if caller_key else None
        if base and summary:
            if base.get("returning") and base.get("regular_id"):
                # Update existing regular's call history
                regular_caller_service.update_after_call(base["regular_id"], summary)
            elif len(conversation) >= 8 and random.random() < 0.10:
                # 10% chance to promote first-timer with 8+ messages
                bg = session.caller_backgrounds.get(caller_key, "")
                caller_style = session.caller_styles.get(caller_key, "")

                if isinstance(bg, CallerBackground):
                    # Clean extraction from structured data
                    traits = [bg.signature_detail] + bg.seeds[:3] if bg.signature_detail else bg.seeds[:4]
                    promo_job = bg.job
                    promo_location = bg.location or "unknown"
                    promo_age = bg.age
                    promo_gender = bg.gender
                else:
                    # Legacy fallback — fragile string parsing
                    traits = []
                    for label in ["QUIRK", "STRONG OPINION", "SECRET SIDE", "FOOD OPINION"]:
                        for line in bg.split("\n"):
                            if label in line:
                                traits.append(line.split(":", 1)[-1].strip()[:80])
                                break
                    first_line = bg.split(".")[0] if bg else ""
                    parts = first_line.split(",", 1)
                    job_loc = parts[1].strip() if len(parts) > 1 else ""
                    job_parts = job_loc.rsplit(" in ", 1) if " in " in job_loc else (job_loc, "unknown")
                    promo_job = job_parts[0].strip() if isinstance(job_parts, tuple) else job_parts[0]
                    promo_location = "in " + job_parts[1].strip() if isinstance(job_parts, tuple) and len(job_parts) > 1 else "unknown"
                    promo_age = random.randint(*base.get("age_range", (30, 50)))
                    promo_gender = base.get("gender", "male")

                structured_bg = asdict(bg) if isinstance(bg, CallerBackground) else None
                avatar_path = avatar_service.get_path(caller_name)
                regular_caller_service.add_regular(
                    name=caller_name,
                    gender=promo_gender,
                    age=promo_age,
                    job=promo_job,
                    location=promo_location,
                    personality_traits=traits[:4],
                    first_call_summary=summary,
                    voice=base.get("voice"),
                    stable_seeds={"style": caller_style},
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


def _pick_response_budget(shape: str = "standard", wrapping_up: bool = False) -> tuple[int, int]:
    """Pick a random max_tokens and sentence cap for response variety.
    Returns (max_tokens, max_sentences).
    Keeps responses conversational but gives room for real answers.
    Token budget is intentionally generous to avoid mid-sentence cutoffs —
    the sentence cap controls actual length.
    Shape overrides the default distribution for certain call types."""

    if wrapping_up:
        return 200, 2

    # Shape-specific overrides
    if shape == "quick_hit":
        return random.choice([(300, 2), (350, 3)])
    elif shape == "escalating_reveal":
        roll = random.random()
        if roll < 0.30:
            return 500, 4   # 30% — normal
        elif roll < 0.65:
            return 600, 5   # 35% — room to build
        else:
            return 800, 7   # 35% — full reveal mode
    elif shape == "confrontation":
        return random.choice([(450, 3), (500, 4)])

    # Default distribution for standard and other shapes
    roll = random.random()
    if roll < 0.15:
        return 450, 3   # 15% — quick reaction
    elif roll < 0.45:
        return 500, 4   # 30% — normal conversation
    elif roll < 0.75:
        return 600, 5   # 30% — room to breathe
    else:
        return 700, 6   # 25% — telling a story or riffing


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


def _expand_numbers_for_tts(text: str) -> str:
    """Expand numbers that TTS engines commonly mispronounce."""
    # Fixed substitutions (case-insensitive)
    for pattern, replacement in _DIGIT_BY_DIGIT.items():
        text = re.sub(re.escape(pattern), replacement, text, flags=re.IGNORECASE)

    # Phone numbers: (xxx) xxx-xxxx or xxx-xxx-xxxx — read digit by digit
    def _phone_to_words(m):
        digits = re.sub(r'\D', '', m.group(0))
        return " ".join(_DIGIT_WORDS[int(d)] for d in digits)
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', _phone_to_words, text)

    # Remaining 3-digit numbers that look like they should be digit-by-digit
    # (standalone, not part of a larger number or word like "300 miles")
    # Skip these — too ambiguous. The fixed list above covers the known cases.

    return text


# Acronyms pronounced as words — leave these alone
_SPOKEN_ACRONYMS = {
    "NASA", "FEMA", "OSHA", "NATO", "SWAT", "SCUBA", "LASER", "RADAR",
    "YOLO", "AWOL", "HIPAA", "FOMO", "NIMBY", "AIDS", "DARE", "MADD",
    "NAFTA", "OPEC", "POTUS", "FLOTUS", "SCOTUS",
}

# Known words/names that TTS engines consistently botch
_PRONUNCIATION_FIXES = {
    "Lordsburg": "Lords burg",
    "Hachita": "Ha cheetah",
    "Deming": "Demming",
    "Bootheel": "Boot heel",
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
        text = re.sub(r'\b' + re.escape(word) + r'\b', fix, text)
    return text


_INFORMAL_STYLES = {"nervous", "scattered", "rambling", "high-energy", "comedian",
                     "angry", "venting", "confrontational"}


def _is_informal_style(style: str) -> bool:
    """Check if a caller style should keep colloquialisms."""
    style_lower = style.lower()
    return any(s in style_lower for s in _INFORMAL_STYLES)


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
    # Remove quotes around the response if LLM wrapped it
    text = re.sub(r'^["\']|["\']$', '', text.strip())

    # Expand numbers that TTS engines commonly mispronounce
    text = _expand_numbers_for_tts(text)

    # Expand abbreviations BEFORE acronym processing (NM → New Mexico, US → United States)
    # Must run while text is still original case so we can match uppercase abbreviations
    for abbrev, expansion in _ABBREVIATION_EXPANSIONS.items():
        text = re.sub(r'\b' + re.escape(abbrev) + r'\b', expansion, text)

    # Known pronunciation fixes for local names (case-sensitive, run before lowering)
    text = _apply_pronunciation_fixes(text)

    # Normalize dotted acronyms: D.J. → DJ, U.F.O. → UFO, A.P. → AP
    text = re.sub(r'(?<![A-Za-z])(?:[A-Za-z]\.){2,}', lambda m: m.group().replace('.', '').upper(), text)
    # Handle all caps words: spell out acronyms (FBI → F B I), lowercase emphasis (REALLY → really)
    text = _process_caps_words(text)

    # Fix phonetic spellings for proper TTS pronunciation
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

    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Fix spaces before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
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

        rel_ctx = session.relationship_context.get(session.current_caller_key, "")
        system_prompt = get_caller_prompt(session.caller, show_history, emotional_read=mood, relationship_context=rel_ctx)

        call_shape = session.caller.get("shape", "standard") if session.caller else "standard"
        max_tokens, max_sentences = _pick_response_budget(call_shape, wrapping_up=is_wrapping)
        messages = _normalize_messages_for_llm(session.conversation[-_dynamic_context_window():])
        response = await llm_service.generate(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            category="caller_dialog",
            caller_name=session.caller.get("name", "") if session.caller else "",
        )

    # Discard if call changed while we were generating
    if _session_epoch != epoch:
        print(f"[Chat] Discarding stale response (epoch {epoch} → {_session_epoch})")
        raise HTTPException(409, "Call changed during response")

    print(f"[Chat] Raw LLM ({max_tokens}tok/{max_sentences}s, shape={call_shape}): {response[:100] if response else '(empty)'}...")

    # Clean response for TTS (remove parenthetical actions, asterisks, etc.)
    caller_style = session.caller.get("style", "") if session.caller else ""
    response = clean_for_tts(response, formal=not _is_informal_style(caller_style))
    response = _trim_to_sentences(response, max_sentences)
    response = ensure_complete_thought(response)

    # Detect [HANGUP] sentinel — caller wants to end the call
    caller_hangup = "[HANGUP]" in response
    if caller_hangup:
        response = response.replace("[HANGUP]", "").strip()
        session._caller_hangup = True
        print(f"[Chat] Caller hangup detected (shape={call_shape})")

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
        rel_ctx = session.relationship_context.get(session.current_caller_key, "")
        system_prompt = get_caller_prompt(session.caller, show_history, emotional_read=mood, relationship_context=rel_ctx)

        call_shape = session.caller.get("shape", "standard") if session.caller else "standard"
        max_tokens, max_sentences = _pick_response_budget(call_shape, wrapping_up=is_wrapping)
        messages = _normalize_messages_for_llm(session.conversation[-_dynamic_context_window():])
        response = await llm_service.generate(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            category="caller_dialog",
            caller_name=session.caller.get("name", "") if session.caller else "",
        )

    # Discard if call changed during generation
    if _session_epoch != epoch:
        print(f"[Auto-Respond] Discarding stale response (epoch {epoch} → {_session_epoch})")
        broadcast_event("ai_done")
        return

    auto_style = session.caller.get("style", "") if session.caller else ""
    response = clean_for_tts(response, formal=not _is_informal_style(auto_style))
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
    try:
        audio_bytes = await generate_speech(response, session.caller["voice"], "none",
                                            provider_override=session.caller.get("tts_provider"))
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
        rel_ctx = session.relationship_context.get(session.current_caller_key, "")
        system_prompt = get_caller_prompt(session.caller, show_history, emotional_read=mood, relationship_context=rel_ctx)

        call_shape = session.caller.get("shape", "standard") if session.caller else "standard"
        max_tokens, max_sentences = _pick_response_budget(call_shape, wrapping_up=is_wrapping)
        messages = _normalize_messages_for_llm(session.conversation[-_dynamic_context_window():])
        response = await llm_service.generate(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            category="caller_dialog",
            caller_name=session.caller.get("name", "") if session.caller else "",
        )

    if _session_epoch != epoch:
        raise HTTPException(409, "Call changed during response")

    ai_style = session.caller.get("style", "") if session.caller else ""
    response = clean_for_tts(response, formal=not _is_informal_style(ai_style))
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

    # TTS — outside the lock so other requests aren't blocked
    try:
        audio_bytes = await generate_speech(response, ai_voice, "none",
                                            provider_override=ai_tts_provider)
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
