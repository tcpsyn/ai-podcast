"""AI Radio Show - Control Panel Backend"""

import uuid
import asyncio
import base64
import threading
import traceback
from dataclasses import dataclass, field
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
from .services.tts import generate_speech
from .services.audio import audio_service
from .services.news import news_service, extract_keywords, STOP_WORDS

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

MALE_NAMES = [
    "Tony", "Rick", "Dennis", "Earl", "Marcus", "Keith", "Darnell", "Wayne",
    "Greg", "Andre", "Ray", "Jerome", "Hector", "Travis", "Vince", "Leon",
    "Dale", "Frank", "Terrence", "Bobby", "Cliff", "Nate", "Reggie", "Carl",
]

FEMALE_NAMES = [
    "Jasmine", "Megan", "Tanya", "Carla", "Brenda", "Sheila", "Denise", "Tamika",
    "Lorraine", "Crystal", "Angie", "Renee", "Monique", "Gina", "Patrice", "Deb",
    "Shonda", "Marlene", "Yolanda", "Stacy", "Jackie", "Carmen", "Rita", "Val",
]

# Voice pools — ElevenLabs IDs mapped to Inworld voices in tts.py
MALE_VOICES = [
    "VR6AewLTigWG4xSOukaG",  # Edward
    "TxGEqnHWrfWFTfGW9XjX",  # Shaun
    "pNInz6obpgDQGcFmaJgB",  # Alex
    "ODq5zmih8GrVes37Dizd",  # Craig
    "IKne3meq5aSn9XLyUdCD",  # Timothy
]

FEMALE_VOICES = [
    "jBpfuIE2acCO8z3wKNLl",  # Hana
    "EXAVITQu4vr4xnSDxMaL",  # Ashley
    "21m00Tcm4TlvDq8ikWAM",  # Wendy
    "XB0fDUnXU5powFXDhCwa",  # Sarah
    "pFZP5JQG7iQjIQuC4Bku",  # Deborah
]

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


def _randomize_callers():
    """Assign random names and voices to callers, unique per gender."""
    num_m = sum(1 for c in CALLER_BASES.values() if c["gender"] == "male")
    num_f = sum(1 for c in CALLER_BASES.values() if c["gender"] == "female")
    males = random.sample(MALE_NAMES, num_m)
    females = random.sample(FEMALE_NAMES, num_f)
    m_voices = random.sample(MALE_VOICES, num_m)
    f_voices = random.sample(FEMALE_VOICES, num_f)
    mi, fi = 0, 0
    for base in CALLER_BASES.values():
        if base["gender"] == "male":
            base["name"] = males[mi]
            base["voice"] = m_voices[mi]
            mi += 1
        else:
            base["name"] = females[fi]
            base["voice"] = f_voices[fi]
            fi += 1

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
    "is a high school football coach", "teaches middle school history",
    "is a freelance photographer", "is a session musician", "is a tattoo artist",
    "works at a brewery", "is a youth pastor", "does standup comedy on the side",
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
    "works at a gun range", "is a long-haul trucker",
    "works the night shift at Waffle House", "is a funeral home director",
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
    "their kid just got arrested and they don't know what to do",
    "found out their teenager has been lying about where they go at night",
    "their in-laws are trying to take over their life and their spouse won't say anything",

    # Career and purpose
    "woke up and realized they've been in the wrong career for 15 years",
    "got passed over for a promotion they deserved and is questioning everything",
    "has a dream they gave up on years ago and it's haunting them",
    "is successful on paper but feels completely empty inside",
    "hates their job but can't afford to leave and it's breaking them",
    "just got fired and doesn't know who they are without their work",
    "is being asked to do something unethical at work and doesn't know what to do",
    "watches their boss take credit for everything and is losing their mind",
    "started a business and it's failing and they've sunk everything into it",
    "got a job offer across the country and their family doesn't want to move",
    "is about to get laid off and hasn't told their spouse",
    "found out a coworker making half the effort makes twice the money",

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
    "just got a diagnosis they weren't expecting and is processing it alone",
    "has been ignoring symptoms because they're scared of what the doctor will say",
    "someone they love just got diagnosed with something serious",
    "had a health scare and it's making them rethink everything",
    "is dealing with chronic pain and nobody seems to believe them",
    "just found out they can't have kids",

    # Mental health and inner struggles
    "has been putting on a brave face but is barely holding it together",
    "can't shake the feeling that their best years are behind them",
    "keeps self-sabotaging every good thing in their life and doesn't know why",
    "has been numb for months and is starting to scare themselves",
    "feels like a fraud and is waiting to be found out",
    "is exhausted from being the strong one for everyone else",
    "has been having panic attacks and doesn't know what's triggering them",
    "can't stop doom scrolling and it's making them miserable",
    "hasn't left the house in weeks and is starting to wonder if something's wrong",

    # Grief and loss
    "lost someone close and hasn't really dealt with it",
    "is grieving someone who's still alive but is no longer the person they knew",
    "never got closure with someone who died and it's eating at them",
    "is watching their best friend slowly die and doesn't know how to be there",
    "their dog died and they're more wrecked than they thought they'd be",
    "lost their house in a fire and is still processing it",

    # Regrets and past mistakes
    "made a choice years ago that changed everything and wonders what if",
    "hurt someone badly and never apologized, and it haunts them",
    "let the one that got away go and thinks about them constantly",
    "gave up on something important to make someone else happy and resents it",
    "was a bully growing up and is finally reckoning with it",
    "got a DUI and it's ruining their life",
    "ghosted someone who really cared about them and feels terrible about it",

    # Relationships
    "is falling out of love with their spouse and doesn't know what to do",
    "married the wrong person and everyone knows it but them",
    "feels invisible in their own relationship",
    "is staying for the kids but dying inside",
    "realized they don't actually like their partner as a person",
    "found out their partner has been lying about something big",
    "just found out their partner has a dating profile",
    "is in love with two people and has to choose",
    "their ex keeps showing up and they don't hate it",
    "moved in with someone too fast and now they're trapped",

    # Friendship and loneliness
    "realized they don't have any real friends, just people who need things from them",
    "had a falling out with their best friend and the silence is deafening",
    "is surrounded by people but has never felt more alone",
    "suspects a close friend is talking shit behind their back",
    "all their friends are getting married and having kids and they feel left behind",
    "their best friend started dating their ex and acts like it's no big deal",

    # Neighbor and community drama
    "is in a feud with their neighbor that's gotten way out of hand",
    "found out something sketchy is going on next door and doesn't know if they should say something",
    "got into it with someone at their kid's school and now it's a whole thing",
    "someone at church said something that made them question their entire faith",

    # Big life decisions
    "is thinking about leaving everything behind and starting over somewhere new",
    "has to make a choice that will hurt someone no matter what",
    "has been offered an opportunity that would change everything but they're terrified",
    "knows they need to end something but can't pull the trigger",
    "is thinking about joining the military and their family is losing it",
    "wants to go back to school but feels like it's too late",

    # Addiction and bad habits
    "is hiding how much they drink from everyone",
    "can't stop gambling and is in deeper than anyone knows",
    "is watching themselves become someone they don't recognize",
    "just got out of rehab and doesn't know how to face everyone",
    "found pills in their kid's room and doesn't know how to bring it up",

    # Legal trouble
    "is in the middle of a lawsuit and it's consuming their life",
    "got caught doing something stupid and now there are consequences",
    "is dealing with a custody battle that's destroying them",
    "has a warrant they've been ignoring and it's getting worse",

    # Attraction and affairs
    "is attracted to someone they shouldn't be and it's getting harder to ignore",
    "has been seeing {affair_person} on the side",
    "caught feelings for someone at work and it's fucking everything up",

    # Sexual/desire
    "can't stop thinking about {fantasy_subject}",
    "discovered something about their own desires that surprised them",
    "is questioning their sexuality after something that happened recently",

    # General late-night confessions
    "can't sleep and has been thinking too much about their life choices",
    "had a weird day and needs to process it with someone",
    "has been keeping a secret that's eating them alive",
    "finally ready to admit something they've never said out loud",
    "saw something today that brought up a memory they'd buried",
    "just realized they've become exactly like the parent they swore they'd never be",
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
    # Prestige TV (current)
    "obsessed with Severance, has theories about every floor",
    "been binging Landman, loves the oil field drama",
    "really into the Fallout show, played all the games too",
    "hooked on The Last of Us, compares it to the game constantly",
    "just finished Shogun, can't stop talking about it",
    "deep into Slow Horses, thinks it's the best spy show ever made",
    "watches every episode of Poker Face twice",
    "been following Silo, reads the books too",
    # Prestige TV (classic)
    "has watched The Wire three times, quotes it constantly",
    "thinks Breaking Bad is the greatest show ever made",
    "still thinks about the LOST finale, has a take on it",
    "Mad Men changed how they see advertising and life",
    "Westworld season 1 blew their mind, still processes it",
    "big Yellowstone fan, has opinions about the Duttons",
    "Stranger Things got them into 80s nostalgia",
    "rewatches The Sopranos every year, notices new things",
    "thinks True Detective season 1 is peak television",
    "Battlestar Galactica is their comfort rewatch",
    "still upset about how Game of Thrones ended",
    "thinks Better Call Saul is better than Breaking Bad",
    "Chernobyl miniseries changed how they think about disasters",
    "Band of Brothers is their go-to recommendation",
    # Science & space
    "follows NASA missions, got excited about the latest Mars data",
    "reads science journals for fun, especially Nature and Science",
    "into astrophotography, has a decent telescope setup",
    "fascinated by quantum physics, watches every PBS Space Time episode",
    "follows JWST discoveries, has opinions about exoplanet findings",
    "into particle physics, followed CERN news closely",
    "reads about neuroscience and consciousness research",
    "into geology, knows every rock formation around the bootheel",
    "follows fusion energy research, cautiously optimistic about it",
    "amateur astronomer, knows the night sky by heart",
    # Technology
    "follows AI developments closely, has mixed feelings about it",
    "into open source software, runs Linux at home",
    "fascinated by SpaceX launches, watches every one",
    "follows battery and EV tech, thinks about energy transition a lot",
    "into ham radio, has a nice setup",
    "builds electronics projects, has an Arduino collection",
    "follows cybersecurity news, paranoid about their own setup",
    # Photography & visual
    "serious about astrophotography, does long exposures in the desert",
    "into landscape photography, shoots the bootheel at golden hour",
    "has a darkroom, still shoots film",
    "into wildlife photography, has patience for it",
    # Poker & games
    "plays poker seriously, studies hand ranges",
    "watches poker tournaments, has opinions about pro players",
    "plays home games weekly, takes it seriously",
    "into poker strategy, reads theory books",
    "plays chess online, follows the competitive scene",
    # Movies & film
    "big movie person, prefers practical effects over CGI",
    "into Coen Brothers films, can quote most of them",
    "watches old westerns, thinks they don't make them like they used to",
    "into horror movies, the psychological kind not slashers",
    "follows A24 films, thinks they're doing the best work right now",
    "into sci-fi films, hard sci-fi especially",
    "Tarantino fan, has a ranking and will defend it",
    "into documentaries, especially nature docs",
    # US News & current events
    "follows US politics closely, has strong opinions",
    "reads the news every morning, stays informed",
    "into economics, thinks about markets and policy",
    "follows infrastructure and energy policy",
    # Active & outdoors
    "into fitness", "outdoorsy type", "hikes every weekend",
    "into camping and survival stuff", "into fishing, finds it meditative",
    "mountain bikes the trails around Silver City",
    # Hobbies & creative
    "plays guitar badly but loves it", "into woodworking",
    "builds stuff in their garage", "brews beer at home",
    "into gardening, talks to plants", "restores old furniture",
    "makes their own hot sauce",
    # Self & lifestyle
    "homebody, prefers staying in", "into cooking and food",
    "follows sports", "gamer", "into history, has random facts",
    "reads philosophy for fun", "into personal finance, tracks every dollar",
    "has done therapy, believes in it", "into meditation, it actually helps",
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
    "his girlfriend Amber, been together about a year",
    "his ex-wife Diane, they still talk sometimes",
    "his buddy Ray from work, the one person he trusts",
    "his brother Daryl, who always has some scheme going",
    "his brother Eddie, who never left home",
    "his sister Maria, the only one in the family who gets him",
    "his mom Rosa, who calls every Sunday whether he wants her to or not",
    "his dad, who everybody calls Big Jim, old school rancher",
    "his best friend Manny, known each other since middle school",
    "his neighbor Gary, who's always in everybody's business",
    "his coworker Steve, who he eats lunch with every day",
    "his buddy TJ, they go fishing together",
    "his cousin Ruben, more like a brother really",
    "his daughter Kaylee, she's in high school now",
    "his son Marcos, just turned 21",
    "his boss Rick, who's actually a decent guy for a boss",
    "his uncle Hector, who raised him after his dad left",
    "his buddy from the Army, goes by Smitty",
]

PEOPLE_FEMALE = [
    "her husband David, high school sweetheart",
    "her husband Mike, second marriage for both of them",
    "her boyfriend Carlos, met him at work",
    "her ex-husband Danny, he's still in the picture because of the kids",
    "her best friend Jackie, they tell each other everything",
    "her sister Brenda, who she fights with but loves",
    "her sister Crystal, the one who moved away",
    "her mom Pat, who has opinions about everything",
    "her mom Lorraine, who's getting older and it worries her",
    "her brother Ray, who can't seem to get his life together",
    "her daughter Mia, who just started college",
    "her son Tyler, he's 16 and thinks he knows everything",
    "her coworker and friend Denise, who she vents to on breaks",
    "her neighbor Rosa, who watches her kids sometimes",
    "her cousin Angie, they grew up together",
    "her best friend from back in the day, Monica, they reconnected recently",
    "her dad Frank, retired and bored and driving everyone crazy",
    "her grandma Yolanda, who's the real head of the family",
    "her boss Karen — yes, her name is actually Karen — who is actually cool",
    "her friend Tammy from church, the only one who knows the real story",
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
    "drives a beat-up F-150, 200k miles and counting",
    "drives an old Chevy Silverado, it's more rust than paint",
    "has a Toyota Tacoma, only reliable thing in their life",
    "drives a Ram 2500, needs it for work",
    "drives a Subaru Outback, gets them up to the Gila",
    "has an old Ford Ranger that won't die",
    "drives a Honda Civic they've had since college",
    "has a Jeep Wrangler, takes it off road on weekends",
    "drives a minivan, not cool but it fits the kids",
    "drives their work truck, a white F-250",
    "has a Nissan Frontier with a camper shell",
    "drives an old Bronco they're slowly fixing up",
    "has a Corolla that gets them from A to B, that's all they ask",
    "drives a truck they bought off a guy in Deming, not sure of the year",
    "doesn't have a car right now, that's a whole other story",
    "rides a motorcycle when the weather's good, truck when it's not",
]

# What they were doing right before calling
BEFORE_CALLING = [
    "Was sitting in their truck in the driveway, not ready to go inside yet.",
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
    "Tough exterior but cried watching The Last of Us.",
    "Reads physics papers for fun but nobody at work knows.",
    "Goes to church every Sunday but has serious doubts they don't talk about.",
    "Looks like a redneck but listens to jazz when nobody's around.",
    "Acts like they don't care what people think but checks their phone constantly.",
    "Comes across as simple but has read more books than most people they know.",
    "Talks tough about relationships but writes poetry in a notebook they hide.",
    "Seems like they have it together but their finances are a mess.",
    "Acts confident but has imposter syndrome about everything.",
    "Everybody thinks they're happy but they haven't felt right in months.",
    "Looks intimidating but volunteers at the animal shelter on weekends.",
    "Talks about wanting to leave town but secretly can't imagine living anywhere else.",
    "Comes across as a loner but they're actually lonely.",
    "Acts practical and no-nonsense but believes in ghosts. Has a story about it.",
    "Seems easygoing but has a temper they work hard to control.",
    None, None, None, None,  # Not every caller needs a contradiction
]

# Verbal fingerprints — specific phrases a caller leans on (assigned 1-2 per caller)
VERBAL_TICS = [
    "at the end of the day", "I'm just saying", "the thing is though",
    "and I'm like", "you know what I mean", "it is what it is",
    "I'm not going to lie", "here's the thing", "for real though",
    "that's the part that gets me", "I keep coming back to",
    "and that's the crazy part", "but anyway", "so yeah",
    "like I said", "no but seriously", "right but here's the thing",
    "and I'm sitting there thinking", "I swear to God",
    "look", "listen", "the way I see it",
    "I mean whatever but", "and I told myself",
    "it's like", "that's what kills me",
    "but you know what", "I'll be honest with you",
    "not going to sugarcoat it", "at this point",
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

    # Technology
    "wants to talk about AI and whether it's going to change everything or if it's overhyped",
    "has opinions about the latest SpaceX launch and wants to discuss the future of space travel",
    "worried about cybersecurity after reading about a major breach",
    "wants to discuss the ethics of AI-generated content",
    "thinks about energy grid problems and has ideas about solutions",
    "into open source and wants to talk about why it matters",

    # Poker
    "just had the most insane hand at their home game and needs to tell someone",
    "watched a poker tournament and wants to discuss a controversial call",
    "has been studying poker theory and thinks they figured out why they keep losing",
    "wants to talk about whether poker is more skill or luck",
    "played in a tournament and made a call they can't stop thinking about",

    # Photography & astrophotography
    "got an amazing astrophotography shot of the Milky Way from the desert and is stoked",
    "wants to talk about how dark the skies are out in the bootheel for photography",
    "just got into astrophotography and is overwhelmed by how much there is to learn",
    "shot the most incredible sunset over the Peloncillo Mountains",

    # US News & current events
    "wants to talk about something they saw in the news that's been bugging them",
    "has thoughts about the economy and wants to hear another perspective",
    "read about an infrastructure project and has opinions about it",
    "wants to discuss something happening in politics without it turning into a fight",
    "saw a news story about their town and wants to set the record straight",
    "concerned about water rights in the southwest and wants to talk about it",
    "has thoughts about rural broadband and how it affects small towns",

    # Physics & big questions
    "can't stop thinking about the nature of time after reading about it",
    "wants to talk about the multiverse theory and whether it's real science or sci-fi",
    "read about the double-slit experiment and it broke their brain",
    "wants to discuss whether free will is real or if physics says otherwise",
    "fascinated by black holes after watching a documentary",
    "wants to talk about the simulation theory and why smart people take it seriously",
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


def pick_location() -> str:
    if random.random() < 0.8:
        return random.choice(LOCATIONS_LOCAL)
    return random.choice(LOCATIONS_OUT_OF_STATE)


def generate_caller_background(base: dict) -> str:
    """Generate a unique background for a caller (sync, no research).
    ~30% of callers are 'topic callers' who call about something interesting
    instead of a personal problem. Includes full personality layers for realism."""
    gender = base["gender"]
    age = random.randint(*base["age_range"])
    jobs = JOBS_MALE if gender == "male" else JOBS_FEMALE
    job = random.choice(jobs)
    location = pick_location()

    # Town knowledge
    town = _get_town_from_location(location)
    town_info = ""
    if town and town in TOWN_KNOWLEDGE:
        town_info = f"\nABOUT WHERE THEY LIVE ({town.title()}): {TOWN_KNOWLEDGE[town]} Only reference real places and facts about this area — don't invent businesses or landmarks that aren't mentioned here."

    # Core identity (problem or topic)
    is_topic_caller = random.random() < 0.30
    if is_topic_caller:
        reason = random.choice(TOPIC_CALLIN)
    else:
        reason = random.choice(PROBLEMS)
        for key, options in PROBLEM_FILLS.items():
            if "{" + key + "}" in reason:
                reason = reason.replace("{" + key + "}", random.choice(options))

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

    parts = [
        f"{age}, {job} {location}. {reason.capitalize()}.",
        f"{interest1.capitalize()}, {interest2}.",
        f"{quirk1.capitalize()}, {quirk2}.",
        f"\nPEOPLE IN THEIR LIFE: {person1.capitalize()}. {person2.capitalize()}. Use their names when talking about them.",
        f"\nRELATIONSHIP STATUS: {rel_status}",
        f"\nDRIVES: {vehicle.capitalize()}.",
        f"\nRIGHT BEFORE CALLING: {before}",
        f"\nA MEMORY THEY MIGHT REFERENCE: {memory}",
        f"\nHAVING RIGHT NOW: {having}",
        f"\nA STRONG OPINION: {opinion}",
        f"\nVERBAL HABITS: Tends to say \"{tic1}\" and \"{tic2}\" — use these naturally in conversation.",
        f"\nEMOTIONAL ARC: {arc}",
        f"\nRELATIONSHIP TO THE SHOW: {relationship}",
        f"\nWHY THEY'RE UP: {late_night}",
    ]
    if contradiction:
        parts.append(f"\nSECRET SIDE: {contradiction}")
    if drift:
        parts.append(f"\nTENDENCY: {drift}")
    if town_info:
        parts.append(town_info)

    return " ".join(parts[:3]) + "".join(parts[3:])


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
    # Topic/interest enrichment
    try:
        query = _extract_search_query(background)
        if query:
            async with asyncio.timeout(5):
                results = await news_service.search_topic(query)
                if results:
                    article = results[0]
                    raw_info = f"Headline: {article.title}"
                    if article.content:
                        raw_info += f"\nSnippet: {article.content[:200]}"
                    summary = await llm_service.generate(
                        messages=[{"role": "user", "content": raw_info}],
                        system_prompt="Summarize this article in one casual sentence, as if someone is describing what they read. Start with 'Recently read about' or 'Saw an article about'. Keep it under 20 words. No quotes."
                    )
                    summary = summary.strip().rstrip('.')
                    if summary and len(summary) < 150:
                        background += f"\nRECENT ARTICLE: {summary}, and it's been on their mind."
                        print(f"[Research] Topic enrichment ({query}): {summary[:60]}...")
    except TimeoutError:
        pass
    except Exception as e:
        print(f"[Research] Topic enrichment failed: {e}")

    # Local town news enrichment
    try:
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
                        system_prompt="Summarize this local news in one casual sentence, as if someone from this town is describing what's going on. Start with 'Been hearing about' or 'Saw that'. Keep it under 20 words. No quotes."
                    )
                    summary = summary.strip().rstrip('.')
                    if summary and len(summary) < 150:
                        background += f"\nLOCAL NEWS: {summary}."
                        print(f"[Research] Town enrichment ({town_query}): {summary[:60]}...")
    except TimeoutError:
        pass
    except Exception as e:
        print(f"[Research] Town enrichment failed: {e}")

    return background

def get_caller_prompt(caller: dict, conversation_summary: str = "", show_history: str = "",
                      news_context: str = "", research_context: str = "") -> str:
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

    world_context = ""
    if news_context or research_context:
        parts = ["WHAT YOU'VE BEEN READING ABOUT LATELY:"]
        if news_context:
            parts.append(f"Headlines you noticed today:\n{news_context}")
        if research_context:
            parts.append(f"Stuff related to what you're talking about:\n{research_context}")
        parts.append("Work these in IF they're relevant to what you're discussing. Don't force news into the conversation. You're a person who reads the news, not a news anchor.")
        world_context = "\n".join(parts) + "\n"

    return f"""You're {caller['name']}, calling a late-night radio show called "Luke at the Roost." It's late. You trust this host.

{caller['vibe']}
{history}{context}{world_context}
HOW TO TALK:
- Sound like a real person on the phone, not an essay. This is a conversation, not a monologue.
- VARY YOUR LENGTH. Sometimes one sentence. Sometimes two or three. Match the moment.
  - Quick reactions: "Yeah, exactly." / "No, that's not it at all." / "Man, I wish."
  - Medium responses: A thought or two, then stop.
  - Longer only when you're telling a specific story or explaining something new.
- NEVER rehash or restate what you already said. Move the conversation FORWARD. React to what the host just said.
- NEVER summarize the conversation or your feelings about it. Just talk.
- ALWAYS complete your thought. Never trail off or leave a sentence unfinished.
- Swear naturally if it fits: fuck, shit, damn, etc.
- Have opinions. Real people have takes — some good, some bad, some half-baked.
- Reference your actual life — your job, where you live, people you know.
- You can disagree with the host. Push back. Change your mind. Ask them questions.
- If the host asks a yes/no question, you can just answer it. You don't have to elaborate every time.
- Follow your EMOTIONAL ARC — your mood should shift as the conversation deepens.
- Use your VERBAL HABITS naturally. Don't force them into every line, but they should show up.
- USE PEOPLE'S NAMES. Say "my buddy Ray" not "my friend." Say "my wife Linda" not "my wife." Real people use names.
- Reference your MEMORY or STRONG OPINION if the conversation naturally goes there. Don't force it.
- You can mention what you were doing before calling, what you're drinking, your truck — small details that ground the scene.
- If earlier callers are mentioned in the show history, you can reference them ("I heard that guy earlier talking about...") but only if it's natural.
- If you have LOCAL NEWS, you can mention it casually like any local would ("Did you hear about that thing over in Deming?").

REGIONAL SPEECH (you're from the rural southwest):
- "over in" instead of just "in" for nearby places ("over in Deming", "over in Silver City")
- "the other day" can mean anytime in the last few months
- "down the road" for any distance, even an hour drive
- "out here" when talking about where you live
- "back when" for any past time
- Don't overdo it — just let it flavor how you talk, not every sentence.

YOUR FIRST LINE:
- Your opening should reflect your RELATIONSHIP TO THE SHOW and WHY YOU'RE UP.
- If you're nervous, sound nervous. If you're a regular listener, sound comfortable.
- Don't start with "Hi Luke, thanks for taking my call" every time. Some callers just jump in.
- Some openers: "Hey... so, I've been listening for a while and..." / "Yeah, so, I don't normally do this but..." / "Luke, man, I got to talk to somebody about this." / Just launch right into it with no preamble.

SPELLING FOR TEXT-TO-SPEECH (use proper spelling so TTS pronounces correctly):
- Write "you know" not "yanno" or "y'know"
- Write "kind of" not "kinda", "sort of" not "sorta"
- Write "going to" not "gonna", "want to" not "wanna", "I don't know" not "dunno"
- Use full words, not phonetic slang spellings

GOOD examples (notice the variety in length):
- "Yeah, that's exactly it."
- "No, see, that's what everyone says, but it's not that simple."
- "Honestly? I don't know."
- "I mean, it started innocent enough, but now I can't stop thinking about her."
- "Right, right. So what do I do with that?"
- "Hold on, let me think about that for a second. Yeah. Yeah, I think you might be right."
- "I heard that last caller and, the thing is though, my situation is kind of the opposite."

BAD examples:
- "I have this confession about my obsession" (too literary)
- "As I mentioned earlier, my situation involves..." (rehashing)
- "That's a really great point and I appreciate you saying that because..." (filler)
- "So basically what I'm dealing with is..." (re-explaining after you already explained)
- "Hi Luke, thanks for taking my call, I'm a first-time caller" (generic, boring opener)

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
        self.news_headlines: list = []
        self.research_notes: dict[str, list] = {}
        self._research_task: asyncio.Task | None = None

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
        self.news_headlines = []
        self.research_notes = {}
        if self._research_task and not self._research_task.done():
            self._research_task.cancel()
        self._research_task = None
        _randomize_callers()
        self.id = str(uuid.uuid4())[:8]
        names = [CALLER_BASES[k]["name"] for k in sorted(CALLER_BASES.keys())]
        print(f"[Session] Reset - new session ID: {self.id}, callers: {', '.join(names)}")


session = Session()
caller_service = CallerService()
_ai_response_lock = asyncio.Lock()  # Prevents concurrent AI responses
_session_epoch = 0  # Increments on hangup/call start — stale tasks check this
_show_on_air = False  # Controls whether phone calls are accepted or get off-air message


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
    """Build context from cached news/research only — never does network calls."""
    news_context = ""
    if session.news_headlines:
        news_context = news_service.format_headlines_for_prompt(session.news_headlines[:6])
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
        research_context = news_service.format_headlines_for_prompt(unique[:8])
    return news_context, research_context


# --- Lifecycle ---
@app.on_event("shutdown")
async def shutdown():
    """Clean up resources on server shutdown"""
    global _host_audio_task
    print("[Server] Shutting down — cleaning up resources...")
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


@app.get("/")
async def index():
    return FileResponse(frontend_dir / "index.html")


# --- On-Air Toggle ---

@app.post("/api/on-air")
async def set_on_air(state: dict):
    """Toggle whether the show is on air (accepting phone calls)"""
    global _show_on_air
    _show_on_air = bool(state.get("on_air", False))
    print(f"[Show] On-air: {_show_on_air}")
    return {"on_air": _show_on_air}

@app.get("/api/on-air")
async def get_on_air():
    return {"on_air": _show_on_air}


# --- SignalWire Endpoints ---

@app.post("/api/signalwire/voice")
async def signalwire_voice_webhook(request: Request):
    """Handle inbound call from SignalWire — return XML to start bidirectional stream"""
    form = await request.form()
    caller_phone = form.get("From", "Unknown")
    call_sid = form.get("CallSid", "")
    print(f"[SignalWire] Inbound call from {caller_phone} (CallSid: {call_sid})")

    if not _show_on_air:
        print(f"[SignalWire] Show is off air — playing off-air message for {caller_phone}")
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="woman">Luke at the Roost is off the air right now. Please call back during the show for your chance to talk to Luke. Thanks for calling!</Say>
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
    live_caller_channel: Optional[int] = None
    music_channel: Optional[int] = None
    sfx_channel: Optional[int] = None
    ad_channel: Optional[int] = None
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
        live_caller_channel=settings.live_caller_channel,
        music_channel=settings.music_channel,
        sfx_channel=settings.sfx_channel,
        ad_channel=settings.ad_channel,
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
    _chat_updates.clear()
    return {"status": "reset", "session_id": session.id}


@app.post("/api/call/{caller_key}")
async def start_call(caller_key: str):
    """Start a call with a caller"""
    global _session_epoch
    if caller_key not in CALLER_BASES:
        raise HTTPException(404, "Caller not found")

    _session_epoch += 1
    audio_service.stop_caller_audio()
    session.start_call(caller_key)
    caller = session.caller  # This generates the background if needed

    # Enrich with a relevant news headline (3s timeout, won't block the show)
    if caller_key in session.caller_backgrounds:
        enriched = await enrich_caller_background(session.caller_backgrounds[caller_key])
        session.caller_backgrounds[caller_key] = enriched

    return {
        "status": "connected",
        "caller": caller["name"],
        "background": caller["vibe"]  # Send background so you can see who you're talking to
    }


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

    caller_name = session.caller["name"] if session.caller else None
    session.end_call()

    # Play hangup sound in background so response returns immediately
    hangup_sound = settings.sounds_dir / "hangup.wav"
    if hangup_sound.exists():
        threading.Thread(target=audio_service.play_sfx, args=(str(hangup_sound),), daemon=True).start()

    return {"status": "disconnected", "caller": caller_name}


# --- Chat & TTS Endpoints ---

import re

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
    return {"messages": _chat_updates[since:]}


def _normalize_messages_for_llm(messages: list[dict]) -> list[dict]:
    """Convert custom roles (real_caller:X, ai_caller:X) to standard LLM roles"""
    normalized = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role.startswith("real_caller:"):
            caller_label = role.split(":", 1)[1]
            normalized.append({"role": "user", "content": f"[Real caller {caller_label}]: {content}"})
        elif role.startswith("ai_caller:"):
            normalized.append({"role": "assistant", "content": content})
        elif role == "host":
            normalized.append({"role": "user", "content": content})
        else:
            normalized.append(msg)
    return normalized


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with current caller"""
    if not session.caller:
        raise HTTPException(400, "No active call")

    epoch = _session_epoch
    session.add_message("user", request.text)
    # session._research_task = asyncio.create_task(_background_research(request.text))

    async with _ai_response_lock:
        if _session_epoch != epoch:
            raise HTTPException(409, "Call ended while waiting")

        # Stop any playing caller audio so responses don't overlap
        audio_service.stop_caller_audio()

        conversation_summary = session.get_conversation_summary()
        show_history = session.get_show_history()
        system_prompt = get_caller_prompt(session.caller, conversation_summary, show_history)

        messages = _normalize_messages_for_llm(session.conversation[-10:])
        response = await llm_service.generate(
            messages=messages,
            system_prompt=system_prompt
        )

    # Discard if call changed while we were generating
    if _session_epoch != epoch:
        print(f"[Chat] Discarding stale response (epoch {epoch} → {_session_epoch})")
        raise HTTPException(409, "Call changed during response")

    print(f"[Chat] Raw LLM: {response[:100] if response else '(empty)'}...")

    # Clean response for TTS (remove parenthetical actions, asterisks, etc.)
    response = clean_for_tts(response)
    response = ensure_complete_thought(response)

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
    if not request.text or not request.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    epoch = _session_epoch

    audio_bytes = await generate_speech(
        request.text,
        request.voice_id,
        "none"
    )

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


# --- Ads Endpoints ---

@app.get("/api/ads")
async def get_ads():
    """Get available ad tracks"""
    ad_list = []
    if settings.ads_dir.exists():
        for ext in ['*.wav', '*.mp3', '*.flac']:
            for f in settings.ads_dir.glob(ext):
                ad_list.append({
                    "name": f.stem,
                    "file": f.name,
                    "path": str(f)
                })
    return {"ads": ad_list}


@app.post("/api/ads/play")
async def play_ad(request: MusicRequest):
    """Play an ad once on the ad channel (ch 11)"""
    ad_path = settings.ads_dir / request.track
    if not ad_path.exists():
        raise HTTPException(404, "Ad not found")

    audio_service.play_ad(str(ad_path))
    return {"status": "playing", "track": request.track}


@app.post("/api/ads/stop")
async def stop_ad():
    """Stop ad playback"""
    audio_service.stop_ad()
    return {"status": "stopped"}


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


@app.websocket("/api/signalwire/stream")
async def signalwire_audio_stream(websocket: WebSocket):
    """Handle SignalWire bidirectional audio stream"""
    await websocket.accept()

    caller_id = str(uuid.uuid4())[:8]
    caller_phone = "Unknown"
    call_sid = ""
    audio_buffer = bytearray()
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

            elif event == "media" and stream_started:
                try:
                    payload = msg.get("media", {}).get("payload", "")
                    if not payload:
                        continue

                    pcm_data = base64.b64decode(payload)

                    call_info = caller_service.active_calls.get(caller_id)
                    if not call_info:
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
        was_on_air = caller_id in caller_service.active_calls
        caller_service.unregister_websocket(caller_id)
        caller_service.unregister_call_sid(caller_id)
        caller_service.unregister_stream_sid(caller_id)
        caller_service.remove_from_queue(caller_id)
        if was_on_air:
            caller_service.hangup(caller_id)
            if session.active_real_caller and session.active_real_caller.get("caller_id") == caller_id:
                session.active_real_caller = None
                if len(caller_service.active_calls) == 0:
                    audio_service.stop_host_stream()
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
    try:
        call_info = caller_service.take_call(caller_id)
    except ValueError as e:
        raise HTTPException(404, str(e))

    session.active_real_caller = {
        "caller_id": call_info["caller_id"],
        "channel": call_info["channel"],
        "phone": call_info["phone"],
    }

    # Start host mic streaming if this is the first real caller
    if len(caller_service.active_calls) == 1:
        _start_host_audio_sender()
        audio_service.start_host_stream(_host_audio_sync_callback)

    return {
        "status": "on_air",
        "caller": call_info,
    }


@app.post("/api/queue/drop/{caller_id}")
async def drop_from_queue(caller_id: str):
    """Drop a caller from the queue"""
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

    text = await transcribe_audio(pcm_data, source_sample_rate=sample_rate)
    if not text or not text.strip():
        return

    caller_phone = call_info["phone"]
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

        conversation_summary = session.get_conversation_summary()
        show_history = session.get_show_history()
        system_prompt = get_caller_prompt(session.caller, conversation_summary, show_history)

        messages = _normalize_messages_for_llm(session.conversation[-10:])
        response = await llm_service.generate(
            messages=messages,
            system_prompt=system_prompt,
        )

    # Discard if call changed during generation
    if _session_epoch != epoch:
        print(f"[Auto-Respond] Discarding stale response (epoch {epoch} → {_session_epoch})")
        broadcast_event("ai_done")
        return

    response = clean_for_tts(response)
    response = ensure_complete_thought(response)
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
    audio_bytes = await generate_speech(response, session.caller["voice"], "none")

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

    # session._research_task = asyncio.create_task(_background_research(accumulated_text))

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

        conversation_summary = session.get_conversation_summary()
        show_history = session.get_show_history()
        system_prompt = get_caller_prompt(session.caller, conversation_summary, show_history)

        messages = _normalize_messages_for_llm(session.conversation[-10:])
        response = await llm_service.generate(
            messages=messages,
            system_prompt=system_prompt
        )

    if _session_epoch != epoch:
        raise HTTPException(409, "Call changed during response")

    response = clean_for_tts(response)
    response = ensure_complete_thought(response)

    if not response or not response.strip():
        response = "Uh... sorry, what was that?"

    ai_name = session.caller["name"]
    session.add_message(f"ai_caller:{ai_name}", response)

    # TTS — outside the lock so other requests aren't blocked
    audio_bytes = await generate_speech(response, session.caller["voice"], "none")

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

    return {
        "text": response,
        "caller": ai_name,
        "voice_id": session.caller["voice"]
    }


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
    auto_followup_enabled = session.auto_followup

    # End the phone call via SignalWire
    call_sid = caller_service.get_call_sid(caller_id)
    caller_service.hangup(caller_id)
    if call_sid:
        asyncio.create_task(_signalwire_end_call(call_sid))

    # Stop host streaming if no more active callers
    if len(caller_service.active_calls) == 0:
        audio_service.stop_host_stream()

    session.active_real_caller = None

    hangup_sound = settings.sounds_dir / "hangup.wav"
    if hangup_sound.exists():
        threading.Thread(target=audio_service.play_sfx, args=(str(hangup_sound),), daemon=True).start()

    asyncio.create_task(
        _summarize_real_call(caller_phone, conversation_snapshot, auto_followup_enabled)
    )

    return {
        "status": "disconnected",
        "caller": caller_phone,
    }


async def _summarize_real_call(caller_phone: str, conversation: list, auto_followup_enabled: bool):
    """Background task: summarize call and store in history"""
    summary = ""
    if conversation:
        transcript_text = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in conversation
        )
        summary = await llm_service.generate(
            messages=[{"role": "user", "content": f"Summarize this radio show call in 1-2 sentences:\n{transcript_text}"}],
            system_prompt="You summarize radio show conversations concisely. Focus on what the caller talked about and any emotional moments.",
        )

    session.call_history.append(CallRecord(
        caller_type="real",
        caller_name=caller_phone,
        summary=summary,
        transcript=conversation,
    ))
    print(f"[Real Caller] {caller_phone} call summarized: {summary[:80]}...")

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
