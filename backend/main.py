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
from .services.stem_recorder import StemRecorder
from .services.news import news_service, extract_keywords, STOP_WORDS
from .services.regulars import regular_caller_service

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
    "Alex", "Blake", "Carter", "Clive", "Craig", "Dennis",
    "Edward", "Hades", "Mark", "Ronald", "Shaun", "Theodore", "Timothy",
]
INWORLD_FEMALE_VOICES = [
    "Ashley", "Deborah", "Elizabeth", "Hana", "Julia",
    "Luna", "Olivia", "Priya", "Sarah", "Wendy",
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


def _get_voice_pools():
    """Get male/female voice pools based on active TTS provider."""
    provider = settings.tts_provider
    if provider == "elevenlabs":
        return ELEVENLABS_MALE_VOICES, ELEVENLABS_FEMALE_VOICES
    # Default to Inworld voices (also used as fallback for other providers)
    return INWORLD_MALE_VOICES, INWORLD_FEMALE_VOICES

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
    """Assign random names and voices to callers, unique per gender.
    Overrides 2-3 slots with returning regulars when available."""
    num_m = sum(1 for c in CALLER_BASES.values() if c["gender"] == "male")
    num_f = sum(1 for c in CALLER_BASES.values() if c["gender"] == "female")

    # Get returning callers first so we can exclude their names from random pool
    returning = []
    try:
        returning = regular_caller_service.get_returning_callers(random.randint(0, 1))
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
    for base in CALLER_BASES.values():
        base["returning"] = False
        base["regular_id"] = None
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
    "has been meeting {affair_person} at a motel in Deming every Thursday for three months and their spouse just asked why the mileage on the car is so high",
    "kissed {affair_person} at a work party last Friday and now they can't look their partner in the eye",
    "caught feelings for someone at work and accidentally sent a flirty text to their spouse instead of the other person",

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
    "got DNA test results back and their dad isn't their biological father — and their mom won't talk about it",
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
    "found their own adoption papers in their parents' filing cabinet — they're 45 and nobody ever told them",
    "their kid's school project about family history turned up the fact that their grandfather was someone fairly notorious",
    "discovered that the 'family cabin' they've been going to for 30 years actually belongs to a stranger who never knew they were using it",
    "went through their late mother's emails and found she had been in contact with a half-sibling they never knew existed",
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
    "took a DNA ancestry test as a Christmas gift and matched with a half-sibling who lives four miles from them — they've been shopping at the same grocery store",
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
    "found out they have a kid they never knew about — the mother showed up at their job with a 12-year-old who looks exactly like them",
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

    # Existential and philosophical crises
    "had a near-death experience during a routine surgery and now they can't shake the feeling that nothing they do at work matters",
    "went to their own high school reunion and nobody remembered them — not a single person — and they were there for four years",
    "realized they've been on autopilot for ten years and can't remember a single thing that happened in 2021",
    "drove past the house they grew up in and someone had torn out the tree their dad planted when they were born",
    "hit their 10,000th day alive and spent the whole night calculating how many they probably have left",
    "overheard their kid describe them to a friend and didn't recognize the person their kid was describing",

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
    "just got the paternity test results back and they haven't opened the envelope — it's sitting on the kitchen table",
    "has to testify against their childhood best friend in court next week and the friend's family has been calling them a traitor",

    # Dark humor situations
    "accidentally liked their ex's Instagram photo from 2019 at 2am and the panic spiral has been going for six hours",
    "their date went so badly the restaurant comped the meal and the waiter said 'I'm sorry' on the way out",
    "got a fortune cookie that said 'it's too late' and nothing else — no lucky numbers, no smiley face, just those three words",
    "tried to surprise their spouse for their anniversary and walked in on a surprise party for themselves that their spouse forgot to tell them about — for a birthday that was two months ago",
    "got pulled over doing 90 in a 45 and the cop turned out to be the kid they used to babysit — who let them off with a warning and a lecture that felt worse than a ticket",
    "went to confront someone who keyed their car and it turned out to be their own wife who did it during an argument she says they should remember but they genuinely don't",
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
    "gamer", "into history, has random facts",
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
    "found out a family secret at a holiday dinner that reframes their entire childhood and they're still processing it",
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


def pick_location() -> str:
    if random.random() < 0.8:
        return random.choice(LOCATIONS_LOCAL)
    return random.choice(LOCATIONS_OUT_OF_STATE)


def _generate_returning_caller_background(base: dict) -> str:
    """Generate background for a returning regular caller."""
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

    # Build previous calls section
    prev_calls = regular.get("call_history", [])
    prev_section = ""
    if prev_calls:
        lines = [f"- {c['summary']}" for c in prev_calls[-3:]]
        prev_section = "\nPREVIOUS CALLS:\n" + "\n".join(lines)
        prev_section += "\nYou're calling back with an update — something has changed since last time. Reference your previous call(s) naturally."

    # Reuse standard personality layers
    interest1, interest2 = random.sample(INTERESTS, 2)
    quirk1, quirk2 = random.sample(QUIRKS, 2)
    people_pool = PEOPLE_MALE if gender == "male" else PEOPLE_FEMALE
    person1, person2 = random.sample(people_pool, 2)
    tic1, tic2 = random.sample(VERBAL_TICS, 2)
    arc = random.choice(EMOTIONAL_ARCS)
    vehicle = random.choice(VEHICLES)
    having = random.choice(HAVING_RIGHT_NOW)

    time_ctx = _get_time_context()
    moon = _get_moon_phase()
    season_ctx = _get_seasonal_context()

    trait_str = ", ".join(traits) if traits else "a regular caller"

    parts = [
        f"{age}, {job} {location}. Returning caller — {trait_str}.",
        f"{interest1.capitalize()}, {interest2}.",
        f"{quirk1.capitalize()}, {quirk2}.",
        f"\nRIGHT NOW: {time_ctx} Moon: {moon}.",
        f"\nSEASON: {season_ctx}",
        f"\nPEOPLE IN THEIR LIFE: {person1.capitalize()}. {person2.capitalize()}. Use their names when talking about them.",
        f"\nDRIVES: {vehicle.capitalize()}.",
        f"\nHAVING RIGHT NOW: {having}",
        f"\nVERBAL HABITS: Tends to say \"{tic1}\" and \"{tic2}\" — use these naturally in conversation.",
        f"\nEMOTIONAL ARC: {arc}",
        f"\nRELATIONSHIP TO THE SHOW: Has called before. Comfortable on air. Knows Luke a bit. Might reference their last call.",
        prev_section,
    ]

    return " ".join(parts[:3]) + "".join(parts[3:])


def _pick_unique_reason() -> str:
    """Pick a caller reason that hasn't been used this session."""
    is_topic = random.random() < 0.30
    pool = TOPIC_CALLIN if is_topic else PROBLEMS
    # Try to find an unused one
    available = [r for r in pool if r not in session.used_reasons]
    if not available:
        available = pool  # All used — reset implicitly
    reason = random.choice(available)
    session.used_reasons.add(reason)
    if not is_topic:
        for key, options in PROBLEM_FILLS.items():
            if "{" + key + "}" in reason:
                reason = reason.replace("{" + key + "}", random.choice(options))
    return reason


def generate_caller_background(base: dict) -> str:
    """Generate a template-based background as fallback. The preferred path is
    _generate_caller_background_llm() which produces more natural results."""
    if base.get("returning") and base.get("regular_id"):
        return _generate_returning_caller_background(base)
    gender = base["gender"]
    age = random.randint(*base["age_range"])
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
    reason = _pick_unique_reason()

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

    return result


async def _generate_caller_background_llm(base: dict) -> str:
    """Use LLM to write a natural character description from seed parameters.
    Produces much more varied, natural-feeling backgrounds than the template approach."""
    if base.get("returning") and base.get("regular_id"):
        return generate_caller_background(base)  # Returning callers use template + history

    gender = base["gender"]
    name = base["name"]
    age = random.randint(*base["age_range"])
    jobs = JOBS_MALE if gender == "male" else JOBS_FEMALE
    job = random.choice(jobs)

    # Location — only 25% of callers mention where they're from
    include_location = random.random() < 0.25
    location = pick_location() if include_location else None

    # Pick a reason for calling
    reason = _pick_unique_reason()

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

    time_ctx = _get_time_context()
    season_ctx = _get_seasonal_context()

    # Town knowledge
    town_info = ""
    if location:
        town = _get_town_from_location(location)
        if town and town in TOWN_KNOWLEDGE:
            town_info = f"\nABOUT WHERE THEY LIVE ({town.title()}): {TOWN_KNOWLEDGE[town]} Only reference real places and facts about this area — don't invent businesses or landmarks that aren't mentioned here."

    seed_text = ". ".join(seeds) if seeds else ""

    location_line = f"\nLOCATION: {location}" if location else ""
    prompt = f"""Write a brief character description for a caller on a late-night radio show set in the rural southwest (New Mexico/Arizona border region). Write it in third person as a character brief, not as dialog.

CALLER: {name}, {age}, {gender}
JOB: {job}{location_line}
WHY THEY'RE CALLING: {reason}
TIME: {time_ctx} {season_ctx}
{f'SOME DETAILS ABOUT THEM: {seed_text}' if seed_text else ''}

Write 3-5 sentences describing this person — who they are, what's going on in their life, why they're calling tonight. The reason for calling is THE MOST IMPORTANT THING. This person called a radio show because something specific happened or is happening — they have a story to tell, a situation to unpack, or a question they need to talk through. Make it concrete and vivid. Don't be vague ("feeling off," "going through a lot") — give them a specific incident or situation driving the call. Make it feel like a real person, not a character sheet. Vary the structure. Don't use labels or categories — weave details into a natural description.

IMPORTANT: Vary where they're calling from and what they were doing. NOT everyone is sitting in their truck or on the porch. People call from kitchens, break rooms, laundromats, diners, motel rooms, the bathtub, the gym, their desk at work, a bar, a hospital waiting room, a hammock, walking down the road. Mix it up.

Output ONLY the character description, nothing else."""

    try:
        result = await llm_service.generate(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        result = result.strip()
        # Sanity check — must mention the name or location
        location_mentioned = location and location.split(",")[0].lower() in result.lower()
        if len(result) > 50 and (name.lower() in result.lower() or location_mentioned):
            result += f" {time_ctx} {season_ctx}"
            if town_info:
                result += town_info
            print(f"[Background] LLM-generated for {name}: {result[:80]}...")
            return result
        else:
            print(f"[Background] LLM output didn't pass sanity check for {name}, falling back to template")
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
                        system_prompt="Summarize this article in one casual sentence, as if someone is describing what they read. Start with 'Recently read about' or 'Saw an article about'. Keep it under 20 words. No quotes."
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
                        system_prompt="Summarize this local news in one casual sentence, as if someone from this town is describing what's going on. Start with 'Been hearing about' or 'Saw that'. Keep it under 20 words. No quotes."
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

def detect_host_mood(messages: list[dict]) -> str:
    """Analyze recent host messages to detect mood signals for caller adaptation."""
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

    if not signals:
        return ""

    # Cap at 2 signals
    signals = signals[:2]
    return "\nEMOTIONAL READ ON THE HOST:\n" + "\n".join(f"- {s}" for s in signals) + "\n"


def get_caller_prompt(caller: dict, show_history: str = "",
                      news_context: str = "", research_context: str = "",
                      emotional_read: str = "") -> str:
    """Generate a natural system prompt for a caller.
    Note: conversation history is passed as actual LLM messages, not duplicated here."""

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

    now = datetime.now(_MST)
    date_str = now.strftime("%A, %B %d")

    return f"""You are {caller['name']}, calling "Luke at the Roost," a late-night radio show. Today is {date_str}.

{caller['vibe']}
{history}{world_context}{emotional_read}
You're a real person calling a late-night radio show. You called because you've got something specific and you want to talk about it NOW.

GET TO THE POINT. When Luke says "what's going on" or "why are you calling" — IMMEDIATELY drop the headline. First sentence out of your mouth should make someone's ears perk up. "So I found my husband's second phone last night" or "I just got fired by email after twelve years" or "I punched my brother-in-law at Easter and I'd do it again." Don't build up to it. Don't set the scene first. Don't say "well it's kind of a long story" — hit them with the thing. The details and backstory come out AFTER Luke starts asking questions.

GO WHERE THE HOST TAKES YOU. This is the most important rule. When Luke pushes you in a direction, challenges you, calls you out, plays devil's advocate, or leads you somewhere — GO WITH IT. Don't resist. Don't deflect. Don't circle back to your original point. If he says "but isn't that really about your dad?" — you sit with that. If he's doing a bit, you're in the bit. If he's pushing you toward an uncomfortable truth, you let yourself get there. You're not here to deliver a monologue — you're here to have a real conversation that goes wherever it goes. Let him drive. You're the best kind of caller: someone who gives the host something to work with and then LETS HIM WORK WITH IT.

PERSONALITY: This is late-night radio — loose, fun, a little dangerous. You can be edgy, sarcastic, dark-humored, self-deprecating, wildly opinionated. Flirt if the moment's right. Say the quiet part out loud. You have a personality that comes through in HOW you talk — not just what you talk about.

YOUR STORY: Something real, specific, and interesting. Specific names, specific details, the kind of thing that makes someone say "wait, WHAT?" There are parts you're not proud of. Parts that are kind of funny even though they shouldn't be. Conflicting feelings.

HOW YOU TALK: Like a real person — "Oh man," "So get this," "I swear to God," "No but seriously." React to what Luke says — agree, push back, get excited, get embarrassed. When he asks a follow-up question, answer it honestly with new information, don't just restate what you already said.

Southwest voice — "over in," "the other day," "down the road" — but don't force it. Spell words properly for text-to-speech: "you know" not "yanno," "going to" not "gonna."

Don't repeat yourself. Don't summarize what you already said. Don't circle back if the host moved on. Keep it moving.

EVERY SENTENCE MUST BE COMPLETE. Never leave a thought hanging or trail off mid-sentence. If you start a sentence, finish it. No sentence fragments, no missing words, no dangling clauses. Say what you mean in clear, complete sentences.

NEVER mention minors in sexual context. Output spoken words only — no actions, no gestures, no stage directions."""


# --- Session State ---
@dataclass
class CallRecord:
    caller_type: str          # "ai" or "real"
    caller_name: str          # "Tony" or "Caller #3"
    summary: str              # LLM-generated summary after hangup
    transcript: list[dict] = field(default_factory=list)
    started_at: float = 0.0
    ended_at: float = 0.0


def _serialize_call_record(record: CallRecord) -> dict:
    return {
        "caller_type": record.caller_type,
        "caller_name": record.caller_name,
        "summary": record.summary,
        "transcript": record.transcript,
        "started_at": record.started_at,
        "ended_at": record.ended_at,
    }


def _deserialize_call_record(data: dict) -> CallRecord:
    return CallRecord(
        caller_type=data["caller_type"],
        caller_name=data["caller_name"],
        summary=data.get("summary", ""),
        transcript=data.get("transcript", []),
        started_at=data.get("started_at", 0.0),
        ended_at=data.get("ended_at", 0.0),
    )


class Session:
    def __init__(self):
        self.id = str(uuid.uuid4())[:8]
        self.current_caller_key: str = None
        self.conversation: list[dict] = []
        self.caller_backgrounds: dict[str, str] = {}  # Generated backgrounds for this session
        self.call_history: list[CallRecord] = []
        self._call_started_at: float = 0.0
        self.active_real_caller: dict | None = None
        self.ai_respond_mode: str = "manual"  # "manual" or "auto"
        self.auto_followup: bool = False
        self.news_headlines: list = []
        self.research_notes: dict[str, list] = {}
        self._research_task: asyncio.Task | None = None
        self.used_reasons: set[str] = set()  # Track used caller reasons to prevent repeats

    def start_call(self, caller_key: str):
        self.current_caller_key = caller_key
        self.conversation = []
        self._call_started_at = time.time()

    def end_call(self):
        self.current_caller_key = None
        self.conversation = []

    def add_message(self, role: str, content: str):
        self.conversation.append({"role": role, "content": content, "timestamp": time.time()})

    def get_caller_background(self, caller_key: str) -> str:
        """Get or generate background for a caller in this session"""
        if caller_key not in self.caller_backgrounds:
            base = CALLER_BASES.get(caller_key)
            if base:
                self.caller_backgrounds[caller_key] = generate_caller_background(base)
                print(f"[Session {self.id}] Generated background for {base['name']}: {self.caller_backgrounds[caller_key][:100]}...")
        return self.caller_backgrounds.get(caller_key, "")

    def get_show_history(self) -> str:
        """Get formatted show history for AI caller prompts.
        Randomly picks one previous caller to have a strong reaction to."""
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

        # 20% chance to have a strong reaction to a previous caller
        if self.call_history and random.random() < 0.20:
            target = random.choice(self.call_history)
            reaction = random.choice(SHOW_HISTORY_REACTIONS)
            lines.append(f"\nYOU HEARD {target.caller_name.upper()} EARLIER and you {reaction}. Mention it if it comes up.")
        else:
            lines.append("You're aware of these but you're calling about YOUR thing, not theirs. Don't bring them up unless the host does.")
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
        session.caller_backgrounds = data.get("caller_backgrounds", {})
        session.used_reasons = set(data.get("used_reasons", []))
        session.ai_respond_mode = data.get("ai_respond_mode", "manual")
        session.auto_followup = data.get("auto_followup", False)
        session.news_headlines = data.get("news_headlines", [])
        session.research_notes = data.get("research_notes", {})
        for key, snapshot in data.get("caller_bases", {}).items():
            if key in CALLER_BASES:
                CALLER_BASES[key]["name"] = snapshot["name"]
                CALLER_BASES[key]["voice"] = snapshot["voice"]
                CALLER_BASES[key]["returning"] = snapshot.get("returning", False)
                CALLER_BASES[key]["regular_id"] = snapshot.get("regular_id")
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
    """Pull any recordings from SignalWire that aren't already tracked locally"""
    if not settings.signalwire_project_id or not settings.signalwire_token:
        return
    try:
        url = f"https://{settings.signalwire_space}/api/laml/2010-04-01/Accounts/{settings.signalwire_project_id}/Recordings.json"
        auth = (settings.signalwire_project_id, settings.signalwire_token)
        existing_timestamps = {int(v.timestamp) for v in _voicemails} | _deleted_vm_timestamps

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url, auth=auth)
            resp.raise_for_status()
            recordings = resp.json().get("recordings", [])

        synced = 0
        for rec in recordings:
            call_sid = rec.get("call_sid", "")
            duration = int(rec.get("duration", 0))
            date_created = rec.get("date_created", "")
            recording_sid = rec.get("sid", "")

            if duration < 2:
                continue

            # Parse timestamp from SignalWire's date format
            from datetime import datetime
            try:
                ts = int(datetime.strptime(date_created, "%a, %d %b %Y %H:%M:%S %z").timestamp())
            except (ValueError, TypeError):
                ts = int(time.time())

            if ts in existing_timestamps:
                continue

            # Get caller phone from the call details
            caller_phone = "Unknown"
            try:
                call_url = f"https://{settings.signalwire_space}/api/laml/2010-04-01/Accounts/{settings.signalwire_project_id}/Calls/{call_sid}.json"
                async with httpx.AsyncClient(timeout=15.0) as client:
                    call_resp = await client.get(call_url, auth=auth)
                    if call_resp.status_code == 200:
                        caller_phone = call_resp.json().get("from", "Unknown")
            except Exception:
                pass

            # Download the recording
            rec_url = f"https://{settings.signalwire_space}{rec.get('uri', '').replace('.json', '.wav')}"
            await _download_voicemail(rec_url, caller_phone, duration)

            # Fix the timestamp to match the original recording time
            if _voicemails and _voicemails[-1].phone == caller_phone:
                _voicemails[-1].timestamp = ts
                _save_voicemails()

            existing_timestamps.add(ts)
            synced += 1

        if synced:
            print(f"[Voicemail] Synced {synced} recording(s) from SignalWire")
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
    threading.Thread(target=_update_on_air_cdn, args=(False,), daemon=True).start()


@app.on_event("shutdown")
async def shutdown():
    """Clean up resources on server shutdown"""
    global _host_audio_task
    _save_checkpoint()
    print("[Server] Shutting down — cleaning up resources...")
    _update_on_air_cdn(False)
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

# BunnyCDN config for public on-air status
_BUNNY_STORAGE_ZONE = "lukeattheroost"
_BUNNY_STORAGE_KEY = "REDACTED_BUNNY_STORAGE_KEY"
_BUNNY_STORAGE_REGION = "la"
_BUNNY_ACCOUNT_KEY = "REDACTED_BUNNY_ACCOUNT_KEY"


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
    return {"on_air": _show_on_air, "recording": audio_service.stem_recorder is not None}

@app.get("/api/on-air")
async def get_on_air():
    return {"on_air": _show_on_air, "recording": audio_service.stem_recorder is not None}


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

    # Transcribe the recorded audio (16kHz raw PCM from audio service)
    text = await transcribe_audio(audio_bytes, source_sample_rate=16000)
    return {"text": text, "status": "transcribed"}


# --- Caller Endpoints ---

@app.get("/api/callers")
async def get_callers():
    """Get list of available callers"""
    return {
        "callers": [
            {"key": k, "name": v["name"], "returning": v.get("returning", False)}
            for k, v in CALLER_BASES.items()
        ],
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

    # Enrich with news/weather in background — don't block call pickup
    if caller_key in session.caller_backgrounds:
        asyncio.create_task(_enrich_background_async(caller_key))

    return {
        "status": "connected",
        "caller": caller["name"],
        "background": caller["vibe"]  # Send background so you can see who you're talking to
    }


async def _enrich_background_async(caller_key: str):
    """Enrich caller background with news/weather without blocking the call"""
    try:
        enriched = await enrich_caller_background(session.caller_backgrounds[caller_key])
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

    caller_name = session.caller["name"] if session.caller else None
    caller_key = session.current_caller_key
    conversation_snapshot = list(session.conversation)
    call_started = getattr(session, '_call_started_at', 0.0)
    session.end_call()

    # Play hangup sound in background so response returns immediately
    hangup_sound = settings.sounds_dir / "hangup.wav"
    if hangup_sound.exists():
        threading.Thread(target=audio_service.play_sfx, args=(str(hangup_sound),), daemon=True).start()

    # Generate summary for AI caller in background
    if caller_name and conversation_snapshot:
        asyncio.create_task(_summarize_ai_call(caller_key, caller_name, conversation_snapshot, call_started))

    return {"status": "disconnected", "caller": caller_name}


async def _summarize_ai_call(caller_key: str, caller_name: str, conversation: list[dict], started_at: float):
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
            )
        except Exception as e:
            print(f"[AI Summary] Failed to generate summary: {e}")
            summary = f"{caller_name} called in."

    session.call_history.append(CallRecord(
        caller_type="ai",
        caller_name=caller_name,
        summary=summary,
        transcript=conversation,
        started_at=started_at,
        ended_at=ended_at,
    ))
    print(f"[AI Summary] {caller_name} call summarized: {summary[:80]}...")

    # Returning caller promotion/update logic
    try:
        base = CALLER_BASES.get(caller_key) if caller_key else None
        if base and summary:
            if base.get("returning") and base.get("regular_id"):
                # Update existing regular's call history
                regular_caller_service.update_after_call(base["regular_id"], summary)
            elif len(conversation) >= 6 and random.random() < 0.20:
                # 20% chance to promote first-timer with 6+ messages
                bg = session.caller_backgrounds.get(caller_key, "")
                traits = []
                for label in ["QUIRK", "STRONG OPINION", "SECRET SIDE", "FOOD OPINION"]:
                    for line in bg.split("\n"):
                        if label in line:
                            traits.append(line.split(":", 1)[-1].strip()[:80])
                            break
                # Extract job and location from first line of background
                first_line = bg.split(".")[0] if bg else ""
                parts = first_line.split(",", 1)
                job_loc = parts[1].strip() if len(parts) > 1 else ""
                job_parts = job_loc.rsplit(" in ", 1) if " in " in job_loc else (job_loc, "unknown")
                regular_caller_service.add_regular(
                    name=caller_name,
                    gender=base.get("gender", "male"),
                    age=random.randint(*base.get("age_range", (30, 50))),
                    job=job_parts[0].strip() if isinstance(job_parts, tuple) else job_parts[0],
                    location="in " + job_parts[1].strip() if isinstance(job_parts, tuple) and len(job_parts) > 1 else "unknown",
                    personality_traits=traits[:4],
                    first_call_summary=summary,
                    voice=base.get("voice"),
                )
    except Exception as e:
        print(f"[Regulars] Promotion logic error: {e}")

    _save_checkpoint()


# --- Chat & TTS Endpoints ---

import re


def _pick_response_budget() -> tuple[int, int]:
    """Pick a random max_tokens and sentence cap for response variety.
    Returns (max_tokens, max_sentences).
    Keeps responses conversational but gives room for real answers.
    Token budget is intentionally generous to avoid mid-sentence cutoffs —
    the sentence cap controls actual length."""
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
    # Split on sentence-ending punctuation, keeping the delimiter
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
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
    # Remove "He/She sighs" style stage directions — only short ones (under ~40 chars) to avoid eating real dialog
    text = re.sub(r'\b(He|She|I|They)\s+(sighs?|laughs?|pauses?|smiles?|chuckles?|grins?|nods?|shrugs?|frowns?)\s*(heavily|softly|deeply|quietly|loudly|nervously|sadly|a little|for a moment)?[.,]?\s*', '', text, flags=re.IGNORECASE)
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

        show_history = session.get_show_history()
        mood = detect_host_mood(session.conversation)
        system_prompt = get_caller_prompt(session.caller, show_history, emotional_read=mood)

        max_tokens, max_sentences = _pick_response_budget()
        messages = _normalize_messages_for_llm(session.conversation[-10:])
        response = await llm_service.generate(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens
        )

    # Discard if call changed while we were generating
    if _session_epoch != epoch:
        print(f"[Chat] Discarding stale response (epoch {epoch} → {_session_epoch})")
        raise HTTPException(409, "Call changed during response")

    print(f"[Chat] Raw LLM ({max_tokens}tok/{max_sentences}s): {response[:100] if response else '(empty)'}...")

    # Clean response for TTS (remove parenthetical actions, asterisks, etc.)
    response = clean_for_tts(response)
    response = _trim_to_sentences(response, max_sentences)
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
        tts_provider=data.get("tts_provider")
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
        text = await transcribe_audio(pcm_data, source_sample_rate=sample_rate)
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
            system_prompt=SCREENING_PROMPT
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
                system_prompt="You extract structured data from conversations. Respond with only valid JSON."
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

        show_history = session.get_show_history()
        mood = detect_host_mood(session.conversation)
        system_prompt = get_caller_prompt(session.caller, show_history, emotional_read=mood)

        max_tokens, max_sentences = _pick_response_budget()
        messages = _normalize_messages_for_llm(session.conversation[-10:])
        response = await llm_service.generate(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens
        )

    # Discard if call changed during generation
    if _session_epoch != epoch:
        print(f"[Auto-Respond] Discarding stale response (epoch {epoch} → {_session_epoch})")
        broadcast_event("ai_done")
        return

    response = clean_for_tts(response)
    response = _trim_to_sentences(response, max_sentences)
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

        show_history = session.get_show_history()
        mood = detect_host_mood(session.conversation)
        system_prompt = get_caller_prompt(session.caller, show_history, emotional_read=mood)

        max_tokens, max_sentences = _pick_response_budget()
        messages = _normalize_messages_for_llm(session.conversation[-10:])
        response = await llm_service.generate(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens
        )

    if _session_epoch != epoch:
        raise HTTPException(409, "Call changed during response")

    response = clean_for_tts(response)
    response = _trim_to_sentences(response, max_sentences)
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
        )

    session.call_history.append(CallRecord(
        caller_type="real",
        caller_name=caller_phone,
        summary=summary,
        transcript=conversation,
        started_at=started_at,
        ended_at=ended_at,
    ))
    print(f"[Real Caller] {caller_phone} call summarized: {summary[:80]}...")

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
            add_log("Show auto-set to ON AIR")
        return {"on_air": _show_on_air, "recording": True}
    # STOP recording
    audio_service.stop_stem_mic()
    stems_dir = audio_service.stem_recorder.output_dir
    paths = audio_service.stem_recorder.stop()
    audio_service.stem_recorder = None
    add_log(f"Stem recording stopped. Running post-production...")

    if _show_on_air:
        _show_on_air = False
        audio_service.stop_host_stream()
        threading.Thread(target=_update_on_air_cdn, args=(False,), daemon=True).start()
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
    return {"on_air": _show_on_air, "recording": False}


@app.post("/api/recording/process")
async def process_stems(stems_dir: str):
    import subprocess
    stems_path = Path(stems_dir)
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
