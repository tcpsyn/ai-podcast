#!/usr/bin/env python3
"""Schedule the X/Twitter launch campaign posts via Postiz.

Schedules 2 weeks of posts from the growth strategy to @lukeattheroost.
All times are ET, converted to UTC for the Postiz API.

Usage:
    python schedule_x_launch.py              # schedule all posts
    python schedule_x_launch.py --dry-run    # preview without scheduling
    python schedule_x_launch.py --week 1     # schedule week 1 only
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# Load .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

POSTIZ_API_KEY = os.getenv("POSTIZ_API_KEY")
POSTIZ_URL = os.getenv("POSTIZ_URL", "https://social.lukeattheroost.com")

SCRIPT_DIR = Path(__file__).parent
X_INTEGRATION_ID = "REDACTED_X_ID"

# ET = UTC-4 (EDT) in March 2026
ET_OFFSET = timedelta(hours=-4)


def et_to_utc(year, month, day, hour, minute=0):
    """Convert ET datetime to UTC ISO string for Postiz."""
    et = datetime(year, month, day, hour, minute, tzinfo=timezone(ET_OFFSET))
    utc = et.astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def get_api_url(path):
    return f"{POSTIZ_URL.rstrip('/')}/api/public/v1{path}"


def api_headers():
    return {"Authorization": POSTIZ_API_KEY, "Content-Type": "application/json"}


def upload_file(file_path):
    headers = {"Authorization": POSTIZ_API_KEY}
    suffix = file_path.suffix.lower()
    if suffix == ".mp4":
        mime = "video/mp4"
    elif suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    else:
        mime = "image/png"

    with open(file_path, "rb") as f:
        resp = requests.post(
            get_api_url("/upload"),
            headers=headers,
            files={"file": (file_path.name, f, mime)},
            timeout=120,
        )
    if resp.status_code not in (200, 201):
        print(f"  Upload failed: {resp.status_code} {resp.text[:200]}")
        return {}
    return resp.json()


def schedule_post(content, media, schedule_time, retries=3):
    payload = {
        "type": "schedule",
        "date": schedule_time,
        "shortLink": False,
        "tags": [],
        "posts": [
            {
                "integration": {"id": X_INTEGRATION_ID},
                "value": [
                    {
                        "content": content,
                        "image": [media] if media else [],
                    }
                ],
                "settings": {"__type": "x", "who_can_reply_post": "everyone"},
            }
        ],
    }
    for attempt in range(retries):
        resp = requests.post(
            get_api_url("/posts"),
            headers=api_headers(),
            json=payload,
            timeout=30,
        )
        if resp.status_code in (200, 201):
            return resp.json()
        if resp.status_code == 429 and attempt < retries - 1:
            wait = 15 * (attempt + 1)
            print(f"(rate limited, waiting {wait}s)...", end=" ", flush=True)
            time.sleep(wait)
            continue
        print(f"  Schedule failed: {resp.status_code} {resp.text[:300]}")
        return {}
    return {}


# ── POST DEFINITIONS ─────────────────────────────────────────────────
# Day 1 = Monday March 17, 2026

WEEK_1 = [
    # Day 1 — Monday March 17
    {
        "label": "W1-Mon-AM (pinned intro)",
        "time": et_to_utc(2026, 3, 17, 10),
        "content": """Every caller on my show is AI-generated.

Every personality. Every voice. Every problem they call in about.

But the conversations are real, the advice is real, and the chaos is very real.

38 episodes. 200+ callers. A cult leader, a guy who opened paternity results live on air, and someone who faked cancer to skip a wedding.

This is Luke at the Roost.

📞 208-439-LUKE (real humans can call in too)
🔗 lukeattheroost.com""",
        "media": "website/images/cover.png",
    },
    {
        "label": "W1-Mon-PM (chili clip)",
        "time": et_to_utc(2026, 3, 17, 14),
        "content": """A guy called in to talk about chili contest cheaters.

Turns out he was really calling about his failing marriage.

#lukeattheroost #podcast #callinshow""",
        "media": "clips/episode-37/clip-1-chili-contest-cheaters-marriage-troubles.mp4",
    },
    {
        "label": "W1-Mon-EVE (intro thread)",
        "time": et_to_utc(2026, 3, 17, 21),
        "content": """People keep asking what Luke at the Roost is.

Short version:
→ AI characters call into my show with real problems
→ I give them actual advice
→ Everything goes off the rails
→ New episode every day

lukeattheroost.com""",
        "media": None,
    },
    # Day 2 — Tuesday March 18
    {
        "label": "W1-Tue-AM (hospice clip)",
        "time": et_to_utc(2026, 3, 18, 12),
        "content": """A caller's mom is in hospice. The nurse asked her to think about final conversations.

Her only wish? To see her kids eat cake together one last time.

#lukeattheroost #podcast""",
        "media": "clips/episode-30/clip-2-mom-s-dying-wish-just-eat-cake-together.mp4",
    },
    {
        "label": "W1-Tue-PM (engagement)",
        "time": et_to_utc(2026, 3, 18, 19),
        "content": """What's the wildest thing you've ever called into a radio show about?

(Or wanted to but chickened out?)""",
        "media": None,
    },
    # Day 3 — Wednesday March 19
    {
        "label": "W1-Wed-AM (cancer clip)",
        "time": et_to_utc(2026, 3, 19, 11),
        "content": """This caller faked having cancer to get out of going to a wedding.

Then his friends staged a coffee enema intervention.

I can't make this up. (Well, the AI did.)

#lukeattheroost #podcast""",
        "media": "clips/episode-32/clip-1-i-faked-cancer-to-skip-a-wedding.mp4",
    },
    {
        "label": "W1-Wed-PM (fakes clip)",
        "time": et_to_utc(2026, 3, 19, 16),
        "content": """"Everybody is a fake. We're all fakes. Nobody knows what's going on and none of us deserve a goddamn thing. We're lucky to be here at all."

— A caller on Episode 2. Still the hardest truth anyone's dropped on my show.

#lukeattheroost #podcast""",
        "media": "clips/episode-2/clip-1-we-re-all-fakes-and-that-s-okay.mp4",
    },
    # Day 4 — Thursday March 20
    {
        "label": "W1-Thu-AM (BTS)",
        "time": et_to_utc(2026, 3, 20, 12),
        "content": """How my show works:

→ AI generates a caller with a full backstory, personality, and voice
→ They call in live
→ I have zero idea what they're going to say
→ I give them real advice
→ Post-production runs automatically
→ Episode publishes

38 episodes. Built the whole thing from scratch.""",
        "media": None,
    },
    {
        "label": "W1-Thu-PM (stakeout clip)",
        "time": et_to_utc(2026, 3, 20, 20),
        "content": """He spent four hours staking out his best friend at Starbucks.

What he found was worse than what he expected.

#lukeattheroost #podcast""",
        "media": "clips/episode-28/clip-3-four-hours-spying-on-his-best-friend.mp4",
    },
    # Day 5 — Friday March 21
    {
        "label": "W1-Fri-AM (review ask)",
        "time": et_to_utc(2026, 3, 21, 11),
        "content": """If you've listened to Luke at the Roost and liked it — a review on Apple Podcasts or Spotify goes further than you'd think.

Not guilt-tripping. Just saying it helps a one-person show more than anything else.

🔗 lukeattheroost.com""",
        "media": "social_posts/x_launch/leave_a_review_twitter.png",
    },
    {
        "label": "W1-Fri-PM (wall clip)",
        "time": et_to_utc(2026, 3, 21, 18),
        "content": """She opened up a mystery wall in her house LIVE on the show.

There were stacks of cash inside.

#lukeattheroost #podcast""",
        "media": "clips/episode-35/clip-3-woman-finds-cash-in-secret-wall.mp4",
    },
    # Day 6 — Saturday March 22
    {
        "label": "W1-Sat-AM (poll)",
        "time": et_to_utc(2026, 3, 22, 12),
        "content": """Which is wilder?

A) Woman hid her daughter from her husband for 8 years
B) Cult leader existential crisis on air
C) Paternity results opened live on the show
D) Faked cancer to skip a wedding""",
        "media": None,
    },
    {
        "label": "W1-Sat-PM (silence clip)",
        "time": et_to_utc(2026, 3, 22, 20),
        "content": """"I told my girlfriend my biggest fantasy and she went completely silent for 10 seconds."

The silence in this clip is brutal.

#lukeattheroost #podcast""",
        "media": "clips/episode-30/clip-3-latex-fetish-confession-goes-silent.mp4",
    },
    # Day 7 — Sunday March 23
    {
        "label": "W1-Sun-PM (week recap)",
        "time": et_to_utc(2026, 3, 23, 15),
        "content": """One week on X. Dropped 38 episodes before making an account.

If any of these clips made you laugh, cringe, or feel something — the full episodes are even wilder.

📞 208-439-LUKE
🔗 lukeattheroost.com
🎧 Spotify · Apple · YouTube""",
        "media": "website/images/cover.png",
    },
]

WEEK_2 = [
    # Day 8 — Monday March 24
    {
        "label": "W2-Mon-AM (second family clip)",
        "time": et_to_utc(2026, 3, 24, 10),
        "content": """A guy called in and found out his dad had a whole second family.

Three kids in Tucson who grew up calling his dad "Dad."

He found out via email from a stranger.

#lukeattheroost #podcast""",
        "media": "clips/episode-20/clip-2-dad-s-secret-second-family-revealed.mp4",
    },
    {
        "label": "W2-Mon-PM (engagement)",
        "time": et_to_utc(2026, 3, 24, 19),
        "content": """What would you do if you found out your dad had a whole second family?

Asking because a caller found out via email from a woman in Tucson.""",
        "media": None,
    },
    # Day 9 — Tuesday March 25
    {
        "label": "W2-Tue-AM (spanish clip)",
        "time": et_to_utc(2026, 3, 25, 11),
        "content": """A caller pretended to speak Spanish at his job for 8 years.

Eight. Years.

#lukeattheroost #podcast""",
        "media": "clips/episode-14/clip-1-i-lied-about-speaking-spanish-for-8-years.mp4",
    },
    {
        "label": "W2-Tue-PM (quote)",
        "time": et_to_utc(2026, 3, 25, 18),
        "content": """"Middle management is plagiarism with a 401k."

AI callers drop better one-liners than most standup specials.

#lukeattheroost #podcast""",
        "media": "social_posts/x_launch/quote_3_stoicism_backwards_twitter.png",
    },
    # Day 10 — Wednesday March 26
    {
        "label": "W2-Wed-AM (BTS)",
        "time": et_to_utc(2026, 3, 26, 12),
        "content": """People ask if I script the show.

I don't even know who's calling until they're on the line. The AI generates the caller, picks a unique voice, gives them a backstory, and dials in.

My job is just to be a good host. The chaos handles itself.""",
        "media": None,
    },
    {
        "label": "W2-Wed-PM (roomba clip)",
        "time": et_to_utc(2026, 3, 26, 20),
        "content": """His neighbor's Roomba broke into his kitchen at 2:30 AM.

This is the content you're here for.

#lukeattheroost #podcast""",
        "media": "clips/episode-26/clip-2-neighbor-s-roomba-breaks-into-kitchen-at-2-30-am.mp4",
    },
    # Day 11 — Thursday March 27
    {
        "label": "W2-Thu-AM (stalking clip)",
        "time": et_to_utc(2026, 3, 27, 11),
        "content": """She sat in a Dairy Queen parking lot for 20 minutes watching her ex's truck at Sonic across the street.

We've all been there. (Right?)

#lukeattheroost #podcast""",
        "media": "clips/episode-22/clip-2-stalking-your-ex-at-sonic.mp4",
    },
    {
        "label": "W2-Thu-PM (engagement)",
        "time": et_to_utc(2026, 3, 27, 19),
        "content": """Be honest: what's a lie you've kept going for way too long?

A caller on my show pretended to speak Spanish at work for 8 years. You can't beat that.""",
        "media": None,
    },
    # Day 12 — Friday March 28
    {
        "label": "W2-Fri-AM (hidden room clip)",
        "time": et_to_utc(2026, 3, 28, 10),
        "content": """A man found an impossible hidden room in a junkyard.

Inside? Beer that was still fresh after 12 years.

Two clips. One mystery.

#lukeattheroost #podcast""",
        "media": "clips/episode-34/clip-1-man-finds-impossible-hidden-room-in-junkyard.mp4",
    },
    {
        "label": "W2-Fri-PM (fix quote)",
        "time": et_to_utc(2026, 3, 28, 18),
        "content": """"You can't fix somebody who doesn't want to be fixed."

A caller said this about their partner and it hit like a truck.

#lukeattheroost #podcast""",
        "media": "clips/episode-30/clip-1-you-can-t-fix-someone-who-won-t-be-fixed.mp4",
    },
    # Day 13 — Saturday March 29
    {
        "label": "W2-Sat-AM (BTS callers)",
        "time": et_to_utc(2026, 3, 29, 12),
        "content": """Each AI caller gets generated with:

• A name, age, job, and hometown
• A reason for calling
• A communication style + energy level
• An emotional state
• A "signature detail" that makes them unique
• A unique AI voice

None of it is scripted. They just... call in and talk.""",
        "media": None,
    },
    {
        "label": "W2-Sat-PM (check clip)",
        "time": et_to_utc(2026, 3, 29, 20),
        "content": """He deposited a $5,000 check instead of $500 three months ago.

Spent it all.

Now his company might find out.

#lukeattheroost #podcast""",
        "media": "clips/episode-25/clip-2-accidentally-kept-4-500-from-work.mp4",
    },
    # Day 14 — Sunday March 30
    {
        "label": "W2-Sun-PM (week 2 recap + review)",
        "time": et_to_utc(2026, 3, 30, 15),
        "content": """Two weeks in. Thank you to everyone who's followed, listened, or dropped a comment.

This show started as a weird experiment — a guy giving life advice to AI-generated callers at 2 AM.

38 episodes later it's still weird. But now people are listening.

If you've been enjoying it, a rating on Apple or Spotify makes a real difference. 🙏

lukeattheroost.com""",
        "media": None,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Schedule X launch campaign via Postiz")
    parser.add_argument("--dry-run", action="store_true", help="Preview without scheduling")
    parser.add_argument("--week", type=int, choices=[1, 2], help="Schedule only week 1 or 2")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N posts (for retrying after partial run)")
    parser.add_argument("--delay", type=int, default=10, help="Seconds between API calls (default 10)")
    args = parser.parse_args()

    if not POSTIZ_API_KEY:
        print("Error: POSTIZ_API_KEY not set in .env")
        sys.exit(1)

    all_posts = []
    if args.week != 2:
        all_posts.extend(WEEK_1)
    if args.week != 1:
        all_posts.extend(WEEK_2)
    posts = all_posts[args.skip:]

    print(f"\n=== X Launch Campaign — {len(posts)} posts ===\n")

    if args.dry_run:
        for i, post in enumerate(posts, 1):
            has_media = "📎" if post["media"] else "  "
            print(f"  {i:2d}. {has_media} {post['label']}")
            print(f"      Schedule: {post['time']}")
            preview = post["content"][:80].replace("\n", " ")
            print(f"      Content: {preview}...")
            if post["media"]:
                print(f"      Media: {post['media']}")
            print()
        print(f"Dry run complete — {len(posts)} posts would be scheduled.")
        return

    # Upload media files first (deduplicate), with disk cache
    cache_file = SCRIPT_DIR / "social_posts" / "x_launch" / ".upload_cache.json"
    media_cache = {}
    if cache_file.exists():
        media_cache = json.loads(cache_file.read_text())
        print(f"Loaded {len(media_cache)} cached uploads from previous run\n")

    media_files = set(p["media"] for p in posts if p["media"])
    to_upload = [m for m in sorted(media_files) if m not in media_cache]

    if to_upload:
        print(f"Uploading {len(to_upload)} new media files ({len(media_files) - len(to_upload)} cached)...\n")
        for media_path in to_upload:
            full_path = SCRIPT_DIR / media_path
            if not full_path.exists():
                print(f"  ✗ {media_path} — FILE NOT FOUND, skipping")
                continue
            print(f"  Uploading {media_path}...", end=" ", flush=True)
            result = upload_file(full_path)
            if result:
                media_cache[media_path] = result
                cache_file.write_text(json.dumps(media_cache, indent=2))
                print("✓")
            else:
                print("✗ FAILED")
            time.sleep(3)
    else:
        print(f"All {len(media_files)} media files already cached, skipping uploads\n")

    # Schedule posts
    print(f"\nScheduling {len(posts)} posts to X...\n")

    success = 0
    failed = 0
    for i, post in enumerate(posts, 1):
        media = media_cache.get(post["media"]) if post["media"] else None

        if post["media"] and not media:
            print(f"  {i:2d}. ✗ {post['label']} — media upload missing, skipping")
            failed += 1
            continue

        print(f"  {i:2d}. Scheduling {post['label']}...", end=" ", flush=True)
        result = schedule_post(post["content"], media, post["time"])
        if result:
            print("✓")
            success += 1
        else:
            print("✗")
            failed += 1
        # Rate limit: pause between API calls
        if i < len(posts):
            time.sleep(5)

    print(f"\n{'='*50}")
    print(f"Scheduled: {success}/{len(posts)}")
    if failed:
        print(f"Failed: {failed}")
    print(f"\nPosts will appear on @lukeattheroost starting Mon March 17")


if __name__ == "__main__":
    main()
