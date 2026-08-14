#!/usr/bin/env python3
"""Post 1000 downloads milestone to all social platforms via Postiz."""

import os
import sys
from datetime import datetime, timezone
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

IMAGE_PATH = Path(__file__).parent / "social_posts" / "1000_milestone" / "main_1000_celebration.jpg"


def get_api_url(path: str) -> str:
    return f"{POSTIZ_URL.rstrip('/')}/api/public/v1{path}"


def api_headers() -> dict:
    return {"Authorization": POSTIZ_API_KEY, "Content-Type": "application/json"}


def fetch_integrations() -> list[dict]:
    resp = requests.get(get_api_url("/integrations"), headers=api_headers(), timeout=15)
    if resp.status_code != 200:
        print(f"Error fetching integrations: {resp.status_code} {resp.text[:200]}")
        sys.exit(1)
    return resp.json()


def find_integration(integrations: list[dict], provider: str) -> dict | None:
    for integ in integrations:
        if integ.get("identifier", "").startswith(provider) and not integ.get("disabled"):
            return integ
    return None


def upload_image(file_path: Path) -> dict:
    headers = {"Authorization": POSTIZ_API_KEY}
    mime = "image/jpeg" if file_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    with open(file_path, "rb") as f:
        resp = requests.post(
            get_api_url("/upload"),
            headers=headers,
            files={"file": (file_path.name, f, mime)},
            timeout=60,
        )
    if resp.status_code not in (200, 201):
        print(f"Upload failed: {resp.status_code} {resp.text[:200]}")
        return {}
    return resp.json()


def create_post(integration_id: str, content: str, media: dict, settings: dict) -> dict:
    date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    payload = {
        "type": "now",
        "date": date,
        "shortLink": False,
        "tags": [],
        "posts": [
            {
                "integration": {"id": integration_id},
                "value": [{"content": content, "image": [media] if media else []}],
                "settings": settings,
            }
        ],
    }
    resp = requests.post(get_api_url("/posts"), headers=api_headers(), json=payload, timeout=30)
    if resp.status_code not in (200, 201):
        print(f"  Post failed: {resp.status_code} {resp.text[:300]}")
        return {}
    return resp.json()


# --- Post content per platform ---

POSTS = {
    "instagram": {
        "content": """1,000 downloads in one month. 🎙️

27 episodes. 200+ callers. 111 unique characters. 13 returning regulars.

Luke at the Roost is a late-night call-in show where AI-generated characters phone in with their problems — relationship drama, moral dilemmas, conspiracy theories, drunk confessions — and I give them real advice, live.

Every caller has a unique voice, a backstory, and a reason for calling. Some of them keep calling back.

Thank you to everyone who's tuned in. This thing was supposed to be a weird experiment. Now it's a weird experiment that 1,000 people have listened to.

New episodes daily. Link in bio.

#podcast #ai #artificialintelligence #sideproject #indieproject #podcastlife #latenightradio #callinshow #milestone #1000downloads""",
        "settings": {"__type": "instagram", "post_type": "post", "collaborators": []},
    },
    "facebook": {
        "content": """🎙️ MILESTONE: 1,000 Downloads

One month ago I launched a weird experiment — a late-night call-in radio show where AI-generated characters phone in with their problems, and I give them real advice.

27 episodes later, 200+ callers have phoned in. Some of them keep calling back. Listeners have favorites. People genuinely care about what happens to these characters.

Thank you to every single person who gave this a listen. It started as a side project and it's become something I look forward to every day.

Listen free: lukeattheroost.com
Call in for real: 208-439-LUKE""",
        "settings": {"__type": "facebook"},
    },
    "threads": {
        "content": """1,000 downloads in one month. 🎙️

27 episodes. 200+ callers. 111 unique characters. 13 returning regulars.

Luke at the Roost is a late-night call-in show where AI-generated characters phone in with their problems and I give them real advice, live.

Thank you to everyone who's tuned in. This weird experiment just hit a milestone.

lukeattheroost.com

#podcast #ai #sideproject #1000downloads""",
        "settings": {"__type": "threads"},
    },
    "linkedin": {
        "content": """1,000 downloads in 30 days — here's what I learned building an AI radio show

A month ago I launched Luke at the Roost, a late-night call-in radio show where every caller is an AI-generated character. I'm the host. They phone in with problems. I give them advice. Every conversation is improvised.

27 episodes and 200+ callers later, the show just hit 1,000 downloads.

Some things that surprised me:

People connect with AI characters. Listeners have favorites. They ask about regulars by name. When a caller's story evolves across episodes, people notice and care. The characters aren't real, but the emotional engagement is.

Constraints drive creativity. Each caller gets a token budget based on their personality type. Emotional callers get more room to ramble. Gossip callers are quick and punchy. This artificial constraint mirrors how real people actually talk — and it makes every call feel distinct.

The tech is the easy part. LLMs, voice synthesis, audio routing — that's engineering. The hard part is being a good host. Knowing when to push, when to listen, when to make a joke. AI handles the callers. The human skill is the conversation.

The full technical breakdown: lukeattheroost.com/how-it-works
Listen: lukeattheroost.com

Thank you to everyone who gave this weird experiment a chance.""",
        "settings": {"__type": "linkedin"},
    },
    "mastodon": {
        "content": """1,000 downloads. 27 episodes. 200+ AI callers given advice on everything from breakups to fish consciousness.

Luke at the Roost hit a milestone today and I just want to say thank you to everyone who's been listening.

This whole thing is self-hosted end-to-end — Castopod on a QNAP NAS, Cloudflare CDN, custom Python pipeline for recording, post-production, and publishing. No big platforms in the loop.

If you haven't heard it: it's a late-night call-in show. AI characters phone in. I talk to them live. It's improvised, weird, and somehow heartfelt.

https://lukeattheroost.com""",
        "settings": {"__type": "mastodon"},
    },
    "tiktok": {
        "content": """1,000 downloads in one month 🎙️

27 episodes. 200+ AI callers. 13 returning regulars.

Luke at the Roost — a late-night call-in show where AI characters phone in with their problems and I give them real advice, live.

Thank you to everyone listening.

#podcast #ai #artificialintelligence #sideproject #latenightradio #callinshow #1000downloads""",
        "settings": {
            "__type": "tiktok",
            "privacy_level": "PUBLIC_TO_EVERYONE",
            "duet": False,
            "stitch": False,
            "comment": True,
            "autoAddMusic": "no",
            "brand_content_toggle": False,
            "brand_organic_toggle": False,
            "content_posting_method": "DIRECT_POST",
        },
    },
    "nostr": {
        "content": """1,000 downloads. 27 episodes. 200+ AI callers given advice.

Luke at the Roost just hit a milestone. Thank you to everyone listening.

It's a late-night call-in show where AI-generated characters phone in with their problems and I give them real advice, live. Every conversation is improvised.

https://lukeattheroost.com""",
        "settings": {"__type": "nostr"},
    },
}


def main():
    dry_run = "--dry-run" in sys.argv

    if not POSTIZ_API_KEY:
        print("Error: POSTIZ_API_KEY not set")
        sys.exit(1)

    if not IMAGE_PATH.exists():
        print(f"Error: Image not found at {IMAGE_PATH}")
        sys.exit(1)

    print("Fetching connected accounts from Postiz...")
    integrations = fetch_integrations()

    available = {}
    for platform in POSTS:
        integ = find_integration(integrations, platform)
        if integ:
            available[platform] = integ
            print(f"  ✓ {platform}: {integ.get('name', 'connected')}")
        else:
            print(f"  ✗ {platform}: not connected, skipping")

    if not available:
        print("\nNo platforms available!")
        sys.exit(1)

    print(f"\nWill post to {len(available)} platform(s) with image: {IMAGE_PATH.name}")

    if dry_run:
        print("\n--- DRY RUN ---")
        for platform in available:
            print(f"\n[{platform.upper()}]")
            print(POSTS[platform]["content"][:200] + "...")
        print("\nDry run complete — nothing posted.")
        return

    # Upload image once
    print(f"\nUploading image...")
    media = upload_image(IMAGE_PATH)
    if not media:
        print("Failed to upload image, aborting")
        sys.exit(1)
    print(f"  Uploaded: {media.get('path', 'ok')}")

    # Post to each platform
    results = {}
    for platform, integ in available.items():
        post_data = POSTS[platform]
        print(f"\nPosting to {platform}...")
        result = create_post(integ["id"], post_data["content"], media, post_data["settings"])
        if result:
            print(f"  ✓ {platform}: Posted!")
            results[platform] = True
        else:
            print(f"  ✗ {platform}: Failed")
            results[platform] = False

    # Summary
    succeeded = [p for p, ok in results.items() if ok]
    failed = [p for p, ok in results.items() if not ok]
    print(f"\n{'='*40}")
    print(f"Posted to {len(succeeded)}/{len(results)} platforms")
    if succeeded:
        print(f"  ✓ {', '.join(succeeded)}")
    if failed:
        print(f"  ✗ {', '.join(failed)}")


if __name__ == "__main__":
    main()
