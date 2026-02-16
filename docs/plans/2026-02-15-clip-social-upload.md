# Clip Social Media Upload Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate social media descriptions/hashtags for podcast clips and upload them to Instagram Reels + YouTube Shorts via Postiz API.

**Architecture:** Two changes — (1) extend `make_clips.py` to add a second LLM call that generates descriptions + hashtags, saved as `clips-metadata.json`, (2) new `upload_clips.py` script that reads that metadata and pushes clips through the self-hosted Postiz instance at `social.lukeattheroost.com`.

**Tech Stack:** Python, OpenRouter API (Claude Sonnet), Postiz REST API, requests library (already installed)

---

### Task 1: Add `generate_social_metadata()` to `make_clips.py`

**Files:**
- Modify: `make_clips.py:231-312` (after `select_clips_with_llm`)

**Step 1: Add the function after `select_clips_with_llm`**

Add this function at line ~314 (after `select_clips_with_llm` returns):

```python
def generate_social_metadata(clips: list[dict], labeled_transcript: str,
                              episode_number: int | None) -> list[dict]:
    """Generate social media descriptions and hashtags for each clip."""
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not set in .env")
        sys.exit(1)

    clips_summary = "\n".join(
        f'{i+1}. "{c["title"]}" — {c["caption_text"]}'
        for i, c in enumerate(clips)
    )

    episode_context = f"This is Episode {episode_number} of " if episode_number else "This is an episode of "

    prompt = f"""{episode_context}the "Luke at the Roost" podcast — a late-night call-in show where AI-generated callers share stories, confessions, and hot takes with host Luke.

Here are {len(clips)} clips selected from this episode:

{clips_summary}

For each clip, generate:
1. description: A short, engaging description for social media (1-2 sentences, hook the viewer, conversational tone). Do NOT include hashtags in the description.
2. hashtags: An array of 5-8 hashtags. Always include #lukeattheroost and #podcast. Add topic-relevant and trending-style tags.

Respond with ONLY a JSON array matching the clip order:
[{{"description": "...", "hashtags": ["#tag1", "#tag2", ...]}}]"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "anthropic/claude-sonnet-4-5",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.7,
        },
    )

    if response.status_code != 200:
        print(f"Error from OpenRouter: {response.text}")
        return clips

    content = response.json()["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\n?", "", content)
        content = re.sub(r"\n?```$", "", content)

    try:
        metadata = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error parsing social metadata: {e}")
        return clips

    for i, clip in enumerate(clips):
        if i < len(metadata):
            clip["description"] = metadata[i].get("description", "")
            clip["hashtags"] = metadata[i].get("hashtags", [])

    return clips
```

**Step 2: Run existing tests to verify no breakage**

Run: `pytest tests/ -v`
Expected: All existing tests pass (this is a new function, no side effects yet)

**Step 3: Commit**

```bash
git add make_clips.py
git commit -m "Add generate_social_metadata() for clip descriptions and hashtags"
```

---

### Task 2: Integrate metadata generation + JSON save into `main()`

**Files:**
- Modify: `make_clips.py:1082-1289` (inside `main()`)

**Step 1: Add metadata generation call and JSON save**

After the LLM clip selection step (~line 1196, after the clip summary print loop), add:

```python
    # Step N: Generate social media metadata
    print(f"\n[{extract_step - 1}/{step_total}] Generating social media descriptions...")
    clips = generate_social_metadata(clips, labeled_transcript, episode_number)
    for i, clip in enumerate(clips):
        if "description" in clip:
            print(f"    Clip {i+1}: {clip['description'][:80]}...")
            print(f"           {' '.join(clip.get('hashtags', []))}")
```

Note: This needs to be inserted BEFORE the audio extraction step, and the step numbering needs to be adjusted (total steps goes from 5/6 to 6/7).

At the end of `main()`, before the summary print, save the metadata JSON:

```python
    # Save clips metadata for social upload
    metadata_path = output_dir / "clips-metadata.json"
    metadata = []
    for i, clip in enumerate(clips):
        slug = slugify(clip["title"])
        metadata.append({
            "title": clip["title"],
            "clip_file": f"clip-{i+1}-{slug}.mp4",
            "audio_file": f"clip-{i+1}-{slug}.mp3",
            "caption_text": clip.get("caption_text", ""),
            "description": clip.get("description", ""),
            "hashtags": clip.get("hashtags", []),
            "start_time": clip["start_time"],
            "end_time": clip["end_time"],
            "duration": round(clip["end_time"] - clip["start_time"], 1),
            "episode_number": episode_number,
        })
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSocial metadata: {metadata_path}")
```

**Step 2: Adjust step numbering**

The pipeline steps need to account for the new metadata step. Update `step_total` calculation:

```python
    step_total = (7 if two_pass else 6)
```

And shift the extract/video step numbers up by 1.

**Step 3: Test manually**

Run: `python make_clips.py --help`
Expected: No import errors, help displays normally

**Step 4: Commit**

```bash
git add make_clips.py
git commit -m "Save clips-metadata.json with social descriptions and hashtags"
```

---

### Task 3: Create `upload_clips.py` — core structure and Postiz API helpers

**Files:**
- Create: `upload_clips.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Upload podcast clips to Instagram Reels and YouTube Shorts via Postiz.

Usage:
    python upload_clips.py clips/episode-12/
    python upload_clips.py clips/episode-12/ --clip 1
    python upload_clips.py clips/episode-12/ --youtube-only
    python upload_clips.py clips/episode-12/ --instagram-only
    python upload_clips.py clips/episode-12/ --schedule "2026-02-16T10:00:00"
    python upload_clips.py clips/episode-12/ --yes  # skip confirmation
"""

import argparse
import json
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / ".env")

POSTIZ_API_KEY = os.getenv("POSTIZ_API_KEY")
POSTIZ_URL = os.getenv("POSTIZ_URL", "https://social.lukeattheroost.com")


def get_api_url(path: str) -> str:
    """Build full Postiz API URL."""
    base = POSTIZ_URL.rstrip("/")
    # Postiz self-hosted API is at /api/public/v1 when NEXT_PUBLIC_BACKEND_URL is the app URL
    # but the docs say /public/v1 relative to backend URL. Try the standard path.
    return f"{base}/api/public/v1{path}"


def api_headers() -> dict:
    return {
        "Authorization": POSTIZ_API_KEY,
        "Content-Type": "application/json",
    }


def fetch_integrations() -> list[dict]:
    """Fetch connected social accounts from Postiz."""
    resp = requests.get(get_api_url("/integrations"), headers=api_headers(), timeout=15)
    if resp.status_code != 200:
        print(f"Error fetching integrations: {resp.status_code} {resp.text[:200]}")
        sys.exit(1)
    return resp.json()


def find_integration(integrations: list[dict], provider: str) -> dict | None:
    """Find integration by provider name (e.g. 'instagram', 'youtube')."""
    for integ in integrations:
        if integ.get("providerIdentifier", "").startswith(provider):
            return integ
        if integ.get("provider", "").startswith(provider):
            return integ
    return None


def upload_file(file_path: Path) -> dict:
    """Upload a file to Postiz. Returns {id, path}."""
    headers = {"Authorization": POSTIZ_API_KEY}
    with open(file_path, "rb") as f:
        resp = requests.post(
            get_api_url("/upload"),
            headers=headers,
            files={"file": (file_path.name, f, "video/mp4")},
            timeout=120,
        )
    if resp.status_code != 200:
        print(f"Upload failed: {resp.status_code} {resp.text[:200]}")
        return {}
    return resp.json()


def create_post(integration_id: str, content: str, media: dict,
                settings: dict, schedule: str | None = None) -> dict:
    """Create a post on Postiz."""
    post_type = "schedule" if schedule else "now"

    payload = {
        "type": post_type,
        "posts": [
            {
                "integration": {"id": integration_id},
                "value": [
                    {
                        "content": content,
                        "image": [media] if media else [],
                    }
                ],
                "settings": settings,
            }
        ],
    }
    if schedule:
        payload["date"] = schedule

    resp = requests.post(
        get_api_url("/posts"),
        headers=api_headers(),
        json=payload,
        timeout=30,
    )
    if resp.status_code not in (200, 201):
        print(f"Post creation failed: {resp.status_code} {resp.text[:300]}")
        return {}
    return resp.json()


def build_instagram_content(clip: dict) -> str:
    """Build Instagram post content: description + hashtags."""
    parts = [clip.get("description", clip.get("caption_text", ""))]
    hashtags = clip.get("hashtags", [])
    if hashtags:
        parts.append("\n\n" + " ".join(hashtags))
    return "".join(parts)


def build_youtube_content(clip: dict) -> str:
    """Build YouTube description."""
    parts = [clip.get("description", clip.get("caption_text", ""))]
    hashtags = clip.get("hashtags", [])
    if hashtags:
        parts.append("\n\n" + " ".join(hashtags))
    parts.append("\n\nListen to the full episode: lukeattheroost.com")
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Upload podcast clips to social media via Postiz")
    parser.add_argument("clips_dir", help="Path to clips directory (e.g. clips/episode-12/)")
    parser.add_argument("--clip", "-c", type=int, help="Upload only clip N (1-indexed)")
    parser.add_argument("--instagram-only", action="store_true", help="Upload to Instagram only")
    parser.add_argument("--youtube-only", action="store_true", help="Upload to YouTube only")
    parser.add_argument("--schedule", "-s", help="Schedule time (ISO 8601, e.g. 2026-02-16T10:00:00)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without posting")
    args = parser.parse_args()

    if not POSTIZ_API_KEY:
        print("Error: POSTIZ_API_KEY not set in .env")
        sys.exit(1)

    clips_dir = Path(args.clips_dir).expanduser().resolve()
    metadata_path = clips_dir / "clips-metadata.json"

    if not metadata_path.exists():
        print(f"Error: No clips-metadata.json found in {clips_dir}")
        print("Run make_clips.py first to generate clips and metadata.")
        sys.exit(1)

    with open(metadata_path) as f:
        clips = json.load(f)

    if args.clip:
        if args.clip < 1 or args.clip > len(clips):
            print(f"Error: Clip {args.clip} not found (have {len(clips)} clips)")
            sys.exit(1)
        clips = [clips[args.clip - 1]]

    # Determine which platforms to post to
    do_instagram = not args.youtube_only
    do_youtube = not args.instagram_only

    # Fetch integrations from Postiz
    print("Fetching connected accounts from Postiz...")
    integrations = fetch_integrations()

    ig_integration = None
    yt_integration = None

    if do_instagram:
        ig_integration = find_integration(integrations, "instagram")
        if not ig_integration:
            print("Warning: No Instagram account connected in Postiz")
            do_instagram = False

    if do_youtube:
        yt_integration = find_integration(integrations, "youtube")
        if not yt_integration:
            print("Warning: No YouTube account connected in Postiz")
            do_youtube = False

    if not do_instagram and not do_youtube:
        print("Error: No platforms available to upload to")
        sys.exit(1)

    # Show summary
    platforms = []
    if do_instagram:
        platforms.append(f"Instagram Reels ({ig_integration.get('name', 'connected')})")
    if do_youtube:
        platforms.append(f"YouTube Shorts ({yt_integration.get('name', 'connected')})")

    print(f"\nUploading {len(clips)} clip(s) to: {', '.join(platforms)}")
    if args.schedule:
        print(f"Scheduled for: {args.schedule}")
    print()

    for i, clip in enumerate(clips):
        print(f"  {i+1}. \"{clip['title']}\" ({clip['duration']:.0f}s)")
        print(f"     {clip.get('description', '')[:80]}")
        print(f"     {' '.join(clip.get('hashtags', []))}")
        print()

    if args.dry_run:
        print("Dry run — nothing uploaded.")
        return

    if not args.yes:
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return

    # Upload each clip
    for i, clip in enumerate(clips):
        clip_file = clips_dir / clip["clip_file"]
        if not clip_file.exists():
            print(f"  Clip {i+1}: Video file not found: {clip_file}")
            continue

        print(f"\n  Clip {i+1}: \"{clip['title']}\"")

        # Upload video to Postiz
        print(f"    Uploading {clip_file.name}...")
        media = upload_file(clip_file)
        if not media:
            print(f"    Failed to upload video, skipping")
            continue
        print(f"    Uploaded: {media.get('path', 'ok')}")

        # Post to Instagram Reels
        if do_instagram:
            print(f"    Posting to Instagram Reels...")
            content = build_instagram_content(clip)
            settings = {
                "__type": "instagram",
                "post_type": "reel",
            }
            result = create_post(
                ig_integration["id"], content, media, settings, args.schedule
            )
            if result:
                print(f"    Instagram: Posted!")
            else:
                print(f"    Instagram: Failed")

        # Post to YouTube Shorts
        if do_youtube:
            print(f"    Posting to YouTube Shorts...")
            content = build_youtube_content(clip)
            settings = {
                "__type": "youtube",
                "title": clip["title"],
                "type": "short",
                "selfDeclaredMadeForKids": False,
                "tags": [h.lstrip("#") for h in clip.get("hashtags", [])],
            }
            result = create_post(
                yt_integration["id"], content, media, settings, args.schedule
            )
            if result:
                print(f"    YouTube: Posted!")
            else:
                print(f"    YouTube: Failed")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
```

**Step 2: Add `POSTIZ_API_KEY` and `POSTIZ_URL` to `.env`**

Add to `.env`:
```
POSTIZ_API_KEY=your-postiz-api-key-here
POSTIZ_URL=https://social.lukeattheroost.com
```

Get your API key from Postiz Settings page.

**Step 3: Test the script loads**

Run: `python upload_clips.py --help`
Expected: Help text displays with all flags

**Step 4: Commit**

```bash
git add upload_clips.py
git commit -m "Add upload_clips.py for posting clips to Instagram/YouTube via Postiz"
```

---

### Task 4: Test with real Postiz instance

**Step 1: Get Postiz API key**

Go to `https://social.lukeattheroost.com` → Settings → API Keys → Generate key. Add to `.env` as `POSTIZ_API_KEY`.

**Step 2: Verify integrations endpoint**

Run: `python -c "from upload_clips import *; print(json.dumps(fetch_integrations(), indent=2))"`

This confirms the API key works and shows connected Instagram/YouTube accounts. Note the integration IDs and provider identifiers — if `find_integration()` doesn't match correctly, adjust the provider string matching.

**Step 3: Dry-run with existing clips**

Run: `python upload_clips.py clips/episode-12/ --dry-run`
Expected: Shows clip summary, "Dry run — nothing uploaded."

**Step 4: Upload a single test clip**

Run: `python upload_clips.py clips/episode-12/ --clip 1 --instagram-only`

Check Postiz dashboard and Instagram to verify it posted as a Reel.

**Step 5: Commit .env update (do NOT commit the key itself)**

The `.env` is gitignored so no action needed. Just ensure the key names are documented in CLAUDE.md if desired.
