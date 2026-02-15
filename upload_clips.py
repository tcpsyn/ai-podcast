#!/usr/bin/env python3
"""Upload podcast clips to social media via Postiz (and direct Bluesky via atproto).

Usage:
    python upload_clips.py clips/episode-12/
    python upload_clips.py clips/episode-12/ --clip 1
    python upload_clips.py clips/episode-12/ --platforms ig,yt
    python upload_clips.py clips/episode-12/ --schedule "2026-02-16T10:00:00"
    python upload_clips.py clips/episode-12/ --yes  # skip confirmation
"""

import argparse
import json
import sys
from pathlib import Path

import requests
from atproto import Client as BskyClient
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / ".env")

POSTIZ_API_KEY = os.getenv("POSTIZ_API_KEY")
POSTIZ_URL = os.getenv("POSTIZ_URL", "https://social.lukeattheroost.com")

BSKY_HANDLE = os.getenv("BSKY_HANDLE", "lukeattheroost.bsky.social")
BSKY_APP_PASSWORD = os.getenv("BSKY_APP_PASSWORD")

PLATFORM_ALIASES = {
    "ig": "instagram", "insta": "instagram", "instagram": "instagram",
    "yt": "youtube", "youtube": "youtube",
    "fb": "facebook", "facebook": "facebook",
    "bsky": "bluesky", "bluesky": "bluesky",
    "masto": "mastodon", "mastodon": "mastodon",
    "nostr": "nostr",
}

PLATFORM_DISPLAY = {
    "instagram": "Instagram Reels",
    "youtube": "YouTube Shorts",
    "facebook": "Facebook Reels",
    "bluesky": "Bluesky",
    "mastodon": "Mastodon",
    "nostr": "Nostr",
}

ALL_PLATFORMS = list(PLATFORM_DISPLAY.keys())


def get_api_url(path: str) -> str:
    base = POSTIZ_URL.rstrip("/")
    return f"{base}/api/public/v1{path}"


def api_headers() -> dict:
    return {
        "Authorization": POSTIZ_API_KEY,
        "Content-Type": "application/json",
    }


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


def upload_file(file_path: Path) -> dict:
    headers = {"Authorization": POSTIZ_API_KEY}
    with open(file_path, "rb") as f:
        resp = requests.post(
            get_api_url("/upload"),
            headers=headers,
            files={"file": (file_path.name, f, "video/mp4")},
            timeout=120,
        )
    if resp.status_code not in (200, 201):
        print(f"Upload failed: {resp.status_code} {resp.text[:200]}")
        return {}
    return resp.json()


def build_content(clip: dict, platform: str) -> str:
    desc = clip.get("description", clip.get("caption_text", ""))
    hashtags = clip.get("hashtags", [])
    hashtag_str = " ".join(hashtags)

    if platform == "bluesky":
        if hashtags and len(desc) + 2 + len(hashtag_str) <= 300:
            return desc + "\n\n" + hashtag_str
        return desc[:300]

    parts = [desc]
    if hashtags:
        parts.append("\n\n" + hashtag_str)
    if platform in ("youtube", "facebook"):
        parts.append("\n\nListen to the full episode: lukeattheroost.com")
    return "".join(parts)


def build_settings(clip: dict, platform: str) -> dict:
    if platform == "instagram":
        return {"__type": "instagram", "post_type": "post", "collaborators": []}
    if platform == "youtube":
        yt_tags = [{"value": h.lstrip("#"), "label": h.lstrip("#")}
                   for h in clip.get("hashtags", [])]
        return {
            "__type": "youtube",
            "title": clip["title"],
            "type": "public",
            "selfDeclaredMadeForKids": "no",
            "thumbnail": None,
            "tags": yt_tags,
        }
    return {"__type": platform}


def post_to_bluesky(clip: dict, clip_file: Path) -> bool:
    """Post a clip directly to Bluesky via atproto (bypasses Postiz)."""
    import time
    import httpx
    from atproto import models

    if not BSKY_APP_PASSWORD:
        print("    Error: BSKY_APP_PASSWORD not set in .env")
        return False

    client = BskyClient()
    client.login(BSKY_HANDLE, BSKY_APP_PASSWORD)
    did = client.me.did
    video_data = clip_file.read_bytes()

    # Get a service auth token scoped to the user's PDS (required by video service)
    from urllib.parse import urlparse
    pds_host = urlparse(client._session.pds_endpoint).hostname
    service_auth = client.com.atproto.server.get_service_auth(
        {"aud": f"did:web:{pds_host}", "lxm": "com.atproto.repo.uploadBlob"}
    )
    token = service_auth.token

    # Upload video to Bluesky's video processing service (not the PDS)
    print(f"    Uploading video ({len(video_data) / 1_000_000:.1f} MB)...")
    upload_resp = httpx.post(
        "https://video.bsky.app/xrpc/app.bsky.video.uploadVideo",
        params={"did": did, "name": clip_file.name},
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "video/mp4",
        },
        content=video_data,
        timeout=120,
    )
    if upload_resp.status_code not in (200, 409):
        print(f"    Upload failed: {upload_resp.status_code} {upload_resp.text[:200]}")
        return False

    upload_data = upload_resp.json()
    job_id = upload_data.get("jobId") or upload_data.get("jobStatus", {}).get("jobId")
    if not job_id:
        print(f"    No jobId returned: {upload_resp.text[:200]}")
        return False
    print(f"    Video processing (job {job_id})...")

    # Poll until video is processed
    session_token = client._session.access_jwt
    blob = None
    while True:
        status_resp = httpx.get(
            "https://video.bsky.app/xrpc/app.bsky.video.getJobStatus",
            params={"jobId": job_id},
            headers={"Authorization": f"Bearer {session_token}"},
            timeout=15,
        )
        resp_data = status_resp.json()
        status = resp_data.get("jobStatus") or resp_data
        state = status.get("state")
        if state == "JOB_STATE_COMPLETED":
            blob = status.get("blob")
            break
        if state == "JOB_STATE_FAILED":
            err = status.get("error") or status.get("message") or "unknown"
            print(f"    Video processing failed: {err}")
            return False
        progress = status.get("progress", 0)
        print(f"    Processing... {progress}%")
        time.sleep(3)

    if not blob:
        print("    No blob returned after processing")
        return False

    text = build_content(clip, "bluesky")

    embed = models.AppBskyEmbedVideo.Main(
        video=models.blob_ref.BlobRef(
            mime_type=blob["mimeType"],
            size=blob["size"],
            ref=models.blob_ref.IpldLink(link=blob["ref"]["$link"]),
        ),
        alt=clip.get("caption_text", clip["title"]),
        aspect_ratio=models.AppBskyEmbedDefs.AspectRatio(width=1080, height=1920),
    )
    client.send_post(text=text, embed=embed)
    return True


def create_post(integration_id: str, content: str, media: dict,
                settings: dict, schedule: str | None = None) -> dict:
    from datetime import datetime, timezone
    post_type = "schedule" if schedule else "now"
    date = schedule or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    payload = {
        "type": post_type,
        "date": date,
        "shortLink": False,
        "tags": [],
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


def main():
    valid_names = sorted(set(PLATFORM_ALIASES.keys()))
    parser = argparse.ArgumentParser(description="Upload podcast clips to social media via Postiz")
    parser.add_argument("clips_dir", help="Path to clips directory (e.g. clips/episode-12/)")
    parser.add_argument("--clip", "-c", type=int, help="Upload only clip N (1-indexed)")
    parser.add_argument("--platforms", "-p",
                        help=f"Comma-separated platforms ({','.join(ALL_PLATFORMS)}). Default: all")
    parser.add_argument("--schedule", "-s", help="Schedule time (ISO 8601, e.g. 2026-02-16T10:00:00)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without posting")
    args = parser.parse_args()

    if not POSTIZ_API_KEY:
        print("Error: POSTIZ_API_KEY not set in .env")
        sys.exit(1)

    if args.platforms:
        requested = []
        for p in args.platforms.split(","):
            p = p.strip().lower()
            if p not in PLATFORM_ALIASES:
                print(f"Unknown platform: {p}")
                print(f"Valid: {', '.join(valid_names)}")
                sys.exit(1)
            requested.append(PLATFORM_ALIASES[p])
        target_platforms = list(dict.fromkeys(requested))
    else:
        target_platforms = ALL_PLATFORMS[:]

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

    needs_postiz = not args.dry_run and any(
        p != "bluesky" for p in target_platforms)
    if needs_postiz:
        print("Fetching connected accounts from Postiz...")
        integrations = fetch_integrations()
    else:
        integrations = []

    active_platforms = {}
    for platform in target_platforms:
        if platform == "bluesky":
            if BSKY_APP_PASSWORD or args.dry_run:
                active_platforms[platform] = {"name": BSKY_HANDLE, "_direct": True}
            else:
                print("Warning: BSKY_APP_PASSWORD not set in .env, skipping Bluesky")
            continue
        if args.dry_run:
            active_platforms[platform] = {"name": PLATFORM_DISPLAY[platform]}
            continue
        integ = find_integration(integrations, platform)
        if integ:
            active_platforms[platform] = integ
        else:
            print(f"Warning: No {PLATFORM_DISPLAY[platform]} account connected in Postiz")

    if not args.dry_run and not active_platforms:
        print("Error: No platforms available to upload to")
        sys.exit(1)

    platform_names = [f"{PLATFORM_DISPLAY[p]} ({integ.get('name', 'connected')})"
                      for p, integ in active_platforms.items()]

    print(f"\nUploading {len(clips)} clip(s) to: {', '.join(platform_names)}")
    if args.schedule:
        print(f"Scheduled for: {args.schedule}")
    print()

    for i, clip in enumerate(clips):
        print(f"  {i+1}. \"{clip['title']}\" ({clip['duration']:.0f}s)")
        desc = clip.get('description', '')
        if len(desc) > 80:
            desc = desc[:desc.rfind(' ', 0, 80)] + '...'
        print(f"     {desc}")
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

    for i, clip in enumerate(clips):
        clip_file = clips_dir / clip["clip_file"]
        if not clip_file.exists():
            print(f"  Clip {i+1}: Video file not found: {clip_file}")
            continue

        print(f"\n  Clip {i+1}: \"{clip['title']}\"")

        postiz_platforms = {p: integ for p, integ in active_platforms.items()
                            if not integ.get("_direct")}

        media = None
        if postiz_platforms:
            print(f"    Uploading {clip_file.name}...")
            media = upload_file(clip_file)
            if not media:
                print("    Failed to upload video to Postiz, skipping Postiz platforms")
                postiz_platforms = {}
            else:
                print(f"    Uploaded: {media.get('path', 'ok')}")

        for platform, integ in postiz_platforms.items():
            display = PLATFORM_DISPLAY[platform]
            print(f"    Posting to {display}...")
            content = build_content(clip, platform)
            settings = build_settings(clip, platform)
            result = create_post(integ["id"], content, media, settings, args.schedule)
            if result:
                print(f"    {display}: Posted!")
            else:
                print(f"    {display}: Failed")

        if "bluesky" in active_platforms:
            print(f"    Posting to Bluesky (direct)...")
            try:
                if post_to_bluesky(clip, clip_file):
                    print(f"    Bluesky: Posted!")
                else:
                    print(f"    Bluesky: Failed")
            except Exception as e:
                print(f"    Bluesky: Failed — {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
