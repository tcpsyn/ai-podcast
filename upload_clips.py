#!/usr/bin/env python3
"""Upload podcast clips to social media (direct YouTube & Bluesky, Postiz for others).

Usage:
    python upload_clips.py                          # interactive: pick episode, clips, platforms
    python upload_clips.py clips/episode-12/        # pick clips and platforms interactively
    python upload_clips.py clips/episode-12/ --clip 1 --platforms ig,yt
    python upload_clips.py clips/episode-12/ --yes  # skip all prompts, upload everything
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

YT_CLIENT_SECRETS = Path(__file__).parent / "youtube_client_secrets.json"
YT_TOKEN_FILE = Path(__file__).parent / "youtube_token.json"

PLATFORM_ALIASES = {
    "ig": "instagram", "insta": "instagram", "instagram": "instagram",
    "yt": "youtube", "youtube": "youtube",
    "fb": "facebook", "facebook": "facebook",
    "bsky": "bluesky", "bluesky": "bluesky",
    "masto": "mastodon", "mastodon": "mastodon",
    "nostr": "nostr",
    "li": "linkedin", "linkedin": "linkedin",
    "threads": "threads",
    "tt": "tiktok", "tiktok": "tiktok",
}

PLATFORM_DISPLAY = {
    "instagram": "Instagram Reels",
    "youtube": "YouTube Shorts",
    "facebook": "Facebook Reels",
    "bluesky": "Bluesky",
    "mastodon": "Mastodon",
    "nostr": "Nostr",
    "linkedin": "LinkedIn",
    "threads": "Threads",
    "tiktok": "TikTok",
}

ALL_PLATFORMS = list(PLATFORM_DISPLAY.keys())

UPLOAD_LEDGER_FILE = "upload-history.json"


def load_upload_history(clips_dir: Path) -> dict:
    """Load upload history for a clips directory.
    Returns dict mapping clip_file -> list of platforms already uploaded to.
    """
    ledger = clips_dir / UPLOAD_LEDGER_FILE
    if ledger.exists():
        with open(ledger) as f:
            return json.load(f)
    return {}


def save_upload_history(clips_dir: Path, history: dict):
    with open(clips_dir / UPLOAD_LEDGER_FILE, "w") as f:
        json.dump(history, f, indent=2)


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
    if platform == "tiktok":
        return {
            "__type": "tiktok",
            "privacy_level": "PUBLIC_TO_EVERYONE",
            "duet": False,
            "stitch": False,
            "comment": True,
            "autoAddMusic": "no",
            "brand_content_toggle": False,
            "brand_organic_toggle": False,
            "content_posting_method": "DIRECT_POST",
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


def get_youtube_service():
    """Authenticate with YouTube API. First run opens a browser, then reuses saved token."""
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build as yt_build

    scopes = ["https://www.googleapis.com/auth/youtube.upload"]
    creds = None

    if YT_TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(YT_TOKEN_FILE), scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not YT_CLIENT_SECRETS.exists():
                print("    Error: youtube_client_secrets.json not found")
                print("    Download OAuth2 Desktop App credentials from Google Cloud Console")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(str(YT_CLIENT_SECRETS), scopes)
            creds = flow.run_local_server(port=8090)

        with open(YT_TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return yt_build("youtube", "v3", credentials=creds)


def post_to_youtube(clip: dict, clip_file: Path) -> bool:
    """Upload a clip directly to YouTube Shorts via the Data API."""
    import time
    import random
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError

    youtube = get_youtube_service()
    if not youtube:
        return False

    title = clip["title"]
    if "#Shorts" not in title:
        title = f"{title} #Shorts"

    description = build_content(clip, "youtube")
    if "#Shorts" not in description:
        description += "\n\n#Shorts"

    tags = [h.lstrip("#") for h in clip.get("hashtags", [])]
    if "Shorts" not in tags:
        tags.insert(0, "Shorts")

    body = {
        "snippet": {
            "title": title[:100],
            "description": description,
            "tags": tags,
            "categoryId": "24",  # Entertainment
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        str(clip_file),
        mimetype="video/mp4",
        chunksize=256 * 1024,
        resumable=True,
    )

    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    file_size = clip_file.stat().st_size / 1_000_000
    print(f"    Uploading video ({file_size:.1f} MB)...")

    response = None
    retry = 0
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                print(f"    Upload {int(status.progress() * 100)}%...")
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504) and retry < 5:
                retry += 1
                wait = random.random() * (2 ** retry)
                print(f"    Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                print(f"    YouTube API error: {e}")
                return False

    video_id = response["id"]
    print(f"    https://youtube.com/shorts/{video_id}")
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
    parser.add_argument("clips_dir", nargs="?", help="Path to clips directory (e.g. clips/episode-12/). If omitted, shows a picker.")
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

    # Resolve clips directory — pick interactively if not provided
    if args.clips_dir:
        clips_dir = Path(args.clips_dir).expanduser().resolve()
    else:
        clips_root = Path(__file__).parent / "clips"
        episode_dirs = sorted(
            [d for d in clips_root.iterdir()
             if d.is_dir() and not d.name.startswith(".") and (d / "clips-metadata.json").exists()],
            key=lambda d: d.name,
        )
        if not episode_dirs:
            print("No clip directories found in clips/. Run make_clips.py first.")
            sys.exit(1)
        print("\nAvailable episodes:\n")
        for i, d in enumerate(episode_dirs):
            with open(d / "clips-metadata.json") as f:
                meta = json.load(f)
            print(f"  {i+1}. {d.name} ({len(meta)} clip{'s' if len(meta) != 1 else ''})")
        print()
        while True:
            try:
                choice = input("Which episode? ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(episode_dirs):
                    clips_dir = episode_dirs[idx]
                    break
                print(f"  Enter 1-{len(episode_dirs)}")
            except (ValueError, EOFError):
                print(f"  Enter an episode number")

    metadata_path = clips_dir / "clips-metadata.json"
    if not metadata_path.exists():
        print(f"Error: No clips-metadata.json found in {clips_dir}")
        print("Run make_clips.py first to generate clips and metadata.")
        sys.exit(1)

    with open(metadata_path) as f:
        clips = json.load(f)

    # Pick clips
    if args.clip:
        if args.clip < 1 or args.clip > len(clips):
            print(f"Error: Clip {args.clip} not found (have {len(clips)} clips)")
            sys.exit(1)
        clips = [clips[args.clip - 1]]
    elif not args.yes:
        print(f"\nFound {len(clips)} clip(s):\n")
        for i, clip in enumerate(clips):
            desc = clip.get('description', clip.get('caption_text', ''))
            if len(desc) > 70:
                desc = desc[:desc.rfind(' ', 0, 70)] + '...'
            print(f"  {i+1}. \"{clip['title']}\" ({clip['duration']:.0f}s)")
            print(f"     {desc}")
        print(f"\n  a. All clips")
        print()
        while True:
            choice = input("Which clips? (e.g. 1,3 or a for all): ").strip().lower()
            if choice in ('a', 'all'):
                break
            try:
                indices = [int(x.strip()) for x in choice.split(",")]
                if all(1 <= x <= len(clips) for x in indices):
                    clips = [clips[x - 1] for x in indices]
                    break
                print(f"  Invalid selection. Enter 1-{len(clips)}, comma-separated, or 'a' for all.")
            except (ValueError, EOFError):
                print(f"  Enter clip numbers (e.g. 1,3) or 'a' for all")

    # Pick platforms
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
    elif not args.yes:
        print(f"\nPlatforms:\n")
        for i, p in enumerate(ALL_PLATFORMS):
            print(f"  {i+1}. {PLATFORM_DISPLAY[p]}")
        print(f"\n  a. All platforms (default)")
        print()
        choice = input("Which platforms? (e.g. 1,3,5 or a for all) [a]: ").strip().lower()
        if choice and choice not in ('a', 'all'):
            try:
                indices = [int(x.strip()) for x in choice.split(",")]
                target_platforms = [ALL_PLATFORMS[x - 1] for x in indices if 1 <= x <= len(ALL_PLATFORMS)]
                if not target_platforms:
                    target_platforms = ALL_PLATFORMS[:]
            except (ValueError, IndexError):
                target_platforms = ALL_PLATFORMS[:]
        else:
            target_platforms = ALL_PLATFORMS[:]
    else:
        target_platforms = ALL_PLATFORMS[:]

    DIRECT_PLATFORMS = {"bluesky", "youtube"}
    needs_postiz = not args.dry_run and any(
        p not in DIRECT_PLATFORMS for p in target_platforms)
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
        if platform == "youtube":
            if YT_CLIENT_SECRETS.exists() or YT_TOKEN_FILE.exists() or args.dry_run:
                active_platforms[platform] = {"name": "YouTube Shorts", "_direct": True}
            else:
                print("Warning: youtube_client_secrets.json not found, skipping YouTube")
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

    upload_history = load_upload_history(clips_dir)

    for i, clip in enumerate(clips):
        clip_file = clips_dir / clip["clip_file"]
        if not clip_file.exists():
            print(f"  Clip {i+1}: Video file not found: {clip_file}")
            continue

        clip_key = clip["clip_file"]
        already_uploaded = set(upload_history.get(clip_key, []))
        remaining_platforms = {p: integ for p, integ in active_platforms.items()
                               if p not in already_uploaded}

        if not remaining_platforms:
            print(f"\n  Clip {i+1}: \"{clip['title']}\" — already uploaded to all selected platforms, skipping")
            continue

        skipped = already_uploaded & set(active_platforms.keys())
        if skipped:
            print(f"\n  Clip {i+1}: \"{clip['title']}\" (skipping already uploaded: {', '.join(sorted(skipped))})")
        else:
            print(f"\n  Clip {i+1}: \"{clip['title']}\"")

        postiz_platforms = {p: integ for p, integ in remaining_platforms.items()
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
                upload_history.setdefault(clip_key, []).append(platform)
                save_upload_history(clips_dir, upload_history)
            else:
                print(f"    {display}: Failed")

        if "youtube" in remaining_platforms:
            print(f"    Posting to YouTube Shorts (direct)...")
            try:
                if post_to_youtube(clip, clip_file):
                    print(f"    YouTube: Posted!")
                    upload_history.setdefault(clip_key, []).append("youtube")
                    save_upload_history(clips_dir, upload_history)
                else:
                    print(f"    YouTube: Failed")
            except Exception as e:
                print(f"    YouTube: Failed — {e}")

        if "bluesky" in remaining_platforms:
            print(f"    Posting to Bluesky (direct)...")
            try:
                if post_to_bluesky(clip, clip_file):
                    print(f"    Bluesky: Posted!")
                    upload_history.setdefault(clip_key, []).append("bluesky")
                    save_upload_history(clips_dir, upload_history)
                else:
                    print(f"    Bluesky: Failed")
            except Exception as e:
                print(f"    Bluesky: Failed — {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
