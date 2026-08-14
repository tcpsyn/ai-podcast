"""Download vocal-free background music from Jamendo (CC-licensed).

Targets late-night talk-radio vibe + hip-hop. Skips tracks shorter than 60s,
dedupes against existing files in music/, and appends CREDITS.txt entries.

Usage:
    python download_music.py            # 100 tracks across all buckets
    python download_music.py --count 30 # smaller batch
    python download_music.py --dry-run  # show what would be downloaded
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlencode

from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("JAMENDO_CLIENT_ID")
if not CLIENT_ID:
    print("ERROR: JAMENDO_CLIENT_ID not set in .env", file=sys.stderr)
    sys.exit(1)

MUSIC_DIR = Path(__file__).parent / "music"
CREDITS_FILE = MUSIC_DIR / "CREDITS.txt"

# (tag_query, genre_label, target_count) — hip-hop weighted heaviest per user pref
BUCKETS = [
    ("hiphop+instrumental", "Hip-Hop",  40),
    ("jazz",                "Jazz",     20),
    ("lofi",                "Lo-Fi",    15),
    ("funk",                "Funk",     15),
    ("soul",                "Soul",     10),
]

# Filenames already on disk — skip duplicates by (artist, title) signature
def _existing_signatures() -> set[str]:
    sigs = set()
    for f in MUSIC_DIR.glob("*.mp3"):
        # "Artist - Title [Genre].mp3" or "Artist - Title.mp3"
        stem = f.stem
        stem = re.sub(r"\s*\[[^\]]+\]\s*$", "", stem)
        sigs.add(stem.lower().strip())
    for f in MUSIC_DIR.glob("*.wav"):
        sigs.add(f.stem.lower().strip())
    return sigs


def _sanitize(s: str) -> str:
    s = s.replace("/", "-").replace("\\", "-")
    s = re.sub(r'[<>:"|?*]', "", s)
    return s.strip()


def _fetch_jamendo_page(tag_query: str, offset: int, limit: int = 50) -> list[dict]:
    params = {
        "client_id": CLIENT_ID,
        "format": "json",
        "limit": limit,
        "offset": offset,
        "vocalinstrumental": "instrumental",
        "fuzzytags": tag_query,
        "audioformat": "mp32",
        "include": "musicinfo+licenses",
        "audiodlallowed": "true",
        "ccnd": "true",      # allow non-derivative (we won't modify)
        "order": "popularity_total",
    }
    url = "https://api.jamendo.com/v3.0/tracks/?" + urlencode(params)
    with urlopen(Request(url, headers={"User-Agent": "ai-podcast-music-fetcher/1.0"}), timeout=30) as r:
        import json
        data = json.load(r)
    if data.get("headers", {}).get("status") != "success":
        print(f"  API error: {data.get('headers', {}).get('error_message')}")
        return []
    return data.get("results", [])


def _download(url: str, dest: Path) -> bool:
    try:
        req = Request(url, headers={"User-Agent": "ai-podcast-music-fetcher/1.0"})
        with urlopen(req, timeout=120) as r, open(dest, "wb") as out:
            while True:
                chunk = r.read(64 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        return True
    except Exception as e:
        print(f"  download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def fetch_bucket(tag_query: str, genre_label: str, target: int, existing: set[str], dry_run: bool) -> list[tuple[Path, dict]]:
    """Returns list of (path, track_info) successfully downloaded."""
    print(f"\n=== {genre_label} (target {target}) ===")
    downloaded: list[tuple[Path, dict]] = []
    offset = 0
    seen_ids = set()
    while len(downloaded) < target and offset < 500:  # cap pagination
        page = _fetch_jamendo_page(tag_query, offset)
        if not page:
            break
        offset += len(page)
        for track in page:
            if len(downloaded) >= target:
                break
            tid = track.get("id")
            if tid in seen_ids:
                continue
            seen_ids.add(tid)

            if track.get("duration", 0) < 60:
                continue
            if not track.get("audiodownload_allowed"):
                continue

            artist = _sanitize(track.get("artist_name", "Unknown"))
            name   = _sanitize(track.get("name", "Untitled"))
            sig = f"{artist} - {name}".lower().strip()
            if sig in existing:
                continue

            filename = f"{artist} - {name} [{genre_label}].mp3"
            dest = MUSIC_DIR / filename

            audio_url = track.get("audiodownload") or track.get("audio")
            if not audio_url:
                continue

            if dry_run:
                print(f"  [DRY] {filename}  ({track.get('duration')}s)")
                downloaded.append((dest, track))
                existing.add(sig)
                continue

            print(f"  ↓ {filename}  ({track.get('duration')}s)")
            if _download(audio_url, dest):
                downloaded.append((dest, track))
                existing.add(sig)
                time.sleep(0.5)  # be polite
        if len(page) < 50:
            break
    return downloaded


def append_credits(entries: list[tuple[Path, dict]]):
    if not entries:
        return
    with open(CREDITS_FILE, "a") as f:
        f.write(f"\n# Added {time.strftime('%Y-%m-%d')} — vocal-free batch via Jamendo API\n")
        for dest, track in entries:
            license_url = track.get("license_ccurl", "")
            share_url = track.get("shareurl", "")
            artist = track.get("artist_name", "")
            name = track.get("name", "")
            f.write(f"{dest.name} | {artist} - {name} | {license_url} | {share_url}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100, help="Total tracks (default 100)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    MUSIC_DIR.mkdir(exist_ok=True)
    existing = _existing_signatures()
    print(f"Existing tracks: {len(existing)}")

    # Scale bucket targets proportionally to --count
    scale = args.count / sum(b[2] for b in BUCKETS)
    all_new: list[tuple[Path, dict]] = []
    for tag, label, weight in BUCKETS:
        target = max(1, round(weight * scale))
        all_new.extend(fetch_bucket(tag, label, target, existing, args.dry_run))

    print(f"\n=== Done. {len(all_new)} new tracks {'planned' if args.dry_run else 'downloaded'}. ===")
    if not args.dry_run:
        append_credits(all_new)
        print(f"CREDITS.txt updated.")


if __name__ == "__main__":
    main()
