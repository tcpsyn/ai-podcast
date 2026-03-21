"""Fetch instrumental background music from Jamendo for the radio show.

Pixabay has no public music API — this uses Jamendo's free API instead.
All tracks are Creative Commons licensed. Attribution is saved to music/CREDITS.txt.

Setup: Get a free client_id at https://devportal.jamendo.com
       Add JAMENDO_CLIENT_ID=your_id to .env

Usage:
    python fetch_music.py                    # download 20 tracks across all genres
    python fetch_music.py --genre jazz       # download jazz only
    python fetch_music.py --count 50         # download 50 tracks
    python fetch_music.py --list             # just list available tracks, don't download
"""

import argparse
import os
import re
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

MUSIC_DIR = Path(__file__).parent / "music"
CREDITS_FILE = MUSIC_DIR / "CREDITS.txt"
API_BASE = "https://api.jamendo.com/v3.0"

# Genres good for a late-night radio show
GENRES = ["jazz", "lofi", "blues", "ambient", "acoustic", "funk", "chill"]

# Map search tags to labels that _detect_genre() in main.py can match
# jazz, blues, funk, lo-fi are already in GENRE_KEYWORDS
# ambient, acoustic, chill would need to be added for auto-detection
GENRE_LABELS = {
    "jazz": "Jazz",
    "lofi": "Lo-Fi",
    "blues": "Blues",
    "ambient": "Ambient",
    "acoustic": "Acoustic",
    "funk": "Funk",
    "chill": "Chill",
}


def get_client_id():
    key = os.getenv("JAMENDO_CLIENT_ID")
    if not key:
        print("Error: JAMENDO_CLIENT_ID not found in .env")
        print("Get one free at https://devportal.jamendo.com")
        sys.exit(1)
    return key


def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', '', name).strip()


def search_tracks(client: httpx.Client, client_id: str, genre: str, limit: int = 20) -> list[dict]:
    params = {
        "client_id": client_id,
        "format": "json",
        "limit": min(limit, 200),
        "vocalinstrumental": "instrumental",
        "fuzzytags": genre,
        "durationbetween": "60_300",
        "include": "musicinfo+licenses",
        "order": "popularity_total",
    }

    resp = client.get(f"{API_BASE}/tracks/", params=params)
    resp.raise_for_status()
    data = resp.json()

    if data["headers"]["status"] != "success":
        print(f"  API error: {data['headers'].get('error_message', 'unknown')}")
        return []

    return data.get("results", [])


def make_filename(track: dict, genre_tag: str) -> str:
    artist = sanitize_filename(track.get("artist_name", "Unknown"))
    title = sanitize_filename(track.get("name", "Untitled"))
    label = GENRE_LABELS.get(genre_tag, genre_tag.title())

    # Include genre tag if not already detectable from artist/title
    lower = f"{artist} {title}".lower()
    needs_tag = not any(kw in lower for kw in [genre_tag, label.lower()])

    if needs_tag:
        return f"{artist} - {title} [{label}].mp3"
    return f"{artist} - {title}.mp3"


def download_track(client: httpx.Client, track: dict, filepath: Path, index: int, total: int) -> bool:
    url = track.get("audiodownload")
    if not url:
        print(f"  [{index}/{total}] SKIP (no download URL): {track['name']}")
        return False

    if not track.get("audiodownload_allowed", True):
        print(f"  [{index}/{total}] SKIP (download not allowed): {track['name']}")
        return False

    print(f"  [{index}/{total}] Downloading: {filepath.name}...", end=" ", flush=True)
    resp = client.get(url, follow_redirects=True)
    resp.raise_for_status()
    filepath.write_bytes(resp.content)
    size_mb = len(resp.content) / (1024 * 1024)
    dur = track.get("duration", 0)
    print(f"{size_mb:.1f} MB, {dur // 60}:{dur % 60:02d}")
    return True


def save_credit(track: dict, filename: str):
    artist = track.get("artist_name", "Unknown")
    title = track.get("name", "Untitled")
    license_url = track.get("license_ccurl", "")
    share_url = track.get("shareurl", "")

    line = f"{filename} | {artist} - {title} | {license_url} | {share_url}\n"

    existing = CREDITS_FILE.read_text() if CREDITS_FILE.exists() else ""
    if filename not in existing:
        with open(CREDITS_FILE, "a") as f:
            if not existing:
                f.write("# Music Credits (Jamendo - Creative Commons)\n")
                f.write("# File | Artist - Title | License | URL\n\n")
            f.write(line)


def main():
    parser = argparse.ArgumentParser(description="Download instrumental music from Jamendo")
    parser.add_argument("--genre", choices=GENRES, help="Download only this genre")
    parser.add_argument("--count", type=int, default=20, help="Total tracks to download (default: 20)")
    parser.add_argument("--list", action="store_true", help="List available tracks without downloading")
    args = parser.parse_args()

    client_id = get_client_id()
    MUSIC_DIR.mkdir(exist_ok=True)

    genres = [args.genre] if args.genre else GENRES
    per_genre = max(1, args.count // len(genres))
    remainder = args.count - per_genre * len(genres)

    all_tracks = []
    seen_ids = set()

    with httpx.Client(timeout=30) as api_client:
        for i, genre in enumerate(genres):
            limit = per_genre + (1 if i < remainder else 0)
            if limit <= 0:
                continue
            print(f"Searching {genre}...", end=" ", flush=True)
            tracks = search_tracks(api_client, client_id, genre, limit)
            # Deduplicate across genres
            added = 0
            for t in tracks:
                if t["id"] not in seen_ids and added < limit:
                    t["_genre_tag"] = genre
                    all_tracks.append(t)
                    seen_ids.add(t["id"])
                    added += 1
            print(f"{added} tracks")

    if not all_tracks:
        print("No tracks found.")
        return

    if args.list:
        print(f"\n{'#':<4} {'Genre':<10} {'Artist':<25} {'Title':<40} {'Duration':<8}")
        print("-" * 90)
        for i, t in enumerate(all_tracks, 1):
            dur = f"{t['duration'] // 60}:{t['duration'] % 60:02d}"
            artist = t["artist_name"][:24]
            title = t["name"][:39]
            label = GENRE_LABELS.get(t["_genre_tag"], t["_genre_tag"])
            print(f"{i:<4} {label:<10} {artist:<25} {title:<40} {dur:<8}")
        print(f"\n{len(all_tracks)} tracks available")
        return

    # Download phase
    downloaded = 0
    skipped_exists = 0
    skipped_error = 0

    with httpx.Client(timeout=120, follow_redirects=True) as dl_client:
        for i, track in enumerate(all_tracks, 1):
            filename = make_filename(track, track["_genre_tag"])
            filepath = MUSIC_DIR / filename

            if filepath.exists():
                print(f"  [{i}/{len(all_tracks)}] EXISTS: {filename}")
                skipped_exists += 1
                continue

            try:
                if download_track(dl_client, track, filepath, i, len(all_tracks)):
                    save_credit(track, filename)
                    downloaded += 1
                else:
                    skipped_error += 1
            except Exception as e:
                print(f"  [{i}/{len(all_tracks)}] ERROR: {e}")
                # Clean up partial download
                if filepath.exists():
                    filepath.unlink()
                skipped_error += 1

    print(f"\nDone: {downloaded} downloaded, {skipped_exists} existed, {skipped_error} skipped")


if __name__ == "__main__":
    main()
