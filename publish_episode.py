#!/usr/bin/env python3
"""
Podcast Episode Publisher
Transcribes audio, generates metadata, and publishes to Castopod.

Usage:
    python publish_episode.py /path/to/episode.mp3
    python publish_episode.py /path/to/episode.mp3 --episode-number 3
    python publish_episode.py /path/to/episode.mp3 --dry-run
"""

import argparse
import json
import os
import re
import subprocess
import sys
import base64
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# Configuration
CASTOPOD_URL = "https://podcast.macneilmediagroup.com"
CASTOPOD_USERNAME = "admin"
CASTOPOD_PASSWORD = "REDACTED_CASTOPOD_PASSWORD"
PODCAST_ID = 1
PODCAST_HANDLE = "LukeAtTheRoost"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large

# NAS Configuration for chapters upload
NAS_HOST = "mmgnas-10g"
NAS_USER = "luke"
NAS_SSH_PORT = 8001
DOCKER_PATH = "/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker"
CASTOPOD_CONTAINER = "castopod-castopod-1"
MARIADB_CONTAINER = "castopod-mariadb-1"
DB_USER = "castopod"
DB_PASS = "REDACTED_DB_PASSWORD"
DB_NAME = "castopod"


def get_auth_header():
    """Get Basic Auth header for Castopod API."""
    credentials = base64.b64encode(
        f"{CASTOPOD_USERNAME}:{CASTOPOD_PASSWORD}".encode()
    ).decode()
    return {"Authorization": f"Basic {credentials}"}


def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio using faster-whisper with timestamps."""
    print(f"[1/5] Transcribing {audio_path}...")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Error: faster-whisper not installed. Run: pip install faster-whisper")
        sys.exit(1)

    model = WhisperModel(WHISPER_MODEL, compute_type="int8")
    segments, info = model.transcribe(audio_path, word_timestamps=True)

    transcript_segments = []
    full_text = []

    for segment in segments:
        transcript_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
        full_text.append(segment.text.strip())

    print(f"    Transcribed {info.duration:.1f} seconds of audio")

    return {
        "segments": transcript_segments,
        "full_text": " ".join(full_text),
        "duration": int(info.duration)
    }


def generate_metadata(transcript: dict, episode_number: int) -> dict:
    """Use LLM to generate title, description, and chapters from transcript."""
    print("[2/5] Generating metadata with LLM...")

    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not set in .env")
        sys.exit(1)

    # Prepare transcript with timestamps for chapter detection
    timestamped_text = ""
    for seg in transcript["segments"]:
        mins = int(seg["start"] // 60)
        secs = int(seg["start"] % 60)
        timestamped_text += f"[{mins:02d}:{secs:02d}] {seg['text']}\n"

    prompt = f"""Analyze this podcast transcript and generate metadata.

TRANSCRIPT:
{timestamped_text}

Generate a JSON response with:
1. "title": A catchy episode title (include "Episode {episode_number}:" prefix)
2. "description": A 2-4 sentence description summarizing the episode's content. Mention callers by name and their topics. End with something engaging.
3. "chapters": An array of chapter objects with "startTime" (in seconds) and "title". Include:
   - "Intro" at 0 seconds
   - A chapter for each caller/topic (use caller names if mentioned)
   - "Outro" near the end

Respond with ONLY valid JSON, no markdown or explanation."""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "anthropic/claude-3-haiku",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
    )

    if response.status_code != 200:
        print(f"Error from OpenRouter: {response.text}")
        sys.exit(1)

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    # Parse JSON from response (handle markdown code blocks)
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\n?", "", content)
        content = re.sub(r"\n?```$", "", content)

    try:
        metadata = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Response was: {content}")
        sys.exit(1)

    print(f"    Title: {metadata['title']}")
    print(f"    Chapters: {len(metadata['chapters'])}")

    return metadata


def create_episode(audio_path: str, metadata: dict, episode_number: int) -> dict:
    """Create episode on Castopod."""
    print("[3/5] Creating episode on Castopod...")

    headers = get_auth_header()
    slug = re.sub(r'[^a-z0-9]+', '-', metadata["title"].lower()).strip('-')

    # Upload audio and create episode
    with open(audio_path, "rb") as f:
        files = {
            "audio_file": (Path(audio_path).name, f, "audio/mpeg")
        }
        data = {
            "title": metadata["title"],
            "slug": slug,
            "description": metadata["description"],
            "parental_advisory": "explicit",
            "type": "full",
            "podcast_id": str(PODCAST_ID),
            "created_by": "1",
            "updated_by": "1",
            "episode_number": str(episode_number),
        }

        response = requests.post(
            f"{CASTOPOD_URL}/api/rest/v1/episodes",
            headers=headers,
            files=files,
            data=data
        )

    if response.status_code not in (200, 201):
        print(f"Error creating episode: {response.status_code} {response.text}")
        sys.exit(1)

    episode = response.json()
    print(f"    Created episode ID: {episode['id']}")
    print(f"    Slug: {episode['slug']}")

    return episode


def publish_episode(episode_id: int) -> dict:
    """Publish the episode."""
    print("[4/5] Publishing episode...")

    headers = get_auth_header()

    response = requests.post(
        f"{CASTOPOD_URL}/api/rest/v1/episodes/{episode_id}/publish",
        headers=headers,
        data={
            "publication_method": "now",
            "created_by": "1"
        }
    )

    if response.status_code != 200:
        print(f"Error publishing: {response.text}")
        sys.exit(1)

    episode = response.json()
    published_at = episode.get("published_at", {})
    if isinstance(published_at, dict):
        print(f"    Published at: {published_at.get('date', 'unknown')}")
    else:
        print(f"    Published at: {published_at}")

    return episode


def save_chapters(metadata: dict, output_path: str):
    """Save chapters to JSON file."""
    chapters_data = {
        "version": "1.2.0",
        "chapters": metadata["chapters"]
    }

    with open(output_path, "w") as f:
        json.dump(chapters_data, f, indent=2)

    print(f"    Chapters saved to: {output_path}")


def run_ssh_command(command: str) -> tuple[bool, str]:
    """Run a command on the NAS via SSH."""
    ssh_cmd = [
        "ssh", "-p", str(NAS_SSH_PORT),
        f"{NAS_USER}@{NAS_HOST}",
        command
    ]
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout.strip() or result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "SSH command timed out"
    except Exception as e:
        return False, str(e)


def upload_chapters_to_castopod(episode_slug: str, episode_id: int, chapters_path: str) -> bool:
    """Upload chapters file to Castopod via SSH and link in database."""
    print("[4.5/5] Uploading chapters to Castopod...")

    chapters_filename = f"{episode_slug}-chapters.json"
    remote_path = f"podcasts/{PODCAST_HANDLE}/{chapters_filename}"

    # Read local chapters file
    with open(chapters_path, "r") as f:
        chapters_content = f.read()

    # Base64 encode for safe transfer
    chapters_b64 = base64.b64encode(chapters_content.encode()).decode()

    # Upload file to container using base64 decode
    upload_cmd = f'echo "{chapters_b64}" | base64 -d | {DOCKER_PATH} exec -i {CASTOPOD_CONTAINER} tee /var/www/castopod/public/media/{remote_path} > /dev/null'
    success, output = run_ssh_command(upload_cmd)
    if not success:
        print(f"    Warning: Failed to upload chapters file: {output}")
        return False

    # Get file size
    file_size = len(chapters_content)

    # Insert into media table
    insert_sql = f"""INSERT INTO cp_media (file_key, file_size, file_mimetype, type, uploaded_by, updated_by, uploaded_at, updated_at)
        VALUES ('{remote_path}', {file_size}, 'application/json', 'chapters', 1, 1, NOW(), NOW())"""
    db_cmd = f'{DOCKER_PATH} exec {MARIADB_CONTAINER} mysql -u {DB_USER} -p{DB_PASS} {DB_NAME} -e "{insert_sql}; SELECT LAST_INSERT_ID();"'
    success, output = run_ssh_command(db_cmd)
    if not success:
        print(f"    Warning: Failed to insert chapters in database: {output}")
        return False

    # Parse media ID from output
    try:
        lines = output.strip().split('\n')
        media_id = int(lines[-1])
    except (ValueError, IndexError):
        print(f"    Warning: Could not parse media ID from: {output}")
        return False

    # Link chapters to episode
    update_sql = f"UPDATE cp_episodes SET chapters_id = {media_id} WHERE id = {episode_id}"
    db_cmd = f'{DOCKER_PATH} exec {MARIADB_CONTAINER} mysql -u {DB_USER} -p{DB_PASS} {DB_NAME} -e "{update_sql}"'
    success, output = run_ssh_command(db_cmd)
    if not success:
        print(f"    Warning: Failed to link chapters to episode: {output}")
        return False

    # Clear Castopod cache
    cache_cmd = f'{DOCKER_PATH} exec {CASTOPOD_CONTAINER} php spark cache:clear'
    run_ssh_command(cache_cmd)

    print(f"    Chapters uploaded and linked (media_id: {media_id})")
    return True


def get_next_episode_number() -> int:
    """Get the next episode number from Castopod."""
    headers = get_auth_header()

    response = requests.get(
        f"{CASTOPOD_URL}/api/rest/v1/episodes",
        headers=headers
    )

    if response.status_code != 200:
        return 1

    episodes = response.json()
    if not episodes:
        return 1

    # Filter to our podcast
    our_episodes = [ep for ep in episodes if ep.get("podcast_id") == PODCAST_ID]
    if not our_episodes:
        return 1

    max_num = max(ep.get("number", 0) or 0 for ep in our_episodes)
    return max_num + 1


def main():
    parser = argparse.ArgumentParser(description="Publish podcast episode to Castopod")
    parser.add_argument("audio_file", help="Path to the audio file (MP3)")
    parser.add_argument("--episode-number", "-n", type=int, help="Episode number (auto-detected if not provided)")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Generate metadata but don't publish")
    parser.add_argument("--title", "-t", help="Override generated title")
    parser.add_argument("--description", help="Override generated description")
    args = parser.parse_args()

    audio_path = Path(args.audio_file).expanduser().resolve()
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    # Determine episode number
    if args.episode_number:
        episode_number = args.episode_number
    else:
        episode_number = get_next_episode_number()
    print(f"Episode number: {episode_number}")

    # Step 1: Transcribe
    transcript = transcribe_audio(str(audio_path))

    # Step 2: Generate metadata
    metadata = generate_metadata(transcript, episode_number)

    # Apply overrides
    if args.title:
        metadata["title"] = args.title
    if args.description:
        metadata["description"] = args.description

    # Save chapters file
    chapters_path = audio_path.with_suffix(".chapters.json")
    save_chapters(metadata, str(chapters_path))

    if args.dry_run:
        print("\n[DRY RUN] Would publish with:")
        print(f"  Title: {metadata['title']}")
        print(f"  Description: {metadata['description']}")
        print(f"  Chapters: {json.dumps(metadata['chapters'], indent=2)}")
        print("\nChapters file saved. Run without --dry-run to publish.")
        return

    # Step 3: Create episode
    episode = create_episode(str(audio_path), metadata, episode_number)

    # Step 4: Publish
    episode = publish_episode(episode["id"])

    # Step 4.5: Upload chapters via SSH
    chapters_uploaded = upload_chapters_to_castopod(
        episode["slug"],
        episode["id"],
        str(chapters_path)
    )

    # Step 5: Summary
    print("\n[5/5] Done!")
    print("=" * 50)
    print(f"Episode URL: {CASTOPOD_URL}/@{PODCAST_HANDLE}/episodes/{episode['slug']}")
    print(f"RSS Feed: {CASTOPOD_URL}/@{PODCAST_HANDLE}/feed.xml")
    print("=" * 50)
    if not chapters_uploaded:
        print("\nNote: Chapters upload failed. Add manually via Castopod admin UI")
        print(f"      Chapters file: {chapters_path}")


if __name__ == "__main__":
    main()
