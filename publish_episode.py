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
import tempfile
import base64
from pathlib import Path

import ssl
import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from dotenv import load_dotenv


class TLSAdapter(HTTPAdapter):
    """Adapter to handle servers with older TLS configurations."""
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.set_ciphers('DEFAULT@SECLEVEL=1')
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)

    def send(self, *args, **kwargs):
        kwargs['verify'] = False
        return super().send(*args, **kwargs)


# Use a session with TLS compatibility for all Castopod requests
_session = requests.Session()
_session.mount('https://', TLSAdapter())

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
# BunnyCDN Storage
BUNNY_STORAGE_ZONE = "lukeattheroost"
BUNNY_STORAGE_KEY = "REDACTED_BUNNY_STORAGE_KEY"
BUNNY_STORAGE_REGION = "la"  # Los Angeles

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
    """Create episode on Castopod using curl (handles large file uploads better)."""
    print("[3/5] Creating episode on Castopod...")

    credentials = base64.b64encode(
        f"{CASTOPOD_USERNAME}:{CASTOPOD_PASSWORD}".encode()
    ).decode()
    slug = re.sub(r'[^a-z0-9]+', '-', metadata["title"].lower()).strip('-')

    cmd = [
        "curl", "-sk", "-X", "POST",
        f"{CASTOPOD_URL}/api/rest/v1/episodes",
        "-H", f"Authorization: Basic {credentials}",
        "-F", f"audio_file=@{audio_path};type=audio/mpeg",
        "-F", f"title={metadata['title']}",
        "-F", f"slug={slug}",
        "-F", f"description={metadata['description']}",
        "-F", "parental_advisory=explicit",
        "-F", "type=full",
        "-F", f"podcast_id={PODCAST_ID}",
        "-F", "created_by=1",
        "-F", "updated_by=1",
        "-F", f"episode_number={episode_number}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"Error uploading: {result.stderr}")
        sys.exit(1)

    try:
        episode = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Error parsing response: {result.stdout[:500]}")
        sys.exit(1)

    if "id" not in episode:
        print(f"Error creating episode: {result.stdout[:500]}")
        sys.exit(1)

    print(f"    Created episode ID: {episode['id']}")
    print(f"    Slug: {episode['slug']}")

    return episode


def publish_episode(episode_id: int) -> dict:
    """Publish the episode."""
    print("[4/5] Publishing episode...")

    headers = get_auth_header()

    response = _session.post(
        f"{CASTOPOD_URL}/api/rest/v1/episodes/{episode_id}/publish",
        headers=headers,
        data={
            "publication_method": "now",
            "created_by": "1"
        },
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


def run_ssh_command(command: str, timeout: int = 30) -> tuple[bool, str]:
    """Run a command on the NAS via SSH."""
    ssh_cmd = [
        "ssh", "-p", str(NAS_SSH_PORT),
        f"{NAS_USER}@{NAS_HOST}",
        command
    ]
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
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


def upload_to_bunny(local_path: str, remote_path: str, content_type: str = None) -> bool:
    """Upload a file to BunnyCDN Storage."""
    if not content_type:
        ext = Path(local_path).suffix.lower()
        content_type = {
            ".mp3": "audio/mpeg", ".png": "image/png", ".jpg": "image/jpeg",
            ".json": "application/json", ".srt": "application/x-subrip",
        }.get(ext, "application/octet-stream")

    url = f"https://{BUNNY_STORAGE_REGION}.storage.bunnycdn.com/{BUNNY_STORAGE_ZONE}/{remote_path}"
    with open(local_path, "rb") as f:
        resp = requests.put(url, data=f, headers={
            "AccessKey": BUNNY_STORAGE_KEY,
            "Content-Type": content_type,
        })
    if resp.status_code == 201:
        return True
    print(f"    Warning: BunnyCDN upload failed ({resp.status_code}): {resp.text[:200]}")
    return False


def download_from_castopod(file_key: str, local_path: str) -> bool:
    """Download a file from Castopod's container storage to local filesystem."""
    remote_filename = Path(file_key).name
    remote_tmp = f"/tmp/castopod_{remote_filename}"
    cp_cmd = f'{DOCKER_PATH} cp {CASTOPOD_CONTAINER}:/var/www/castopod/public/media/{file_key} {remote_tmp}'
    success, _ = run_ssh_command(cp_cmd, timeout=120)
    if not success:
        return False
    scp_cmd = [
        "scp", "-P", str(NAS_SSH_PORT),
        f"{NAS_USER}@{NAS_HOST}:{remote_tmp}",
        local_path
    ]
    try:
        result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=300)
        ok = result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        ok = False
    run_ssh_command(f"rm -f {remote_tmp}")
    return ok


def sync_episode_media_to_bunny(episode_id: int, already_uploaded: set):
    """Ensure all media linked to an episode exists on BunnyCDN."""
    ep_id = episode_id
    query = (
        "SELECT DISTINCT m.file_key FROM cp_media m WHERE m.id IN ("
        f"SELECT audio_id FROM cp_episodes WHERE id = {ep_id} "
        f"UNION ALL SELECT cover_id FROM cp_episodes WHERE id = {ep_id} AND cover_id IS NOT NULL "
        f"UNION ALL SELECT transcript_id FROM cp_episodes WHERE id = {ep_id} AND transcript_id IS NOT NULL "
        f"UNION ALL SELECT chapters_id FROM cp_episodes WHERE id = {ep_id} AND chapters_id IS NOT NULL)"
    )
    cmd = f'{DOCKER_PATH} exec {MARIADB_CONTAINER} mysql -u {DB_USER} -p{DB_PASS} {DB_NAME} -N -e "{query};"'
    success, output = run_ssh_command(cmd)
    if not success or not output:
        return
    file_keys = [line.strip() for line in output.strip().split('\n') if line.strip()]
    for file_key in file_keys:
        if file_key in already_uploaded:
            continue
        cdn_url = f"https://cdn.lukeattheroost.com/media/{file_key}"
        try:
            resp = requests.head(cdn_url, timeout=10)
            if resp.status_code == 200:
                continue
        except Exception:
            pass
        with tempfile.NamedTemporaryFile(suffix=Path(file_key).suffix, delete=False) as tmp:
            tmp_path = tmp.name
        try:
            if download_from_castopod(file_key, tmp_path):
                print(f"    Syncing to CDN: {file_key}")
                upload_to_bunny(tmp_path, f"media/{file_key}")
            else:
                print(f"    Warning: Could not sync {file_key} to CDN")
        finally:
            Path(tmp_path).unlink(missing_ok=True)


def get_next_episode_number() -> int:
    """Get the next episode number from Castopod."""
    headers = get_auth_header()

    response = _session.get(
        f"{CASTOPOD_URL}/api/rest/v1/episodes",
        headers=headers,
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
    parser.add_argument("--session-data", "-s", help="Path to session export JSON (from /api/session/export)")
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

    # Load session data if provided
    session_data = None
    if args.session_data:
        session_path = Path(args.session_data).expanduser().resolve()
        if session_path.exists():
            with open(session_path) as f:
                session_data = json.load(f)
            print(f"Loaded session data: {session_data.get('call_count', 0)} calls")
        else:
            print(f"Warning: Session data file not found: {session_path}")

    # Step 1: Transcribe
    transcript = transcribe_audio(str(audio_path))

    # Step 2: Generate metadata
    metadata = generate_metadata(transcript, episode_number)

    # Use session chapters if available (more accurate than LLM-generated)
    if session_data and session_data.get("chapters"):
        metadata["chapters"] = session_data["chapters"]
        print(f"    Using {len(metadata['chapters'])} chapters from session data")

    # Apply overrides
    if args.title:
        metadata["title"] = args.title
    if args.description:
        metadata["description"] = args.description

    # Save chapters file
    chapters_path = audio_path.with_suffix(".chapters.json")
    save_chapters(metadata, str(chapters_path))

    # Save transcript alongside episode if session data available
    if session_data and session_data.get("transcript"):
        transcript_path = audio_path.with_suffix(".transcript.txt")
        with open(transcript_path, "w") as f:
            f.write(session_data["transcript"])
        print(f"    Transcript saved to: {transcript_path}")

    if args.dry_run:
        print("\n[DRY RUN] Would publish with:")
        print(f"  Title: {metadata['title']}")
        print(f"  Description: {metadata['description']}")
        print(f"  Chapters: {json.dumps(metadata['chapters'], indent=2)}")
        print("\nChapters file saved. Run without --dry-run to publish.")
        return

    # Step 3: Create episode
    episode = create_episode(str(audio_path), metadata, episode_number)

    # Step 3.5: Upload to BunnyCDN
    print("[3.5/5] Uploading to BunnyCDN...")
    uploaded_keys = set()

    # Audio: download Castopod's copy (ensures byte-exact match with RSS metadata)
    ep_id = episode["id"]
    audio_media_cmd = f'{DOCKER_PATH} exec {MARIADB_CONTAINER} mysql -u {DB_USER} -p{DB_PASS} {DB_NAME} -N -e "SELECT m.file_key FROM cp_media m JOIN cp_episodes e ON e.audio_id = m.id WHERE e.id = {ep_id};"'
    success, audio_file_key = run_ssh_command(audio_media_cmd)
    if success and audio_file_key:
        audio_file_key = audio_file_key.strip()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_audio = tmp.name
        try:
            print(f"    Downloading from Castopod: {audio_file_key}")
            if download_from_castopod(audio_file_key, tmp_audio):
                print(f"    Uploading audio to BunnyCDN")
                upload_to_bunny(tmp_audio, f"media/{audio_file_key}", "audio/mpeg")
            else:
                print(f"    Castopod download failed, uploading original file")
                upload_to_bunny(str(audio_path), f"media/{audio_file_key}", "audio/mpeg")
        finally:
            Path(tmp_audio).unlink(missing_ok=True)
        uploaded_keys.add(audio_file_key)
    else:
        print(f"    Error: Could not determine audio file_key from Castopod DB")
        print(f"    Audio will be served from Castopod directly (not CDN)")

    # Chapters
    chapters_key = f"podcasts/{PODCAST_HANDLE}/{episode['slug']}-chapters.json"
    print(f"    Uploading chapters to BunnyCDN")
    upload_to_bunny(str(chapters_path), f"media/{chapters_key}")
    uploaded_keys.add(chapters_key)

    # Step 4: Publish
    episode = publish_episode(episode["id"])

    # Step 4.5: Upload chapters via SSH
    chapters_uploaded = upload_chapters_to_castopod(
        episode["slug"],
        episode["id"],
        str(chapters_path)
    )

    # Sync any remaining episode media to BunnyCDN (cover art, transcripts, etc.)
    print("    Syncing episode media to CDN...")
    sync_episode_media_to_bunny(episode["id"], uploaded_keys)

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
