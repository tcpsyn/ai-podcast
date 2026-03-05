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
import base64
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
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

WHISPER_MODEL = "distil-large-v3"

# YouTube
YT_CLIENT_SECRETS = Path(__file__).parent / "youtube_client_secrets.json"
YT_TOKEN_FILE = Path(__file__).parent / "youtube_token.json"
YT_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]
YT_PODCAST_PLAYLIST = "PLGq4uZyNV1yYH_rcitTTPVysPbC6-7pe-"

# Postiz (social media posting)
POSTIZ_URL = "https://social.lukeattheroost.com"
POSTIZ_JWT_SECRET = "REDACTED_POSTIZ_JWT_SECRET"
POSTIZ_USER_ID = "REDACTED_POSTIZ_USER_ID"
POSTIZ_INTEGRATIONS = {
    "facebook": {"id": "REDACTED_FB_ID"},
    "instagram": {"id": "REDACTED_IG_ID"},
    "discord": {"id": "REDACTED_DISCORD_ID", "channel": "REDACTED_DISCORD_CHANNEL"},
    "bluesky": {"id": "REDACTED_BSKY_ID"},
    "mastodon": {"id": "REDACTED_MASTODON_ID"},
    "nostr": {"id": "REDACTED_NOSTR_ID"},
    "linkedin": {"id": "REDACTED_LINKEDIN_ID"},
    "threads": {"id": "REDACTED_THREADS_ID"},
    # TikTok excluded — requires video, not image posts. Use upload_clips.py instead.
    # "tiktok": {"id": "REDACTED_TIKTOK_ID"},
}

# NAS Configuration for chapters upload
# BunnyCDN Storage
BUNNY_STORAGE_ZONE = "lukeattheroost"
BUNNY_STORAGE_KEY = "REDACTED_BUNNY_STORAGE_KEY"
BUNNY_STORAGE_REGION = "la"  # Los Angeles

NAS_HOST = "mmgnas"
NAS_USER = "luke"
NAS_SSH_PORT = 8001
DOCKER_PATH = "/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker"
CASTOPOD_CONTAINER = "castopod-castopod-1"
MARIADB_CONTAINER = "castopod-mariadb-1"
DB_USER = "castopod"
DB_PASS = "REDACTED_DB_PASSWORD"
DB_NAME = "castopod"

LOCK_FILE = Path(__file__).parent / ".publish.lock"


def get_auth_header():
    """Get Basic Auth header for Castopod API."""
    credentials = base64.b64encode(
        f"{CASTOPOD_USERNAME}:{CASTOPOD_PASSWORD}".encode()
    ).decode()
    return {"Authorization": f"Basic {credentials}"}


def label_transcript_speakers(text):
    """Add LUKE:/CALLER: speaker labels to transcript using LLM."""
    import time as _time

    prompt = """Insert speaker labels into this radio show transcript. The show is "Luke at the Roost". The host is LUKE. Callers call in one at a time.

CRITICAL: Output EVERY SINGLE WORD from the input. Do NOT summarize, shorten, paraphrase, or skip ANY text. The output must contain the EXACT SAME words as the input, with ONLY speaker labels and line breaks added.

At each speaker change, insert a blank line and the new speaker's label (e.g., "LUKE:" or "REGGIE:").

Speaker identification:
- LUKE is the host — he introduces callers, asks questions, does sponsor reads, opens and closes the show
- Callers are introduced by name by Luke (e.g., "let's talk to Earl", "next up Brenda")
- Use caller FIRST NAME in caps as the label
- When Luke says "Tell me about..." or asks a question, that's LUKE
- When someone responds with their story/opinion/answer, that's the CALLER

Output format — ONLY the labeled transcript with blank lines between turns. No notes, no commentary. Do NOT add any bracketed notes like [Continued...], [Note:...], [Sponsor read], etc. Do NOT add meta-commentary about the transcript. ONLY output the spoken words with speaker labels.

TRANSCRIPT:
"""
    # Chunk text into ~8000 char segments
    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= 8000:
            if chunks and len(remaining) < 1000:
                chunks[-1] = chunks[-1] + " " + remaining
            else:
                chunks.append(remaining)
            break
        pos = remaining[:8000].rfind('. ')
        if pos < 4000:
            pos = remaining[:8000].rfind('? ')
        if pos < 4000:
            pos = remaining[:8000].rfind('! ')
        if pos < 4000:
            pos = 8000
        chunks.append(remaining[:pos + 1].strip())
        remaining = remaining[pos + 1:].strip()

    labeled_parts = []
    context = ""
    for i, chunk in enumerate(chunks):
        full_prompt = prompt + chunk
        if context:
            full_prompt += f"\n\nCONTEXT: The previous section ended with speaker {context}"

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "anthropic/claude-3.5-sonnet",
                    "messages": [{"role": "user", "content": full_prompt}],
                    "max_tokens": 8192,
                    "temperature": 0
                },
                timeout=120
            )
        except requests.exceptions.Timeout:
            print(f"    Warning: Speaker labeling timed out for chunk {i+1}, using raw text")
            labeled_parts.append(chunk)
            continue
        if response.status_code != 200:
            print(f"    Warning: Speaker labeling failed for chunk {i+1}, using raw text")
            labeled_parts.append(chunk)
        else:
            content = response.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = re.sub(r'^```\w*\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            labeled_parts.append(content)

            # Extract last speaker for context
            for line in reversed(content.strip().split('\n')):
                m = re.match(r'^([A-Z][A-Z\s\'-]+?):', line.strip())
                if m:
                    context = m.group(1)
                    break

        if i < len(chunks) - 1:
            _time.sleep(0.5)

    result = "\n\n".join(labeled_parts)
    # Strip LLM-inserted bracketed notes like [Continued...], [Note:...], [Sponsor read]
    result = re.sub(r'^\[.*?\]\s*$', '', result, flags=re.MULTILINE)
    result = re.sub(r'\n{3,}', '\n\n', result)
    # Normalize: SPEAKER:\ntext -> SPEAKER: text
    result = re.sub(r'^([A-Z][A-Z\s\'-]+?):\s*\n(?!\n)', r'\1: ', result, flags=re.MULTILINE)
    return result


def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio using Lightning Whisper MLX (Apple Silicon GPU)."""
    print(f"[1/5] Transcribing {audio_path} (MLX GPU)...")

    try:
        from lightning_whisper_mlx import LightningWhisperMLX
    except ImportError:
        print("Error: lightning-whisper-mlx not installed. Run: pip install lightning-whisper-mlx")
        sys.exit(1)

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path],
        capture_output=True, text=True
    )
    duration = int(float(probe.stdout.strip())) if probe.returncode == 0 else 0

    whisper = LightningWhisperMLX(model=WHISPER_MODEL, batch_size=12, quant=None)
    result = whisper.transcribe(audio_path=audio_path, language="en")

    transcript_segments = []
    full_text = []

    for segment in result.get("segments", []):
        start_ms, end_ms, text = segment[0], segment[1], segment[2]
        transcript_segments.append({
            "start": start_ms / 1000.0,
            "end": end_ms / 1000.0,
            "text": text.strip()
        })
        full_text.append(text.strip())
    print(f"    Transcribed {duration} seconds of audio ({len(transcript_segments)} segments)")

    return {
        "segments": transcript_segments,
        "full_text": " ".join(full_text),
        "duration": duration
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
            "model": "anthropic/claude-3.5-haiku",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        },
        timeout=300
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


CLOUDFLARE_UPLOAD_LIMIT = 100 * 1024 * 1024  # 100 MB


def create_episode(audio_path: str, metadata: dict, episode_number: int, duration: int = 0) -> dict:
    """Create episode on Castopod. Bypasses Cloudflare for large files."""
    file_size = os.path.getsize(audio_path)

    if file_size > CLOUDFLARE_UPLOAD_LIMIT:
        print(f"[3/5] Creating episode on Castopod (direct, {file_size / 1024 / 1024:.0f} MB > 100 MB limit)...")
        return _create_episode_direct(audio_path, metadata, episode_number, file_size, duration)

    print("[3/5] Creating episode on Castopod...")
    return _create_episode_api(audio_path, metadata, episode_number)


def _create_episode_api(audio_path: str, metadata: dict, episode_number: int) -> dict:
    """Create episode via Castopod REST API (through Cloudflare)."""
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

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
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


def _create_episode_direct(audio_path: str, metadata: dict, episode_number: int,
                           file_size: int, duration: int) -> dict:
    """Create episode by uploading directly to NAS and inserting into DB."""
    import time as _time
    slug = re.sub(r'[^a-z0-9]+', '-', metadata["title"].lower()).strip('-')
    timestamp = int(_time.time())
    rand_hex = os.urandom(10).hex()
    filename = f"{timestamp}_{rand_hex}.mp3"
    file_key = f"podcasts/{PODCAST_HANDLE}/{filename}"
    nas_tmp = f"/share/CACHEDEV1_DATA/tmp/{filename}"
    guid = f"{CASTOPOD_URL}/@{PODCAST_HANDLE}/episodes/{slug}"
    desc_md = metadata["description"]
    desc_html = f"<p>{desc_md}</p>"
    duration_json = json.dumps({"playtime_seconds": duration, "avdataoffset": 85})

    # SCP audio to NAS
    print("    Uploading audio to NAS...")
    scp_cmd = ["scp", "-P", str(NAS_SSH_PORT), audio_path, f"{NAS_USER}@{NAS_HOST}:{nas_tmp}"]
    result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"Error: SCP failed: {result.stderr}")
        sys.exit(1)

    # Docker cp into Castopod container
    print("    Copying into Castopod container...")
    media_path = f"/var/www/castopod/public/media/{file_key}"
    cp_cmd = f'{DOCKER_PATH} cp {nas_tmp} {CASTOPOD_CONTAINER}:{media_path}'
    success, output = run_ssh_command(cp_cmd, timeout=120)
    if not success:
        print(f"Error: docker cp failed: {output}")
        sys.exit(1)
    run_ssh_command(f'{DOCKER_PATH} exec {CASTOPOD_CONTAINER} chown www-data:www-data {media_path}')
    run_ssh_command(f"rm -f {nas_tmp}")

    # Build SQL and transfer via base64 to avoid shell escaping issues
    print("    Inserting media and episode records...")

    def _mysql_escape(s: str) -> str:
        """Escape a string for MySQL single-quoted literals."""
        return s.replace("\\", "\\\\").replace("'", "\\'")

    title_esc = _mysql_escape(metadata["title"])
    desc_md_esc = _mysql_escape(desc_md)
    desc_html_esc = _mysql_escape(desc_html)
    duration_json_esc = _mysql_escape(duration_json)

    sql = (
        f"INSERT INTO cp_media (file_key, file_size, file_mimetype, file_metadata, type, "
        f"uploaded_by, updated_by, uploaded_at, updated_at) VALUES "
        f"('{file_key}', {file_size}, 'audio/mpeg', '{duration_json_esc}', 'audio', 1, 1, NOW(), NOW());\n"
        f"SET @audio_id = LAST_INSERT_ID();\n"
        f"INSERT INTO cp_episodes (podcast_id, guid, title, slug, audio_id, "
        f"description_markdown, description_html, parental_advisory, number, type, "
        f"is_blocked, is_published_on_hubs, is_premium, created_by, updated_by, "
        f"published_at, created_at, updated_at) VALUES "
        f"(1, '{guid}', '{title_esc}', '{slug}', @audio_id, "
        f"'{desc_md_esc}', '{desc_html_esc}', 'explicit', {episode_number}, 'full', "
        f"0, 0, 0, 1, 1, NOW(), NOW(), NOW());\n"
        f"SELECT LAST_INSERT_ID();\n"
    )

    # Write SQL to local temp file, SCP to NAS, docker cp into MariaDB
    local_sql_path = "/tmp/_castopod_insert.sql"
    nas_sql_path = "/share/CACHEDEV1_DATA/tmp/_castopod_insert.sql"
    with open(local_sql_path, "w") as f:
        f.write(sql)
    scp_sql = ["scp", "-P", str(NAS_SSH_PORT), local_sql_path, f"{NAS_USER}@{NAS_HOST}:{nas_sql_path}"]
    result = subprocess.run(scp_sql, capture_output=True, text=True, timeout=30)
    os.remove(local_sql_path)
    if result.returncode != 0:
        print(f"Error: failed to SCP SQL file: {result.stderr}")
        sys.exit(1)

    # Copy SQL into MariaDB container and execute
    run_ssh_command(f'{DOCKER_PATH} cp {nas_sql_path} {MARIADB_CONTAINER}:/tmp/_insert.sql')
    exec_cmd = f'{DOCKER_PATH} exec {MARIADB_CONTAINER} sh -c "mysql -u {DB_USER} -p{DB_PASS} {DB_NAME} -N < /tmp/_insert.sql"'
    success, output = run_ssh_command(exec_cmd, timeout=30)
    run_ssh_command(f'rm -f {nas_sql_path}')
    run_ssh_command(f'{DOCKER_PATH} exec {MARIADB_CONTAINER} rm -f /tmp/_insert.sql')

    if not success:
        print(f"Error: DB insert failed: {output}")
        sys.exit(1)

    episode_id = int(output.strip().split('\n')[-1])
    # Get the audio media ID for CDN upload
    audio_id_cmd = f'{DOCKER_PATH} exec {MARIADB_CONTAINER} mysql -u {DB_USER} -p{DB_PASS} {DB_NAME} -N -e "SELECT audio_id FROM cp_episodes WHERE id = {episode_id};"'
    success, audio_id_str = run_ssh_command(audio_id_cmd)
    audio_id = int(audio_id_str.strip()) if success else None
    if audio_id:
        print(f"    Audio media ID: {audio_id}")

    # Clear cache
    run_ssh_command(f'{DOCKER_PATH} exec {CASTOPOD_CONTAINER} php spark cache:clear')

    print(f"    Created episode ID: {episode_id}")
    print(f"    Slug: {slug}")

    return {"id": episode_id, "slug": slug}


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


def generate_srt(segments: list, output_path: str):
    """Generate SRT subtitle file from whisper segments."""
    with open(output_path, "w") as f:
        for i, seg in enumerate(segments, 1):
            start = seg["start"]
            end = seg["end"]
            sh, sm, ss = int(start // 3600), int((start % 3600) // 60), start % 60
            eh, em, es = int(end // 3600), int((end % 3600) // 60), end % 60
            f.write(f"{i}\n")
            f.write(f"{sh:02d}:{sm:02d}:{ss:06.3f} --> {eh:02d}:{em:02d}:{es:06.3f}\n")
            f.write(f"{seg['text']}\n\n")


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


def _check_episode_exists_in_db(episode_number: int) -> bool:
    """Check if an episode with this number already exists in Castopod DB."""
    cmd = (f'{DOCKER_PATH} exec {MARIADB_CONTAINER} mysql -u {DB_USER} -p{DB_PASS} {DB_NAME} '
           f'-N -e "SELECT COUNT(*) FROM cp_episodes WHERE number = {episode_number};"')
    success, output = run_ssh_command(cmd)
    if success and output.strip():
        return int(output.strip()) > 0
    return False


def _srt_to_castopod_json(srt_path: str) -> str:
    """Parse SRT to JSON matching Castopod's TranscriptParser format."""
    with open(srt_path, "r") as f:
        srt_text = f.read()

    subs = []
    blocks = re.split(r'\n\n+', srt_text.strip())
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        try:
            num = int(lines[0].strip())
        except ValueError:
            continue
        time_match = re.match(
            r'(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})',
            lines[1].strip()
        )
        if not time_match:
            continue
        text = '\n'.join(lines[2:]).strip()

        def ts_to_seconds(ts):
            ts = ts.replace(',', '.')
            parts = ts.split(':')
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

        subs.append({
            "number": num,
            "startTime": ts_to_seconds(time_match.group(1)),
            "endTime": ts_to_seconds(time_match.group(2)),
            "text": text,
        })
    return json.dumps(subs, indent=4)


def upload_transcript_to_castopod(episode_slug: str, episode_id: int, transcript_path: str) -> bool:
    """Upload SRT transcript + JSON to Castopod via SSH and link in database."""
    print("    Uploading transcript to Castopod...")

    is_srt = transcript_path.endswith(".srt")
    ext = ".srt" if is_srt else ".txt"
    mimetype = "application/x-subrip" if is_srt else "text/plain"

    transcript_filename = f"{episode_slug}{ext}"
    remote_path = f"podcasts/{PODCAST_HANDLE}/{transcript_filename}"
    json_key = f"podcasts/{PODCAST_HANDLE}/{episode_slug}.json"

    # Upload SRT via SCP + docker cp (handles large files)
    nas_tmp = f"/share/CACHEDEV1_DATA/tmp/_transcript_{episode_slug}{ext}"
    scp_cmd = ["scp", "-P", str(NAS_SSH_PORT), transcript_path, f"{NAS_USER}@{NAS_HOST}:{nas_tmp}"]
    result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"    Warning: SCP transcript failed: {result.stderr}")
        return False

    media_path = f"/var/www/castopod/public/media/{remote_path}"
    run_ssh_command(f'{DOCKER_PATH} cp {nas_tmp} {CASTOPOD_CONTAINER}:{media_path}', timeout=60)
    run_ssh_command(f'{DOCKER_PATH} exec {CASTOPOD_CONTAINER} chown www-data:www-data {media_path}')
    run_ssh_command(f'rm -f {nas_tmp}')

    # Generate and upload JSON for Castopod's frontend rendering
    if is_srt:
        json_content = _srt_to_castopod_json(transcript_path)
        json_tmp_local = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json_tmp_local.write(json_content)
        json_tmp_local.close()

        nas_json_tmp = f"/share/CACHEDEV1_DATA/tmp/_transcript_{episode_slug}.json"
        scp_json = ["scp", "-P", str(NAS_SSH_PORT), json_tmp_local.name, f"{NAS_USER}@{NAS_HOST}:{nas_json_tmp}"]
        subprocess.run(scp_json, capture_output=True, text=True, timeout=60)
        os.remove(json_tmp_local.name)

        json_media_path = f"/var/www/castopod/public/media/{json_key}"
        run_ssh_command(f'{DOCKER_PATH} cp {nas_json_tmp} {CASTOPOD_CONTAINER}:{json_media_path}', timeout=60)
        run_ssh_command(f'{DOCKER_PATH} exec {CASTOPOD_CONTAINER} chown www-data:www-data {json_media_path}')
        run_ssh_command(f'rm -f {nas_json_tmp}')

    with open(transcript_path, "rb") as f:
        file_size = len(f.read())

    # Build file_metadata with json_key — escape double quotes for shell embedding
    metadata_json = json.dumps({"json_key": json_key}) if is_srt else "NULL"
    metadata_sql = f"'{metadata_json}'" if is_srt else "NULL"
    metadata_sql_escaped = metadata_sql.replace('"', '\\"')

    insert_sql = (
        f"INSERT INTO cp_media (file_key, file_size, file_mimetype, file_metadata, type, "
        f"uploaded_by, updated_by, uploaded_at, updated_at) VALUES "
        f"('{remote_path}', {file_size}, '{mimetype}', {metadata_sql_escaped}, 'transcript', 1, 1, NOW(), NOW())"
    )
    db_cmd = f'{DOCKER_PATH} exec {MARIADB_CONTAINER} mysql -u {DB_USER} -p{DB_PASS} {DB_NAME} -e "{insert_sql}; SELECT LAST_INSERT_ID();"'
    success, output = run_ssh_command(db_cmd)
    if not success:
        print(f"    Warning: Failed to insert transcript in database: {output}")
        return False

    try:
        lines = output.strip().split('\n')
        media_id = int(lines[-1])
    except (ValueError, IndexError):
        print(f"    Warning: Could not parse media ID from: {output}")
        return False

    update_sql = f"UPDATE cp_episodes SET transcript_id = {media_id} WHERE id = {episode_id}"
    db_cmd = f'{DOCKER_PATH} exec {MARIADB_CONTAINER} mysql -u {DB_USER} -p{DB_PASS} {DB_NAME} -e "{update_sql}"'
    success, output = run_ssh_command(db_cmd)
    if not success:
        print(f"    Warning: Failed to link transcript to episode: {output}")
        return False

    cache_cmd = f'{DOCKER_PATH} exec {CASTOPOD_CONTAINER} php spark cache:clear'
    run_ssh_command(cache_cmd)

    print(f"    Transcript uploaded and linked (media_id: {media_id})")
    return True


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
        }, timeout=600)
    if resp.status_code == 201:
        return True
    print(f"    Warning: BunnyCDN upload failed ({resp.status_code}): {resp.text[:200]}")
    return False


def download_from_castopod(file_key: str, local_path: str) -> bool:
    """Download a file from Castopod's container storage to local filesystem."""
    remote_filename = Path(file_key).name
    remote_tmp = f"/share/CACHEDEV1_DATA/tmp/castopod_{remote_filename}"
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


def add_episode_to_sitemap(slug: str):
    """Add episode transcript page to sitemap.xml."""
    sitemap_path = Path(__file__).parent / "website" / "sitemap.xml"
    if not sitemap_path.exists():
        return

    url = f"https://lukeattheroost.com/episode.html?slug={slug}"
    content = sitemap_path.read_text()

    if url in content:
        print(f"    Episode already in sitemap")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    new_entry = f"""  <url>
    <loc>{url}</loc>
    <lastmod>{today}</lastmod>
    <changefreq>never</changefreq>
    <priority>0.7</priority>
  </url>
</urlset>"""

    content = content.replace("</urlset>", new_entry)
    sitemap_path.write_text(content)
    print(f"    Added episode to sitemap.xml")



def generate_social_image(episode_number: int, description: str, output_path: str) -> str:
    """Generate a social media image with cover art, episode number, and description."""
    from PIL import Image, ImageDraw, ImageFont
    import textwrap

    COVER_ART = Path(__file__).parent / "website" / "images" / "cover.png"
    SIZE = 1080

    img = Image.open(COVER_ART).convert("RGBA")
    img = img.resize((SIZE, SIZE), Image.LANCZOS)

    # Dark gradient overlay on the bottom ~45%
    gradient = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw_grad = ImageDraw.Draw(gradient)
    gradient_start = int(SIZE * 0.50)
    for y in range(gradient_start, SIZE):
        progress = (y - gradient_start) / (SIZE - gradient_start)
        alpha = int(210 * progress)
        draw_grad.line([(0, y), (SIZE, y)], fill=(0, 0, 0, alpha))

    img = Image.alpha_composite(img, gradient)
    draw = ImageDraw.Draw(img)

    # Fonts
    try:
        font_episode = ImageFont.truetype("/Library/Fonts/Montserrat-ExtraBold.ttf", 64)
        font_desc = ImageFont.truetype("/Library/Fonts/Montserrat-Medium.ttf", 36)
        font_url = ImageFont.truetype("/Library/Fonts/Montserrat-SemiBold.ttf", 28)
    except OSError:
        font_episode = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 64)
        font_desc = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 36)
        font_url = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 28)

    margin = 60
    max_width = SIZE - margin * 2

    # Episode number
    ep_text = f"EPISODE {episode_number}"
    draw.text((margin, SIZE - 300), ep_text, font=font_episode, fill=(255, 200, 80))

    # Description — word-wrap to fit
    wrapped = textwrap.fill(description, width=45)
    lines = wrapped.split("\n")[:4]  # max 4 lines
    if len(wrapped.split("\n")) > 4:
        lines[-1] = lines[-1][:lines[-1].rfind(" ")] + "..."
    desc_text = "\n".join(lines)
    draw.text((margin, SIZE - 220), desc_text, font=font_desc, fill=(255, 255, 255, 230),
              spacing=8)

    # Website URL — bottom right
    url_text = "lukeattheroost.com"
    bbox = draw.textbbox((0, 0), url_text, font=font_url)
    url_width = bbox[2] - bbox[0]
    draw.text((SIZE - margin - url_width, SIZE - 50), url_text, font=font_url,
              fill=(255, 200, 80, 200))

    img = img.convert("RGB")
    img.save(output_path, "JPEG", quality=92)
    print(f"    Social image saved: {output_path}")
    return output_path


def _get_postiz_token():
    """Generate a JWT token for Postiz API authentication."""
    import jwt
    return jwt.encode(
        {"id": POSTIZ_USER_ID, "email": "luke@macneilmediagroup.com",
         "providerName": "LOCAL", "activated": True, "isSuperAdmin": False},
        POSTIZ_JWT_SECRET, algorithm="HS256"
    )


def upload_image_to_postiz(image_path: str) -> dict | None:
    """Upload an image to Postiz and return the media object."""
    token = _get_postiz_token()
    try:
        with open(image_path, "rb") as f:
            resp = requests.post(
                f"{POSTIZ_URL}/api/media/upload-simple",
                headers={"auth": token},
                files={"file": ("social.jpg", f, "image/jpeg")},
                timeout=30,
            )
        if resp.status_code in (200, 201):
            media = resp.json()
            print(f"    Uploaded image to Postiz (id: {media.get('id', 'unknown')})")
            return media
        else:
            print(f"    Warning: Postiz image upload returned {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"    Warning: Postiz image upload failed: {e}")
    return None


def post_to_social(metadata: dict, episode_slug: str, image_path: str = None):
    """Post episode announcement to all connected social channels via Postiz."""
    print("[5.5/5] Posting to social media...")

    token = _get_postiz_token()

    # Upload image if provided
    image_ids = []
    if image_path:
        media = upload_image_to_postiz(image_path)
        if media and media.get("id"):
            image_ids = [{"id": media["id"], "path": media.get("path", "")}]

    episode_url = f"https://lukeattheroost.com/episode.html?slug={episode_slug}"
    base_content = f"{metadata['title']}\n\n{metadata['description']}\n\n{episode_url}"

    hashtags = "#podcast #LukeAtTheRoost #talkradio #callinshow #newepisode"
    hashtag_platforms = {"instagram", "facebook", "bluesky", "mastodon", "nostr", "linkedin", "threads", "tiktok"}

    # Platform-specific content length limits
    PLATFORM_MAX_LENGTH = {"bluesky": 300, "threads": 500, "tiktok": 2200}

    # Post to each platform individually so one failure doesn't block others
    posted = 0
    for platform, intg_config in POSTIZ_INTEGRATIONS.items():
        content = base_content
        if platform in hashtag_platforms:
            content += f"\n\n{hashtags}"

        # Truncate for platforms with short limits
        max_len = PLATFORM_MAX_LENGTH.get(platform)
        if max_len and len(content) > max_len:
            # Keep title + URL, truncate description
            short = f"{metadata['title']}\n\n{episode_url}"
            if platform in hashtag_platforms:
                short += f"\n\n{hashtags}"
            content = short[:max_len]

        settings = {"post_type": "post"}
        if "channel" in intg_config:
            settings["channel"] = intg_config["channel"]

        post = {
            "integration": {"id": intg_config["id"]},
            "value": [{"content": content, "image": image_ids}],
            "settings": settings,
        }

        payload = {
            "type": "now",
            "shortLink": False,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "tags": [],
            "posts": [post],
        }

        try:
            resp = requests.post(
                f"{POSTIZ_URL}/api/posts",
                headers={"auth": token, "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            if resp.status_code in (200, 201):
                posted += 1
                print(f"    Posted to {platform}")
            else:
                print(f"    Warning: {platform} failed ({resp.status_code}): {resp.text[:150]}")
        except Exception as e:
            print(f"    Warning: {platform} failed: {e}")

    print(f"    Posted to {posted}/{len(POSTIZ_INTEGRATIONS)} channels")


def get_youtube_service():
    """Authenticate with YouTube API. Reuses saved token."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build as yt_build

    creds = None
    if YT_TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(YT_TOKEN_FILE), YT_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(YT_TOKEN_FILE, "w") as f:
                f.write(creds.to_json())
        else:
            print("    Warning: YouTube token missing or invalid. Run: python yt_auth.py")
            return None

    return yt_build("youtube", "v3", credentials=creds)


def upload_to_youtube(audio_path: str, metadata: dict, chapters: list,
                      episode_slug: str) -> str | None:
    """Convert audio to video with cover art, upload to YouTube, add to podcast playlist."""
    import time as _time
    import random
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError

    youtube = get_youtube_service()
    if not youtube:
        return None

    cover_art = Path(__file__).parent / "website" / "images" / "cover.png"
    video_path = Path(audio_path).with_suffix(".yt.mp4")

    # Convert MP3 + cover art to MP4
    print("    Converting audio to video...")
    result = subprocess.run([
        "ffmpeg", "-y", "-loop", "1",
        "-i", str(cover_art), "-i", audio_path,
        "-c:v", "libx264", "-tune", "stillimage",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p", "-shortest",
        "-movflags", "+faststart", str(video_path)
    ], capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        print(f"    Warning: ffmpeg failed: {result.stderr[-200:]}")
        return None

    # Build chapter timestamps for description
    chapter_lines = []
    for ch in chapters:
        t = int(ch["startTime"])
        h, m, s = t // 3600, (t % 3600) // 60, t % 60
        ts = f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"
        chapter_lines.append(f"{ts} {ch['title']}")

    episode_url = f"https://lukeattheroost.com/episode.html?slug={episode_slug}"
    description = (
        f"{metadata['description']}\n\n"
        + "\n".join(chapter_lines) + "\n\n"
        f"Listen on your favorite podcast app: {episode_url}\n\n"
        f"#podcast #LukeAtTheRoost #talkradio #callinshow"
    )

    body = {
        "snippet": {
            "title": metadata["title"][:100],
            "description": description,
            "tags": ["podcast", "Luke at the Roost", "talk radio", "call-in show",
                     "talk show", "comedy"],
            "categoryId": "22",
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        str(video_path), mimetype="video/mp4",
        chunksize=5 * 1024 * 1024, resumable=True,
    )

    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    file_mb = video_path.stat().st_size / 1_000_000
    print(f"    Uploading to YouTube ({file_mb:.0f} MB)...")

    response = None
    retry = 0
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                if pct % 25 == 0:
                    print(f"    Upload {pct}%...")
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504) and retry < 5:
                retry += 1
                wait = random.random() * (2 ** retry)
                print(f"    Retrying in {wait:.1f}s...")
                _time.sleep(wait)
            else:
                print(f"    YouTube API error: {e}")
                video_path.unlink(missing_ok=True)
                return None

    video_id = response["id"]
    video_path.unlink(missing_ok=True)

    # Add to podcast playlist
    try:
        youtube.playlistItems().insert(part="snippet", body={
            "snippet": {
                "playlistId": YT_PODCAST_PLAYLIST,
                "resourceId": {"kind": "youtube#video", "videoId": video_id},
            }
        }).execute()
        print(f"    Added to podcast playlist")
    except HttpError as e:
        print(f"    Warning: Could not add to playlist: {e}")
        print(f"    Add manually in YouTube Studio")

    print(f"    https://youtube.com/watch?v={video_id}")
    return video_id


def get_next_episode_number() -> int:
    """Get the next episode number from Castopod (DB first, API fallback)."""
    # Query DB directly — the REST API is unreliable
    cmd = (f'{DOCKER_PATH} exec {MARIADB_CONTAINER} mysql -u {DB_USER} -p{DB_PASS} {DB_NAME} '
           f'-N -e "SELECT COALESCE(MAX(number), 0) FROM cp_episodes WHERE podcast_id = {PODCAST_ID};"')
    success, output = run_ssh_command(cmd)
    if success and output.strip():
        try:
            return int(output.strip()) + 1
        except ValueError:
            pass

    # Fallback to API
    headers = get_auth_header()
    response = _session.get(
        f"{CASTOPOD_URL}/api/rest/v1/episodes",
        headers=headers,
    )

    if response.status_code != 200:
        print("Warning: Could not determine episode number from API or DB")
        sys.exit(1)

    episodes = response.json()
    if not episodes:
        return 1

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

    # Acquire exclusive lock to prevent concurrent/duplicate runs
    lock_fp = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Error: Another publish is already running (lock file held)")
        sys.exit(1)
    lock_fp.write(str(os.getpid()))
    lock_fp.flush()

    # Kill the backend server to free memory for transcription
    server_was_running = False
    try:
        result = subprocess.run(
            ["lsof", "-ti", ":8000"], capture_output=True, text=True
        )
        pids = result.stdout.strip().split('\n') if result.stdout.strip() else []
        if pids:
            server_was_running = True
            print("Stopping backend server for resources...")
            for pid in pids:
                try:
                    os.kill(int(pid), 9)
                except (ProcessLookupError, ValueError):
                    pass
            import time as _time
            _time.sleep(1)
    except Exception:
        pass

    # Determine episode number
    if args.episode_number:
        episode_number = args.episode_number
    else:
        episode_number = get_next_episode_number()
    print(f"Episode number: {episode_number}")

    # Guard against duplicate publish
    if not args.dry_run and _check_episode_exists_in_db(episode_number):
        print(f"Error: Episode {episode_number} already exists in Castopod. "
              f"Use --episode-number to specify a different number, or remove the existing episode first.")
        lock_fp.close()
        LOCK_FILE.unlink(missing_ok=True)
        sys.exit(1)

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

    # Save transcript text file with LUKE:/CALLER: speaker labels
    transcript_path = audio_path.with_suffix(".transcript.txt")
    raw_text = transcript["full_text"]
    labeled_text = label_transcript_speakers(raw_text)
    with open(transcript_path, "w") as f:
        f.write(labeled_text)
    print(f"    Transcript saved to: {transcript_path}")

    # Generate SRT from whisper segments (for Castopod/podcast apps)
    srt_path = audio_path.with_suffix(".srt")
    generate_srt(transcript["segments"], str(srt_path))
    print(f"    SRT saved to: {srt_path}")

    # Save session transcript alongside episode if available (has speaker labels)
    if session_data and session_data.get("transcript"):
        session_transcript_path = audio_path.with_suffix(".session_transcript.txt")
        with open(session_transcript_path, "w") as f:
            f.write(session_data["transcript"])
        print(f"    Session transcript saved to: {session_transcript_path}")

    if args.dry_run:
        print("\n[DRY RUN] Would publish with:")
        print(f"  Title: {metadata['title']}")
        print(f"  Description: {metadata['description']}")
        print(f"  Chapters: {json.dumps(metadata['chapters'], indent=2)}")
        print("\nChapters file saved. Run without --dry-run to publish.")
        return

    # Step 3: Create episode
    direct_upload = os.path.getsize(str(audio_path)) > CLOUDFLARE_UPLOAD_LIMIT
    episode = create_episode(str(audio_path), metadata, episode_number, duration=transcript["duration"])

    # Step 3.5: Upload to BunnyCDN
    print("[3.5/5] Uploading to BunnyCDN...")
    uploaded_keys = set()

    # Audio: query file_key from DB, then upload to CDN
    ep_id = episode["id"]
    audio_media_cmd = f'{DOCKER_PATH} exec {MARIADB_CONTAINER} mysql -u {DB_USER} -p{DB_PASS} {DB_NAME} -N -e "SELECT m.file_key FROM cp_media m JOIN cp_episodes e ON e.audio_id = m.id WHERE e.id = {ep_id};"'
    success, audio_file_key = run_ssh_command(audio_media_cmd)
    if success and audio_file_key:
        audio_file_key = audio_file_key.strip()
        if direct_upload:
            # Direct upload: we have the original file locally, upload straight to CDN
            print(f"    Uploading audio to BunnyCDN")
            upload_to_bunny(str(audio_path), f"media/{audio_file_key}", "audio/mpeg")
        else:
            # API upload: download Castopod's copy (ensures byte-exact match with RSS metadata)
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

    # Transcript
    print(f"    Uploading transcript to BunnyCDN")
    upload_to_bunny(str(transcript_path), f"transcripts/{episode['slug']}.txt", "text/plain")

    # Copy transcript to website dir for Cloudflare Pages
    website_transcript_dir = Path(__file__).parent / "website" / "transcripts"
    website_transcript_dir.mkdir(exist_ok=True)
    website_transcript_path = website_transcript_dir / f"{episode['slug']}.txt"
    shutil.copy2(str(transcript_path), str(website_transcript_path))
    print(f"    Transcript copied to website/transcripts/")

    # Add to sitemap
    add_episode_to_sitemap(episode["slug"])

    # Step 4: Publish via API (triggers RSS rebuild, federation, etc.)
    try:
        published = publish_episode(episode["id"])
        if "slug" in published:
            episode = published
    except SystemExit:
        if direct_upload:
            print("    Warning: Publish API failed, but episode is in DB with published_at set")
        else:
            raise

    # Step 4.5: Upload chapters and transcript via SSH
    chapters_uploaded = upload_chapters_to_castopod(
        episode["slug"],
        episode["id"],
        str(chapters_path)
    )

    # Upload SRT transcript to Castopod (preferred for podcast apps)
    transcript_uploaded = upload_transcript_to_castopod(
        episode["slug"],
        episode["id"],
        str(srt_path)
    )

    # Sync any remaining episode media to BunnyCDN (cover art, transcripts, etc.)
    print("    Syncing episode media to CDN...")
    sync_episode_media_to_bunny(episode["id"], uploaded_keys)

    # Step 5: Deploy website (transcript + sitemap must be live before social links go out)
    print("[5/5] Deploying website...")
    project_dir = Path(__file__).parent
    deploy_result = subprocess.run(
        ["npx", "wrangler", "pages", "deploy", "website/",
         "--project-name=lukeattheroost", "--branch=main", "--commit-dirty=true"],
        capture_output=True, text=True, cwd=project_dir, timeout=120
    )
    if deploy_result.returncode == 0:
        print("    Website deployed")
    else:
        print(f"    Warning: Website deploy failed: {deploy_result.stderr[:200]}")

    # Step 5.5: Upload to YouTube
    print("[5.5] Uploading to YouTube...")
    yt_video_id = upload_to_youtube(
        str(audio_path), metadata, metadata["chapters"], episode["slug"]
    )

    # Step 5.7: Generate social image and post
    social_image_path = str(audio_path.with_suffix(".social.jpg"))
    generate_social_image(episode_number, metadata["description"], social_image_path)
    post_to_social(metadata, episode["slug"], social_image_path)

    # Step 6: Summary
    print("\n[6/6] Done!")
    print("=" * 50)
    print(f"Episode URL: {CASTOPOD_URL}/@{PODCAST_HANDLE}/episodes/{episode['slug']}")
    print(f"RSS Feed: {CASTOPOD_URL}/@{PODCAST_HANDLE}/feed.xml")
    if yt_video_id:
        print(f"YouTube: https://youtube.com/watch?v={yt_video_id}")
    print("=" * 50)
    if not chapters_uploaded:
        print("\nNote: Chapters upload failed. Add manually via Castopod admin UI")
        print(f"      Chapters file: {chapters_path}")
    if not transcript_uploaded:
        print("\nNote: Transcript upload to Castopod failed")
        print(f"      Transcript file: {srt_path}")
    if not yt_video_id:
        print("\nNote: YouTube upload failed. Run 'python yt_auth.py' if token expired")

    # Restart the backend server if it was running before
    if server_was_running:
        print("Restarting backend server...")
        project_dir = Path(__file__).parent
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.main:app",
             "--reload", "--reload-dir", "backend", "--host", "0.0.0.0", "--port", "8000"],
            cwd=project_dir,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print("    Server restarted on port 8000")

    # Release lock
    lock_fp.close()
    LOCK_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
