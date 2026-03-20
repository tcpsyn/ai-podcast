#!/usr/bin/env python3
"""Extract the best short-form clips from a podcast episode.

Two-pass pipeline (default):
  1. Fast Whisper model (base) transcribes full episode for clip identification
  2. LLM selects best moments
  3. Quality Whisper model (large-v3) re-transcribes only selected clips for precise timestamps

Usage:
    python make_clips.py ~/Desktop/episode12.mp3 --count 3
    python make_clips.py ~/Desktop/episode12.mp3 --transcript website/transcripts/episode-12-love-lies-and-loyalty.txt
    python make_clips.py ~/Desktop/episode12.mp3 --fast-model small --quality-model large-v3
    python make_clips.py ~/Desktop/episode12.mp3 --single-pass  # skip two-pass, use quality model only
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import time

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
RSS_FEED_URL = "https://podcast.macneilmediagroup.com/@LukeAtTheRoost/feed.xml"
EPISODE_CACHE_DIR = Path(__file__).parent / "clips" / ".episode-cache"
WHISPER_MODEL_FAST = "distil-large-v3"
WHISPER_MODEL_QUALITY = "distil-large-v3"
COVER_ART = Path(__file__).parent / "website" / "images" / "cover.png"
REMOTION_DIR = Path(__file__).parent / "remotion-demo"

# Fonts
FONT_BOLD = "/Library/Fonts/Montserrat-ExtraBold.ttf"
FONT_MEDIUM = "/Library/Fonts/Montserrat-Medium.ttf"
FONT_SEMIBOLD = "/Library/Fonts/Montserrat-SemiBold.ttf"

# Video dimensions (9:16 vertical)
WIDTH = 1080
HEIGHT = 1920


def _llm_request(prompt: str, max_tokens: int = 2048, temperature: float = 0.3,
                  timeout: int = 60) -> str | None:
    """Make an LLM API call with timeout and retry. Returns content or None on failure."""
    for attempt in range(2):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "anthropic/claude-sonnet-4-5",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=timeout,
            )
            if response.status_code != 200:
                print(f"    LLM error (HTTP {response.status_code}): {response.text[:200]}")
                if attempt == 0:
                    print(f"    Retrying in 5s...")
                    time.sleep(5)
                    continue
                return None
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.Timeout:
            print(f"    LLM request timed out ({timeout}s)")
            if attempt == 0:
                print(f"    Retrying in 5s...")
                time.sleep(5)
                continue
            return None
        except Exception as e:
            print(f"    LLM request failed: {e}")
            if attempt == 0:
                print(f"    Retrying in 5s...")
                time.sleep(5)
                continue
            return None
    return None


def _build_whisper_prompt(labeled_transcript: str) -> str:
    """Build an initial_prompt for Whisper from the labeled transcript.

    Whisper's initial_prompt conditions the model to recognize specific names
    and vocabulary. We extract speaker names and the first few lines of dialog.
    """
    prompt_parts = ["Luke at the Roost podcast. Host: Luke."]

    if labeled_transcript:
        # Extract speaker names
        names = set(re.findall(r'^([A-Z][A-Z\s\'-]+?):', labeled_transcript, re.MULTILINE))
        caller_names = [n.strip().title() for n in names if n.strip() != "LUKE"]
        if caller_names:
            prompt_parts.append(f"Callers: {', '.join(caller_names)}.")

        # First ~500 chars of transcript as context (stripped of labels)
        stripped = re.sub(r'^[A-Z][A-Z\s\'-]+?:\s*', '', labeled_transcript[:800], flags=re.MULTILINE)
        stripped = re.sub(r'\n+', ' ', stripped).strip()[:500]
        if stripped:
            prompt_parts.append(stripped)

    return " ".join(prompt_parts)


def transcribe_with_timestamps(audio_path: str, whisper_model: str = None,
                               labeled_transcript: str = "") -> list[dict]:
    """Transcribe audio with word-level timestamps using mlx-whisper (Apple Silicon GPU).

    Returns list of segments: [{start, end, text, words: [{word, start, end}]}]
    """
    model_name = whisper_model or WHISPER_MODEL_QUALITY
    cache_path = Path(audio_path).with_suffix(f".whisper_cache_{model_name}.json")
    if cache_path.exists():
        print(f"    Using cached Whisper output ({model_name})")
        with open(cache_path) as f:
            return json.load(f)

    try:
        import mlx_whisper
    except ImportError:
        print("Error: mlx-whisper not installed. Run: pip install mlx-whisper")
        sys.exit(1)

    MODEL_HF_REPOS = {
        "distil-large-v3": "mlx-community/distil-whisper-large-v3",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "base": "mlx-community/whisper-base-mlx",
    }
    hf_repo = MODEL_HF_REPOS.get(model_name, f"mlx-community/whisper-{model_name}-mlx")

    initial_prompt = _build_whisper_prompt(labeled_transcript)
    print(f"    Model: {model_name} (MLX GPU)")
    if labeled_transcript:
        print(f"    Prompt: {initial_prompt[:100]}...")

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=hf_repo,
        language="en",
        word_timestamps=True,
        initial_prompt=initial_prompt,
    )

    segments = []
    for seg in result.get("segments", []):
        words = []
        for w in seg.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "start": round(w["start"], 3),
                "end": round(w["end"], 3),
            })
        segments.append({
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip(),
            "words": words,
        })

    duration = segments[-1]["end"] if segments else 0
    print(f"    Transcribed {duration:.1f}s ({len(segments)} segments)")

    with open(cache_path, "w") as f:
        json.dump(segments, f)
    print(f"    Cached to {cache_path}")

    return segments


def refine_clip_timestamps(audio_path: str, clips: list[dict],
                           quality_model: str, labeled_transcript: str = "",
                           ) -> dict[int, list[dict]]:
    """Re-transcribe just the selected clip ranges with mlx-whisper (GPU).

    Extracts each clip segment, runs the quality model on it, and returns
    refined segments with word-level timestamps mapped back to the original timeline.

    Returns: {clip_index: [segments]} keyed by clip index
    """
    try:
        import mlx_whisper
    except ImportError:
        print("Error: mlx-whisper not installed. Run: pip install mlx-whisper")
        sys.exit(1)

    MODEL_HF_REPOS = {
        "distil-large-v3": "mlx-community/distil-whisper-large-v3",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "base": "mlx-community/whisper-base-mlx",
    }
    hf_repo = MODEL_HF_REPOS.get(quality_model, f"mlx-community/whisper-{quality_model}-mlx")

    print(f"    Refinement model: {quality_model} (MLX GPU)")

    initial_prompt = _build_whisper_prompt(labeled_transcript)
    refined = {}

    with tempfile.TemporaryDirectory() as tmp:
        for i, clip in enumerate(clips):
            pad = 3.0
            seg_start = max(0, clip["start_time"] - pad)
            seg_end = clip["end_time"] + pad

            cache_key = f"{Path(audio_path).stem}_clip{i}_{seg_start:.1f}-{seg_end:.1f}"
            cache_path = Path(audio_path).parent / f".whisper_refine_{quality_model}_{cache_key}.json"
            if cache_path.exists():
                print(f"      Clip {i+1}: Using cached refinement")
                with open(cache_path) as f:
                    refined[i] = json.load(f)
                continue

            seg_path = os.path.join(tmp, f"segment_{i}.wav")
            cmd = [
                "ffmpeg", "-y", "-ss", str(seg_start), "-t", str(seg_end - seg_start),
                "-i", audio_path, "-ar", "16000", "-ac", "1", seg_path,
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                print(f"      Clip {i+1}: ffmpeg timed out (120s), skipping")
                refined[i] = []
                continue
            if result.returncode != 0:
                print(f"      Clip {i+1}: Failed to extract segment")
                refined[i] = []
                continue

            mlx_result = mlx_whisper.transcribe(
                seg_path,
                path_or_hf_repo=hf_repo,
                language="en",
                word_timestamps=True,
                initial_prompt=initial_prompt,
            )

            segments = []
            for seg_data in mlx_result.get("segments", []):
                text = seg_data["text"].strip()
                words = []
                for w in seg_data.get("words", []):
                    words.append({
                        "word": w["word"].strip(),
                        "start": round(w["start"] + seg_start, 3),
                        "end": round(w["end"] + seg_start, 3),
                    })

                segments.append({
                    "start": round(seg_data["start"] + seg_start, 3),
                    "end": round(seg_data["end"] + seg_start, 3),
                    "text": text,
                    "words": words,
                })

            refined[i] = segments
            seg_duration = segments[-1]["end"] - segments[0]["start"] if segments else 0
            print(f"      Clip {i+1}: Refined {seg_duration:.1f}s → {len(segments)} segments")

            with open(cache_path, "w") as f:
                json.dump(segments, f)

    return refined


def get_transcript_text(segments: list[dict]) -> str:
    """Build timestamped transcript text for the LLM."""
    lines = []
    for seg in segments:
        mins = int(seg["start"] // 60)
        secs = int(seg["start"] % 60)
        lines.append(f"[{mins:02d}:{secs:02d}] {seg['text']}")
    return "\n".join(lines)


def select_clips_with_llm(transcript_text: str, labeled_transcript: str,
                           chapters_json: str | None, count: int) -> list[dict]:
    """Ask LLM to pick the best clip-worthy moments."""
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not set in .env")
        sys.exit(1)

    chapters_context = ""
    if chapters_json:
        chapters_context = f"\nCHAPTERS:\n{chapters_json}\n"

    labeled_context = ""
    if labeled_transcript:
        # Truncate if too long — LLM needs the gist, not every word
        if len(labeled_transcript) > 12000:
            labeled_context = f"\nSPEAKER-LABELED TRANSCRIPT (truncated):\n{labeled_transcript[:12000]}...\n"
        else:
            labeled_context = f"\nSPEAKER-LABELED TRANSCRIPT:\n{labeled_transcript}\n"

    prompt = f"""You are selecting the {count} best moments from a podcast episode for short-form video clips (TikTok/YouTube Shorts/Reels).

Each clip should be 30-60 seconds long and contain a single compelling moment — a funny exchange, an emotional beat, a surprising take, or an interesting story.

TIMESTAMPED TRANSCRIPT:
{transcript_text}
{chapters_context}{labeled_context}
Pick the {count} best moments. For each, return:
- title: A catchy, short title for the clip (max 8 words)
- start_time: Start timestamp in seconds (float). Start a few seconds before the key moment for context.
- end_time: End timestamp in seconds (float). 30-60 seconds after start_time.
- caption_text: The key quote or line that makes this moment clip-worthy (1-2 sentences max)

IMPORTANT:
- Use the timestamps from the transcript to set precise start/end times
- Ensure clips don't overlap
- Prefer moments with back-and-forth dialog over monologues
- Avoid intro/outro segments

Respond with ONLY a JSON array, no markdown or explanation:
[{{"title": "...", "start_time": 0.0, "end_time": 0.0, "caption_text": "..."}}]"""

    content = _llm_request(prompt, max_tokens=2048, temperature=0.3, timeout=60)
    if content is None:
        print("    Failed to get clip selections from LLM — aborting")
        return []

    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\n?", "", content)
        content = re.sub(r"\n?```$", "", content)

    try:
        clips = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Response was: {content[:500]}")
        return []

    # Validate and clamp durations
    validated = []
    for clip in clips:
        duration = clip["end_time"] - clip["start_time"]
        if duration < 15:
            clip["end_time"] = clip["start_time"] + 30
        elif duration > 75:
            clip["end_time"] = clip["start_time"] + 60
        validated.append(clip)

    return validated


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

    content = _llm_request(prompt, max_tokens=2048, temperature=0.7, timeout=60)
    if content is None:
        print("    Failed to generate social metadata — skipping")
        return clips

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


def snap_to_sentences(clips: list[dict], segments: list[dict]) -> list[dict]:
    """Snap clip start/end times to sentence boundaries.

    Uses Whisper segment boundaries and punctuation to find the nearest
    sentence start/end so clips don't begin or end mid-sentence.
    """
    # Build a list of sentence boundary timestamps from Whisper segments.
    # A sentence boundary is: the start of a segment, or a word right after .?!
    sentence_starts = []
    sentence_ends = []

    for seg in segments:
        sentence_starts.append(seg["start"])
        sentence_ends.append(seg["end"])

        # Also find sentence breaks within segments using word punctuation
        words = seg.get("words", [])
        for i, w in enumerate(words):
            if w["word"].rstrip().endswith(('.', '?', '!')):
                sentence_ends.append(w["end"])
                if i + 1 < len(words):
                    sentence_starts.append(words[i + 1]["start"])

    sentence_starts.sort()
    sentence_ends.sort()

    for clip in clips:
        original_start = clip["start_time"]
        original_end = clip["end_time"]

        # Find nearest sentence start at or before the clip start
        # Look up to 5s back for a sentence boundary
        best_start = original_start
        best_start_dist = float('inf')
        for s in sentence_starts:
            dist = abs(s - original_start)
            if dist < best_start_dist and s <= original_start + 1:
                best_start = s
                best_start_dist = dist
            if s > original_start + 1:
                break

        # Find nearest sentence end at or after the clip end
        # Look up to 5s forward for a sentence boundary
        best_end = original_end
        best_end_dist = float('inf')
        for e in sentence_ends:
            if e < original_end - 5:
                continue
            dist = abs(e - original_end)
            if dist < best_end_dist:
                best_end = e
                best_end_dist = dist
            if e > original_end + 5:
                break

        # Make sure we didn't create a clip that's too short or too long
        duration = best_end - best_start
        if duration < 20:
            # Too short — extend end to next sentence boundary
            for e in sentence_ends:
                if e > best_start + 25:
                    best_end = e
                    break
        elif duration > 75:
            # Too long — pull end back
            for e in reversed(sentence_ends):
                if best_start + 30 <= e <= best_start + 65:
                    best_end = e
                    break

        clip["start_time"] = best_start
        clip["end_time"] = best_end

    return clips


def get_words_in_range(segments: list[dict], start: float, end: float) -> list[dict]:
    """Extract word-level timestamps for a time range from Whisper segments."""
    words = []
    for seg in segments:
        if seg["end"] < start or seg["start"] > end:
            continue
        for w in seg.get("words", []):
            if w["start"] >= start - 0.5 and w["end"] <= end + 0.5:
                words.append(w)
    return words


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if abs(len(a) - len(b)) > 5:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        curr = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[len(b)]


def _word_score(a: str, b: str) -> int:
    """Alignment score: +2 exact, +1 fuzzy (edit dist ≤2), -1 mismatch."""
    if a == b:
        return 2
    if len(a) >= 3 and len(b) >= 3 and _edit_distance(a, b) <= 2:
        return 1
    return -1


def _align_sequences(whisper_words: list[str],
                     labeled_words: list[str]) -> list[tuple[int | None, int | None]]:
    """Needleman-Wunsch DP alignment between whisper and labeled word sequences.

    Returns list of (whisper_idx, labeled_idx) pairs where None = gap.
    """
    n = len(whisper_words)
    m = len(labeled_words)
    GAP = -1

    # Build score matrix
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + GAP
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + GAP

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score[i - 1][j - 1] + _word_score(whisper_words[i - 1], labeled_words[j - 1])
            delete = score[i - 1][j] + GAP
            insert = score[i][j - 1] + GAP
            score[i][j] = max(match, delete, insert)

    # Traceback
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and score[i][j] == score[i - 1][j - 1] + _word_score(whisper_words[i - 1], labeled_words[j - 1]):
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and score[i][j] == score[i - 1][j] + GAP:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1

    pairs.reverse()
    return pairs


def _parse_full_transcript(labeled_transcript: str) -> list[dict]:
    """Parse entire labeled transcript into flat word list with speaker metadata.

    Returns list of {word: str, clean: str, speaker: str} for every word.
    """
    result = []
    for m in re.finditer(r'^([A-Z][A-Z\s\'-]+?):\s*(.+?)(?=\n[A-Z][A-Z\s\'-]+?:|\n\n|\Z)',
                         labeled_transcript, re.MULTILINE | re.DOTALL):
        speaker = m.group(1).strip()
        text = m.group(2)
        for w in text.split():
            original = w.strip()
            clean = re.sub(r"[^\w']", '', original.lower())
            if clean:
                result.append({"word": original, "clean": clean, "speaker": speaker})
    return result


def _find_transcript_region(labeled_words: list[dict], whisper_words: list[str],
                            ) -> tuple[int, int] | None:
    """Find the region of labeled_words that best matches the whisper words.

    Uses multi-anchor matching: tries phrases from start, middle, and end
    of the whisper words to find a consensus region.
    """
    if not whisper_words or not labeled_words:
        return None

    labeled_clean = [w["clean"] for w in labeled_words]
    n_labeled = len(labeled_clean)

    def find_phrase(phrase_words: list[str], search_start: int = 0,
                    search_end: int | None = None) -> int | None:
        """Find a phrase in labeled_clean, return index of first word or None."""
        if search_end is None:
            search_end = n_labeled
        plen = len(phrase_words)
        for i in range(search_start, min(search_end, n_labeled - plen + 1)):
            match = True
            for k in range(plen):
                if _word_score(phrase_words[k], labeled_clean[i + k]) < 1:
                    match = False
                    break
            if match:
                return i
        return None

    # Try anchors from different positions in the whisper words
    anchors = []
    n_whisper = len(whisper_words)
    anchor_positions = [0, n_whisper // 2, max(0, n_whisper - 5)]
    # Deduplicate positions
    anchor_positions = sorted(set(anchor_positions))

    for pos in anchor_positions:
        for phrase_len in [5, 4, 3]:
            phrase = whisper_words[pos:pos + phrase_len]
            if len(phrase) < 3:
                continue
            idx = find_phrase(phrase)
            if idx is not None:
                # Estimate region start based on anchor's position in whisper
                region_start = max(0, idx - pos)
                anchors.append(region_start)
                break

    if not anchors:
        return None

    # Use median anchor as region start for robustness
    anchors.sort()
    region_start = anchors[len(anchors) // 2]

    # Region extends to cover all whisper words plus margin
    margin = max(20, n_whisper // 4)
    region_start = max(0, region_start - margin)
    region_end = min(n_labeled, region_start + n_whisper + 2 * margin)

    return (region_start, region_end)


def add_speaker_labels(words: list[dict], labeled_transcript: str,
                       start_time: float, end_time: float,
                       segments: list[dict]) -> list[dict]:
    """Replace Whisper text with labeled transcript text, keeping Whisper timestamps.

    The labeled transcript is the source of truth for TEXT. Whisper is only used
    for TIMESTAMPS. Uses DP alignment to map between the two, then rebuilds the
    word list from the labeled transcript with interpolated timestamps for any
    words Whisper missed.
    """
    if not labeled_transcript or not words:
        return words

    all_labeled = _parse_full_transcript(labeled_transcript)
    if not all_labeled:
        return words

    whisper_clean = []
    for w in words:
        clean = re.sub(r"[^\w']", '', w["word"].lower())
        whisper_clean.append(clean if clean else w["word"].lower())

    region = _find_transcript_region(all_labeled, whisper_clean)
    if region is None:
        return words

    region_start, region_end = region
    region_words = all_labeled[region_start:region_end]
    region_clean = [w["clean"] for w in region_words]

    pairs = _align_sequences(whisper_clean, region_clean)

    # Build mapping: labeled_idx -> whisper_idx (for timestamp lookup)
    labeled_to_whisper = {}
    for w_idx, l_idx in pairs:
        if w_idx is not None and l_idx is not None:
            score = _word_score(whisper_clean[w_idx], region_clean[l_idx])
            if score > 0:
                labeled_to_whisper[l_idx] = w_idx

    # Find the range of labeled words that actually overlap with this clip
    # Use only labeled indices that have a whisper match to determine boundaries
    matched_labeled_indices = sorted(labeled_to_whisper.keys())
    if not matched_labeled_indices:
        return words

    first_labeled = matched_labeled_indices[0]
    last_labeled = matched_labeled_indices[-1]

    # Build output from labeled transcript words with whisper timestamps
    result = []
    corrections = 0
    for l_idx in range(first_labeled, last_labeled + 1):
        labeled_word = region_words[l_idx]
        word_text = re.sub(r'[^\w\s\'-]', '', labeled_word["word"]).strip()
        if not word_text:
            continue

        if l_idx in labeled_to_whisper:
            w_idx = labeled_to_whisper[l_idx]
            ts_start = words[w_idx]["start"]
            ts_end = words[w_idx]["end"]
            if word_text.lower() != whisper_clean[w_idx]:
                corrections += 1
        else:
            # Interpolate timestamp from neighbors
            ts_start, ts_end = _interpolate_timestamp(l_idx, labeled_to_whisper, words)

        result.append({
            "word": word_text,
            "start": ts_start,
            "end": ts_end,
            "speaker": labeled_word["speaker"],
        })

    if corrections:
        print(f"      Corrected {corrections} words from labeled transcript")
    if len(result) != len(words):
        print(f"      Word count: {len(words)} (whisper) -> {len(result)} (labeled)")

    return result


def _interpolate_speaker(idx: int, matched: dict, n_words: int) -> str | None:
    """Find speaker from nearest matched neighbor."""
    for dist in range(1, n_words):
        before = idx - dist
        after = idx + dist
        if before >= 0 and before in matched:
            return matched[before][0]["speaker"]
        if after < n_words and after in matched:
            return matched[after][0]["speaker"]
    return None


def _interpolate_timestamp(labeled_idx: int, labeled_to_whisper: dict,
                           words: list[dict]) -> tuple[float, float]:
    """Interpolate timestamp for a labeled word with no direct whisper match.

    Finds the nearest matched neighbors before and after, then linearly
    interpolates based on position.
    """
    before_l = after_l = None
    for dist in range(1, len(labeled_to_whisper) + 10):
        if before_l is None and (labeled_idx - dist) in labeled_to_whisper:
            before_l = labeled_idx - dist
        if after_l is None and (labeled_idx + dist) in labeled_to_whisper:
            after_l = labeled_idx + dist
        if before_l is not None and after_l is not None:
            break

    if before_l is not None and after_l is not None:
        w_before = words[labeled_to_whisper[before_l]]
        w_after = words[labeled_to_whisper[after_l]]
        span = after_l - before_l
        frac = (labeled_idx - before_l) / span
        start = w_before["end"] + frac * (w_after["start"] - w_before["end"])
        duration = (w_after["start"] - w_before["end"]) / span
        return start, start + max(duration, 0.1)
    elif before_l is not None:
        w = words[labeled_to_whisper[before_l]]
        offset = (labeled_idx - before_l) * 0.3
        return w["end"] + offset, w["end"] + offset + 0.3
    elif after_l is not None:
        w = words[labeled_to_whisper[after_l]]
        offset = (after_l - labeled_idx) * 0.3
        return w["start"] - offset - 0.3, w["start"] - offset
    else:
        return 0.0, 0.3


def polish_clip_words(words: list[dict], labeled_transcript: str = "") -> list[dict]:
    """Use LLM to add punctuation and fix capitalization.

    The word text is already correct (from the labeled transcript). This step
    only adds sentence punctuation and proper capitalization.
    """
    if not words or not OPENROUTER_API_KEY:
        return words

    raw_text = " ".join(w["word"] for w in words)

    prompt = f"""Add punctuation and capitalization to this podcast transcript excerpt so it reads as proper sentences.

RULES:
- Keep the EXACT same number of words in the EXACT same order
- The words themselves are already correct — do NOT change any word's spelling
- Only add punctuation (periods, commas, question marks, exclamation marks) and fix capitalization
- Do NOT add, remove, merge, or reorder words
- Contractions count as one word (don't = 1 word)
- Return ONLY the corrected text, nothing else

RAW TEXT ({len(words)} words):
{raw_text}"""

    polished = _llm_request(prompt, max_tokens=2048, temperature=0, timeout=30)
    if polished is None:
        print(f"      Polish failed, using raw text")
        return words

    polished_words = polished.split()

    if len(polished_words) != len(words):
        print(f"      Polish word count mismatch ({len(polished_words)} vs {len(words)}), using raw text")
        return words

    changes = 0
    for i, pw in enumerate(polished_words):
        if pw != words[i]["word"]:
            changes += 1
            words[i]["word"] = pw

    if changes:
        print(f"      Polished {changes} words")

    return words


def group_words_into_lines(words: list[dict], clip_start: float,
                           clip_duration: float) -> list[dict]:
    """Group words into timed caption lines for rendering.

    Splits at speaker changes so each line has a single, correct speaker label.
    Returns list of: {start, end, speaker, words: [{word, highlighted}]}
    """
    if not words:
        return []

    # First split at speaker boundaries, then group into display lines
    speaker_groups = []
    current_group = []
    current_speaker = words[0].get("speaker", "")
    for w in words:
        speaker = w.get("speaker", "")
        if speaker and speaker != current_speaker and current_group:
            speaker_groups.append((current_speaker, current_group))
            current_group = []
            current_speaker = speaker
        current_group.append(w)
    if current_group:
        speaker_groups.append((current_speaker, current_group))

    # Now group each speaker's words into display lines (5-7 words)
    raw_lines = []
    for speaker, group_words in speaker_groups:
        current_line = []
        for w in group_words:
            current_line.append(w)
            if len(current_line) >= 6 or w["word"].rstrip().endswith(('.', '?', '!', ',')):
                if len(current_line) >= 3:
                    raw_lines.append((speaker, current_line))
                    current_line = []
        if current_line:
            if raw_lines and len(current_line) < 3 and raw_lines[-1][0] == speaker:
                raw_lines[-1] = (speaker, raw_lines[-1][1] + current_line)
            else:
                raw_lines.append((speaker, current_line))

    lines = []
    for speaker, line_words in raw_lines:
        line_start = line_words[0]["start"] - clip_start
        line_end = line_words[-1]["end"] - clip_start

        if line_start < 0:
            line_start = 0
        if line_end > clip_duration:
            line_end = clip_duration
        if line_end <= line_start:
            continue

        lines.append({
            "start": line_start,
            "end": line_end,
            "speaker": speaker,
            "words": line_words,
        })

    return lines


def extract_clip_audio(audio_path: str, start: float, end: float,
                       output_path: str) -> bool:
    """Extract audio clip with fade in/out."""
    duration = end - start
    fade_in = 0.3
    fade_out = 0.5

    af = f"afade=t=in:d={fade_in},afade=t=out:st={duration - fade_out}:d={fade_out}"
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", audio_path,
        "-af", af,
        "-ab", "192k",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"    ffmpeg audio extraction timed out (120s)")
        return False


def generate_background_image(episode_number: int, clip_title: str,
                              output_path: str) -> bool:
    """Generate 9:16 vertical background with blurred/cropped cover art."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    if not COVER_ART.exists():
        print(f"    Warning: Cover art not found at {COVER_ART}")
        # Create solid dark background fallback
        img = Image.new("RGB", (WIDTH, HEIGHT), (20, 15, 30))
        img.save(output_path)
        return True

    cover = Image.open(COVER_ART).convert("RGB")

    # Scale cover to fill 1080x1920 (crop to fit)
    cover_ratio = cover.width / cover.height
    target_ratio = WIDTH / HEIGHT

    if cover_ratio > target_ratio:
        new_h = HEIGHT
        new_w = int(HEIGHT * cover_ratio)
    else:
        new_w = WIDTH
        new_h = int(WIDTH / cover_ratio)

    cover = cover.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - WIDTH) // 2
    top = (new_h - HEIGHT) // 2
    cover = cover.crop((left, top, left + WIDTH, top + HEIGHT))

    # Heavy blur + darken for background
    bg = cover.filter(ImageFilter.GaussianBlur(radius=30))
    from PIL import ImageEnhance
    bg = ImageEnhance.Brightness(bg).enhance(0.3)

    # Place sharp cover art centered, sized to ~60% width
    art_size = int(WIDTH * 0.6)
    art = Image.open(COVER_ART).convert("RGB")
    art = art.resize((art_size, art_size), Image.LANCZOS)

    # Add rounded shadow effect (just darken behind)
    art_x = (WIDTH - art_size) // 2
    art_y = int(HEIGHT * 0.18)
    bg.paste(art, (art_x, art_y))

    # Draw text overlays
    draw = ImageDraw.Draw(bg)

    try:
        font_ep = ImageFont.truetype(FONT_BOLD, 42)
        font_title = ImageFont.truetype(FONT_BOLD, 56)
        font_url = ImageFont.truetype(FONT_SEMIBOLD, 32)
    except OSError:
        font_ep = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 42)
        font_title = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 56)
        font_url = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 32)

    margin = 60

    # Episode label at top
    ep_text = f"EPISODE {episode_number}" if episode_number else "LUKE AT THE ROOST"
    draw.text((margin, 80), ep_text, font=font_ep, fill=(255, 200, 80))

    # Clip title below episode label
    # Word wrap the title
    import textwrap
    wrapped_title = textwrap.fill(clip_title, width=22)
    draw.text((margin, 140), wrapped_title, font=font_title, fill=(255, 255, 255))

    # Watermark at bottom
    url_text = "lukeattheroost.com"
    bbox = draw.textbbox((0, 0), url_text, font=font_url)
    url_w = bbox[2] - bbox[0]
    draw.text(((WIDTH - url_w) // 2, HEIGHT - 80), url_text,
              font=font_url, fill=(255, 200, 80, 200))

    bg.save(output_path, "PNG")
    return True


def generate_caption_frames(bg_path: str, caption_lines: list[dict],
                            clip_start: float, duration: float,
                            tmp_dir: Path, fps: int = 10) -> str:
    """Generate caption frame PNGs and a concat file for ffmpeg.

    Uses a low FPS (10) since the background is static — only captions change.
    Returns path to the concat file.
    """
    from PIL import Image, ImageDraw, ImageFont

    bg = Image.open(bg_path).convert("RGB")

    try:
        font_caption = ImageFont.truetype(FONT_BOLD, 52)
        font_speaker = ImageFont.truetype(FONT_SEMIBOLD, 40)
    except OSError:
        font_caption = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 52)
        font_speaker = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 40)

    frames_dir = tmp_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    n_frames = int(duration * fps)
    frame_duration = 1.0 / fps

    concat_lines = []

    prev_state = None  # (line_idx, highlighted_word_idx) — only reuse when both match
    prev_frame_path = None

    for frame_num in range(n_frames):
        t = frame_num * frame_duration

        # Find active caption line
        active_idx = -1
        active_line = None
        for i, line in enumerate(caption_lines):
            if line["start"] <= t <= line["end"]:
                active_idx = i
                active_line = line
                break

        # Find which word is currently highlighted
        highlight_idx = -1
        if active_line:
            for wi, w in enumerate(active_line["words"]):
                word_rel_start = w["start"] - clip_start
                word_rel_end = w["end"] - clip_start
                if word_rel_start <= t <= word_rel_end:
                    highlight_idx = wi
                    break
            if highlight_idx == -1:
                # Between words — highlight the last spoken word
                for wi in range(len(active_line["words"]) - 1, -1, -1):
                    if t > active_line["words"][wi]["end"] - clip_start:
                        highlight_idx = wi
                        break

        # Reuse previous frame only if same line AND same highlighted word
        state = (active_idx, highlight_idx)
        if state == prev_state and prev_frame_path:
            concat_lines.append(f"file '{prev_frame_path}'")
            concat_lines.append(f"duration {frame_duration:.4f}")
            continue

        frame = bg.copy()

        if active_line:
            draw = ImageDraw.Draw(frame)
            margin = 60
            caption_y = int(HEIGHT * 0.78)

            # Speaker label
            if active_line.get("speaker"):
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    draw.text((margin + dx, caption_y - 55 + dy),
                              active_line["speaker"],
                              font=font_speaker, fill=(0, 0, 0))
                draw.text((margin, caption_y - 55), active_line["speaker"],
                          font=font_speaker, fill=(255, 200, 80))

            # Caption text — all words visible, current word highlighted yellow
            x = margin
            y = caption_y
            for wi, w in enumerate(active_line["words"]):
                word_text = w["word"] + " "

                if wi == highlight_idx:
                    color = (255, 200, 80)  # Yellow — currently spoken
                elif wi < highlight_idx or (highlight_idx == -1 and t > w["end"] - clip_start):
                    color = (255, 255, 255)  # White — already spoken
                else:
                    color = (180, 180, 180)  # Gray — upcoming

                bbox = draw.textbbox((0, 0), word_text, font=font_caption)
                w_width = bbox[2] - bbox[0]

                # Wrap line
                if x + w_width > WIDTH - margin:
                    x = margin
                    y += 65

                # Outline
                for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                    draw.text((x + dx, y + dy), w["word"],
                              font=font_caption, fill=(0, 0, 0))

                draw.text((x, y), w["word"], font=font_caption, fill=color)
                x += w_width

        frame_path = str(frames_dir / f"frame_{frame_num:05d}.png")
        frame.save(frame_path, "PNG")

        concat_lines.append(f"file '{frame_path}'")
        concat_lines.append(f"duration {frame_duration:.4f}")

        prev_state = state
        prev_frame_path = frame_path

    # Final frame needs duration too
    if prev_frame_path:
        concat_lines.append(f"file '{prev_frame_path}'")
        concat_lines.append(f"duration {frame_duration:.4f}")

    concat_path = str(tmp_dir / "concat.txt")
    with open(concat_path, "w") as f:
        f.write("\n".join(concat_lines))

    return concat_path


def generate_clip_video(audio_path: str, background_path: str,
                        caption_lines: list[dict], clip_start: float,
                        output_path: str, duration: float,
                        tmp_dir: Path) -> bool:
    """Generate clip video with burned-in captions using Pillow + ffmpeg."""
    if caption_lines:
        # Generate frames with captions
        concat_path = generate_caption_frames(
            background_path, caption_lines, clip_start, duration, tmp_dir
        )

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_path,
            "-i", audio_path,
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-t", str(duration),
            "-shortest",
            "-r", "30",
            output_path,
        ]
    else:
        # No captions — just static image + audio
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", background_path,
            "-i", audio_path,
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-t", str(duration),
            "-shortest",
            "-r", "30",
            output_path,
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"    ffmpeg video generation timed out (300s)")
        return False
    if result.returncode != 0:
        print(f"    ffmpeg error: {result.stderr[-300:]}")
        return False
    return True


def generate_clip_video_remotion(
    audio_path: str,
    caption_lines: list[dict],
    clip_start: float,
    clip_title: str,
    episode_number: int | None,
    output_path: str,
    duration: float,
) -> bool:
    """Generate clip video using Remotion (animated captions, waveform, dynamic background)."""
    if not REMOTION_DIR.exists():
        print(f"    Remotion project not found at {REMOTION_DIR}")
        return False

    # Copy assets to Remotion public/ dir
    public_dir = REMOTION_DIR / "public"
    public_dir.mkdir(exist_ok=True)

    # Copy audio
    audio_dest = public_dir / "clip-audio.mp3"
    import shutil
    shutil.copy2(audio_path, audio_dest)

    # Copy cover art
    cover_dest = public_dir / "cover.png"
    if COVER_ART.exists() and (not cover_dest.exists()
                                or cover_dest.stat().st_mtime < COVER_ART.stat().st_mtime):
        shutil.copy2(COVER_ART, cover_dest)

    # Build caption data for Remotion — convert word timestamps to clip-relative
    remotion_lines = []
    for line in caption_lines:
        remotion_words = []
        for w in line["words"]:
            w_start = max(0, round(w["start"] - clip_start, 3))
            w_end = min(round(duration, 3), round(w["end"] - clip_start, 3))
            if w_end <= w_start:
                w_end = w_start + 0.1
            remotion_words.append({
                "word": w["word"].strip(),
                "start": w_start,
                "end": w_end,
            })
        remotion_lines.append({
            "start": round(line["start"], 3),
            "end": round(line["end"], 3),
            "speaker": line.get("speaker", ""),
            "words": remotion_words,
        })

    episode_label = f"EPISODE {episode_number}" if episode_number else "LUKE AT THE ROOST"

    props = {
        "captionLines": remotion_lines,
        "clipTitle": clip_title,
        "episodeLabel": episode_label,
        "durationSeconds": round(duration + 0.5, 1),  # small padding
        "audioFile": "clip-audio.mp3",
        "coverFile": "cover.png",
    }

    # Write props to temp file
    props_path = REMOTION_DIR / "render-props.json"
    with open(props_path, "w") as f:
        json.dump(props, f)

    cmd = [
        "npx", "remotion", "render",
        "src/index.ts", "PodcastClipDemo",
        f"--props={props_path}",
        "--timeout=60000",
        "--log=verbose",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REMOTION_DIR), timeout=180)
    except subprocess.TimeoutExpired:
        props_path.unlink(missing_ok=True)
        print(f"    Remotion render timed out (180s)")
        return False
    props_path.unlink(missing_ok=True)

    if result.returncode != 0:
        stderr = result.stderr
        # Show head (error message) and tail (stack trace) of stderr
        if len(stderr) > 1000:
            print(f"    Remotion error (first 500 chars): {stderr[:500]}")
            print(f"    ... (last 500 chars): {stderr[-500:]}")
        else:
            print(f"    Remotion error: {stderr}")
        return False
    return True


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
    return slug[:50]


def detect_episode_number(audio_path: str) -> int | None:
    """Try to detect episode number from filename."""
    name = Path(audio_path).stem
    m = re.search(r'(?:episode|ep|podcast)[-_]?(\d+)', name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)', name)
    if m:
        return int(m.group(1))
    return None


def fetch_episodes() -> list[dict]:
    """Fetch episode list from Castopod RSS feed."""
    print("Fetching episodes from Castopod...")
    try:
        resp = requests.get(RSS_FEED_URL, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching RSS feed: {e}")
        sys.exit(1)

    root = ET.fromstring(resp.content)
    ns = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}
    episodes = []

    for item in root.findall(".//item"):
        title = item.findtext("title", "")
        enclosure = item.find("enclosure")
        audio_url = enclosure.get("url", "") if enclosure is not None else ""
        duration = item.findtext("itunes:duration", "", ns)
        ep_num = item.findtext("itunes:episode", "", ns)
        pub_date = item.findtext("pubDate", "")

        if not audio_url:
            continue

        episodes.append({
            "title": title,
            "audio_url": audio_url,
            "duration": duration,
            "episode_number": int(ep_num) if ep_num and ep_num.isdigit() else None,
            "pub_date": pub_date,
        })

    return episodes


def pick_episode(episodes: list[dict]) -> dict:
    """Display episode list and let user pick one."""
    if not episodes:
        print("No episodes found.")
        sys.exit(1)

    # Sort by episode number (episodes without numbers go to the end)
    episodes.sort(key=lambda e: (e["episode_number"] is None, e["episode_number"] or 0))

    print(f"\nFound {len(episodes)} episodes:\n")
    for ep in episodes:
        num = ep['episode_number']
        label = f"Ep{num}" if num else "  "
        dur = ep['duration'] or "?"
        display_num = f"{num:>2}" if num else " ?"
        print(f"  {display_num}. [{label:>4}] {ep['title']}  ({dur})")

    print()
    while True:
        try:
            choice = input("Select episode number (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                sys.exit(0)
            num = int(choice)
            # Match by episode number first
            match = next((ep for ep in episodes if ep["episode_number"] == num), None)
            if match:
                return match
            print(f"  No episode #{num} found. Episodes: {', '.join(str(e['episode_number']) for e in episodes if e['episode_number'])}")
        except (ValueError, EOFError):
            print("  Enter an episode number")


def download_episode(episode: dict) -> Path:
    """Download episode audio, using a cache to avoid re-downloading."""
    EPISODE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Build a filename from episode number or title slug
    if episode["episode_number"]:
        filename = f"episode-{episode['episode_number']}.mp3"
    else:
        filename = slugify(episode["title"]) + ".mp3"

    cached = EPISODE_CACHE_DIR / filename
    if cached.exists():
        size_mb = cached.stat().st_size / (1024 * 1024)
        print(f"Using cached: {cached.name} ({size_mb:.1f} MB)")
        return cached

    print(f"Downloading: {episode['title']}...")
    try:
        resp = requests.get(episode["audio_url"], stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(cached, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / (1024*1024):.1f} / {total / (1024*1024):.1f} MB ({pct:.0f}%)", end="", flush=True)
                else:
                    print(f"\r  {downloaded / (1024*1024):.1f} MB", end="", flush=True)
        print()
    except requests.RequestException as e:
        if cached.exists():
            cached.unlink()
        print(f"\nError downloading episode: {e}")
        sys.exit(1)

    size_mb = cached.stat().st_size / (1024 * 1024)
    print(f"Saved: {cached.name} ({size_mb:.1f} MB)")
    return cached


def main():
    parser = argparse.ArgumentParser(description="Extract short-form clips from podcast episodes")
    parser.add_argument("audio_file", nargs="?", help="Path to episode MP3 (optional if using --pick)")
    parser.add_argument("--pick", "-p", action="store_true",
                        help="Pick an episode from Castopod to clip")
    parser.add_argument("--transcript", "-t", help="Path to labeled transcript (.txt)")
    parser.add_argument("--chapters", "-c", help="Path to chapters JSON")
    parser.add_argument("--count", "-n", type=int, default=3, help="Number of clips to extract (default: 3)")
    parser.add_argument("--episode-number", "-e", type=int, help="Episode number (auto-detected from filename)")
    parser.add_argument("--output-dir", "-o", help="Output directory (default: clips/episode-N/)")
    parser.add_argument("--audio-only", action="store_true", help="Only extract audio clips, skip video")
    parser.add_argument("--fast-model", default=WHISPER_MODEL_FAST,
                        help=f"Fast Whisper model for clip identification (default: {WHISPER_MODEL_FAST})")
    parser.add_argument("--quality-model", default=WHISPER_MODEL_QUALITY,
                        help=f"Quality Whisper model for clip refinement (default: {WHISPER_MODEL_QUALITY})")
    parser.add_argument("--single-pass", action="store_true",
                        help="Use quality model for everything (slower, no two-pass)")
    parser.add_argument("--legacy-video", action="store_true",
                        help="Use old PIL+ffmpeg video renderer instead of Remotion")
    args = parser.parse_args()

    # Default to --pick when no audio file provided
    if not args.audio_file and not args.pick:
        args.pick = True

    if args.pick:
        episodes = fetch_episodes()
        selected = pick_episode(episodes)
        audio_path = download_episode(selected)
        episode_number = selected["episode_number"] or args.episode_number
    else:
        audio_path = Path(args.audio_file).expanduser().resolve()
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}")
            sys.exit(1)
        episode_number = None

    # Detect episode number
    if not args.pick:
        episode_number = args.episode_number or detect_episode_number(str(audio_path))
    if args.episode_number:
        episode_number = args.episode_number

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif episode_number:
        output_dir = Path(__file__).parent / "clips" / f"episode-{episode_number}"
    else:
        output_dir = Path(__file__).parent / "clips" / audio_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Clip extraction: {audio_path.name}")
    if episode_number:
        print(f"Episode: {episode_number}")
    print(f"Output: {output_dir}")
    print(f"Clips requested: {args.count}")

    # Step 1: Load labeled transcript (needed to prime Whisper with names)
    print(f"\n[1] Loading labeled transcript...")
    labeled_transcript = ""
    if args.transcript:
        transcript_path = Path(args.transcript).expanduser().resolve()
        if transcript_path.exists():
            labeled_transcript = transcript_path.read_text()
            print(f"    Loaded: {transcript_path.name} ({len(labeled_transcript)} chars)")
        else:
            print(f"    Warning: Transcript not found: {transcript_path}")
    else:
        # Auto-detect from website/transcripts/
        transcripts_dir = Path(__file__).parent / "website" / "transcripts"
        if episode_number and transcripts_dir.exists():
            for f in transcripts_dir.iterdir():
                if f.suffix == ".txt" and f"episode-{episode_number}" in f.name:
                    labeled_transcript = f.read_text()
                    print(f"    Auto-detected: {f.name}")
                    break
        if not labeled_transcript:
            print("    No labeled transcript found (names may be inaccurate)")

    # Step 2: Fast transcription for clip identification
    two_pass = not args.single_pass and args.fast_model != args.quality_model
    if two_pass:
        print(f"\n[2/7] Fast transcription for clip identification ({args.fast_model})...")
    else:
        print(f"\n[2/6] Transcribing with word-level timestamps ({args.quality_model})...")
    identify_model = args.fast_model if two_pass else args.quality_model
    segments = transcribe_with_timestamps(
        str(audio_path), identify_model, labeled_transcript
    )

    # Build timestamped transcript for LLM
    transcript_text = get_transcript_text(segments)

    # Load chapters if provided
    chapters_json = None
    if args.chapters:
        chapters_path = Path(args.chapters).expanduser().resolve()
        if chapters_path.exists():
            with open(chapters_path) as f:
                chapters_json = f.read()
            print(f"    Chapters loaded: {chapters_path.name}")

    # Step 3: LLM selects best moments
    step_total = 7 if two_pass else 6
    print(f"\n[3/{step_total}] Selecting {args.count} best moments with LLM...")
    clips = select_clips_with_llm(transcript_text, labeled_transcript,
                                   chapters_json, args.count)
    if not clips:
        print("\nNo clips selected — aborting.")
        return

    # Snap to sentence boundaries so clips don't start/end mid-sentence
    clips = snap_to_sentences(clips, segments)

    for i, clip in enumerate(clips):
        duration = clip["end_time"] - clip["start_time"]
        print(f"    Clip {i+1}: \"{clip['title']}\" "
              f"({clip['start_time']:.1f}s - {clip['end_time']:.1f}s, {duration:.0f}s)")
        print(f"           \"{clip['caption_text']}\"")

    # Generate social media metadata
    meta_step = 4
    print(f"\n[{meta_step}/{step_total}] Generating social media descriptions...")
    clips = generate_social_metadata(clips, labeled_transcript, episode_number)
    for i, clip in enumerate(clips):
        if "description" in clip:
            print(f"    Clip {i+1}: {clip['description'][:80]}...")
            print(f"           {' '.join(clip.get('hashtags', []))}")

    # Step 5: Refine clip timestamps with quality model (two-pass only)
    refined = {}
    if two_pass:
        print(f"\n[5/{step_total}] Refining clips with {args.quality_model}...")
        refined = refine_clip_timestamps(
            str(audio_path), clips, args.quality_model, labeled_transcript
        )
        # Re-snap to sentence boundaries using refined segments
        for i, clip in enumerate(clips):
            if i in refined and refined[i]:
                clip_segments = refined[i]
                clips[i:i+1] = snap_to_sentences([clip], clip_segments)

    # Step N: Extract audio clips
    extract_step = 6 if two_pass else 5
    print(f"\n[{extract_step}/{step_total}] Extracting audio clips...")
    for i, clip in enumerate(clips):
        print(f"    [{i+1}/{len(clips)}] \"{clip['title']}\"...")
        slug = slugify(clip["title"])
        mp3_path = output_dir / f"clip-{i+1}-{slug}.mp3"

        try:
            if extract_clip_audio(str(audio_path), clip["start_time"], clip["end_time"],
                                  str(mp3_path)):
                print(f"    Clip {i+1} audio: {mp3_path.name}")
            else:
                print(f"    Error extracting clip {i+1} audio — skipping")
        except Exception as e:
            print(f"    Clip {i+1} audio failed: {e} — skipping")

    video_step = 7 if two_pass else 6
    if args.audio_only:
        print(f"\n[{video_step}/{step_total}] Skipped video generation (--audio-only)")
        print(f"\nDone! {len(clips)} audio clips saved to {output_dir}")
        return

    # Step N: Generate video clips
    use_remotion = REMOTION_DIR.exists() and not args.legacy_video
    renderer = "Remotion" if use_remotion else "PIL+ffmpeg"
    print(f"\n[{video_step}/{step_total}] Generating video clips ({renderer})...")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        for i, clip in enumerate(clips):
            slug = slugify(clip["title"])
            mp3_path = output_dir / f"clip-{i+1}-{slug}.mp3"
            mp4_path = output_dir / f"clip-{i+1}-{slug}.mp4"
            duration = clip["end_time"] - clip["start_time"]

            print(f"    [{i+1}/{len(clips)}] \"{clip['title']}\" ({duration:.0f}s)...")

            try:
                # Get word timestamps — use refined segments if available
                word_source = refined[i] if (two_pass and i in refined and refined[i]) else segments
                clip_words = get_words_in_range(word_source, clip["start_time"], clip["end_time"])

                # Add speaker labels
                clip_words = add_speaker_labels(clip_words, labeled_transcript,
                                                clip["start_time"], clip["end_time"],
                                                word_source)

                # Polish text with LLM (fix punctuation, capitalization, mishearings)
                clip_words = polish_clip_words(clip_words, labeled_transcript)

                # Group words into timed caption lines
                caption_lines = group_words_into_lines(
                    clip_words, clip["start_time"], duration
                )

                if use_remotion:
                    if generate_clip_video_remotion(
                        str(mp3_path), caption_lines, clip["start_time"],
                        clip["title"], episode_number, str(mp4_path), duration
                    ):
                        file_size = mp4_path.stat().st_size / (1024 * 1024)
                        print(f"    Clip {i+1} video: {mp4_path.name} ({file_size:.1f} MB)")
                    else:
                        print(f"    Clip {i+1} video failed (Remotion) — skipping")
                else:
                    # Legacy PIL+ffmpeg renderer
                    bg_path = str(tmp_dir / f"bg_{i}.png")
                    generate_background_image(episode_number, clip["title"], bg_path)

                    clip_tmp = tmp_dir / f"clip_{i}"
                    clip_tmp.mkdir(exist_ok=True)

                    if generate_clip_video(str(mp3_path), bg_path, caption_lines,
                                           clip["start_time"], str(mp4_path),
                                           duration, clip_tmp):
                        file_size = mp4_path.stat().st_size / (1024 * 1024)
                        print(f"    Clip {i+1} video: {mp4_path.name} ({file_size:.1f} MB)")
                    else:
                        print(f"    Clip {i+1} video failed (ffmpeg) — skipping")
            except Exception as e:
                print(f"    Clip {i+1} video failed: {e} — skipping")

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

    # Summary
    print(f"\nDone! {len(clips)} clips saved to {output_dir}")
    for i, clip in enumerate(clips):
        slug = slugify(clip["title"])
        mp4 = output_dir / f"clip-{i+1}-{slug}.mp4"
        mp3 = output_dir / f"clip-{i+1}-{slug}.mp3"
        print(f"  {i+1}. \"{clip['title']}\"")
        if mp4.exists():
            print(f"     Video: {mp4}")
        if mp3.exists():
            print(f"     Audio: {mp3}")


if __name__ == "__main__":
    main()
