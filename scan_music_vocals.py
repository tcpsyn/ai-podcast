"""Scan music directory for tracks that contain vocals/lyrics.

Uses Whisper to transcribe a sample from each track — if it picks up
actual words, the track likely has vocals.

Usage:
    python scan_music_vocals.py              # scan and report
    python scan_music_vocals.py --delete     # scan and delete vocal tracks
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
from faster_whisper import WhisperModel

MUSIC_DIR = Path(__file__).parent / "music"
WHISPER_MODEL = "distil-large-v3"

# Words Whisper hallucinates on silence/instrumental — ignore these
HALLUCINATION_PHRASES = {
    "thank you", "thanks for watching", "subscribe", "like and subscribe",
    "please subscribe", "thank you for watching", "thanks for listening",
    "you", "the end", "bye", "okay",
}


def scan_track(model: WhisperModel, filepath: Path) -> tuple[bool, str]:
    """Check a single track for vocals. Returns (has_vocals, transcription)."""
    try:
        audio, sr = librosa.load(str(filepath), sr=16000, mono=True)
    except Exception as e:
        return False, f"[load error: {e}]"

    duration = len(audio) / sr
    if duration < 10:
        return False, "[too short]"

    # Sample 30s from the middle (most likely to have vocals)
    mid = len(audio) // 2
    half_window = int(15 * sr)  # 15s each side
    start = max(0, mid - half_window)
    end = min(len(audio), mid + half_window)
    sample = audio[start:end]

    segments, info = model.transcribe(
        sample,
        beam_size=3,
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_speech_duration_ms=500),
    )
    segments_list = list(segments)
    text = " ".join(s.text for s in segments_list).strip()

    # Filter out Whisper hallucinations
    text_lower = text.lower().strip()
    if text_lower in HALLUCINATION_PHRASES or len(text_lower) < 4:
        return False, ""

    # If Whisper found substantial text, it's likely vocals
    word_count = len(text.split())
    has_vocals = word_count >= 3

    return has_vocals, text


def main():
    parser = argparse.ArgumentParser(description="Scan music for vocal tracks")
    parser.add_argument("--delete", action="store_true", help="Delete tracks with vocals")
    args = parser.parse_args()

    audio_files = sorted(
        f for f in MUSIC_DIR.iterdir()
        if f.suffix.lower() in {".mp3", ".wav", ".ogg", ".flac"}
    )

    if not audio_files:
        print("No audio files found in music/")
        return

    print(f"Loading Whisper {WHISPER_MODEL}...")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    print(f"Scanning {len(audio_files)} tracks for vocals...\n")

    vocal_tracks = []
    for i, f in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] {f.name}...", end=" ", flush=True)
        has_vocals, text = scan_track(model, f)
        if has_vocals:
            print(f"VOCALS: {text[:80]}")
            vocal_tracks.append((f, text))
        else:
            print("OK")

    print(f"\n{'='*60}")
    print(f"Results: {len(vocal_tracks)} tracks with vocals out of {len(audio_files)}\n")

    if not vocal_tracks:
        print("All tracks appear to be instrumental!")
        return

    for f, text in vocal_tracks:
        print(f"  {f.name}")
        print(f"    Lyrics: {text[:120]}")
        print()

    if args.delete:
        print(f"Deleting {len(vocal_tracks)} vocal tracks...")
        for f, _ in vocal_tracks:
            f.unlink()
            print(f"  Deleted: {f.name}")
        print("Done.")
    else:
        print("Run with --delete to remove these tracks.")


if __name__ == "__main__":
    main()
