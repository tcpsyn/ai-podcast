#!/usr/bin/env python3
"""Analyze silence gaps in podcast stems to find optimal strip-silence thresholds.

Usage: python analyze_gaps.py recordings/2026-03-17_235137/
"""
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

BLOCK_SEC = 0.1
SILENCE_DB = -30
THRESHOLD = 10 ** (SILENCE_DB / 20)
MIN_VOICE_SEC = 0.3


def load_stem(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, sr


def compute_rms_blocks(audio: np.ndarray, sr: int) -> np.ndarray:
    block_samples = int(sr * BLOCK_SEC)
    n_blocks = len(audio) // block_samples
    if n_blocks == 0:
        return np.array([0.0])
    trimmed = audio[:n_blocks * block_samples].reshape(n_blocks, block_samples)
    return np.sqrt(np.mean(trimmed ** 2, axis=1))


def compute_peak_blocks(audio: np.ndarray, sr: int) -> np.ndarray:
    block_samples = int(sr * BLOCK_SEC)
    n_blocks = len(audio) // block_samples
    if n_blocks == 0:
        return np.array([0.0])
    trimmed = audio[:n_blocks * block_samples].reshape(n_blocks, block_samples)
    return np.max(np.abs(trimmed), axis=1)


def analyze(stems_dir: Path):
    stems_dir = Path(stems_dir)
    voice_stems = {}
    for name in ["host", "devon", "caller"]:
        path = stems_dir / f"{name}.wav"
        if path.exists():
            print(f"Loading {name}...", end=" ", flush=True)
            audio, sr = load_stem(path)
            voice_stems[name] = audio
            print(f"{len(audio)/sr:.0f}s @ {sr}Hz")

    if not voice_stems:
        print("No voice stems found")
        return

    sr_val = sr
    duration = max(len(a) for a in voice_stems.values()) / sr_val
    print(f"\nTotal duration: {duration/60:.1f} min")

    # Compute per-track RMS and peak blocks
    track_rms = {}
    track_peak = {}
    for name, audio in voice_stems.items():
        track_rms[name] = compute_rms_blocks(audio, sr_val)
        track_peak[name] = compute_peak_blocks(audio, sr_val)

    n_blocks = min(len(v) for v in track_peak.values())

    # Detect gaps using same logic as Lua script (RMS for speaker ID, peak for silence)
    min_voice_blocks = int(MIN_VOICE_SEC / BLOCK_SEC)
    track_names = list(voice_stems.keys())

    gaps = []
    in_silence = False
    silence_start = 0
    track_before = None
    last_active = None
    voice_run = 0
    voice_run_track = None

    for i in range(n_blocks):
        # Peak for silence detection
        best_peak = max(track_peak[name][i] for name in track_names)
        # RMS for speaker identification
        best_rms = 0
        best_track = None
        for name in track_names:
            r = track_rms[name][i]
            if r > best_rms:
                best_rms = r
                best_track = name

        all_silent = best_peak < THRESHOLD

        if not all_silent:
            last_active = best_track

        if in_silence:
            if all_silent:
                voice_run = 0
                voice_run_track = None
            else:
                if voice_run == 0:
                    voice_run_track = best_track
                voice_run += 1
                if voice_run >= min_voice_blocks:
                    voice_start_block = i - (voice_run - 1)
                    gap_start = silence_start * BLOCK_SEC
                    gap_end = voice_start_block * BLOCK_SEC
                    dur = gap_end - gap_start
                    if dur >= 0.5:  # log gaps >= 0.5s
                        gaps.append({
                            "start": gap_start,
                            "end": gap_end,
                            "dur": dur,
                            "before": track_before or "?",
                            "after": voice_run_track or "?",
                        })
                    in_silence = False
                    voice_run = 0
                    voice_run_track = None
        else:
            if all_silent:
                in_silence = True
                silence_start = i
                track_before = last_active
                voice_run = 0
                voice_run_track = None

    # Trailing silence
    if in_silence:
        dur = (n_blocks - silence_start) * BLOCK_SEC
        if dur >= 0.5:
            gaps.append({
                "start": silence_start * BLOCK_SEC,
                "end": n_blocks * BLOCK_SEC,
                "dur": dur,
                "before": track_before or "?",
                "after": "end",
            })

    if not gaps:
        print("No gaps detected")
        return

    # Categorize gaps
    categories = {
        "host_self": [],      # Host -> Host
        "host_to_caller": [], # Host -> Caller (TTS latency)
        "caller_to_host": [], # Caller -> Host
        "host_to_devon": [],  # Host -> Devon (TTS latency)
        "devon_to_host": [],  # Devon -> Host
        "caller_to_devon": [],# Caller -> Devon (interjection)
        "devon_to_caller": [],# Devon -> Caller
        "other": [],
    }

    for g in gaps:
        b, a = g["before"], g["after"]
        if b == "host" and a == "host":
            categories["host_self"].append(g)
        elif b == "host" and a == "caller":
            categories["host_to_caller"].append(g)
        elif b == "caller" and a == "host":
            categories["caller_to_host"].append(g)
        elif b == "host" and a == "devon":
            categories["host_to_devon"].append(g)
        elif b == "devon" and a == "host":
            categories["devon_to_host"].append(g)
        elif b == "caller" and a == "devon":
            categories["caller_to_devon"].append(g)
        elif b == "devon" and a == "caller":
            categories["devon_to_caller"].append(g)
        else:
            categories["other"].append(g)

    # Print results
    print(f"\n{'='*70}")
    print(f"GAP ANALYSIS — {len(gaps)} gaps detected")
    print(f"{'='*70}")

    total_silence = sum(g["dur"] for g in gaps)
    print(f"Total silence: {total_silence:.0f}s ({total_silence/60:.1f} min)")
    print(f"Content after removal: ~{(duration - total_silence)/60:.1f} min")

    for cat_name, cat_gaps in sorted(categories.items(), key=lambda x: -len(x[1])):
        if not cat_gaps:
            continue
        durs = sorted([g["dur"] for g in cat_gaps])
        print(f"\n--- {cat_name} ({len(cat_gaps)} gaps) ---")
        print(f"  Range: {durs[0]:.1f}s - {durs[-1]:.1f}s")
        print(f"  Median: {np.median(durs):.1f}s  Mean: {np.mean(durs):.1f}s")
        if len(durs) >= 5:
            print(f"  P25: {np.percentile(durs, 25):.1f}s  P75: {np.percentile(durs, 75):.1f}s")

        # Histogram
        brackets = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 18), (18, 30), (30, 60), (60, 999)]
        print(f"  Distribution:")
        for lo, hi in brackets:
            count = sum(1 for d in durs if lo <= d < hi)
            if count > 0:
                bar = "#" * count
                label = f"{lo}-{hi}s" if hi < 999 else f"{lo}s+"
                print(f"    {label:>8s}: {bar} ({count})")

    # Find natural clusters and suggest thresholds
    print(f"\n{'='*70}")
    print("SUGGESTED THRESHOLDS")
    print(f"{'='*70}")

    # For each Devon-involved category, find the gap between interjection and TTS gaps
    devon_gaps = categories["host_to_devon"] + categories["devon_to_host"] + categories["caller_to_devon"] + categories["devon_to_caller"]
    if devon_gaps:
        devon_durs = sorted([g["dur"] for g in devon_gaps])
        # Look for a natural break between short (interjection) and long (TTS) gaps
        short = [d for d in devon_durs if d < 5]
        long = [d for d in devon_durs if d >= 5]
        if short and long:
            suggested = (max(short) + min(long)) / 2
            print(f"Devon threshold: {suggested:.1f}s  (short gaps: {len(short)} up to {max(short):.1f}s, long gaps: {len(long)} from {min(long):.1f}s)")
        elif short:
            print(f"Devon threshold: {max(short) + 1:.1f}s  (all gaps are short, max {max(short):.1f}s)")
        else:
            print(f"Devon threshold: 3.0s  (all gaps are long, min {min(long):.1f}s)")

    caller_gaps = categories["host_to_caller"] + categories["caller_to_host"]
    if caller_gaps:
        caller_durs = sorted([g["dur"] for g in caller_gaps])
        short = [d for d in caller_durs if d < 5]
        long = [d for d in caller_durs if d >= 5]
        if short and long:
            suggested = (max(short) + min(long)) / 2
            print(f"Caller transition threshold: {suggested:.1f}s  (short: {len(short)} up to {max(short):.1f}s, long: {len(long)} from {min(long):.1f}s)")
        elif long:
            print(f"Caller transition threshold: {min(long) - 1:.1f}s  (all gaps >= {min(long):.1f}s)")

    host_self = categories["host_self"]
    if host_self:
        host_durs = sorted([g["dur"] for g in host_self])
        short = [d for d in host_durs if d < 5]
        long = [d for d in host_durs if d >= 5]
        if short and long:
            suggested = (max(short) + min(long)) / 2
            print(f"Same-speaker threshold: {suggested:.1f}s  (short: {len(short)} up to {max(short):.1f}s, long: {len(long)} from {min(long):.1f}s)")
        elif long:
            print(f"Same-speaker threshold: {min(long) - 1:.1f}s  (all gaps >= {min(long):.1f}s)")

    all_durs = sorted([g["dur"] for g in gaps])
    would_cut = [d for d in all_durs if d >= 3.0]
    print(f"\nWith current thresholds (Devon=3s, others=6s):")
    print(f"  Would cut: ~{len(would_cut)} gaps, ~{sum(would_cut):.0f}s ({sum(would_cut)/60:.1f} min)")
    print(f"  Result: ~{(duration - sum(would_cut))/60:.1f} min")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_gaps.py <stems_dir>")
        sys.exit(1)
    analyze(Path(sys.argv[1]))
