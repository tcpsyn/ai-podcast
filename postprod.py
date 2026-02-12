#!/usr/bin/env python3
"""Post-production pipeline for AI podcast stems.

Usage: python postprod.py recordings/2026-02-07_213000/ -o episode.mp3

Processes 5 aligned WAV stems (host, caller, music, sfx, ads) into a
broadcast-ready MP3 with gap removal, voice compression, music ducking,
and loudness normalization.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

STEM_NAMES = ["host", "caller", "music", "sfx", "ads"]


def load_stems(stems_dir: Path) -> tuple[dict[str, np.ndarray], int]:
    stems = {}
    sample_rate = None
    for name in STEM_NAMES:
        path = stems_dir / f"{name}.wav"
        if not path.exists():
            print(f"  {name}.wav not found, creating empty stem")
            stems[name] = None
            continue
        data, sr = sf.read(str(path), dtype="float32")
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            print(f"  WARNING: {name}.wav has sample rate {sr}, expected {sample_rate}")
        stems[name] = data
        print(f"  {name}: {len(data)} samples ({len(data)/sr:.1f}s)")

    if sample_rate is None:
        print("ERROR: No valid stems found")
        sys.exit(1)

    # Pad all stems to same length
    max_len = max(len(s) for s in stems.values() if s is not None)
    for name in STEM_NAMES:
        if stems[name] is None:
            stems[name] = np.zeros(max_len, dtype=np.float32)
        elif len(stems[name]) < max_len:
            stems[name] = np.pad(stems[name], (0, max_len - len(stems[name])))

    return stems, sample_rate


def compute_rms(audio: np.ndarray, window_samples: int) -> np.ndarray:
    n_windows = len(audio) // window_samples
    if n_windows == 0:
        return np.array([0.0])
    trimmed = audio[:n_windows * window_samples].reshape(n_windows, window_samples)
    return np.sqrt(np.mean(trimmed ** 2, axis=1))


def remove_gaps(stems: dict[str, np.ndarray], sr: int,
                threshold_s: float = 2.0, max_gap_s: float = 8.0,
                crossfade_ms: float = 30, pad_s: float = 0.5) -> dict[str, np.ndarray]:
    window_ms = 50
    window_samples = int(sr * window_ms / 1000)
    crossfade_samples = int(sr * crossfade_ms / 1000)

    # Detect gaps in everything except music (which always plays).
    # This catches TTS latency gaps while protecting ad breaks and SFX transitions.
    content = stems["host"] + stems["caller"] + stems["sfx"] + stems["ads"]
    rms = compute_rms(content, window_samples)

    # Threshold: percentile-based to sit above the mic noise floor
    nonzero_rms = rms[rms > 0]
    if len(nonzero_rms) == 0:
        print("  No audio detected")
        return stems
    noise_floor = np.percentile(nonzero_rms, 20)
    silence_thresh = noise_floor * 3

    is_silent = rms < silence_thresh
    min_silent_windows = int(threshold_s / (window_ms / 1000))
    max_silent_windows = int(max_gap_s / (window_ms / 1000))

    # Only cut gaps between 1.5-8s — targets TTS latency, not long breaks
    cuts = []
    i = 0
    while i < len(is_silent):
        if is_silent[i]:
            start = i
            while i < len(is_silent) and is_silent[i]:
                i += 1
            length = i - start
            if min_silent_windows <= length <= max_silent_windows:
                # Leave pad_s of silence so the edit sounds natural
                pad_samples = int(pad_s * sr)
                cut_start = (start + 1) * window_samples + pad_samples
                cut_end = (i - 1) * window_samples - pad_samples
                if cut_end > cut_start + crossfade_samples * 2:
                    cuts.append((cut_start, cut_end))
        else:
            i += 1

    if not cuts:
        print("  No gaps to remove")
        return stems

    total_cut = sum(end - start for start, end in cuts) / sr
    print(f"  Removing {len(cuts)} gaps ({total_cut:.1f}s total)")

    # Cut dialog/sfx/ads at gap points. Leave music uncut — just trim to fit.
    result = {}

    for name in STEM_NAMES:
        if name == "music":
            continue  # handled below
        audio = stems[name]
        pieces = []
        prev_end = 0
        for cut_start, cut_end in cuts:
            if prev_end < cut_start:
                piece = audio[prev_end:cut_start].copy()
                if pieces and len(piece) > crossfade_samples:
                    fade_in = np.linspace(0, 1, crossfade_samples, dtype=np.float32)
                    piece[:crossfade_samples] *= fade_in
                if len(pieces) > 0 and len(pieces[-1]) > crossfade_samples:
                    fade_out = np.linspace(1, 0, crossfade_samples, dtype=np.float32)
                    pieces[-1][-crossfade_samples:] *= fade_out
                pieces.append(piece)
            prev_end = cut_end

        if prev_end < len(audio):
            piece = audio[prev_end:].copy()
            if pieces and len(piece) > crossfade_samples:
                fade_in = np.linspace(0, 1, crossfade_samples, dtype=np.float32)
                piece[:crossfade_samples] *= fade_in
            if len(pieces) > 0 and len(pieces[-1]) > crossfade_samples:
                fade_out = np.linspace(1, 0, crossfade_samples, dtype=np.float32)
                pieces[-1][-crossfade_samples:] *= fade_out
            pieces.append(piece)

        result[name] = np.concatenate(pieces) if pieces else np.array([], dtype=np.float32)

    # Music: leave uncut, just trim to match new duration with fade-out
    new_len = len(result["host"])
    music = stems["music"]
    if len(music) >= new_len:
        music = music[:new_len].copy()
    else:
        music = np.pad(music, (0, new_len - len(music)))
    fade_samples = int(sr * 3)
    if len(music) > fade_samples:
        music[-fade_samples:] *= np.linspace(1, 0, fade_samples, dtype=np.float32)
    result["music"] = music

    return result


def denoise(audio: np.ndarray, sr: int, tmp_dir: Path) -> np.ndarray:
    """High-quality noise reduction using ffmpeg afftdn (adaptive Wiener filter)."""
    in_path = tmp_dir / "host_pre_denoise.wav"
    out_path = tmp_dir / "host_post_denoise.wav"
    sf.write(str(in_path), audio, sr)

    # afftdn: adaptive FFT denoiser with Wiener filter
    #   nt=w  - Wiener filter (best quality)
    #   om=o  - output cleaned signal
    #   nr=10 - noise reduction in dB (10 = moderate, preserves voice naturalness)
    #   nf=-30 - noise floor estimate in dB
    # anlmdn: non-local means denoiser for residual broadband noise
    #   s=4   - patch size
    #   p=0.002 - strength (gentle to avoid artifacts)
    af = (
        "afftdn=nt=w:om=o:nr=12:nf=-30,"
        "anlmdn=s=4:p=0.002"
    )
    cmd = ["ffmpeg", "-y", "-i", str(in_path), "-af", af, str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: denoise failed: {result.stderr[:200]}")
        return audio

    denoised, _ = sf.read(str(out_path), dtype="float32")
    return denoised


def compress_voice(audio: np.ndarray, sr: int, tmp_dir: Path,
                   stem_name: str) -> np.ndarray:
    in_path = tmp_dir / f"{stem_name}_pre_comp.wav"
    out_path = tmp_dir / f"{stem_name}_post_comp.wav"

    sf.write(str(in_path), audio, sr)

    cmd = [
        "ffmpeg", "-y", "-i", str(in_path),
        "-af", "acompressor=threshold=-24dB:ratio=2.5:attack=10:release=800:makeup=6dB",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: compression failed for {stem_name}: {result.stderr[:200]}")
        return audio

    compressed, _ = sf.read(str(out_path), dtype="float32")
    return compressed


def phone_eq(audio: np.ndarray, sr: int, tmp_dir: Path) -> np.ndarray:
    """Apply telephone EQ to make caller sound like a phone call."""
    in_path = tmp_dir / "caller_pre_phone.wav"
    out_path = tmp_dir / "caller_post_phone.wav"
    sf.write(str(in_path), audio, sr)

    # Bandpass 300-3400Hz (telephone bandwidth) + slight mid boost for presence
    af = (
        "highpass=f=300:poles=2,"
        "lowpass=f=3400:poles=2,"
        "equalizer=f=1000:t=q:w=0.8:g=4"
    )
    cmd = ["ffmpeg", "-y", "-i", str(in_path), "-af", af, str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: phone EQ failed: {result.stderr[:200]}")
        return audio

    filtered, _ = sf.read(str(out_path), dtype="float32")
    return filtered


def apply_ducking(music: np.ndarray, dialog: np.ndarray, sr: int,
                  duck_db: float = -20, attack_ms: float = 200,
                  release_ms: float = 3000,
                  mute_signal: np.ndarray | None = None) -> np.ndarray:
    window_ms = 50
    window_samples = int(sr * window_ms / 1000)
    rms = compute_rms(dialog, window_samples)

    # Speech detection threshold
    mean_rms = np.mean(rms[rms > 0]) if np.any(rms > 0) else 1e-4
    speech_thresh = mean_rms * 0.1

    # Build gain envelope (per window)
    duck_gain = 10 ** (duck_db / 20)
    is_speech = rms > speech_thresh
    target_gain = np.where(is_speech, duck_gain, 1.0).astype(np.float32)

    # Mute music completely during ads with lookahead and tail
    if mute_signal is not None:
        mute_rms = compute_rms(mute_signal, window_samples)
        mute_thresh = np.mean(mute_rms[mute_rms > 0]) * 0.1 if np.any(mute_rms > 0) else 1e-4
        is_ads = mute_rms > mute_thresh
        # Expand ad regions: 2s before (fade out music before ad) and 2s after (don't resume immediately)
        lookahead_windows = int(2000 / window_ms)
        tail_windows = int(2000 / window_ms)
        expanded_ads = is_ads.copy()
        for i in range(len(is_ads)):
            if is_ads[i]:
                start = max(0, i - lookahead_windows)
                end = min(len(expanded_ads), i + tail_windows + 1)
                expanded_ads[start:end] = True
        target_gain[expanded_ads] = 0.0

    # Smooth the envelope
    attack_windows = max(1, int(attack_ms / window_ms))
    release_windows = max(1, int(release_ms / window_ms))
    smoothed = np.ones_like(target_gain)
    for i in range(1, len(target_gain)):
        if target_gain[i] < smoothed[i - 1]:
            alpha = 1.0 / attack_windows
            smoothed[i] = smoothed[i - 1] + alpha * (target_gain[i] - smoothed[i - 1])
        else:
            alpha = 1.0 / release_windows
            smoothed[i] = smoothed[i - 1] + alpha * (target_gain[i] - smoothed[i - 1])

    # Expand envelope to sample level
    gain_samples = np.repeat(smoothed, window_samples)
    if len(gain_samples) < len(music):
        gain_samples = np.pad(gain_samples, (0, len(music) - len(gain_samples)), constant_values=1.0)
    else:
        gain_samples = gain_samples[:len(music)]

    return music * gain_samples


def match_voice_levels(stems: dict[str, np.ndarray], target_rms: float = 0.1) -> dict[str, np.ndarray]:
    """Normalize host, caller, and ads stems to the same RMS level."""
    for name in ["host", "caller", "ads"]:
        audio = stems[name]
        # Only measure non-silent portions
        active = audio[np.abs(audio) > 0.001]
        if len(active) == 0:
            continue
        current_rms = np.sqrt(np.mean(active ** 2))
        if current_rms < 1e-6:
            continue
        gain = target_rms / current_rms
        # Clamp gain to avoid extreme boosts on very quiet stems
        gain = min(gain, 10.0)
        stems[name] = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
        db_change = 20 * np.log10(gain) if gain > 0 else 0
        print(f"  {name}: RMS {current_rms:.4f} -> {target_rms:.4f} ({db_change:+.1f}dB)")
    return stems


def mix_stems(stems: dict[str, np.ndarray],
              levels: dict[str, float] | None = None) -> np.ndarray:
    if levels is None:
        levels = {"host": 0, "caller": 0, "music": -6, "sfx": -6, "ads": 0}

    gains = {name: 10 ** (db / 20) for name, db in levels.items()}

    # Find max length
    max_len = max(len(s) for s in stems.values())

    mix = np.zeros(max_len, dtype=np.float64)
    for name in STEM_NAMES:
        audio = stems[name]
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))
        mix += audio.astype(np.float64) * gains.get(name, 1.0)

    # Stereo (mono duplicated to both channels)
    mix = np.clip(mix, -1.0, 1.0).astype(np.float32)
    stereo = np.column_stack([mix, mix])
    return stereo


def normalize_and_export(audio: np.ndarray, sr: int, output_path: Path,
                         target_lufs: float = -16, bitrate: str = "128k",
                         tmp_dir: Path = None):
    tmp_wav = tmp_dir / "pre_loudnorm.wav"
    sf.write(str(tmp_wav), audio, sr)

    # Pass 1: measure loudness
    measure_cmd = [
        "ffmpeg", "-y", "-i", str(tmp_wav),
        "-af", f"loudnorm=I={target_lufs}:TP=-1:LRA=11:print_format=json",
        "-f", "null", "-",
    ]
    result = subprocess.run(measure_cmd, capture_output=True, text=True)
    stderr = result.stderr

    # Parse loudnorm output
    import json
    json_start = stderr.rfind("{")
    json_end = stderr.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        stats = json.loads(stderr[json_start:json_end])
    else:
        print("  WARNING: couldn't parse loudnorm stats, using defaults")
        stats = {
            "input_i": "-23", "input_tp": "-1", "input_lra": "11",
            "input_thresh": "-34",
        }

    # Pass 2: apply normalization + limiter + export MP3
    loudnorm_filter = (
        f"loudnorm=I={target_lufs}:TP=-1:LRA=11"
        f":measured_I={stats['input_i']}"
        f":measured_TP={stats['input_tp']}"
        f":measured_LRA={stats['input_lra']}"
        f":measured_thresh={stats['input_thresh']}"
        f":linear=true"
    )
    export_cmd = [
        "ffmpeg", "-y", "-i", str(tmp_wav),
        "-af", f"{loudnorm_filter},alimiter=limit=-1dB:level=false",
        "-ab", bitrate, "-ar", str(sr),
        str(output_path),
    ]
    result = subprocess.run(export_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: export failed: {result.stderr[:300]}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Post-production for AI podcast stems")
    parser.add_argument("stems_dir", type=Path, help="Directory containing stem WAV files")
    parser.add_argument("-o", "--output", type=str, default="episode.mp3", help="Output filename")
    parser.add_argument("--gap-threshold", type=float, default=2.0, help="Min silence to cut (seconds)")
    parser.add_argument("--duck-amount", type=float, default=-20, help="Music duck in dB")
    parser.add_argument("--target-lufs", type=float, default=-16, help="Target loudness (LUFS)")
    parser.add_argument("--bitrate", type=str, default="128k", help="MP3 bitrate")
    parser.add_argument("--no-gap-removal", action="store_true", help="Skip gap removal")
    parser.add_argument("--no-compression", action="store_true", help="Skip voice compression")
    parser.add_argument("--no-ducking", action="store_true", help="Skip music ducking")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    stems_dir = args.stems_dir
    if not stems_dir.exists():
        print(f"ERROR: directory not found: {stems_dir}")
        sys.exit(1)

    # Resolve output path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = stems_dir / output_path

    print(f"Post-production: {stems_dir} -> {output_path}")
    print(f"  Gap removal: {'skip' if args.no_gap_removal else f'threshold={args.gap_threshold}s'}")
    print(f"  Compression: {'skip' if args.no_compression else 'on'}")
    print(f"  Ducking: {'skip' if args.no_ducking else f'{args.duck_amount}dB'}")
    print(f"  Loudness: {args.target_lufs} LUFS, bitrate: {args.bitrate}")

    if args.dry_run:
        print("Dry run — exiting")
        return

    # Step 1: Load
    print("\n[1/9] Loading stems...")
    stems, sr = load_stems(stems_dir)

    # Step 2: Gap removal
    print("\n[2/9] Gap removal...")
    if not args.no_gap_removal:
        stems = remove_gaps(stems, sr, threshold_s=args.gap_threshold)
    else:
        print("  Skipped")

    # Step 3: Host mic noise reduction
    print("\n[3/9] Host mic noise reduction...")
    if np.any(stems["host"] != 0):
        with tempfile.TemporaryDirectory() as tmp:
            stems["host"] = denoise(stems["host"], sr, Path(tmp))
            print("  Applied")
    else:
        print("  No host audio")

    # Step 4: Voice compression
    print("\n[4/9] Voice compression...")
    if not args.no_compression:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for name in ["host", "caller"]:
                if np.any(stems[name] != 0):
                    print(f"  Compressing {name}...")
                    stems[name] = compress_voice(stems[name], sr, tmp_dir, name)
    else:
        print("  Skipped")

    # Step 5: Phone EQ on caller
    print("\n[5/9] Phone EQ on caller...")
    if np.any(stems["caller"] != 0):
        with tempfile.TemporaryDirectory() as tmp:
            stems["caller"] = phone_eq(stems["caller"], sr, Path(tmp))
            print("  Applied")
    else:
        print("  No caller audio")

    # Step 6: Match voice levels
    print("\n[6/9] Matching voice levels...")
    stems = match_voice_levels(stems)

    # Step 7: Music ducking
    print("\n[7/9] Music ducking...")
    if not args.no_ducking:
        dialog = stems["host"] + stems["caller"]
        if np.any(dialog != 0) and np.any(stems["music"] != 0):
            stems["music"] = apply_ducking(stems["music"], dialog, sr, duck_db=args.duck_amount,
                                           mute_signal=stems["ads"])
            print("  Applied")
        else:
            print("  No dialog or music to duck")
    else:
        print("  Skipped")

    # Step 8: Mix
    print("\n[8/9] Mixing...")
    stereo = mix_stems(stems)
    print(f"  Mixed to stereo: {len(stereo)} samples ({len(stereo)/sr:.1f}s)")

    # Step 9: Normalize + export
    print("\n[9/9] Loudness normalization + export...")
    with tempfile.TemporaryDirectory() as tmp:
        normalize_and_export(stereo, sr, output_path,
                             target_lufs=args.target_lufs,
                             bitrate=args.bitrate,
                             tmp_dir=Path(tmp))

    print(f"\nDone! Output: {output_path}")


if __name__ == "__main__":
    main()
