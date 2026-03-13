#!/usr/bin/env python3
"""Post-production pipeline for AI podcast stems.

Usage: python postprod.py recordings/2026-02-07_213000/ -o episode.mp3

Processes 6 aligned WAV stems (host, caller, music, sfx, ads, idents) into a
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

STEM_NAMES = ["host", "caller", "music", "sfx", "ads", "idents"]


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
                threshold_s: float = 2.0, max_gap_s: float = 15.0,
                crossfade_ms: float = 30, pad_s: float = 0.5) -> dict[str, np.ndarray]:
    window_ms = 50
    window_samples = int(sr * window_ms / 1000)
    crossfade_samples = int(sr * crossfade_ms / 1000)

    # Detect gaps in everything except music (which always plays).
    # This catches TTS latency gaps while protecting ad breaks and SFX transitions.
    content = stems["host"] + stems["caller"] + stems["sfx"] + stems["ads"] + stems["idents"]
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

    # Only cut gaps between threshold-8s — targets TTS latency, not long breaks
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
    """HPF to cut rumble below 80Hz (plosives, HVAC, handling noise)."""
    in_path = tmp_dir / "host_pre_denoise.wav"
    out_path = tmp_dir / "host_post_denoise.wav"
    sf.write(str(in_path), audio, sr)

    af = "highpass=f=80:poles=2"
    cmd = ["ffmpeg", "-y", "-i", str(in_path), "-af", af, str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: denoise failed: {result.stderr[:200]}")
        return audio

    denoised, _ = sf.read(str(out_path), dtype="float32")
    return denoised


def deess(audio: np.ndarray, sr: int, tmp_dir: Path) -> np.ndarray:
    """Reduce sibilance (harsh s/sh/ch sounds) in voice audio."""
    in_path = tmp_dir / "host_pre_deess.wav"
    out_path = tmp_dir / "host_post_deess.wav"
    sf.write(str(in_path), audio, sr)

    # Gentle high-shelf reduction at 5kHz (-4dB) to tame sibilance
    # Single-pass, no phase issues unlike split-band approaches
    af = "equalizer=f=5500:t=h:w=2000:g=-4"
    cmd = ["ffmpeg", "-y", "-i", str(in_path), "-af", af, str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: de-essing failed: {result.stderr[:200]}")
        return audio

    deessed, _ = sf.read(str(out_path), dtype="float32")
    return deessed


def reduce_breaths(audio: np.ndarray, sr: int, reduction_db: float = -12) -> np.ndarray:
    """Reduce loud breaths between speech phrases."""
    window_ms = 30
    window_samples = int(sr * window_ms / 1000)
    rms = compute_rms(audio, window_samples)

    if not np.any(rms > 0):
        return audio

    # Speech threshold: breaths are quieter than speech but louder than silence
    nonzero = rms[rms > 0]
    speech_level = np.percentile(nonzero, 70)
    silence_level = np.percentile(nonzero, 10)
    breath_upper = speech_level * 0.3  # below 30% of speech level
    breath_lower = silence_level * 2   # above 2x silence

    if breath_upper <= breath_lower:
        return audio

    # Detect breath-length bursts (0.15-0.8s) in the breath amplitude range
    min_windows = int(150 / window_ms)
    max_windows = int(800 / window_ms)

    breath_gain = 10 ** (reduction_db / 20)
    gain_envelope = np.ones(len(rms), dtype=np.float32)

    i = 0
    breath_count = 0
    while i < len(rms):
        if breath_lower < rms[i] < breath_upper:
            start = i
            while i < len(rms) and breath_lower < rms[i] < breath_upper:
                i += 1
            length = i - start
            if min_windows <= length <= max_windows:
                gain_envelope[start:i] = breath_gain
                breath_count += 1
        else:
            i += 1

    if breath_count == 0:
        return audio

    print(f"  Reduced {breath_count} breaths by {reduction_db}dB")

    # Smooth transitions (10ms ramp)
    ramp = max(1, int(10 / window_ms))
    smoothed = gain_envelope.copy()
    for i in range(1, len(smoothed)):
        if smoothed[i] < smoothed[i - 1]:
            smoothed[i] = smoothed[i - 1] + (smoothed[i] - smoothed[i - 1]) / ramp
        elif smoothed[i] > smoothed[i - 1]:
            smoothed[i] = smoothed[i - 1] + (smoothed[i] - smoothed[i - 1]) / ramp

    # Expand to sample level
    gain_samples = np.repeat(smoothed, window_samples)[:len(audio)]
    if len(gain_samples) < len(audio):
        gain_samples = np.pad(gain_samples, (0, len(audio) - len(gain_samples)), constant_values=1.0)

    return (audio * gain_samples).astype(np.float32)


def limit_stem(audio: np.ndarray, sr: int, tmp_dir: Path,
               stem_name: str) -> np.ndarray:
    """Hard-limit a stem to -1dB true peak to prevent clipping."""
    peak = np.max(np.abs(audio))
    if peak <= 0.89:  # already below -1dB
        return audio
    in_path = tmp_dir / f"{stem_name}_pre_limit.wav"
    out_path = tmp_dir / f"{stem_name}_post_limit.wav"
    sf.write(str(in_path), audio, sr)
    cmd = [
        "ffmpeg", "-y", "-i", str(in_path),
        "-af", "alimiter=limit=-1dB:level=false:attack=0.1:release=50",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: limiting failed for {stem_name}: {result.stderr[:200]}")
        return audio
    limited, _ = sf.read(str(out_path), dtype="float32")
    peak_db = 20 * np.log10(peak)
    print(f"  {stem_name}: peak was {peak_db:+.1f}dB, limited to -1dB")
    return limited


def compress_voice(audio: np.ndarray, sr: int, tmp_dir: Path,
                   stem_name: str) -> np.ndarray:
    in_path = tmp_dir / f"{stem_name}_pre_comp.wav"
    out_path = tmp_dir / f"{stem_name}_post_comp.wav"

    sf.write(str(in_path), audio, sr)

    if stem_name == "host":
        # Spoken word compression: lower threshold, higher ratio, more makeup
        af = "acompressor=threshold=-28dB:ratio=4:attack=5:release=600:makeup=8dB"
    else:
        af = "acompressor=threshold=-24dB:ratio=2.5:attack=10:release=800:makeup=6dB"

    cmd = [
        "ffmpeg", "-y", "-i", str(in_path),
        "-af", af,
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
    for name in ["host", "caller", "ads", "idents"]:
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
              levels: dict[str, float] | None = None,
              stereo_imaging: bool = True) -> np.ndarray:
    if levels is None:
        levels = {"host": 0, "caller": 0, "music": -6, "sfx": -10, "ads": 0, "idents": 0}

    gains = {name: 10 ** (db / 20) for name, db in levels.items()}

    max_len = max(len(s) for s in stems.values())

    if stereo_imaging:
        # Pan positions: -1.0 = full left, 0.0 = center, 1.0 = full right
        # Using constant-power panning law
        pans = {"host": 0.0, "caller": 0.15, "music": 0.0, "sfx": 0.0, "ads": 0.0, "idents": 0.0}
        # Music gets stereo width via slight L/R decorrelation
        music_width = 0.3

        left = np.zeros(max_len, dtype=np.float64)
        right = np.zeros(max_len, dtype=np.float64)

        for name in STEM_NAMES:
            audio = stems[name]
            if len(audio) < max_len:
                audio = np.pad(audio, (0, max_len - len(audio)))
            signal = audio.astype(np.float64) * gains.get(name, 1.0)

            if name == "music" and music_width > 0:
                # Widen music: delay right channel by ~0.5ms for Haas effect
                delay_samples = int(0.0005 * sr)  # ~22 samples at target sample rate
                left += signal * (1 + music_width * 0.5)
                right_delayed = np.zeros_like(signal)
                right_delayed[delay_samples:] = signal[:-delay_samples] if delay_samples > 0 else signal
                right += right_delayed * (1 + music_width * 0.5)
            else:
                pan = pans.get(name, 0.0)
                # Constant-power pan: L = cos(angle), R = sin(angle)
                angle = (pan + 1) * np.pi / 4  # 0 to pi/2
                l_gain = np.cos(angle)
                r_gain = np.sin(angle)
                left += signal * l_gain
                right += signal * r_gain

        left = np.clip(left, -1.0, 1.0).astype(np.float32)
        right = np.clip(right, -1.0, 1.0).astype(np.float32)
        stereo = np.column_stack([left, right])
    else:
        mix = np.zeros(max_len, dtype=np.float64)
        for name in STEM_NAMES:
            audio = stems[name]
            if len(audio) < max_len:
                audio = np.pad(audio, (0, max_len - len(audio)))
            mix += audio.astype(np.float64) * gains.get(name, 1.0)
        mix = np.clip(mix, -1.0, 1.0).astype(np.float32)
        stereo = np.column_stack([mix, mix])

    return stereo


def bus_compress(audio: np.ndarray, sr: int, tmp_dir: Path) -> np.ndarray:
    """Gentle bus compression on the final stereo mix to glue everything together."""
    in_path = tmp_dir / "bus_pre.wav"
    out_path = tmp_dir / "bus_post.wav"
    sf.write(str(in_path), audio, sr)

    # Gentle glue compressor: slow attack lets transients through,
    # low ratio just levels out the overall dynamics
    af = "acompressor=threshold=-20dB:ratio=2:attack=20:release=300:makeup=2dB"
    cmd = ["ffmpeg", "-y", "-i", str(in_path), "-af", af, str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: bus compression failed: {result.stderr[:200]}")
        return audio

    compressed, _ = sf.read(str(out_path), dtype="float32")
    return compressed


def trim_silence(audio: np.ndarray, sr: int, pad_s: float = 0.5,
                 threshold_db: float = -50) -> np.ndarray:
    """Trim leading and trailing silence from stereo audio."""
    threshold = 10 ** (threshold_db / 20)
    # Use the louder channel for detection
    mono = np.max(np.abs(audio), axis=1) if audio.ndim > 1 else np.abs(audio)

    # Smoothed envelope for more reliable detection
    window = int(sr * 0.05)  # 50ms window
    if len(mono) > window:
        kernel = np.ones(window) / window
        envelope = np.convolve(mono, kernel, mode='same')
    else:
        envelope = mono

    above = np.where(envelope > threshold)[0]
    if len(above) == 0:
        return audio

    pad_samples = int(pad_s * sr)
    start = max(0, above[0] - pad_samples)
    end = min(len(audio), above[-1] + pad_samples)

    trimmed_start = start / sr
    trimmed_end = (len(audio) - end) / sr
    if trimmed_start > 0.1 or trimmed_end > 0.1:
        print(f"  Trimmed {trimmed_start:.1f}s from start, {trimmed_end:.1f}s from end")
    else:
        print("  No significant silence to trim")

    return audio[start:end]


def apply_fades(audio: np.ndarray, sr: int,
                fade_in_s: float = 1.5, fade_out_s: float = 3.0) -> np.ndarray:
    """Apply fade in/out to stereo audio using equal-power curve."""
    audio = audio.copy()

    # Fade in
    fade_in_samples = int(fade_in_s * sr)
    if fade_in_samples > 0 and fade_in_samples < len(audio):
        # Equal-power: sine curve for smooth perceived volume change
        curve = np.sin(np.linspace(0, np.pi / 2, fade_in_samples)).astype(np.float32)
        if audio.ndim > 1:
            audio[:fade_in_samples] *= curve[:, np.newaxis]
        else:
            audio[:fade_in_samples] *= curve

    # Fade out
    fade_out_samples = int(fade_out_s * sr)
    if fade_out_samples > 0 and fade_out_samples < len(audio):
        curve = np.sin(np.linspace(np.pi / 2, 0, fade_out_samples)).astype(np.float32)
        if audio.ndim > 1:
            audio[-fade_out_samples:] *= curve[:, np.newaxis]
        else:
            audio[-fade_out_samples:] *= curve

    print(f"  Fade in: {fade_in_s}s, fade out: {fade_out_s}s")
    return audio


def detect_chapters(stems: dict[str, np.ndarray], sr: int) -> list[dict]:
    """Auto-detect chapter boundaries from stem activity."""
    window_s = 2  # 2-second analysis windows
    window_samples = int(sr * window_s)
    n_windows = min(len(s) for s in stems.values()) // window_samples

    if n_windows == 0:
        return []

    chapters = []
    current_type = None
    chapter_start = 0

    for w in range(n_windows):
        start = w * window_samples
        end = start + window_samples

        ads_rms = np.sqrt(np.mean(stems["ads"][start:end] ** 2))
        caller_rms = np.sqrt(np.mean(stems["caller"][start:end] ** 2))
        host_rms = np.sqrt(np.mean(stems["host"][start:end] ** 2))

        # Classify this window
        if ads_rms > 0.005:
            seg_type = "Ad Break"
        elif caller_rms > 0.005:
            seg_type = "Caller"
        elif host_rms > 0.005:
            seg_type = "Host"
        else:
            seg_type = current_type  # keep current during silence

        if seg_type != current_type and seg_type is not None:
            if current_type is not None:
                chapters.append({
                    "title": current_type,
                    "start_ms": int(chapter_start * 1000),
                    "end_ms": int(w * window_s * 1000),
                })
            current_type = seg_type
            chapter_start = w * window_s

    # Final chapter
    if current_type is not None:
        chapters.append({
            "title": current_type,
            "start_ms": int(chapter_start * 1000),
            "end_ms": int(n_windows * window_s * 1000),
        })

    # Merge consecutive chapters of same type
    merged = []
    for ch in chapters:
        if merged and merged[-1]["title"] == ch["title"]:
            merged[-1]["end_ms"] = ch["end_ms"]
        else:
            merged.append(ch)

    # Number duplicate types (Caller 1, Caller 2, etc.)
    type_counts = {}
    for ch in merged:
        base = ch["title"]
        type_counts[base] = type_counts.get(base, 0) + 1
        if type_counts[base] > 1 or base in ("Caller", "Ad Break"):
            ch["title"] = f"{base} {type_counts[base]}"

    # Filter out very short chapters (< 10s)
    merged = [ch for ch in merged if ch["end_ms"] - ch["start_ms"] >= 10000]

    return merged


def write_ffmpeg_chapters(chapters: list[dict], output_path: Path):
    """Write an ffmpeg-format metadata file with chapter markers."""
    lines = [";FFMETADATA1"]
    for ch in chapters:
        lines.append("[CHAPTER]")
        lines.append("TIMEBASE=1/1000")
        lines.append(f"START={ch['start_ms']}")
        lines.append(f"END={ch['end_ms']}")
        lines.append(f"title={ch['title']}")
    output_path.write_text("\n".join(lines) + "\n")


def normalize_and_export(audio: np.ndarray, sr: int, output_path: Path,
                         target_lufs: float = -16, bitrate: str = "128k",
                         tmp_dir: Path = None,
                         metadata: dict | None = None,
                         chapters_file: Path | None = None):
    import json
    import shutil

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

    # Pass 2: normalize + limiter + export MP3
    loudnorm_filter = (
        f"loudnorm=I={target_lufs}:TP=-1:LRA=11"
        f":measured_I={stats['input_i']}"
        f":measured_TP={stats['input_tp']}"
        f":measured_LRA={stats['input_lra']}"
        f":measured_thresh={stats['input_thresh']}"
        f":linear=true"
    )

    export_cmd = ["ffmpeg", "-y", "-i", str(tmp_wav)]

    if chapters_file and chapters_file.exists():
        export_cmd += ["-i", str(chapters_file), "-map_metadata", "1"]

    export_cmd += [
        "-af", f"{loudnorm_filter},alimiter=limit=-1dB:level=false",
        "-ab", bitrate, "-ar", str(sr),
    ]

    if metadata:
        for key, value in metadata.items():
            if value and not key.startswith("_"):
                export_cmd += ["-metadata", f"{key}={value}"]

    export_cmd.append(str(output_path))
    result = subprocess.run(export_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: export failed: {result.stderr[:300]}")
        sys.exit(1)

    # Embed artwork as a second pass (avoids complex multi-input mapping)
    artwork = metadata.get("_artwork") if metadata else None
    if artwork and Path(artwork).exists():
        tmp_mp3 = tmp_dir / "with_art.mp3"
        art_cmd = [
            "ffmpeg", "-y", "-i", str(output_path), "-i", artwork,
            "-map", "0:a", "-map", "1:0",
            "-c:a", "copy", "-id3v2_version", "3",
            "-metadata:s:v", "title=Album cover",
            "-metadata:s:v", "comment=Cover (front)",
            "-disposition:v", "attached_pic",
            str(tmp_mp3),
        ]
        art_result = subprocess.run(art_cmd, capture_output=True, text=True)
        if art_result.returncode == 0:
            shutil.move(str(tmp_mp3), str(output_path))
            print(f"  Embedded artwork: {artwork}")
        else:
            print(f"  WARNING: artwork embedding failed: {art_result.stderr[:200]}")


def main():
    parser = argparse.ArgumentParser(description="Post-production for AI podcast stems")
    parser.add_argument("stems_dir", type=Path, help="Directory containing stem WAV files")
    parser.add_argument("-o", "--output", type=str, default="episode.mp3", help="Output filename")
    parser.add_argument("--gap-threshold", type=float, default=2.0, help="Min silence to cut (seconds)")
    parser.add_argument("--duck-amount", type=float, default=-20, help="Music duck in dB")
    parser.add_argument("--target-lufs", type=float, default=-16, help="Target loudness (LUFS)")
    parser.add_argument("--bitrate", type=str, default="128k", help="MP3 bitrate")
    parser.add_argument("--fade-in", type=float, default=1.5, help="Fade in duration (seconds)")
    parser.add_argument("--fade-out", type=float, default=3.0, help="Fade out duration (seconds)")

    # Metadata
    parser.add_argument("--title", type=str, help="Episode title (ID3 tag)")
    parser.add_argument("--artist", type=str, default="Luke at the Roost", help="Artist name")
    parser.add_argument("--album", type=str, default="Luke at the Roost", help="Album/show name")
    parser.add_argument("--episode-num", type=str, help="Episode number (track tag)")
    parser.add_argument("--artwork", type=str, help="Path to artwork image (embedded in MP3)")

    # Skip flags
    parser.add_argument("--no-gap-removal", action="store_true", help="Skip gap removal")
    parser.add_argument("--no-denoise", action="store_true", help="Skip noise reduction + HPF")
    parser.add_argument("--no-deess", action="store_true", help="Skip de-essing")
    parser.add_argument("--no-breath-reduction", action="store_true", help="Skip breath reduction")
    parser.add_argument("--no-compression", action="store_true", help="Skip voice compression")
    parser.add_argument("--no-phone-eq", action="store_true", help="Skip caller phone EQ")
    parser.add_argument("--no-ducking", action="store_true", help="Skip music ducking")
    parser.add_argument("--no-stereo", action="store_true", help="Skip stereo imaging (mono mix)")
    parser.add_argument("--no-trim", action="store_true", help="Skip silence trimming")
    parser.add_argument("--no-fade", action="store_true", help="Skip fade in/out")
    parser.add_argument("--no-chapters", action="store_true", help="Skip chapter markers")
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

    if args.dry_run:
        print("Dry run — exiting")
        return

    total_steps = 15

    # Step 1: Load
    print(f"\n[1/{total_steps}] Loading stems...")
    stems, sr = load_stems(stems_dir)

    # Step 2: Gap removal
    print(f"\n[2/{total_steps}] Gap removal...")
    if not args.no_gap_removal:
        stems = remove_gaps(stems, sr, threshold_s=args.gap_threshold)
    else:
        print("  Skipped")

    # Step 3: Limit ads + SFX (prevent clipping)
    print(f"\n[3/{total_steps}] Limiting ads + SFX...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for name in ["ads", "sfx", "idents"]:
            if np.any(stems[name] != 0):
                stems[name] = limit_stem(stems[name], sr, tmp_dir, name)

    # Step 4: Host mic noise reduction + HPF
    print(f"\n[4/{total_steps}] Host noise reduction + HPF...")
    if not args.no_denoise and np.any(stems["host"] != 0):
        with tempfile.TemporaryDirectory() as tmp:
            stems["host"] = denoise(stems["host"], sr, Path(tmp))
            print("  Applied")
    else:
        print("  Skipped" if args.no_denoise else "  No host audio")

    # Step 5: De-essing
    print(f"\n[5/{total_steps}] De-essing host...")
    if not args.no_deess and np.any(stems["host"] != 0):
        with tempfile.TemporaryDirectory() as tmp:
            stems["host"] = deess(stems["host"], sr, Path(tmp))
            print("  Applied")
    else:
        print("  Skipped" if args.no_deess else "  No host audio")

    # Step 6: Breath reduction
    print(f"\n[6/{total_steps}] Breath reduction...")
    if not args.no_breath_reduction and np.any(stems["host"] != 0):
        stems["host"] = reduce_breaths(stems["host"], sr)
    else:
        print("  Skipped" if args.no_breath_reduction else "  No host audio")

    # Step 7: Voice compression
    print(f"\n[7/{total_steps}] Voice compression...")
    if not args.no_compression:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for name in ["host", "caller"]:
                if np.any(stems[name] != 0):
                    print(f"  Compressing {name}...")
                    stems[name] = compress_voice(stems[name], sr, tmp_dir, name)
    else:
        print("  Skipped")

    # Step 8: Phone EQ on caller
    print(f"\n[8/{total_steps}] Phone EQ on caller...")
    if not args.no_phone_eq and np.any(stems["caller"] != 0):
        with tempfile.TemporaryDirectory() as tmp:
            stems["caller"] = phone_eq(stems["caller"], sr, Path(tmp))
            print("  Applied")
    else:
        print("  Skipped" if args.no_phone_eq else "  No caller audio")

    # Step 9: Match voice levels
    print(f"\n[9/{total_steps}] Matching voice levels...")
    stems = match_voice_levels(stems)

    # Step 10: Music ducking
    print(f"\n[10/{total_steps}] Music ducking...")
    if not args.no_ducking:
        dialog = stems["host"] + stems["caller"]
        if np.any(dialog != 0) and np.any(stems["music"] != 0):
            stems["music"] = apply_ducking(stems["music"], dialog, sr, duck_db=args.duck_amount,
                                           mute_signal=stems["ads"] + stems["idents"])
            print("  Applied")
        else:
            print("  No dialog or music to duck")
    else:
        print("  Skipped")

    # Step 11: Stereo mix
    print(f"\n[11/{total_steps}] Mixing...")
    stereo = mix_stems(stems, stereo_imaging=not args.no_stereo)
    imaging = "stereo" if not args.no_stereo else "mono"
    print(f"  Mixed to {imaging}: {len(stereo)} samples ({len(stereo)/sr:.1f}s)")

    # Step 12: Bus compression
    print(f"\n[12/{total_steps}] Bus compression...")
    with tempfile.TemporaryDirectory() as tmp:
        stereo = bus_compress(stereo, sr, Path(tmp))
        print("  Applied")

    # Step 13: Silence trimming
    print(f"\n[13/{total_steps}] Trimming silence...")
    if not args.no_trim:
        stereo = trim_silence(stereo, sr)
    else:
        print("  Skipped")

    # Step 14: Fade in/out
    print(f"\n[14/{total_steps}] Fades...")
    if not args.no_fade:
        stereo = apply_fades(stereo, sr, fade_in_s=args.fade_in, fade_out_s=args.fade_out)
    else:
        print("  Skipped")

    # Step 15: Normalize + export with metadata and chapters
    print(f"\n[15/{total_steps}] Loudness normalization + export...")

    # Build metadata dict
    meta = {}
    if args.title:
        meta["title"] = args.title
    if args.artist:
        meta["artist"] = args.artist
    if args.album:
        meta["album"] = args.album
    if args.episode_num:
        meta["track"] = args.episode_num
    if args.artwork:
        meta["_artwork"] = args.artwork

    # Auto-detect chapters
    chapters = []
    if not args.no_chapters:
        chapters = detect_chapters(stems, sr)
        if chapters:
            print(f"  Detected {len(chapters)} chapters:")
            for ch in chapters:
                start_s = ch["start_ms"] / 1000
                end_s = ch["end_ms"] / 1000
                print(f"    {start_s:6.1f}s - {end_s:6.1f}s  {ch['title']}")
        else:
            print("  No chapters detected")
    else:
        print("  Skipped")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        chapters_file = None
        if chapters:
            chapters_file = tmp_dir / "chapters.txt"
            write_ffmpeg_chapters(chapters, chapters_file)

        normalize_and_export(stereo, sr, output_path,
                             target_lufs=args.target_lufs,
                             bitrate=args.bitrate,
                             tmp_dir=tmp_dir,
                             metadata=meta if meta else None,
                             chapters_file=chapters_file)

    print(f"\nDone! Output: {output_path}")


if __name__ == "__main__":
    main()
