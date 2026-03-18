"""Records separate audio stems during a live show for post-production"""

import time
import threading
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import deque

STEM_NAMES = ["host", "caller", "devon", "music", "sfx", "ads", "idents"]


class StemRecorder:
    def __init__(self, output_dir: str | Path, sample_rate: int = 48000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self._running = False
        self._queues: dict[str, deque] = {}
        self._writer_thread: threading.Thread | None = None
        self._start_time: float = 0.0
        self._write_errors: int = 0

    def start(self):
        self._start_time = time.time()
        self._running = True
        self._write_errors = 0
        for name in STEM_NAMES:
            self._queues[name] = deque()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=False)
        self._writer_thread.start()
        print(f"[StemRecorder] Recording started -> {self.output_dir}")

    def write(self, stem_name: str, audio_data: np.ndarray, source_sr: int):
        """Non-blocking write for continuous streams (host mic, music, ads).
        Safe to call from audio callbacks."""
        if not self._running or stem_name not in self._queues:
            return
        self._queues[stem_name].append(("audio", audio_data.copy(), source_sr))

    def write_sporadic(self, stem_name: str, audio_data: np.ndarray, source_sr: int):
        """Write for burst sources (caller TTS, SFX). Pads silence to current time."""
        if not self._running or stem_name not in self._queues:
            return
        self._queues[stem_name].append(("sporadic", audio_data.copy(), source_sr))

    def _resample(self, audio_data: np.ndarray, source_sr: int) -> np.ndarray:
        if source_sr == self.sample_rate:
            return audio_data.astype(np.float32)
        ratio = self.sample_rate / source_sr
        num_samples = int(len(audio_data) * ratio)
        if num_samples <= 0:
            return np.array([], dtype=np.float32)
        indices = (np.arange(num_samples) / ratio).astype(int)
        indices = np.clip(indices, 0, len(audio_data) - 1)
        return audio_data[indices].astype(np.float32)

    def _writer_loop(self):
        """Background thread that drains queues and writes to WAV files."""
        files: dict[str, sf.SoundFile] = {}
        positions: dict[str, int] = {}

        for name in STEM_NAMES:
            path = self.output_dir / f"{name}.wav"
            files[name] = sf.SoundFile(
                str(path), mode="w",
                samplerate=self.sample_rate,
                channels=1, subtype="FLOAT",
            )
            positions[name] = 0

        try:
            while self._running or any(len(q) > 0 for q in self._queues.values()):
                did_work = False
                for name in STEM_NAMES:
                    q = self._queues[name]
                    while q:
                        did_work = True
                        msg_type, audio_data, source_sr = q.popleft()
                        resampled = self._resample(audio_data, source_sr)
                        if len(resampled) == 0:
                            continue

                        try:
                            if msg_type == "sporadic":
                                elapsed = time.time() - self._start_time
                                expected_pos = int(elapsed * self.sample_rate)
                                if expected_pos > positions[name]:
                                    gap = expected_pos - positions[name]
                                    files[name].write(np.zeros(gap, dtype=np.float32))
                                    positions[name] = expected_pos

                            files[name].write(resampled)
                            positions[name] += len(resampled)
                        except Exception as e:
                            self._write_errors += 1
                            if self._write_errors <= 5:
                                print(f"[StemRecorder] Write error on {name}: {e}")
                            elif self._write_errors == 6:
                                print(f"[StemRecorder] Suppressing further write errors")

                if not did_work:
                    time.sleep(0.02)

            # Pad all stems to same length
            max_pos = max(positions.values()) if positions else 0
            for name in STEM_NAMES:
                try:
                    if positions[name] < max_pos:
                        files[name].write(np.zeros(max_pos - positions[name], dtype=np.float32))
                except Exception as e:
                    print(f"[StemRecorder] Final pad error on {name}: {e}")
        finally:
            for name, f in files.items():
                try:
                    f.close()
                except Exception as e:
                    print(f"[StemRecorder] Error closing {name}.wav: {e}")

        total_errors = self._write_errors
        err_msg = f", {total_errors} write errors" if total_errors else ""
        print(f"[StemRecorder] Writer done. {max_pos} samples ({max_pos / self.sample_rate:.1f}s{err_msg})")

    def stop(self) -> dict[str, str]:
        if not self._running:
            return {}

        self._running = False
        if self._writer_thread:
            self._writer_thread.join(timeout=30.0)
            if self._writer_thread.is_alive():
                print("[StemRecorder] Warning: writer thread still running after 30s")
            self._writer_thread = None

        paths = {}
        for name in STEM_NAMES:
            paths[name] = str(self.output_dir / f"{name}.wav")

        self._queues.clear()
        return paths
