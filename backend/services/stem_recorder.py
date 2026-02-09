"""Records separate audio stems during a live show for post-production"""

import time
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy import signal as scipy_signal

STEM_NAMES = ["host", "caller", "music", "sfx", "ads"]


class StemRecorder:
    def __init__(self, output_dir: str | Path, sample_rate: int = 48000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self._files: dict[str, sf.SoundFile] = {}
        self._write_positions: dict[str, int] = {}
        self._start_time: float = 0.0
        self._running = False

    def start(self):
        self._start_time = time.time()
        self._running = True
        for name in STEM_NAMES:
            path = self.output_dir / f"{name}.wav"
            f = sf.SoundFile(
                str(path), mode="w",
                samplerate=self.sample_rate,
                channels=1, subtype="FLOAT",
            )
            self._files[name] = f
            self._write_positions[name] = 0
        print(f"[StemRecorder] Recording started -> {self.output_dir}")

    def write(self, stem_name: str, audio_data: np.ndarray, source_sr: int):
        if not self._running or stem_name not in self._files:
            return

        # Resample to target rate if needed
        if source_sr != self.sample_rate:
            num_samples = int(len(audio_data) * self.sample_rate / source_sr)
            if num_samples > 0:
                audio_data = scipy_signal.resample(audio_data, num_samples).astype(np.float32)
            else:
                return

        # Fill silence gap based on elapsed time
        elapsed = time.time() - self._start_time
        expected_pos = int(elapsed * self.sample_rate)
        current_pos = self._write_positions[stem_name]

        if expected_pos > current_pos:
            gap = expected_pos - current_pos
            silence = np.zeros(gap, dtype=np.float32)
            self._files[stem_name].write(silence)
            self._write_positions[stem_name] = expected_pos

        self._files[stem_name].write(audio_data.astype(np.float32))
        self._write_positions[stem_name] += len(audio_data)

    def stop(self) -> dict[str, str]:
        if not self._running:
            return {}

        self._running = False

        # Pad all stems to the same length
        max_pos = max(self._write_positions.values()) if self._write_positions else 0
        for name in STEM_NAMES:
            pos = self._write_positions[name]
            if pos < max_pos:
                silence = np.zeros(max_pos - pos, dtype=np.float32)
                self._files[name].write(silence)

        # Close all files
        paths = {}
        for name in STEM_NAMES:
            self._files[name].close()
            paths[name] = str(self.output_dir / f"{name}.wav")

        self._files.clear()
        self._write_positions.clear()

        print(f"[StemRecorder] Recording stopped. {max_pos} samples ({max_pos/self.sample_rate:.1f}s)")
        return paths
