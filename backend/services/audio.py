"""Server-side audio service for Loopback routing"""

import sounddevice as sd
import numpy as np
import threading
import queue
import json
from pathlib import Path
from typing import Optional, Callable
import wave
import time

# Settings file path
SETTINGS_FILE = Path(__file__).parent.parent.parent / "audio_settings.json"


class AudioService:
    """Manages audio I/O with multi-channel support for Loopback routing"""

    def __init__(self):
        # Device configuration
        self.input_device: Optional[int] = None
        self.input_channel: int = 1  # 1-indexed channel

        self.output_device: Optional[int] = None  # Single output device (multi-channel)
        self.caller_channel: int = 1   # Channel for caller TTS
        self.live_caller_channel: int = 9  # Channel for live caller audio
        self.music_channel: int = 2    # Channel for music
        self.sfx_channel: int = 3      # Channel for SFX
        self.ad_channel: int = 11      # Channel for ads
        self.phone_filter: bool = False  # Phone filter on caller voices

        # Ad playback state
        self._ad_stream: Optional[sd.OutputStream] = None
        self._ad_data: Optional[np.ndarray] = None
        self._ad_resampled: Optional[np.ndarray] = None
        self._ad_position: int = 0
        self._ad_playing: bool = False

        # Recording state
        self._recording = False
        self._record_thread: Optional[threading.Thread] = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._recorded_audio: list = []
        self._record_device_sr: int = 48000

        # Music playback state
        self._music_stream: Optional[sd.OutputStream] = None
        self._music_data: Optional[np.ndarray] = None
        self._music_resampled: Optional[np.ndarray] = None
        self._music_position: int = 0
        self._music_playing: bool = False
        self._music_volume: float = 0.3
        self._music_loop: bool = True

        # Caller playback state
        self._caller_stop_event = threading.Event()
        self._caller_thread: Optional[threading.Thread] = None

        # Host mic streaming state
        self._host_stream: Optional[sd.InputStream] = None
        self._host_send_callback: Optional[Callable] = None
        self._host_device_sr: int = 48000

        # Live caller routing state
        self._live_caller_stream: Optional[sd.OutputStream] = None
        self._live_caller_write: Optional[Callable] = None

        # Sample rates
        self.input_sample_rate = 16000  # For Whisper
        self.output_sample_rate = 24000  # For TTS

        # Load saved settings
        self._load_settings()

    def _load_settings(self):
        """Load settings from disk"""
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE) as f:
                    data = json.load(f)
                self.input_device = data.get("input_device")
                self.input_channel = data.get("input_channel", 1)
                self.output_device = data.get("output_device")
                self.caller_channel = data.get("caller_channel", 1)
                self.live_caller_channel = data.get("live_caller_channel", 4)
                self.music_channel = data.get("music_channel", 2)
                self.sfx_channel = data.get("sfx_channel", 3)
                self.ad_channel = data.get("ad_channel", 11)
                self.phone_filter = data.get("phone_filter", False)
                print(f"Loaded audio settings: output={self.output_device}, channels={self.caller_channel}/{self.live_caller_channel}/{self.music_channel}/{self.sfx_channel}/ad:{self.ad_channel}, phone_filter={self.phone_filter}")
            except Exception as e:
                print(f"Failed to load audio settings: {e}")

    def _save_settings(self):
        """Save settings to disk"""
        try:
            data = {
                "input_device": self.input_device,
                "input_channel": self.input_channel,
                "output_device": self.output_device,
                "caller_channel": self.caller_channel,
                "live_caller_channel": self.live_caller_channel,
                "music_channel": self.music_channel,
                "sfx_channel": self.sfx_channel,
                "ad_channel": self.ad_channel,
                "phone_filter": self.phone_filter,
            }
            with open(SETTINGS_FILE, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved audio settings")
        except Exception as e:
            print(f"Failed to save audio settings: {e}")

    def list_devices(self) -> list[dict]:
        """List all available audio devices"""
        devices = sd.query_devices()
        result = []
        for i, d in enumerate(devices):
            result.append({
                "id": i,
                "name": d["name"],
                "inputs": d["max_input_channels"],
                "outputs": d["max_output_channels"],
                "default_sr": d["default_samplerate"]
            })
        return result

    def set_devices(
        self,
        input_device: Optional[int] = None,
        input_channel: Optional[int] = None,
        output_device: Optional[int] = None,
        caller_channel: Optional[int] = None,
        live_caller_channel: Optional[int] = None,
        music_channel: Optional[int] = None,
        sfx_channel: Optional[int] = None,
        phone_filter: Optional[bool] = None
    ):
        """Configure audio devices and channels"""
        if input_device is not None:
            self.input_device = input_device
        if input_channel is not None:
            self.input_channel = input_channel
        if output_device is not None:
            self.output_device = output_device
        if caller_channel is not None:
            self.caller_channel = caller_channel
        if live_caller_channel is not None:
            self.live_caller_channel = live_caller_channel
        if music_channel is not None:
            self.music_channel = music_channel
        if sfx_channel is not None:
            self.sfx_channel = sfx_channel
        if phone_filter is not None:
            self.phone_filter = phone_filter

        # Persist to disk
        self._save_settings()

    def get_device_settings(self) -> dict:
        """Get current device configuration"""
        return {
            "input_device": self.input_device,
            "input_channel": self.input_channel,
            "output_device": self.output_device,
            "caller_channel": self.caller_channel,
            "live_caller_channel": self.live_caller_channel,
            "music_channel": self.music_channel,
            "sfx_channel": self.sfx_channel,
            "phone_filter": self.phone_filter,
        }

    # --- Recording ---

    def start_recording(self) -> bool:
        """Start recording from input device"""
        if self._recording:
            return False

        if self.input_device is None:
            print("No input device configured")
            return False

        self._recording = True
        self._recorded_audio = []

        if self._host_stream is not None:
            # Host stream already capturing — piggyback on it
            self._record_device_sr = self._host_device_sr
            print(f"Recording started (piggybacking on host stream @ {self._host_device_sr}Hz)")
            return True

        self._record_thread = threading.Thread(target=self._record_worker)
        self._record_thread.start()
        print(f"Recording started from device {self.input_device}")
        return True

    def stop_recording(self) -> bytes:
        """Stop recording and return audio data resampled to 16kHz for Whisper"""
        import librosa

        if not self._recording:
            return b""

        self._recording = False
        if self._record_thread:
            self._record_thread.join(timeout=2.0)
            self._record_thread = None
        else:
            # Piggybacking on host stream — give callback a moment to finish
            time.sleep(0.05)

        if not self._recorded_audio:
            piggyback = self._host_stream is not None
            # Check what other streams might be active
            active_streams = []
            if self._music_stream:
                active_streams.append("music")
            if self._live_caller_stream:
                active_streams.append("live_caller")
            if self._host_stream:
                active_streams.append("host")
            streams_info = f", active_streams=[{','.join(active_streams)}]" if active_streams else ""
            print(f"Recording stopped: NO audio chunks captured (piggyback={piggyback}, device={self.input_device}, ch={self.input_channel}{streams_info})")
            return b""

        # Combine all chunks
        audio = np.concatenate(self._recorded_audio)
        device_sr = getattr(self, '_record_device_sr', 48000)
        print(f"Recording stopped: {len(audio)} samples @ {device_sr}Hz ({len(audio)/device_sr:.2f}s), chunks={len(self._recorded_audio)}, peak={np.abs(audio).max():.4f}")

        # Resample to 16kHz for Whisper
        if device_sr != 16000:
            audio = librosa.resample(audio, orig_sr=device_sr, target_sr=16000)
            print(f"Resampled to 16kHz: {len(audio)} samples")

        # Convert to bytes (16-bit PCM)
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def _record_worker(self):
        """Background thread for recording from specific channel"""
        try:
            # Get device info
            device_info = sd.query_devices(self.input_device)
            max_channels = device_info['max_input_channels']
            device_sr = int(device_info['default_samplerate'])
            record_channel = min(self.input_channel, max_channels) - 1

            if max_channels == 0:
                print(f"Recording error: device {self.input_device} has no input channels")
                self._recording = False
                return

            # Store device sample rate for later resampling
            self._record_device_sr = device_sr

            stream_ready = threading.Event()
            callback_count = [0]

            def callback(indata, frames, time_info, status):
                if status:
                    print(f"Record status: {status}")
                callback_count[0] += 1
                if not stream_ready.is_set():
                    stream_ready.set()
                if self._recording:
                    self._recorded_audio.append(indata[:, record_channel].copy())

            print(f"Recording: opening stream on device {self.input_device} ch {self.input_channel} @ {device_sr}Hz ({max_channels} ch)")

            with sd.InputStream(
                device=self.input_device,
                channels=max_channels,
                samplerate=device_sr,  # Use device's native rate
                dtype=np.float32,
                callback=callback,
                blocksize=1024
            ):
                # Wait for stream to actually start capturing
                if not stream_ready.wait(timeout=1.0):
                    print(f"Recording WARNING: stream opened but callback not firing after 1s")

                while self._recording:
                    time.sleep(0.05)

            print(f"Recording: stream closed, {callback_count[0]} callbacks fired, {len(self._recorded_audio)} chunks captured")

        except Exception as e:
            print(f"Recording error: {e}")
            import traceback
            traceback.print_exc()
            self._recording = False

    # --- Caller TTS Playback ---

    def _apply_fade(self, audio: np.ndarray, sample_rate: int, fade_ms: int = 15) -> np.ndarray:
        """Apply fade-in and fade-out to avoid clicks"""
        fade_samples = int(sample_rate * fade_ms / 1000)
        if len(audio) < fade_samples * 2:
            return audio

        # Fade in
        fade_in = np.linspace(0, 1, fade_samples)
        audio[:fade_samples] *= fade_in

        # Fade out
        fade_out = np.linspace(1, 0, fade_samples)
        audio[-fade_samples:] *= fade_out

        return audio

    def play_caller_audio(self, audio_bytes: bytes, sample_rate: int = 24000):
        """Play caller TTS audio to specific channel of output device (interruptible)"""
        import librosa

        # Stop any existing caller audio
        self.stop_caller_audio()
        self._caller_stop_event.clear()

        # Convert bytes to numpy
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if self.output_device is None:
            print("No output device configured, using default")
            audio = self._apply_fade(audio, sample_rate)
            with sd.OutputStream(samplerate=sample_rate, channels=1, dtype=np.float32) as stream:
                stream.write(audio.reshape(-1, 1))
            return

        try:
            # Get device info and resample to device's native rate
            device_info = sd.query_devices(self.output_device)
            num_channels = device_info['max_output_channels']
            device_sr = int(device_info['default_samplerate'])
            channel_idx = min(self.caller_channel, num_channels) - 1

            # Resample if needed
            if sample_rate != device_sr:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=device_sr)

            # Apply fade to prevent clicks
            audio = self._apply_fade(audio, device_sr)

            # Create multi-channel output with audio only on target channel
            multi_ch = np.zeros((len(audio), num_channels), dtype=np.float32)
            multi_ch[:, channel_idx] = audio

            print(f"Playing caller audio to device {self.output_device} ch {self.caller_channel} @ {device_sr}Hz")

            # Play in chunks so we can interrupt
            chunk_size = int(device_sr * 0.1)  # 100ms chunks
            pos = 0

            with sd.OutputStream(
                device=self.output_device,
                samplerate=device_sr,
                channels=num_channels,
                dtype=np.float32
            ) as stream:
                while pos < len(multi_ch) and not self._caller_stop_event.is_set():
                    end = min(pos + chunk_size, len(multi_ch))
                    stream.write(multi_ch[pos:end])
                    pos = end

            if self._caller_stop_event.is_set():
                print("Caller audio stopped early")
            else:
                print(f"Played caller audio: {len(audio)/device_sr:.2f}s")

        except Exception as e:
            print(f"Caller playback error: {e}")

    def stop_caller_audio(self):
        """Stop any playing caller audio"""
        self._caller_stop_event.set()

    def _start_live_caller_stream(self):
        """Start persistent output stream with ring buffer jitter absorption"""
        if self._live_caller_stream is not None:
            return

        if self.output_device is None:
            return

        device_info = sd.query_devices(self.output_device)
        num_channels = device_info['max_output_channels']
        device_sr = int(device_info['default_samplerate'])
        channel_idx = min(self.live_caller_channel, num_channels) - 1

        self._live_caller_device_sr = device_sr
        self._live_caller_num_channels = num_channels
        self._live_caller_channel_idx = channel_idx

        # Ring buffer: 3 seconds capacity, 80ms pre-buffer before playback starts
        ring_size = int(device_sr * 3)
        ring = np.zeros(ring_size, dtype=np.float32)
        prebuffer_samples = int(device_sr * 0.08)
        # Mutable state shared between writer (main thread) and reader (audio callback)
        # CPython GIL makes individual int reads/writes atomic
        state = {"write_pos": 0, "read_pos": 0, "avail": 0, "started": False}

        def write_audio(data):
            n = len(data)
            wp = state["write_pos"]
            if wp + n <= ring_size:
                ring[wp:wp + n] = data
            else:
                first = ring_size - wp
                ring[wp:] = data[:first]
                ring[:n - first] = data[first:]
            state["write_pos"] = (wp + n) % ring_size
            state["avail"] += n

        def callback(outdata, frames, time_info, status):
            outdata.fill(0)
            avail = state["avail"]

            if not state["started"]:
                if avail >= prebuffer_samples:
                    state["started"] = True
                else:
                    return

            if avail < frames:
                # Underrun — stop and re-buffer
                state["started"] = False
                return

            rp = state["read_pos"]
            if rp + frames <= ring_size:
                outdata[:frames, channel_idx] = ring[rp:rp + frames]
            else:
                first = ring_size - rp
                outdata[:first, channel_idx] = ring[rp:]
                outdata[first:frames, channel_idx] = ring[:frames - first]
            state["read_pos"] = (rp + frames) % ring_size
            state["avail"] -= frames

        self._live_caller_write = write_audio

        self._live_caller_stream = sd.OutputStream(
            device=self.output_device,
            samplerate=device_sr,
            channels=num_channels,
            dtype=np.float32,
            callback=callback,
            blocksize=1024,
        )
        self._live_caller_stream.start()
        print(f"[Audio] Live caller stream started on ch {self.live_caller_channel} @ {device_sr}Hz (prebuffer {prebuffer_samples} samples)")

    def _stop_live_caller_stream(self):
        """Stop persistent live caller output stream"""
        if self._live_caller_stream:
            self._live_caller_stream.stop()
            self._live_caller_stream.close()
            self._live_caller_stream = None
            self._live_caller_write = None
            print("[Audio] Live caller stream stopped")

    def route_real_caller_audio(self, pcm_data: bytes, sample_rate: int):
        """Route real caller PCM audio to the configured live caller Loopback channel"""
        if self.output_device is None:
            return

        if self._live_caller_stream is None:
            self._start_live_caller_stream()

        try:
            audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

            device_sr = self._live_caller_device_sr
            if sample_rate != device_sr:
                ratio = device_sr / sample_rate
                out_len = int(len(audio) * ratio)
                indices = (np.arange(out_len) / ratio).astype(int)
                indices = np.clip(indices, 0, len(audio) - 1)
                audio = audio[indices]

            if self._live_caller_write:
                self._live_caller_write(audio)

        except Exception as e:
            print(f"Real caller audio routing error: {e}")

    # --- Host Mic Streaming ---

    def start_host_stream(self, send_callback: Callable):
        """Start continuous host mic capture for streaming to real callers"""
        if self.input_device is None:
            print("[Audio] No input device configured for host streaming")
            return

        self._host_send_callback = send_callback

        def _start():
            device_info = sd.query_devices(self.input_device)
            max_channels = device_info['max_input_channels']
            device_sr = int(device_info['default_samplerate'])
            record_channel = min(self.input_channel, max_channels) - 1
            step = max(1, int(device_sr / 16000))

            # Buffer host mic to send ~100ms chunks (reduces WebSocket frame rate)
            host_accum = []
            host_accum_samples = [0]
            send_threshold = 1600  # 100ms at 16kHz

            def callback(indata, frames, time_info, status):
                # Capture for push-to-talk recording if active
                if self._recording and self._recorded_audio is not None:
                    self._recorded_audio.append(indata[:, record_channel].copy())

                if not self._host_send_callback:
                    return
                mono = indata[:, record_channel]
                # Downsample to ~16kHz with averaging (anti-aliased)
                if step > 1:
                    n = len(mono) // step * step
                    mono = mono[:n].reshape(-1, step).mean(axis=1)

                host_accum.append(mono.copy())
                host_accum_samples[0] += len(mono)

                if host_accum_samples[0] >= send_threshold:
                    combined = np.concatenate(host_accum)
                    pcm = (combined * 32767).astype(np.int16).tobytes()
                    host_accum.clear()
                    host_accum_samples[0] = 0
                    self._host_send_callback(pcm)

            self._host_device_sr = device_sr
            self._host_stream = sd.InputStream(
                device=self.input_device,
                channels=max_channels,
                samplerate=device_sr,
                dtype=np.float32,
                blocksize=1024,
                callback=callback,
            )
            self._host_stream.start()
            print(f"[Audio] Host mic streaming started (device {self.input_device} ch {self.input_channel} @ {device_sr}Hz)")

        threading.Thread(target=_start, daemon=True).start()

    def stop_host_stream(self):
        """Stop host mic streaming and live caller output"""
        if self._host_stream:
            self._host_stream.stop()
            self._host_stream.close()
            self._host_stream = None
            self._host_send_callback = None
            print("[Audio] Host mic streaming stopped")
        self._stop_live_caller_stream()

    # --- Music Playback ---

    def load_music(self, file_path: str) -> bool:
        """Load a music file for playback"""
        path = Path(file_path)
        if not path.exists():
            print(f"Music file not found: {file_path}")
            return False

        try:
            import librosa
            audio, sr = librosa.load(str(path), sr=self.output_sample_rate, mono=True)
            self._music_data = audio.astype(np.float32)
            self._music_position = 0
            print(f"Loaded music: {path.name} ({len(audio)/sr:.1f}s)")
            return True
        except Exception as e:
            print(f"Failed to load music: {e}")
            return False

    def play_music(self):
        """Start music playback to specific channel"""
        import librosa

        if self._music_data is None:
            print("No music loaded")
            return

        if self._music_playing:
            self.stop_music()

        self._music_playing = True
        self._music_position = 0

        if self.output_device is None:
            print("No output device configured, using default")
            num_channels = 2
            device = None
            device_sr = self.output_sample_rate
            channel_idx = 0
        else:
            device_info = sd.query_devices(self.output_device)
            num_channels = device_info['max_output_channels']
            device_sr = int(device_info['default_samplerate'])
            device = self.output_device
            channel_idx = min(self.music_channel, num_channels) - 1

        # Resample music to device sample rate if needed
        if self.output_sample_rate != device_sr:
            self._music_resampled = librosa.resample(
                self._music_data, orig_sr=self.output_sample_rate, target_sr=device_sr
            )
        else:
            self._music_resampled = self._music_data.copy()

        # Apply fade-in at start of track
        fade_samples = int(device_sr * 0.015)  # 15ms fade
        if len(self._music_resampled) > fade_samples:
            fade_in = np.linspace(0, 1, fade_samples).astype(np.float32)
            self._music_resampled[:fade_samples] *= fade_in

        def callback(outdata, frames, time_info, status):
            outdata.fill(0)

            if not self._music_playing or self._music_resampled is None:
                return

            end_pos = self._music_position + frames

            if end_pos <= len(self._music_resampled):
                outdata[:, channel_idx] = self._music_resampled[self._music_position:end_pos] * self._music_volume
                self._music_position = end_pos
            else:
                remaining = len(self._music_resampled) - self._music_position
                if remaining > 0:
                    outdata[:remaining, channel_idx] = self._music_resampled[self._music_position:] * self._music_volume

                if self._music_loop:
                    self._music_position = 0
                    wrap_frames = frames - remaining
                    if wrap_frames > 0:
                        outdata[remaining:, channel_idx] = self._music_resampled[:wrap_frames] * self._music_volume
                    self._music_position = wrap_frames
                else:
                    self._music_playing = False

        try:
            self._music_stream = sd.OutputStream(
                device=device,
                channels=num_channels,
                samplerate=device_sr,
                dtype=np.float32,
                callback=callback,
                blocksize=2048
            )
            self._music_stream.start()
            print(f"Music playback started on ch {self.music_channel} @ {device_sr}Hz")
        except Exception as e:
            print(f"Music playback error: {e}")
            self._music_playing = False

    def stop_music(self):
        """Stop music playback"""
        self._music_playing = False
        if self._music_stream:
            self._music_stream.stop()
            self._music_stream.close()
            self._music_stream = None
        self._music_position = 0
        print("Music stopped")

    def play_ad(self, file_path: str):
        """Load and play an ad file once (no loop) on the ad channel"""
        import librosa

        path = Path(file_path)
        if not path.exists():
            print(f"Ad file not found: {file_path}")
            return

        self.stop_ad()

        try:
            audio, sr = librosa.load(str(path), sr=self.output_sample_rate, mono=True)
            self._ad_data = audio.astype(np.float32)
        except Exception as e:
            print(f"Failed to load ad: {e}")
            return

        self._ad_playing = True
        self._ad_position = 0

        if self.output_device is None:
            num_channels = 2
            device = None
            device_sr = self.output_sample_rate
            channel_idx = 0
        else:
            device_info = sd.query_devices(self.output_device)
            num_channels = device_info['max_output_channels']
            device_sr = int(device_info['default_samplerate'])
            device = self.output_device
            channel_idx = min(self.ad_channel, num_channels) - 1

        if self.output_sample_rate != device_sr:
            self._ad_resampled = librosa.resample(
                self._ad_data, orig_sr=self.output_sample_rate, target_sr=device_sr
            ).astype(np.float32)
        else:
            self._ad_resampled = self._ad_data

        def callback(outdata, frames, time_info, status):
            outdata[:] = 0
            if not self._ad_playing or self._ad_resampled is None:
                return

            remaining = len(self._ad_resampled) - self._ad_position
            if remaining >= frames:
                outdata[:, channel_idx] = self._ad_resampled[self._ad_position:self._ad_position + frames]
                self._ad_position += frames
            else:
                if remaining > 0:
                    outdata[:remaining, channel_idx] = self._ad_resampled[self._ad_position:]
                # Ad finished — no loop
                self._ad_playing = False

        try:
            self._ad_stream = sd.OutputStream(
                device=device,
                channels=num_channels,
                samplerate=device_sr,
                dtype=np.float32,
                callback=callback,
                blocksize=2048
            )
            self._ad_stream.start()
            print(f"Ad playback started on ch {self.ad_channel} @ {device_sr}Hz")
        except Exception as e:
            print(f"Ad playback error: {e}")
            self._ad_playing = False

    def stop_ad(self):
        """Stop ad playback"""
        self._ad_playing = False
        if self._ad_stream:
            self._ad_stream.stop()
            self._ad_stream.close()
            self._ad_stream = None
        self._ad_position = 0

    def set_music_volume(self, volume: float):
        """Set music volume (0.0 to 1.0)"""
        self._music_volume = max(0.0, min(1.0, volume))

    def is_music_playing(self) -> bool:
        """Check if music is currently playing"""
        return self._music_playing

    # --- SFX Playback ---

    def play_sfx(self, file_path: str):
        """Play a sound effect to specific channel using dedicated stream"""
        path = Path(file_path)
        if not path.exists():
            print(f"SFX file not found: {file_path}")
            return

        try:
            import librosa

            if self.output_device is None:
                audio, sr = librosa.load(str(path), sr=None, mono=True)
                audio = self._apply_fade(audio, sr)
                def play():
                    # Use a dedicated stream instead of sd.play()
                    with sd.OutputStream(samplerate=sr, channels=1, dtype=np.float32) as stream:
                        stream.write(audio.reshape(-1, 1))
            else:
                device_info = sd.query_devices(self.output_device)
                num_channels = device_info['max_output_channels']
                device_sr = int(device_info['default_samplerate'])
                channel_idx = min(self.sfx_channel, num_channels) - 1

                audio, _ = librosa.load(str(path), sr=device_sr, mono=True)
                audio = self._apply_fade(audio, device_sr)

                multi_ch = np.zeros((len(audio), num_channels), dtype=np.float32)
                multi_ch[:, channel_idx] = audio

                def play():
                    # Use dedicated stream to avoid interrupting other audio
                    with sd.OutputStream(
                        device=self.output_device,
                        samplerate=device_sr,
                        channels=num_channels,
                        dtype=np.float32
                    ) as stream:
                        stream.write(multi_ch)

            threading.Thread(target=play, daemon=True).start()
            print(f"Playing SFX: {path.name} on ch {self.sfx_channel}")
        except Exception as e:
            print(f"SFX playback error: {e}")


# Global instance
audio_service = AudioService()
