"""Browser caller queue and audio stream service"""

import asyncio
import time
import threading
from typing import Optional


class CallerService:
    """Manages browser caller queue, channel allocation, and WebSocket streams"""

    FIRST_REAL_CHANNEL = 3

    def __init__(self):
        self._queue: list[dict] = []
        self.active_calls: dict[str, dict] = {}
        self._allocated_channels: set[int] = set()
        self._caller_counter: int = 0
        self._lock = threading.Lock()
        self._websockets: dict[str, any] = {}  # caller_id -> WebSocket

    def add_to_queue(self, caller_id: str, name: str):
        with self._lock:
            self._queue.append({
                "caller_id": caller_id,
                "name": name,
                "queued_at": time.time(),
            })
        print(f"[Caller] {name} added to queue (ID: {caller_id})")

    def remove_from_queue(self, caller_id: str):
        with self._lock:
            self._queue = [c for c in self._queue if c["caller_id"] != caller_id]
        print(f"[Caller] {caller_id} removed from queue")

    def get_queue(self) -> list[dict]:
        now = time.time()
        with self._lock:
            return [
                {
                    "caller_id": c["caller_id"],
                    "name": c["name"],
                    "wait_time": int(now - c["queued_at"]),
                }
                for c in self._queue
            ]

    def allocate_channel(self) -> int:
        with self._lock:
            ch = self.FIRST_REAL_CHANNEL
            while ch in self._allocated_channels:
                ch += 1
            self._allocated_channels.add(ch)
            return ch

    def release_channel(self, channel: int):
        with self._lock:
            self._allocated_channels.discard(channel)

    def take_call(self, caller_id: str) -> dict:
        caller = None
        with self._lock:
            for c in self._queue:
                if c["caller_id"] == caller_id:
                    caller = c
                    break
            if caller:
                self._queue = [c for c in self._queue if c["caller_id"] != caller_id]

        if not caller:
            raise ValueError(f"Caller {caller_id} not in queue")

        channel = self.allocate_channel()
        self._caller_counter += 1
        name = caller["name"]

        call_info = {
            "caller_id": caller_id,
            "name": name,
            "channel": channel,
            "started_at": time.time(),
        }
        self.active_calls[caller_id] = call_info
        print(f"[Caller] {name} taken on air — channel {channel}")
        return call_info

    def hangup(self, caller_id: str):
        call_info = self.active_calls.pop(caller_id, None)
        if call_info:
            self.release_channel(call_info["channel"])
            print(f"[Caller] {call_info['name']} hung up — channel {call_info['channel']} released")
        self._websockets.pop(caller_id, None)

    def reset(self):
        with self._lock:
            for call_info in self.active_calls.values():
                self._allocated_channels.discard(call_info["channel"])
            self._queue.clear()
            self.active_calls.clear()
            self._allocated_channels.clear()
            self._caller_counter = 0
            self._websockets.clear()
        print("[Caller] Service reset")

    def register_websocket(self, caller_id: str, websocket):
        """Register a WebSocket for a caller"""
        self._websockets[caller_id] = websocket

    def unregister_websocket(self, caller_id: str):
        """Unregister a WebSocket"""
        self._websockets.pop(caller_id, None)

    async def send_audio_to_caller(self, caller_id: str, pcm_data: bytes, sample_rate: int):
        """Send audio to real caller via WebSocket binary frame"""
        ws = self._websockets.get(caller_id)
        if not ws:
            return

        try:
            if sample_rate != 16000:
                import numpy as np
                import librosa
                audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                pcm_data = (audio * 32767).astype(np.int16).tobytes()
            await ws.send_bytes(pcm_data)
        except Exception as e:
            print(f"[Caller] Failed to send audio: {e}")

    async def notify_caller(self, caller_id: str, message: dict):
        """Send JSON control message to caller"""
        ws = self._websockets.get(caller_id)
        if ws:
            import json
            await ws.send_text(json.dumps(message))

    async def disconnect_caller(self, caller_id: str):
        """Disconnect a caller's WebSocket"""
        ws = self._websockets.get(caller_id)
        if ws:
            try:
                import json
                await ws.send_text(json.dumps({"status": "disconnected"}))
                await ws.close()
            except Exception:
                pass
        self._websockets.pop(caller_id, None)
