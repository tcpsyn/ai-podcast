"""Phone caller queue and audio stream service"""

import asyncio
import time
import threading
import numpy as np
from typing import Optional


class CallerService:
    """Manages phone caller queue, channel allocation, and WebSocket streams"""

    FIRST_REAL_CHANNEL = 3

    def __init__(self):
        self._queue: list[dict] = []
        self.active_calls: dict[str, dict] = {}
        self._allocated_channels: set[int] = set()
        self._caller_counter: int = 0
        self._lock = threading.Lock()
        self._websockets: dict[str, any] = {}  # caller_id -> WebSocket
        self._call_sids: dict[str, str] = {}  # caller_id -> SignalWire callSid
        self.streaming_tts: bool = False  # True while TTS audio is being streamed

    def add_to_queue(self, caller_id: str, phone: str):
        with self._lock:
            self._queue.append({
                "caller_id": caller_id,
                "phone": phone,
                "queued_at": time.time(),
            })
        print(f"[Caller] {phone} added to queue (ID: {caller_id})")

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
                    "phone": c["phone"],
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
        phone = caller["phone"]

        call_info = {
            "caller_id": caller_id,
            "phone": phone,
            "channel": channel,
            "started_at": time.time(),
        }
        self.active_calls[caller_id] = call_info
        print(f"[Caller] {phone} taken on air — channel {channel}")
        return call_info

    def hangup(self, caller_id: str):
        call_info = self.active_calls.pop(caller_id, None)
        if call_info:
            self.release_channel(call_info["channel"])
            print(f"[Caller] {call_info['phone']} hung up — channel {call_info['channel']} released")
        self._websockets.pop(caller_id, None)
        self._call_sids.pop(caller_id, None)

    def reset(self):
        with self._lock:
            for call_info in self.active_calls.values():
                self._allocated_channels.discard(call_info["channel"])
            self._queue.clear()
            self.active_calls.clear()
            self._allocated_channels.clear()
            self._caller_counter = 0
            self._websockets.clear()
            self._call_sids.clear()
        print("[Caller] Service reset")

    def register_websocket(self, caller_id: str, websocket):
        """Register a WebSocket for a caller"""
        self._websockets[caller_id] = websocket

    def unregister_websocket(self, caller_id: str):
        """Unregister a WebSocket"""
        self._websockets.pop(caller_id, None)

    async def send_audio_to_caller(self, caller_id: str, pcm_data: bytes, sample_rate: int):
        """Send small audio chunk to caller via SignalWire WebSocket.
        Encodes L16 PCM as base64 JSON per SignalWire protocol.
        """
        ws = self._websockets.get(caller_id)
        if not ws:
            return

        try:
            import base64
            if sample_rate != 16000:
                audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                ratio = 16000 / sample_rate
                out_len = int(len(audio) * ratio)
                indices = (np.arange(out_len) / ratio).astype(int)
                indices = np.clip(indices, 0, len(audio) - 1)
                audio = audio[indices]
                pcm_data = (audio * 32767).astype(np.int16).tobytes()

            payload = base64.b64encode(pcm_data).decode('ascii')
            import json
            await ws.send_text(json.dumps({
                "event": "media",
                "media": {"payload": payload}
            }))
        except Exception as e:
            print(f"[Caller] Failed to send audio: {e}")

    async def stream_audio_to_caller(self, caller_id: str, pcm_data: bytes, sample_rate: int):
        """Stream large audio (TTS) to caller in real-time chunks via SignalWire WebSocket."""
        ws = self._websockets.get(caller_id)
        if not ws:
            return

        self.streaming_tts = True
        try:
            import base64
            import json
            audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            if sample_rate != 16000:
                ratio = 16000 / sample_rate
                out_len = int(len(audio) * ratio)
                indices = (np.arange(out_len) / ratio).astype(int)
                indices = np.clip(indices, 0, len(audio) - 1)
                audio = audio[indices]

            chunk_samples = 960
            for i in range(0, len(audio), chunk_samples):
                if caller_id not in self._websockets:
                    break
                chunk = audio[i:i + chunk_samples]
                pcm_chunk = (chunk * 32767).astype(np.int16).tobytes()
                payload = base64.b64encode(pcm_chunk).decode('ascii')
                await ws.send_text(json.dumps({
                    "event": "media",
                    "media": {"payload": payload}
                }))
                await asyncio.sleep(0.055)

        except Exception as e:
            print(f"[Caller] Failed to stream audio: {e}")
        finally:
            self.streaming_tts = False

    def register_call_sid(self, caller_id: str, call_sid: str):
        """Track SignalWire callSid for a caller"""
        self._call_sids[caller_id] = call_sid

    def get_call_sid(self, caller_id: str) -> str | None:
        """Get SignalWire callSid for a caller"""
        return self._call_sids.get(caller_id)

    def unregister_call_sid(self, caller_id: str):
        """Remove callSid tracking"""
        self._call_sids.pop(caller_id, None)
