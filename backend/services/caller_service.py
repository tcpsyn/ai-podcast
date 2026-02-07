"""Phone caller queue and audio stream service"""

import asyncio
import base64
import json
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
        self._stream_sids: dict[str, str] = {}  # caller_id -> SignalWire streamSid
        self._send_locks: dict[str, asyncio.Lock] = {}  # per-caller send lock
        self._streaming_tts: set[str] = set()  # caller_ids currently receiving TTS
        self._screening_state: dict[str, dict] = {}  # caller_id -> screening conversation

    def _get_send_lock(self, caller_id: str) -> asyncio.Lock:
        if caller_id not in self._send_locks:
            self._send_locks[caller_id] = asyncio.Lock()
        return self._send_locks[caller_id]

    def is_streaming_tts(self, caller_id: str) -> bool:
        return caller_id in self._streaming_tts

    def is_streaming_tts_any(self) -> bool:
        return len(self._streaming_tts) > 0

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
        self._stream_sids.pop(caller_id, None)
        self._send_locks.pop(caller_id, None)
        self._screening_state.pop(caller_id, None)

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
            self._stream_sids.clear()
            self._send_locks.clear()
            self._streaming_tts.clear()
            self._screening_state.clear()
        print("[Caller] Service reset")

    # --- Screening ---

    def start_screening(self, caller_id: str):
        """Initialize screening state for a queued caller"""
        self._screening_state[caller_id] = {
            "conversation": [],
            "caller_name": None,
            "topic": None,
            "status": "screening",  # screening, complete
            "response_count": 0,
        }
        print(f"[Screening] Started for {caller_id}")

    def get_screening_state(self, caller_id: str) -> Optional[dict]:
        return self._screening_state.get(caller_id)

    def update_screening(self, caller_id: str, caller_text: str = None,
                         screener_text: str = None, caller_name: str = None,
                         topic: str = None):
        """Update screening conversation and extracted info"""
        state = self._screening_state.get(caller_id)
        if not state:
            return
        if caller_text:
            state["conversation"].append({"role": "caller", "content": caller_text})
            state["response_count"] += 1
        if screener_text:
            state["conversation"].append({"role": "screener", "content": screener_text})
        if caller_name:
            state["caller_name"] = caller_name
        if topic:
            state["topic"] = topic

    def end_screening(self, caller_id: str):
        """Mark screening as complete"""
        state = self._screening_state.get(caller_id)
        if state:
            state["status"] = "complete"
            print(f"[Screening] Complete for {caller_id}: name={state.get('caller_name')}, topic={state.get('topic')}")

    def get_queue(self) -> list[dict]:
        """Get queue with screening info enrichment"""
        now = time.time()
        with self._lock:
            result = []
            for c in self._queue:
                entry = {
                    "caller_id": c["caller_id"],
                    "phone": c["phone"],
                    "wait_time": int(now - c["queued_at"]),
                }
                screening = self._screening_state.get(c["caller_id"])
                if screening:
                    entry["screening_status"] = screening["status"]
                    entry["caller_name"] = screening.get("caller_name")
                    entry["screening_summary"] = screening.get("topic")
                else:
                    entry["screening_status"] = None
                    entry["caller_name"] = None
                    entry["screening_summary"] = None
                result.append(entry)
            return result

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
        if caller_id in self._streaming_tts:
            return  # Don't send host audio during TTS streaming

        ws = self._websockets.get(caller_id)
        if not ws:
            return

        lock = self._get_send_lock(caller_id)
        async with lock:
            try:
                if sample_rate != 16000:
                    audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                    ratio = 16000 / sample_rate
                    out_len = int(len(audio) * ratio)
                    indices = (np.arange(out_len) / ratio).astype(int)
                    indices = np.clip(indices, 0, len(audio) - 1)
                    audio = audio[indices]
                    pcm_data = (audio * 32767).astype(np.int16).tobytes()

                payload = base64.b64encode(pcm_data).decode('ascii')
                stream_sid = self._stream_sids.get(caller_id, "")
                await ws.send_text(json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": payload}
                }))
            except Exception as e:
                print(f"[Caller] Failed to send audio: {e}")

    async def stream_audio_to_caller(self, caller_id: str, pcm_data: bytes, sample_rate: int):
        """Stream large audio (TTS) to caller in real-time chunks via SignalWire WebSocket."""
        ws = self._websockets.get(caller_id)
        if not ws:
            return

        lock = self._get_send_lock(caller_id)
        self._streaming_tts.add(caller_id)
        chunks_sent = 0
        try:
            audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            if sample_rate != 16000:
                ratio = 16000 / sample_rate
                out_len = int(len(audio) * ratio)
                indices = (np.arange(out_len) / ratio).astype(int)
                indices = np.clip(indices, 0, len(audio) - 1)
                audio = audio[indices]

            total_chunks = (len(audio) + 959) // 960
            duration_s = len(audio) / 16000
            print(f"[Caller] TTS stream starting: {duration_s:.1f}s audio, {total_chunks} chunks")

            chunk_samples = 960
            chunk_duration = chunk_samples / 16000  # 60ms per chunk

            for i in range(0, len(audio), chunk_samples):
                if caller_id not in self._websockets:
                    print(f"[Caller] TTS stream aborted: caller {caller_id} disconnected at chunk {chunks_sent}/{total_chunks}")
                    break
                t0 = time.time()
                chunk = audio[i:i + chunk_samples]
                pcm_chunk = (chunk * 32767).astype(np.int16).tobytes()
                payload = base64.b64encode(pcm_chunk).decode('ascii')
                stream_sid = self._stream_sids.get(caller_id, "")
                async with lock:
                    await ws.send_text(json.dumps({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": payload}
                    }))
                chunks_sent += 1
                # Sleep to match real-time playback rate
                elapsed = time.time() - t0
                sleep_time = max(0, chunk_duration - elapsed)
                await asyncio.sleep(sleep_time)

            print(f"[Caller] TTS stream finished: {chunks_sent}/{total_chunks} chunks sent")

        except Exception as e:
            print(f"[Caller] TTS stream failed at chunk {chunks_sent}: {e}")
        finally:
            self._streaming_tts.discard(caller_id)

    def register_call_sid(self, caller_id: str, call_sid: str):
        """Track SignalWire callSid for a caller"""
        self._call_sids[caller_id] = call_sid

    def get_call_sid(self, caller_id: str) -> str | None:
        """Get SignalWire callSid for a caller"""
        return self._call_sids.get(caller_id)

    def unregister_call_sid(self, caller_id: str):
        """Remove callSid tracking"""
        self._call_sids.pop(caller_id, None)

    def register_stream_sid(self, caller_id: str, stream_sid: str):
        """Track SignalWire streamSid for a caller"""
        self._stream_sids[caller_id] = stream_sid

    def unregister_stream_sid(self, caller_id: str):
        """Remove streamSid tracking"""
        self._stream_sids.pop(caller_id, None)
