"""Twilio call queue and media stream service"""

import time
import threading
from typing import Optional


class TwilioService:
    """Manages Twilio call queue, channel allocation, and media streams"""

    FIRST_REAL_CHANNEL = 3

    def __init__(self):
        self._queue: list[dict] = []
        self.active_calls: dict[str, dict] = {}
        self._allocated_channels: set[int] = set()
        self._caller_counter: int = 0
        self._lock = threading.Lock()

    def add_to_queue(self, call_sid: str, phone: str):
        with self._lock:
            self._queue.append({
                "call_sid": call_sid,
                "phone": phone,
                "queued_at": time.time(),
            })
        print(f"[Twilio] Caller {phone} added to queue (SID: {call_sid})")

    def remove_from_queue(self, call_sid: str):
        with self._lock:
            self._queue = [c for c in self._queue if c["call_sid"] != call_sid]
        print(f"[Twilio] Caller {call_sid} removed from queue")

    def get_queue(self) -> list[dict]:
        now = time.time()
        with self._lock:
            return [
                {
                    "call_sid": c["call_sid"],
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

    def take_call(self, call_sid: str) -> dict:
        caller = None
        with self._lock:
            for c in self._queue:
                if c["call_sid"] == call_sid:
                    caller = c
                    break
            if caller:
                self._queue = [c for c in self._queue if c["call_sid"] != call_sid]

        if not caller:
            raise ValueError(f"Call {call_sid} not in queue")

        channel = self.allocate_channel()
        self._caller_counter += 1
        name = f"Caller #{self._caller_counter}"

        call_info = {
            "call_sid": call_sid,
            "phone": caller["phone"],
            "channel": channel,
            "name": name,
            "started_at": time.time(),
        }
        self.active_calls[call_sid] = call_info
        print(f"[Twilio] {name} ({caller['phone']}) taken on air — channel {channel}")
        return call_info

    def hangup(self, call_sid: str):
        call_info = self.active_calls.pop(call_sid, None)
        if call_info:
            self.release_channel(call_info["channel"])
            print(f"[Twilio] {call_info['name']} hung up — channel {call_info['channel']} released")

    def reset(self):
        with self._lock:
            for call_info in self.active_calls.values():
                self._allocated_channels.discard(call_info["channel"])
            self._queue.clear()
            self.active_calls.clear()
            self._allocated_channels.clear()
            self._caller_counter = 0
        print("[Twilio] Service reset")
