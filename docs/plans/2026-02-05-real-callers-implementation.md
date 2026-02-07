# Real Callers + AI Follow-Up Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Twilio phone call-in support with hold queue, three-way calls (host + real caller + AI), AI auto-respond mode, and AI follow-up callers that reference real caller conversations.

**Architecture:** Twilio Media Streams deliver real caller audio via WebSocket. Audio is decoded from mulaw 8kHz, routed to a dedicated Loopback channel, and transcribed in real-time. Host + AI TTS audio is mixed and streamed back to the caller. Session model tracks multi-party conversations and show history for AI follow-up context.

**Tech Stack:** Python/FastAPI, Twilio (twilio package + Media Streams WebSocket), sounddevice, faster-whisper, existing LLM/TTS services, vanilla JS frontend.

**Design doc:** `docs/plans/2026-02-05-real-callers-design.md`

---

## Task 1: Config and Dependencies

**Files:**
- Modify: `backend/config.py`
- Modify: `.env`

**Step 1: Install twilio package**

```bash
pip install twilio
```

**Step 2: Add Twilio settings to config**

In `backend/config.py`, add to the `Settings` class after the existing API key fields:

```python
# Twilio Settings
twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
twilio_phone_number: str = os.getenv("TWILIO_PHONE_NUMBER", "")
twilio_webhook_base_url: str = os.getenv("TWILIO_WEBHOOK_BASE_URL", "")
```

**Step 3: Add placeholder env vars to `.env`**

```
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=
TWILIO_WEBHOOK_BASE_URL=
```

**Step 4: Verify server starts**

```bash
cd /Users/lukemacneil/ai-podcast && python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Expected: Server starts without errors.

**Step 5: Commit**

```bash
git add backend/config.py .env
git commit -m "Add Twilio config and dependencies"
```

---

## Task 2: Session Model — Multi-Party Calls and Show History

**Files:**
- Modify: `backend/main.py` (Session class, lines 296-356)
- Create: `tests/test_session.py`

**Step 1: Write tests for new session model**

Create `tests/test_session.py`:

```python
import sys
sys.path.insert(0, "/Users/lukemacneil/ai-podcast")

from backend.main import Session, CallRecord


def test_call_record_creation():
    record = CallRecord(
        caller_type="real",
        caller_name="Dave",
        summary="Called about his wife leaving",
        transcript=[{"role": "host", "content": "What happened?"}],
    )
    assert record.caller_type == "real"
    assert record.caller_name == "Dave"


def test_session_call_history():
    s = Session()
    assert s.call_history == []
    record = CallRecord(
        caller_type="ai", caller_name="Tony",
        summary="Talked about gambling", transcript=[],
    )
    s.call_history.append(record)
    assert len(s.call_history) == 1


def test_session_active_real_caller():
    s = Session()
    assert s.active_real_caller is None
    s.active_real_caller = {
        "call_sid": "CA123", "phone": "+15125550142",
        "channel": 3, "name": "Caller #1",
    }
    assert s.active_real_caller["channel"] == 3


def test_session_three_party_conversation():
    s = Session()
    s.start_call("1")  # AI caller Tony
    s.add_message("host", "Hey Tony")
    s.add_message("ai_caller:Tony", "What's up man")
    s.add_message("real_caller:Dave", "Yeah I agree with Tony")
    assert len(s.conversation) == 3
    assert s.conversation[2]["role"] == "real_caller:Dave"


def test_session_get_show_history_summary():
    s = Session()
    s.call_history.append(CallRecord(
        caller_type="real", caller_name="Dave",
        summary="Called about his wife leaving after 12 years",
        transcript=[],
    ))
    s.call_history.append(CallRecord(
        caller_type="ai", caller_name="Jasmine",
        summary="Talked about her boss hitting on her",
        transcript=[],
    ))
    summary = s.get_show_history()
    assert "Dave" in summary
    assert "Jasmine" in summary


def test_session_reset_clears_history():
    s = Session()
    s.call_history.append(CallRecord(
        caller_type="real", caller_name="Dave",
        summary="test", transcript=[],
    ))
    s.active_real_caller = {"call_sid": "CA123"}
    s.ai_respond_mode = "auto"
    s.reset()
    assert s.call_history == []
    assert s.active_real_caller is None
    assert s.ai_respond_mode == "manual"


def test_session_conversation_summary_three_party():
    s = Session()
    s.start_call("1")
    s.add_message("host", "Tell me what happened")
    s.add_message("real_caller:Dave", "She just left man")
    s.add_message("ai_caller:Tony", "Same thing happened to me")
    summary = s.get_conversation_summary()
    assert "Dave" in summary
    assert "Tony" in summary
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/lukemacneil/ai-podcast && python -m pytest tests/test_session.py -v
```

Expected: Failures — `CallRecord` doesn't exist, new fields missing.

**Step 3: Implement CallRecord and extend Session**

In `backend/main.py`, add `CallRecord` dataclass above the `Session` class:

```python
from dataclasses import dataclass, field

@dataclass
class CallRecord:
    caller_type: str          # "ai" or "real"
    caller_name: str          # "Tony" or "Caller #3"
    summary: str              # LLM-generated summary after hangup
    transcript: list[dict] = field(default_factory=list)
```

Extend `Session.__init__` to add:

```python
self.call_history: list[CallRecord] = []
self.active_real_caller: dict | None = None
self.ai_respond_mode: str = "manual"  # "manual" or "auto"
self.auto_followup: bool = False
```

Add `get_show_history()` method to Session:

```python
def get_show_history(self) -> str:
    """Get formatted show history for AI caller prompts"""
    if not self.call_history:
        return ""
    lines = ["EARLIER IN THE SHOW:"]
    for record in self.call_history:
        caller_type_label = "(real caller)" if record.caller_type == "real" else "(AI)"
        lines.append(f"- {record.caller_name} {caller_type_label}: {record.summary}")
    lines.append("You can reference these if it feels natural. Don't force it.")
    return "\n".join(lines)
```

Update `get_conversation_summary()` to handle three-party roles — replace the role label logic:

```python
def get_conversation_summary(self) -> str:
    if len(self.conversation) <= 2:
        return ""
    summary_parts = []
    for msg in self.conversation[-6:]:
        role = msg["role"]
        if role == "user" or role == "host":
            label = "Host"
        elif role.startswith("real_caller:"):
            label = role.split(":", 1)[1]
        elif role.startswith("ai_caller:"):
            label = role.split(":", 1)[1]
        elif role == "assistant":
            label = self.caller["name"] if self.caller else "Caller"
        else:
            label = role
        content = msg["content"]
        summary_parts.append(
            f'{label}: "{content[:100]}..."' if len(content) > 100
            else f'{label}: "{content}"'
        )
    return "\n".join(summary_parts)
```

Update `reset()` to clear new fields:

```python
def reset(self):
    self.caller_backgrounds = {}
    self.current_caller_key = None
    self.conversation = []
    self.call_history = []
    self.active_real_caller = None
    self.ai_respond_mode = "manual"
    self.auto_followup = False
    self.id = str(uuid.uuid4())[:8]
    print(f"[Session] Reset - new session ID: {self.id}")
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/lukemacneil/ai-podcast && python -m pytest tests/test_session.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add backend/main.py tests/test_session.py
git commit -m "Add CallRecord model and multi-party session support"
```

---

## Task 3: Twilio Call Queue Service

**Files:**
- Create: `backend/services/twilio_service.py`
- Create: `tests/test_twilio_service.py`

**Step 1: Write tests for call queue**

Create `tests/test_twilio_service.py`:

```python
import sys
sys.path.insert(0, "/Users/lukemacneil/ai-podcast")

from backend.services.twilio_service import TwilioService


def test_queue_starts_empty():
    svc = TwilioService()
    assert svc.get_queue() == []


def test_add_caller_to_queue():
    svc = TwilioService()
    svc.add_to_queue("CA123", "+15125550142")
    q = svc.get_queue()
    assert len(q) == 1
    assert q[0]["call_sid"] == "CA123"
    assert q[0]["phone"] == "+15125550142"
    assert "wait_time" in q[0]


def test_remove_caller_from_queue():
    svc = TwilioService()
    svc.add_to_queue("CA123", "+15125550142")
    svc.remove_from_queue("CA123")
    assert svc.get_queue() == []


def test_allocate_channel():
    svc = TwilioService()
    ch1 = svc.allocate_channel()
    ch2 = svc.allocate_channel()
    assert ch1 == 3  # First real caller channel
    assert ch2 == 4
    svc.release_channel(ch1)
    ch3 = svc.allocate_channel()
    assert ch3 == 3  # Reuses released channel


def test_take_call():
    svc = TwilioService()
    svc.add_to_queue("CA123", "+15125550142")
    result = svc.take_call("CA123")
    assert result["call_sid"] == "CA123"
    assert result["channel"] >= 3
    assert svc.get_queue() == []  # Removed from queue
    assert svc.active_calls["CA123"]["channel"] == result["channel"]


def test_hangup_real_caller():
    svc = TwilioService()
    svc.add_to_queue("CA123", "+15125550142")
    svc.take_call("CA123")
    ch = svc.active_calls["CA123"]["channel"]
    svc.hangup("CA123")
    assert "CA123" not in svc.active_calls
    # Channel is released back to pool
    assert ch not in svc._allocated_channels


def test_caller_counter_increments():
    svc = TwilioService()
    svc.add_to_queue("CA1", "+15125550001")
    svc.add_to_queue("CA2", "+15125550002")
    r1 = svc.take_call("CA1")
    r2 = svc.take_call("CA2")
    assert r1["name"] == "Caller #1"
    assert r2["name"] == "Caller #2"
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/lukemacneil/ai-podcast && python -m pytest tests/test_twilio_service.py -v
```

Expected: ImportError — module doesn't exist.

**Step 3: Implement TwilioService**

Create `backend/services/twilio_service.py`:

```python
"""Twilio call queue and media stream service"""

import time
import threading
from typing import Optional


class TwilioService:
    """Manages Twilio call queue, channel allocation, and media streams"""

    # Real caller channels start at 3 (1=host, 2=AI callers)
    FIRST_REAL_CHANNEL = 3

    def __init__(self):
        self._queue: list[dict] = []  # Waiting callers
        self.active_calls: dict[str, dict] = {}  # call_sid -> {phone, channel, name, stream}
        self._allocated_channels: set[int] = set()
        self._caller_counter: int = 0
        self._lock = threading.Lock()

    def add_to_queue(self, call_sid: str, phone: str):
        """Add incoming caller to hold queue"""
        with self._lock:
            self._queue.append({
                "call_sid": call_sid,
                "phone": phone,
                "queued_at": time.time(),
            })
        print(f"[Twilio] Caller {phone} added to queue (SID: {call_sid})")

    def remove_from_queue(self, call_sid: str):
        """Remove caller from queue without taking them"""
        with self._lock:
            self._queue = [c for c in self._queue if c["call_sid"] != call_sid]
        print(f"[Twilio] Caller {call_sid} removed from queue")

    def get_queue(self) -> list[dict]:
        """Get current queue with wait times"""
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
        """Allocate the next available Loopback channel for a real caller"""
        with self._lock:
            ch = self.FIRST_REAL_CHANNEL
            while ch in self._allocated_channels:
                ch += 1
            self._allocated_channels.add(ch)
            return ch

    def release_channel(self, channel: int):
        """Release a channel back to the pool"""
        with self._lock:
            self._allocated_channels.discard(channel)

    def take_call(self, call_sid: str) -> dict:
        """Take a caller off hold — allocate channel and mark active"""
        # Find in queue
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
        """Hang up on a real caller — release channel"""
        call_info = self.active_calls.pop(call_sid, None)
        if call_info:
            self.release_channel(call_info["channel"])
            print(f"[Twilio] {call_info['name']} hung up — channel {call_info['channel']} released")

    def reset(self):
        """Reset all state"""
        with self._lock:
            for call_info in self.active_calls.values():
                self._allocated_channels.discard(call_info["channel"])
            self._queue.clear()
            self.active_calls.clear()
            self._allocated_channels.clear()
            self._caller_counter = 0
        print("[Twilio] Service reset")
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/lukemacneil/ai-podcast && python -m pytest tests/test_twilio_service.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add backend/services/twilio_service.py tests/test_twilio_service.py
git commit -m "Add Twilio call queue service with channel allocation"
```

---

## Task 4: Twilio Webhook Endpoints

**Files:**
- Modify: `backend/main.py`

**Step 1: Add Twilio webhook imports and service instance**

At the top of `backend/main.py`, add:

```python
from twilio.twiml.voice_response import VoiceResponse
from .services.twilio_service import TwilioService
```

After `session = Session()`, add:

```python
twilio_service = TwilioService()
```

**Step 2: Add the voice webhook endpoint**

This is what Twilio calls when someone dials your number:

```python
from fastapi import Form

@app.post("/api/twilio/voice")
async def twilio_voice_webhook(
    CallSid: str = Form(...),
    From: str = Form(...),
):
    """Handle incoming Twilio call — greet and enqueue"""
    twilio_service.add_to_queue(CallSid, From)

    response = VoiceResponse()
    response.say("You're calling Luke at the Roost. Hold tight, we'll get to you.", voice="alice")
    response.enqueue(
        "radio_show",
        wait_url="/api/twilio/hold-music",
        wait_url_method="POST",
    )
    return Response(content=str(response), media_type="application/xml")
```

**Step 3: Add hold music endpoint**

```python
@app.post("/api/twilio/hold-music")
async def twilio_hold_music():
    """Serve hold music TwiML for queued callers"""
    response = VoiceResponse()
    # Play hold music in a loop — Twilio will re-request this URL periodically
    music_files = list(settings.music_dir.glob("*.mp3")) + list(settings.music_dir.glob("*.wav"))
    if music_files:
        # Use first available track via public URL
        response.say("Please hold, you'll be on air shortly.", voice="alice")
        response.pause(length=30)
    else:
        response.say("Please hold.", voice="alice")
        response.pause(length=30)
    return Response(content=str(response), media_type="application/xml")
```

**Step 4: Add queue management endpoints**

```python
@app.get("/api/queue")
async def get_call_queue():
    """Get list of callers waiting in queue"""
    return {"queue": twilio_service.get_queue()}


@app.post("/api/queue/take/{call_sid}")
async def take_call_from_queue(call_sid: str):
    """Take a caller off hold and put them on air"""
    try:
        call_info = twilio_service.take_call(call_sid)
    except ValueError as e:
        raise HTTPException(404, str(e))

    session.active_real_caller = {
        "call_sid": call_info["call_sid"],
        "phone": call_info["phone"],
        "channel": call_info["channel"],
        "name": call_info["name"],
    }

    # Connect Twilio media stream by updating the call
    # This redirects the call from the queue to a media stream
    from twilio.rest import Client as TwilioClient
    if settings.twilio_account_sid and settings.twilio_auth_token:
        client = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)
        twiml = VoiceResponse()
        connect = twiml.connect()
        connect.stream(
            url=f"wss://{settings.twilio_webhook_base_url.replace('https://', '')}/api/twilio/stream",
            name=call_sid,
        )
        client.calls(call_sid).update(twiml=str(twiml))

    return {
        "status": "on_air",
        "caller": call_info,
    }


@app.post("/api/queue/drop/{call_sid}")
async def drop_from_queue(call_sid: str):
    """Drop a caller from the queue"""
    twilio_service.remove_from_queue(call_sid)

    # Hang up the Twilio call
    from twilio.rest import Client as TwilioClient
    if settings.twilio_account_sid and settings.twilio_auth_token:
        try:
            client = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)
            client.calls(call_sid).update(status="completed")
        except Exception as e:
            print(f"[Twilio] Failed to end call {call_sid}: {e}")

    return {"status": "dropped"}
```

**Step 5: Add Response import**

```python
from fastapi.responses import FileResponse, Response
```

(Modify the existing `FileResponse` import line to include `Response`.)

**Step 6: Verify server starts**

```bash
cd /Users/lukemacneil/ai-podcast && python -c "from backend.main import app; print('OK')"
```

Expected: `OK`

**Step 7: Commit**

```bash
git add backend/main.py
git commit -m "Add Twilio webhook and queue management endpoints"
```

---

## Task 5: WebSocket Media Stream Handler

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/services/twilio_service.py`
- Modify: `backend/services/audio.py`

This is the core of real caller audio — bidirectional streaming via Twilio Media Streams.

**Step 1: Add WebSocket endpoint to main.py**

```python
from fastapi import WebSocket, WebSocketDisconnect
import json
import base64
import audioop
import asyncio
import struct

@app.websocket("/api/twilio/stream")
async def twilio_media_stream(websocket: WebSocket):
    """Handle Twilio Media Streams WebSocket — bidirectional audio"""
    await websocket.accept()
    print("[Twilio WS] Media stream connected")

    call_sid = None
    stream_sid = None
    audio_buffer = bytearray()
    CHUNK_DURATION_S = 3  # Transcribe every 3 seconds of audio
    MULAW_SAMPLE_RATE = 8000
    chunk_samples = CHUNK_DURATION_S * MULAW_SAMPLE_RATE

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            event = msg.get("event")

            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                call_sid = msg["start"]["callSid"]
                print(f"[Twilio WS] Stream started: {stream_sid} for call {call_sid}")

            elif event == "media":
                # Decode mulaw audio from base64
                payload = base64.b64decode(msg["media"]["payload"])
                # Convert mulaw to 16-bit PCM
                pcm_data = audioop.ulaw2lin(payload, 2)
                audio_buffer.extend(pcm_data)

                # Get channel for this caller
                call_info = twilio_service.active_calls.get(call_sid)
                if call_info:
                    channel = call_info["channel"]
                    # Route PCM to the caller's dedicated Loopback channel
                    audio_service.route_real_caller_audio(pcm_data, channel, MULAW_SAMPLE_RATE)

                # When we have enough audio, transcribe
                if len(audio_buffer) >= chunk_samples * 2:  # 2 bytes per sample
                    pcm_chunk = bytes(audio_buffer[:chunk_samples * 2])
                    audio_buffer = audio_buffer[chunk_samples * 2:]

                    # Transcribe in background
                    asyncio.create_task(
                        _handle_real_caller_transcription(call_sid, pcm_chunk, MULAW_SAMPLE_RATE)
                    )

            elif event == "stop":
                print(f"[Twilio WS] Stream stopped: {stream_sid}")
                break

    except WebSocketDisconnect:
        print(f"[Twilio WS] Disconnected: {call_sid}")
    except Exception as e:
        print(f"[Twilio WS] Error: {e}")
    finally:
        # Transcribe any remaining audio
        if audio_buffer and call_sid:
            asyncio.create_task(
                _handle_real_caller_transcription(call_sid, bytes(audio_buffer), MULAW_SAMPLE_RATE)
            )


async def _handle_real_caller_transcription(call_sid: str, pcm_data: bytes, sample_rate: int):
    """Transcribe a chunk of real caller audio and add to conversation"""
    call_info = twilio_service.active_calls.get(call_sid)
    if not call_info:
        return

    text = await transcribe_audio(pcm_data, source_sample_rate=sample_rate)
    if not text or not text.strip():
        return

    caller_name = call_info["name"]
    print(f"[Real Caller] {caller_name}: {text}")

    # Add to conversation with real_caller role
    session.add_message(f"real_caller:{caller_name}", text)

    # If AI auto-respond mode is on and an AI caller is active, check if AI should respond
    if session.ai_respond_mode == "auto" and session.current_caller_key:
        asyncio.create_task(_check_ai_auto_respond(text, caller_name))


async def _check_ai_auto_respond(real_caller_text: str, real_caller_name: str):
    """Check if AI caller should jump in, and generate response if so"""
    if not session.caller:
        return

    # Cooldown check
    if hasattr(session, '_last_ai_auto_respond') and \
       time.time() - session._last_ai_auto_respond < 10:
        return

    ai_name = session.caller["name"]

    # Quick "should I respond?" check with minimal LLM call
    should_respond = await llm_service.generate(
        messages=[{"role": "user", "content": f'Someone just said: "{real_caller_text}". Should {ai_name} jump in? Reply only YES or NO.'}],
        system_prompt=f"You're deciding if {ai_name} should respond to what was just said on a radio show. Say YES if it's interesting or relevant to them, NO if not.",
    )

    if "YES" not in should_respond.upper():
        return

    print(f"[Auto-Respond] {ai_name} is jumping in...")
    session._last_ai_auto_respond = time.time()

    # Generate full response
    conversation_summary = session.get_conversation_summary()
    show_history = session.get_show_history()
    system_prompt = get_caller_prompt(session.caller, conversation_summary)
    if show_history:
        system_prompt += f"\n\n{show_history}"

    response = await llm_service.generate(
        messages=session.conversation[-10:],
        system_prompt=system_prompt,
    )
    response = clean_for_tts(response)
    if not response or not response.strip():
        return

    session.add_message(f"ai_caller:{ai_name}", response)

    # Generate TTS and play
    audio_bytes = await generate_speech(response, session.caller["voice"], "none")

    import threading
    thread = threading.Thread(
        target=audio_service.play_caller_audio,
        args=(audio_bytes, 24000),
        daemon=True,
    )
    thread.start()

    # Also send to Twilio so real caller hears the AI
    # (handled in Task 6 - outbound audio mixing)
```

**Step 2: Add `route_real_caller_audio` to AudioService**

In `backend/services/audio.py`, add this method to `AudioService`:

```python
def route_real_caller_audio(self, pcm_data: bytes, channel: int, sample_rate: int):
    """Route real caller PCM audio to a specific Loopback channel"""
    import librosa

    if self.output_device is None:
        return

    try:
        # Convert bytes to float32
        audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

        device_info = sd.query_devices(self.output_device)
        num_channels = device_info['max_output_channels']
        device_sr = int(device_info['default_samplerate'])
        channel_idx = min(channel, num_channels) - 1

        # Resample from Twilio's 8kHz to device sample rate
        if sample_rate != device_sr:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=device_sr)

        # Create multi-channel output
        multi_ch = np.zeros((len(audio), num_channels), dtype=np.float32)
        multi_ch[:, channel_idx] = audio

        # Write to output device (non-blocking, small chunks)
        with sd.OutputStream(
            device=self.output_device,
            samplerate=device_sr,
            channels=num_channels,
            dtype=np.float32,
        ) as stream:
            stream.write(multi_ch)

    except Exception as e:
        print(f"Real caller audio routing error: {e}")
```

**Step 3: Add `import time` at the top of `main.py`** (if not already present)

**Step 4: Verify server starts**

```bash
cd /Users/lukemacneil/ai-podcast && python -c "from backend.main import app; print('OK')"
```

Expected: `OK`

**Step 5: Commit**

```bash
git add backend/main.py backend/services/audio.py
git commit -m "Add Twilio WebSocket media stream handler with real-time transcription"
```

---

## Task 6: Outbound Audio to Real Caller (Host + AI TTS)

**Files:**
- Modify: `backend/services/twilio_service.py`
- Modify: `backend/main.py`

The real caller needs to hear the host's voice and the AI caller's TTS through the phone.

**Step 1: Add WebSocket registry to TwilioService**

In `backend/services/twilio_service.py`, add:

```python
import asyncio
import base64
import audioop

class TwilioService:
    def __init__(self):
        # ... existing init ...
        self._websockets: dict[str, any] = {}  # call_sid -> WebSocket

    def register_websocket(self, call_sid: str, websocket):
        """Register a WebSocket for a call"""
        self._websockets[call_sid] = websocket

    def unregister_websocket(self, call_sid: str):
        """Unregister a WebSocket"""
        self._websockets.pop(call_sid, None)

    async def send_audio_to_caller(self, call_sid: str, pcm_data: bytes, sample_rate: int):
        """Send audio back to real caller via Twilio WebSocket"""
        ws = self._websockets.get(call_sid)
        if not ws:
            return

        call_info = self.active_calls.get(call_sid)
        if not call_info or "stream_sid" not in call_info:
            return

        try:
            # Resample to 8kHz if needed
            if sample_rate != 8000:
                import numpy as np
                import librosa
                audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=8000)
                pcm_data = (audio * 32767).astype(np.int16).tobytes()

            # Convert PCM to mulaw
            mulaw_data = audioop.lin2ulaw(pcm_data, 2)

            # Send as Twilio media message
            import json
            await ws.send_text(json.dumps({
                "event": "media",
                "streamSid": call_info["stream_sid"],
                "media": {
                    "payload": base64.b64encode(mulaw_data).decode("ascii"),
                },
            }))
        except Exception as e:
            print(f"[Twilio] Failed to send audio to caller: {e}")
```

**Step 2: Update WebSocket handler in main.py to register/unregister**

In the `twilio_media_stream` function, after the `event == "start"` block, add:

```python
if event == "start":
    stream_sid = msg["start"]["streamSid"]
    call_sid = msg["start"]["callSid"]
    twilio_service.register_websocket(call_sid, websocket)
    if call_sid in twilio_service.active_calls:
        twilio_service.active_calls[call_sid]["stream_sid"] = stream_sid
    print(f"[Twilio WS] Stream started: {stream_sid} for call {call_sid}")
```

In the `finally` block, add:

```python
finally:
    if call_sid:
        twilio_service.unregister_websocket(call_sid)
```

**Step 3: Send AI TTS audio to real caller**

In the `/api/tts` endpoint, after starting the playback thread, add code to also stream to any active real callers:

```python
# Also send to active real callers so they hear the AI
if session.active_real_caller:
    call_sid = session.active_real_caller["call_sid"]
    asyncio.create_task(
        twilio_service.send_audio_to_caller(call_sid, audio_bytes, 24000)
    )
```

**Step 4: Commit**

```bash
git add backend/main.py backend/services/twilio_service.py
git commit -m "Add outbound audio streaming to real callers"
```

---

## Task 7: AI Follow-Up System

**Files:**
- Modify: `backend/main.py`
- Create: `tests/test_followup.py`

**Step 1: Write tests**

Create `tests/test_followup.py`:

```python
import sys
sys.path.insert(0, "/Users/lukemacneil/ai-podcast")

from backend.main import Session, CallRecord, get_caller_prompt


def test_caller_prompt_includes_show_history():
    s = Session()
    s.call_history.append(CallRecord(
        caller_type="real", caller_name="Dave",
        summary="Called about his wife leaving after 12 years",
        transcript=[],
    ))

    # Simulate an active AI caller
    s.start_call("1")  # Tony
    caller = s.caller
    prompt = get_caller_prompt(caller, "", s.get_show_history())
    assert "Dave" in prompt
    assert "wife leaving" in prompt
    assert "EARLIER IN THE SHOW" in prompt
```

**Step 2: Update `get_caller_prompt` to accept show history**

In `backend/main.py`, modify `get_caller_prompt` signature and body:

```python
def get_caller_prompt(caller: dict, conversation_summary: str = "", show_history: str = "") -> str:
    context = ""
    if conversation_summary:
        context = f"""
CONVERSATION SO FAR:
{conversation_summary}
Continue naturally. Don't repeat yourself.
"""
    history = ""
    if show_history:
        history = f"\n{show_history}\n"

    return f"""You're {caller['name']}, calling a late-night radio show. You trust this host.

{caller['vibe']}
{history}{context}
HOW TO TALK:
...  # rest of the existing prompt unchanged
"""
```

**Step 3: Update `/api/chat` to include show history**

In the `/api/chat` endpoint:

```python
@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not session.caller:
        raise HTTPException(400, "No active call")

    session.add_message("user", request.text)

    conversation_summary = session.get_conversation_summary()
    show_history = session.get_show_history()
    system_prompt = get_caller_prompt(session.caller, conversation_summary, show_history)

    # ... rest unchanged
```

**Step 4: Add hangup endpoint for real callers with summarization**

```python
@app.post("/api/hangup/real")
async def hangup_real_caller():
    """Hang up on real caller — summarize call and store in history"""
    if not session.active_real_caller:
        raise HTTPException(400, "No active real caller")

    call_sid = session.active_real_caller["call_sid"]
    caller_name = session.active_real_caller["name"]

    # Summarize the conversation
    summary = ""
    if session.conversation:
        transcript_text = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in session.conversation
        )
        summary = await llm_service.generate(
            messages=[{"role": "user", "content": f"Summarize this radio show call in 1-2 sentences:\n{transcript_text}"}],
            system_prompt="You summarize radio show conversations concisely. Focus on what the caller talked about and any emotional moments.",
        )

    # Store in call history
    session.call_history.append(CallRecord(
        caller_type="real",
        caller_name=caller_name,
        summary=summary,
        transcript=list(session.conversation),
    ))

    # Clean up
    twilio_service.hangup(call_sid)

    # End the Twilio call
    from twilio.rest import Client as TwilioClient
    if settings.twilio_account_sid and settings.twilio_auth_token:
        try:
            client = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)
            client.calls(call_sid).update(status="completed")
        except Exception as e:
            print(f"[Twilio] Failed to end call: {e}")

    session.active_real_caller = None
    # Don't clear conversation — AI follow-up might reference it
    # Conversation gets cleared when next call starts

    # Play hangup sound
    hangup_sound = settings.sounds_dir / "hangup.wav"
    if hangup_sound.exists():
        audio_service.play_sfx(str(hangup_sound))

    # Auto follow-up?
    auto_followup_triggered = False
    if session.auto_followup:
        auto_followup_triggered = True
        asyncio.create_task(_auto_followup(summary))

    return {
        "status": "disconnected",
        "caller": caller_name,
        "summary": summary,
        "auto_followup": auto_followup_triggered,
    }


async def _auto_followup(last_call_summary: str):
    """Automatically pick an AI caller and connect them as follow-up"""
    await asyncio.sleep(7)  # Brief pause before follow-up

    # Ask LLM to pick best AI caller for follow-up
    caller_list = ", ".join(
        f'{k}: {v["name"]} ({v["gender"]}, {v["age_range"][0]}-{v["age_range"][1]})'
        for k, v in CALLER_BASES.items()
    )
    pick = await llm_service.generate(
        messages=[{"role": "user", "content": f'A caller just talked about: "{last_call_summary}". Which AI caller should follow up? Available: {caller_list}. Reply with just the key number.'}],
        system_prompt="Pick the most interesting AI caller to follow up on this topic. Just reply with the number key.",
    )

    # Extract key from response
    import re
    match = re.search(r'\d+', pick)
    if match:
        caller_key = match.group()
        if caller_key in CALLER_BASES:
            session.start_call(caller_key)
            print(f"[Auto Follow-Up] {CALLER_BASES[caller_key]['name']} is calling in about: {last_call_summary[:50]}...")
```

**Step 5: Add manual follow-up endpoint**

```python
@app.post("/api/followup/generate")
async def generate_followup():
    """Generate an AI follow-up caller based on recent show history"""
    if not session.call_history:
        raise HTTPException(400, "No call history to follow up on")

    last_record = session.call_history[-1]
    await _auto_followup(last_record.summary)

    return {
        "status": "followup_triggered",
        "based_on": last_record.caller_name,
    }
```

**Step 6: Run tests**

```bash
cd /Users/lukemacneil/ai-podcast && python -m pytest tests/test_followup.py -v
```

Expected: All PASS.

**Step 7: Commit**

```bash
git add backend/main.py tests/test_followup.py
git commit -m "Add AI follow-up system with call summarization and show history"
```

---

## Task 8: Frontend — Call Queue Panel

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/js/app.js`
- Modify: `frontend/css/style.css`

**Step 1: Add queue panel HTML**

In `frontend/index.html`, after the callers section (`</section>` at line 27) and before the chat section, add:

```html
<!-- Call Queue -->
<section class="queue-section">
    <h2>Incoming Calls</h2>
    <div id="call-queue" class="call-queue">
        <div class="queue-empty">No callers waiting</div>
    </div>
</section>
```

**Step 2: Add queue polling and UI to app.js**

Add to `initEventListeners()`:

```javascript
// Start queue polling
startQueuePolling();
```

Add new functions:

```javascript
// --- Call Queue ---
let queuePollInterval = null;

function startQueuePolling() {
    queuePollInterval = setInterval(fetchQueue, 3000);
    fetchQueue();
}

async function fetchQueue() {
    try {
        const res = await fetch('/api/queue');
        const data = await res.json();
        renderQueue(data.queue);
    } catch (err) {
        // Server might be down
    }
}

function renderQueue(queue) {
    const el = document.getElementById('call-queue');
    if (!el) return;

    if (queue.length === 0) {
        el.innerHTML = '<div class="queue-empty">No callers waiting</div>';
        return;
    }

    el.innerHTML = queue.map(caller => {
        const mins = Math.floor(caller.wait_time / 60);
        const secs = caller.wait_time % 60;
        const waitStr = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
        return `
            <div class="queue-item">
                <span class="queue-phone">${caller.phone}</span>
                <span class="queue-wait">waiting ${waitStr}</span>
                <button class="queue-take-btn" onclick="takeCall('${caller.call_sid}')">Take Call</button>
                <button class="queue-drop-btn" onclick="dropCall('${caller.call_sid}')">Drop</button>
            </div>
        `;
    }).join('');
}

async function takeCall(callSid) {
    try {
        const res = await fetch(`/api/queue/take/${callSid}`, { method: 'POST' });
        const data = await res.json();
        if (data.status === 'on_air') {
            log(`${data.caller.name} (${data.caller.phone}) is on air — Channel ${data.caller.channel}`);
            // Update active call UI
            updateActiveCallIndicator();
        }
    } catch (err) {
        log('Failed to take call: ' + err.message);
    }
}

async function dropCall(callSid) {
    try {
        await fetch(`/api/queue/drop/${callSid}`, { method: 'POST' });
        fetchQueue();
    } catch (err) {
        log('Failed to drop call: ' + err.message);
    }
}
```

**Step 3: Add queue CSS to style.css**

```css
/* Call Queue */
.queue-section { margin: 1rem 0; }

.call-queue {
    border: 1px solid #333;
    border-radius: 4px;
    padding: 0.5rem;
    max-height: 150px;
    overflow-y: auto;
}

.queue-empty {
    color: #666;
    text-align: center;
    padding: 0.5rem;
}

.queue-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.4rem 0.5rem;
    border-bottom: 1px solid #222;
}

.queue-item:last-child { border-bottom: none; }

.queue-phone {
    font-family: monospace;
    color: #4fc3f7;
}

.queue-wait {
    color: #999;
    font-size: 0.85rem;
    flex: 1;
}

.queue-take-btn {
    background: #2e7d32;
    color: white;
    border: none;
    padding: 0.25rem 0.75rem;
    border-radius: 3px;
    cursor: pointer;
}

.queue-take-btn:hover { background: #388e3c; }

.queue-drop-btn {
    background: #c62828;
    color: white;
    border: none;
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    cursor: pointer;
}

.queue-drop-btn:hover { background: #d32f2f; }
```

**Step 4: Commit**

```bash
git add frontend/index.html frontend/js/app.js frontend/css/style.css
git commit -m "Add call queue UI with take/drop controls"
```

---

## Task 9: Frontend — Active Call Indicator and AI Controls

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/js/app.js`
- Modify: `frontend/css/style.css`

**Step 1: Replace the existing call-status div with active call indicator**

In `frontend/index.html`, replace the call-status area in the callers section:

```html
<!-- Active Call Indicator -->
<div id="active-call" class="active-call hidden">
    <div id="real-caller-info" class="caller-info hidden">
        <span class="caller-type real">LIVE</span>
        <span id="real-caller-name"></span>
        <span id="real-caller-channel" class="channel-badge"></span>
        <span id="real-caller-duration" class="call-duration"></span>
        <button id="hangup-real-btn" class="hangup-btn small">Hang Up</button>
    </div>
    <div id="ai-caller-info" class="caller-info hidden">
        <span class="caller-type ai">AI</span>
        <span id="ai-caller-name"></span>
        <div class="ai-controls">
            <div class="mode-toggle">
                <button id="mode-manual" class="mode-btn active">Manual</button>
                <button id="mode-auto" class="mode-btn">Auto</button>
            </div>
            <button id="ai-respond-btn" class="respond-btn">Let them respond</button>
        </div>
        <button id="hangup-ai-btn" class="hangup-btn small">Hang Up</button>
    </div>
    <label class="auto-followup-label">
        <input type="checkbox" id="auto-followup"> Auto Follow-Up
    </label>
</div>
<div id="call-status" class="call-status">No active call</div>
```

**Step 2: Add active call indicator JS**

```javascript
// --- Active Call Indicator ---
let realCallerTimer = null;
let realCallerStartTime = null;

function updateActiveCallIndicator() {
    const container = document.getElementById('active-call');
    const realInfo = document.getElementById('real-caller-info');
    const aiInfo = document.getElementById('ai-caller-info');
    const statusEl = document.getElementById('call-status');

    const hasReal = !!document.getElementById('real-caller-name')?.textContent;
    const hasAi = !!currentCaller;

    if (hasReal || hasAi) {
        container?.classList.remove('hidden');
        statusEl?.classList.add('hidden');
    } else {
        container?.classList.add('hidden');
        statusEl?.classList.remove('hidden');
        statusEl.textContent = 'No active call';
    }
}

function showRealCaller(callerInfo) {
    const nameEl = document.getElementById('real-caller-name');
    const chEl = document.getElementById('real-caller-channel');
    if (nameEl) nameEl.textContent = `${callerInfo.name} (${callerInfo.phone})`;
    if (chEl) chEl.textContent = `Ch ${callerInfo.channel}`;

    document.getElementById('real-caller-info')?.classList.remove('hidden');
    realCallerStartTime = Date.now();

    // Start duration timer
    if (realCallerTimer) clearInterval(realCallerTimer);
    realCallerTimer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - realCallerStartTime) / 1000);
        const mins = Math.floor(elapsed / 60);
        const secs = elapsed % 60;
        const durEl = document.getElementById('real-caller-duration');
        if (durEl) durEl.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
    }, 1000);

    updateActiveCallIndicator();
}

function hideRealCaller() {
    document.getElementById('real-caller-info')?.classList.add('hidden');
    if (realCallerTimer) clearInterval(realCallerTimer);
    realCallerTimer = null;
    updateActiveCallIndicator();
}

// Wire up hangup-real-btn
document.getElementById('hangup-real-btn')?.addEventListener('click', async () => {
    await fetch('/api/hangup/real', { method: 'POST' });
    hideRealCaller();
    log('Real caller disconnected');
});

// Wire up AI respond mode toggle
document.getElementById('mode-manual')?.addEventListener('click', () => {
    document.getElementById('mode-manual')?.classList.add('active');
    document.getElementById('mode-auto')?.classList.remove('active');
    document.getElementById('ai-respond-btn')?.classList.remove('hidden');
    fetch('/api/session/ai-mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'manual' }),
    });
});

document.getElementById('mode-auto')?.addEventListener('click', () => {
    document.getElementById('mode-auto')?.classList.add('active');
    document.getElementById('mode-manual')?.classList.remove('active');
    document.getElementById('ai-respond-btn')?.classList.add('hidden');
    fetch('/api/session/ai-mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'auto' }),
    });
});

// Auto follow-up toggle
document.getElementById('auto-followup')?.addEventListener('change', (e) => {
    fetch('/api/session/auto-followup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: e.target.checked }),
    });
});
```

**Step 3: Add session control endpoints in main.py**

```python
@app.post("/api/session/ai-mode")
async def set_ai_mode(data: dict):
    """Set AI respond mode (manual or auto)"""
    mode = data.get("mode", "manual")
    session.ai_respond_mode = mode
    print(f"[Session] AI respond mode: {mode}")
    return {"mode": mode}


@app.post("/api/session/auto-followup")
async def set_auto_followup(data: dict):
    """Toggle auto follow-up"""
    session.auto_followup = data.get("enabled", False)
    print(f"[Session] Auto follow-up: {session.auto_followup}")
    return {"enabled": session.auto_followup}
```

**Step 4: Update the `takeCall` JS function to show real caller indicator**

In the `takeCall` function, after the success check:

```javascript
if (data.status === 'on_air') {
    showRealCaller(data.caller);
    log(`${data.caller.name} (${data.caller.phone}) is on air — Channel ${data.caller.channel}`);
}
```

**Step 5: Add CSS for active call indicator**

```css
/* Active Call Indicator */
.active-call {
    border: 1px solid #444;
    border-radius: 4px;
    padding: 0.75rem;
    margin: 0.5rem 0;
    background: #1a1a2e;
}

.caller-info {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}

.caller-info:last-of-type { margin-bottom: 0; }

.caller-type {
    font-size: 0.7rem;
    font-weight: bold;
    padding: 0.15rem 0.4rem;
    border-radius: 3px;
    text-transform: uppercase;
}

.caller-type.real { background: #c62828; color: white; }
.caller-type.ai { background: #1565c0; color: white; }

.channel-badge {
    font-size: 0.75rem;
    color: #999;
    background: #222;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
}

.call-duration {
    font-family: monospace;
    color: #4fc3f7;
}

.ai-controls {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-left: auto;
}

.mode-toggle {
    display: flex;
    border: 1px solid #444;
    border-radius: 3px;
    overflow: hidden;
}

.mode-btn {
    background: #222;
    color: #999;
    border: none;
    padding: 0.2rem 0.5rem;
    font-size: 0.75rem;
    cursor: pointer;
}

.mode-btn.active {
    background: #1565c0;
    color: white;
}

.respond-btn {
    background: #2e7d32;
    color: white;
    border: none;
    padding: 0.25rem 0.75rem;
    border-radius: 3px;
    font-size: 0.8rem;
    cursor: pointer;
}

.hangup-btn.small {
    font-size: 0.75rem;
    padding: 0.2rem 0.5rem;
}

.auto-followup-label {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
    color: #999;
    margin-top: 0.5rem;
}
```

**Step 6: Commit**

```bash
git add frontend/index.html frontend/js/app.js frontend/css/style.css backend/main.py
git commit -m "Add active call indicator with AI mode toggle and auto follow-up"
```

---

## Task 10: Frontend — Three-Party Chat Log

**Files:**
- Modify: `frontend/js/app.js`
- Modify: `frontend/css/style.css`

**Step 1: Update `addMessage` to support three-party roles**

Replace the existing `addMessage` function:

```javascript
function addMessage(sender, text) {
    const chat = document.getElementById('chat');
    if (!chat) {
        console.log(`[${sender}]: ${text}`);
        return;
    }
    const div = document.createElement('div');

    let className = 'message';
    if (sender === 'You') {
        className += ' host';
    } else if (sender === 'System') {
        className += ' system';
    } else if (sender.includes('(caller)') || sender.includes('Caller #')) {
        className += ' real-caller';
    } else {
        className += ' ai-caller';
    }

    div.className = className;
    div.innerHTML = `<strong>${sender}:</strong> ${text}`;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}
```

**Step 2: Add chat role colors to CSS**

```css
.message.real-caller {
    border-left: 3px solid #c62828;
    padding-left: 0.5rem;
}

.message.ai-caller {
    border-left: 3px solid #1565c0;
    padding-left: 0.5rem;
}

.message.host {
    border-left: 3px solid #2e7d32;
    padding-left: 0.5rem;
}
```

**Step 3: Commit**

```bash
git add frontend/js/app.js frontend/css/style.css
git commit -m "Add three-party chat log with color-coded roles"
```

---

## Task 11: Frontend — Caller Grid Three-Way Support

**Files:**
- Modify: `frontend/js/app.js`

**Step 1: Modify `startCall` to support adding AI as third party**

When a real caller is active and you click an AI caller, it should add the AI as a third party instead of replacing the call:

```javascript
async function startCall(key, name) {
    if (isProcessing) return;

    const res = await fetch(`/api/call/${key}`, { method: 'POST' });
    const data = await res.json();

    currentCaller = { key, name };

    // If real caller is active, show as three-way
    const realCallerActive = !document.getElementById('real-caller-info')?.classList.contains('hidden');

    if (realCallerActive) {
        document.getElementById('call-status').textContent = `Three-way: ${name} (AI) + Real Caller`;
    } else {
        document.getElementById('call-status').textContent = `On call: ${name}`;
    }

    document.getElementById('hangup-btn').disabled = false;

    // Show AI caller in active call indicator
    const aiInfo = document.getElementById('ai-caller-info');
    const aiName = document.getElementById('ai-caller-name');
    if (aiInfo) aiInfo.classList.remove('hidden');
    if (aiName) aiName.textContent = name;

    // Show caller background
    const bgEl = document.getElementById('caller-background');
    if (bgEl && data.background) {
        bgEl.textContent = data.background;
        bgEl.classList.remove('hidden');
    }

    document.querySelectorAll('.caller-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.key === key);
    });

    log(`Connected to ${name}` + (realCallerActive ? ' (three-way)' : ''));
    if (!realCallerActive) clearChat();

    updateActiveCallIndicator();
}
```

**Step 2: Commit**

```bash
git add frontend/js/app.js
git commit -m "Support three-way calls when clicking AI caller with real caller active"
```

---

## Task 12: Cloudflare Tunnel Setup

**Files:**
- Create: `docs/twilio-setup.md` (setup instructions, not code)

**Step 1: Document setup steps**

Create `docs/twilio-setup.md`:

```markdown
# Twilio + Cloudflare Tunnel Setup

## 1. Twilio Account
- Sign up at twilio.com
- Buy a phone number (~$1.15/mo)
- Note your Account SID and Auth Token from the dashboard

## 2. Environment Variables
Add to `.env`:
```
TWILIO_ACCOUNT_SID=ACxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxx
TWILIO_PHONE_NUMBER=+1xxxxxxxxxx
TWILIO_WEBHOOK_BASE_URL=https://radio.yourdomain.com
```

## 3. Cloudflare Tunnel
Create a tunnel that routes to your local server:

```bash
cloudflared tunnel create radio-show
cloudflared tunnel route dns radio-show radio.yourdomain.com
```

Run during shows:
```bash
cloudflared tunnel --url http://localhost:8000 run radio-show
```

Or add to your NAS Cloudflare tunnel config.

## 4. Twilio Webhook Config
In the Twilio console, configure your phone number:
- Voice webhook URL: `https://radio.yourdomain.com/api/twilio/voice`
- Method: POST

## 5. Test
1. Start the server: `./run.sh`
2. Start the tunnel: `cloudflared tunnel run radio-show`
3. Call your Twilio number from a phone
4. You should see the caller appear in the queue panel
```

**Step 2: Commit**

```bash
git add docs/twilio-setup.md
git commit -m "Add Twilio and Cloudflare tunnel setup docs"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Config + deps | `config.py`, `.env` |
| 2 | Session model (multi-party, history) | `main.py`, `tests/test_session.py` |
| 3 | Call queue service | `twilio_service.py`, `tests/test_twilio_service.py` |
| 4 | Twilio webhook endpoints | `main.py` |
| 5 | WebSocket media stream handler | `main.py`, `audio.py` |
| 6 | Outbound audio to real callers | `twilio_service.py`, `main.py` |
| 7 | AI follow-up system | `main.py`, `tests/test_followup.py` |
| 8 | Frontend: queue panel | `index.html`, `app.js`, `style.css` |
| 9 | Frontend: active call indicator | `index.html`, `app.js`, `style.css` |
| 10 | Frontend: three-party chat | `app.js`, `style.css` |
| 11 | Frontend: three-way caller grid | `app.js` |
| 12 | Cloudflare tunnel setup docs | `docs/twilio-setup.md` |

Tasks 1-7 are backend (do in order). Tasks 8-11 are frontend (can be done in parallel after task 7). Task 12 is independent.
