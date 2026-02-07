# SignalWire Phone Call-In Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace browser-based WebSocket call-in with real phone calls via SignalWire (208-439-5853).

**Architecture:** SignalWire hits our webhook on inbound calls, we return XML to open a bidirectional WebSocket stream with L16@16kHz audio. The existing queue, channel allocation, transcription, host mic streaming, and TTS streaming are reused — only the WebSocket message format changes (base64 JSON instead of raw binary).

**Tech Stack:** Python/FastAPI, SignalWire Compatibility API (LaML XML + WebSocket), httpx for REST calls, existing audio pipeline.

---

## Task 1: Add SignalWire Config

**Files:**
- Modify: `backend/config.py`
- Modify: `.env`

**Step 1: Add SignalWire settings to config.py**

In `backend/config.py`, add these fields to the `Settings` class after the existing API keys block (after line 16):

```python
    # SignalWire
    signalwire_project_id: str = os.getenv("SIGNALWIRE_PROJECT_ID", "")
    signalwire_space: str = os.getenv("SIGNALWIRE_SPACE", "")
    signalwire_token: str = os.getenv("SIGNALWIRE_TOKEN", "")
    signalwire_phone: str = os.getenv("SIGNALWIRE_PHONE", "")
```

**Step 2: Add SignalWire vars to .env**

Append to `.env`:

```
# SignalWire
SIGNALWIRE_PROJECT_ID=8eb54732-ade3-4487-8b40-ecd2cd680df7
SIGNALWIRE_SPACE=macneil-media-group-llc.signalwire.com
SIGNALWIRE_TOKEN=PT9c9b61f44ee49914c614fed32aa5c3d7b9372b5199d81dec
SIGNALWIRE_PHONE=+12084395853
```

**Step 3: Verify config loads**

```bash
cd /Users/lukemacneil/ai-podcast && python -c "from backend.config import settings; print(settings.signalwire_space)"
```

Expected: `macneil-media-group-llc.signalwire.com`

**Step 4: Commit**

```bash
git add backend/config.py .env
git commit -m "Add SignalWire configuration"
```

---

## Task 2: Update CallerService for SignalWire Protocol

**Files:**
- Modify: `backend/services/caller_service.py`

The CallerService currently sends raw binary PCM frames. SignalWire needs base64-encoded L16 PCM wrapped in JSON. Also swap `name` field to `phone` since callers now have phone numbers.

**Step 1: Update queue to use `phone` instead of `name`**

In `caller_service.py`, make these changes:

1. Update docstring (line 1): `"""Phone caller queue and audio stream service"""`

2. In `add_to_queue` (line 24): Change parameter `name` to `phone`, and update the dict:

```python
    def add_to_queue(self, caller_id: str, phone: str):
        with self._lock:
            self._queue.append({
                "caller_id": caller_id,
                "phone": phone,
                "queued_at": time.time(),
            })
        print(f"[Caller] {phone} added to queue (ID: {caller_id})")
```

3. In `get_queue` (line 38): Return `phone` instead of `name`:

```python
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
```

4. In `take_call` (line 62): Use `phone` instead of `name`:

```python
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
```

5. In `hangup` (line 89): Use `phone` instead of `name`:

```python
    def hangup(self, caller_id: str):
        call_info = self.active_calls.pop(caller_id, None)
        if call_info:
            self.release_channel(call_info["channel"])
            print(f"[Caller] {call_info['phone']} hung up — channel {call_info['channel']} released")
        self._websockets.pop(caller_id, None)
```

**Step 2: Update `send_audio_to_caller` for SignalWire JSON format**

Replace the existing `send_audio_to_caller` method with:

```python
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
```

**Step 3: Update `stream_audio_to_caller` for SignalWire JSON format**

Replace the existing `stream_audio_to_caller` method with:

```python
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
```

**Step 4: Remove `notify_caller` and `disconnect_caller` methods**

These sent browser-specific JSON control messages. SignalWire callers are disconnected via REST API (handled in main.py). Delete methods `notify_caller` (line 168) and `disconnect_caller` (line 175). They will be replaced with a REST-based hangup in Task 4.

**Step 5: Add `call_sid` tracking for SignalWire call hangup**

Add a dict to track SignalWire call SIDs so we can end calls via REST:

In `__init__`, after `self._websockets` line, add:

```python
        self._call_sids: dict[str, str] = {}  # caller_id -> SignalWire callSid
```

Add methods:

```python
    def register_call_sid(self, caller_id: str, call_sid: str):
        """Track SignalWire callSid for a caller"""
        self._call_sids[caller_id] = call_sid

    def get_call_sid(self, caller_id: str) -> str | None:
        """Get SignalWire callSid for a caller"""
        return self._call_sids.get(caller_id)

    def unregister_call_sid(self, caller_id: str):
        """Remove callSid tracking"""
        self._call_sids.pop(caller_id, None)
```

In `reset`, also clear `self._call_sids`:

```python
            self._call_sids.clear()
```

In `hangup`, also clean up call_sid:

```python
        self._call_sids.pop(caller_id, None)
```

**Step 6: Run existing tests**

```bash
cd /Users/lukemacneil/ai-podcast && python -m pytest tests/test_caller_service.py -v
```

Tests will likely need updates due to `name` → `phone` rename. Fix any failures.

**Step 7: Commit**

```bash
git add backend/services/caller_service.py
git commit -m "Update CallerService for SignalWire protocol"
```

---

## Task 3: Add SignalWire Voice Webhook

**Files:**
- Modify: `backend/main.py`

**Step 1: Add the voice webhook endpoint**

Add after the existing route definitions (after line 421), replacing the `/call-in` route:

```python
# --- SignalWire Endpoints ---

from fastapi import Request, Response

@app.post("/api/signalwire/voice")
async def signalwire_voice_webhook(request: Request):
    """Handle inbound call from SignalWire — return XML to start bidirectional stream"""
    form = await request.form()
    caller_phone = form.get("From", "Unknown")
    call_sid = form.get("CallSid", "")
    print(f"[SignalWire] Inbound call from {caller_phone} (CallSid: {call_sid})")

    # Build WebSocket URL from the request
    ws_scheme = "wss"
    host = request.headers.get("host", "radioshow.macneilmediagroup.com")
    stream_url = f"{ws_scheme}://{host}/api/signalwire/stream"

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_url}" codec="L16@16000h">
            <Parameter name="caller_phone" value="{caller_phone}"/>
            <Parameter name="call_sid" value="{call_sid}"/>
        </Stream>
    </Connect>
</Response>"""

    return Response(content=xml, media_type="application/xml")
```

**Step 2: Remove the `/call-in` route**

Delete these lines (around line 419-421):

```python
@app.get("/call-in")
async def call_in_page():
    return FileResponse(frontend_dir / "call-in.html")
```

**Step 3: Verify server starts**

```bash
cd /Users/lukemacneil/ai-podcast && python -c "from backend.main import app; print('OK')"
```

**Step 4: Commit**

```bash
git add backend/main.py
git commit -m "Add SignalWire voice webhook, remove call-in route"
```

---

## Task 4: Add SignalWire WebSocket Stream Handler

**Files:**
- Modify: `backend/main.py`

This replaces the browser caller WebSocket handler at `/api/caller/stream`.

**Step 1: Replace the browser WebSocket handler**

Delete the entire `caller_audio_stream` function (the `@app.websocket("/api/caller/stream")` handler, lines 807-887).

Add the new SignalWire WebSocket handler:

```python
@app.websocket("/api/signalwire/stream")
async def signalwire_audio_stream(websocket: WebSocket):
    """Handle SignalWire bidirectional audio stream"""
    await websocket.accept()

    caller_id = str(uuid.uuid4())[:8]
    caller_phone = "Unknown"
    call_sid = ""
    audio_buffer = bytearray()
    CHUNK_DURATION_S = 3
    SAMPLE_RATE = 16000
    chunk_samples = CHUNK_DURATION_S * SAMPLE_RATE
    stream_started = False

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "start":
                # Extract caller info from stream parameters
                params = {}
                for p in msg.get("start", {}).get("customParameters", {}):
                    pass
                # customParameters comes as a dict
                custom = msg.get("start", {}).get("customParameters", {})
                caller_phone = custom.get("caller_phone", "Unknown")
                call_sid = custom.get("call_sid", "")

                stream_started = True
                print(f"[SignalWire WS] Stream started: {caller_phone} (CallSid: {call_sid})")

                # Add to queue and register
                caller_service.add_to_queue(caller_id, caller_phone)
                caller_service.register_websocket(caller_id, websocket)
                if call_sid:
                    caller_service.register_call_sid(caller_id, call_sid)

            elif event == "media" and stream_started:
                # Decode base64 L16 PCM audio
                import base64
                payload = msg.get("media", {}).get("payload", "")
                if not payload:
                    continue

                pcm_data = base64.b64decode(payload)

                # Only process audio if caller is on air
                call_info = caller_service.active_calls.get(caller_id)
                if not call_info:
                    continue

                audio_buffer.extend(pcm_data)

                # Route to configured live caller Loopback channel
                audio_service.route_real_caller_audio(pcm_data, SAMPLE_RATE)

                # Transcribe when we have enough audio
                if len(audio_buffer) >= chunk_samples * 2:
                    pcm_chunk = bytes(audio_buffer[:chunk_samples * 2])
                    audio_buffer = audio_buffer[chunk_samples * 2:]
                    asyncio.create_task(
                        _handle_real_caller_transcription(caller_id, pcm_chunk, SAMPLE_RATE)
                    )

            elif event == "stop":
                print(f"[SignalWire WS] Stream stopped: {caller_phone}")
                break

    except WebSocketDisconnect:
        print(f"[SignalWire WS] Disconnected: {caller_id} ({caller_phone})")
    except Exception as e:
        print(f"[SignalWire WS] Error: {e}")
    finally:
        caller_service.unregister_websocket(caller_id)
        caller_service.unregister_call_sid(caller_id)
        caller_service.remove_from_queue(caller_id)
        if caller_id in caller_service.active_calls:
            caller_service.hangup(caller_id)
            if session.active_real_caller and session.active_real_caller.get("caller_id") == caller_id:
                session.active_real_caller = None
                if len(caller_service.active_calls) == 0:
                    audio_service.stop_host_stream()
        if audio_buffer:
            asyncio.create_task(
                _handle_real_caller_transcription(caller_id, bytes(audio_buffer), SAMPLE_RATE)
            )
```

**Step 2: Commit**

```bash
git add backend/main.py
git commit -m "Add SignalWire WebSocket stream handler, remove browser handler"
```

---

## Task 5: Update Hangup and Queue Endpoints for SignalWire

**Files:**
- Modify: `backend/main.py`

When the host hangs up or drops a caller, we need to end the actual phone call via SignalWire's REST API.

**Step 1: Add SignalWire hangup helper**

Add this function near the top of `main.py` (after imports):

```python
async def _signalwire_end_call(call_sid: str):
    """End a phone call via SignalWire REST API"""
    if not call_sid or not settings.signalwire_space:
        return
    try:
        url = f"https://{settings.signalwire_space}/api/laml/2010-04-01/Accounts/{settings.signalwire_project_id}/Calls/{call_sid}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                url,
                data={"Status": "completed"},
                auth=(settings.signalwire_project_id, settings.signalwire_token),
            )
            print(f"[SignalWire] End call {call_sid}: {response.status_code}")
    except Exception as e:
        print(f"[SignalWire] Failed to end call {call_sid}: {e}")
```

Also add `import httpx` at the top of main.py if not already present.

**Step 2: Update `take_call_from_queue`**

In the `take_call_from_queue` endpoint, update `name` references to `phone`:

```python
@app.post("/api/queue/take/{caller_id}")
async def take_call_from_queue(caller_id: str):
    """Take a caller off hold and put them on air"""
    try:
        call_info = caller_service.take_call(caller_id)
    except ValueError as e:
        raise HTTPException(404, str(e))

    session.active_real_caller = {
        "caller_id": call_info["caller_id"],
        "channel": call_info["channel"],
        "phone": call_info["phone"],
    }

    # Start host mic streaming if this is the first real caller
    if len(caller_service.active_calls) == 1:
        _start_host_audio_sender()
        audio_service.start_host_stream(_host_audio_sync_callback)

    return {
        "status": "on_air",
        "caller": call_info,
    }
```

Note: The `notify_caller` call is removed — SignalWire callers don't need a JSON status message, they're already connected via the phone.

**Step 3: Update `drop_from_queue`**

End the phone call when dropping:

```python
@app.post("/api/queue/drop/{caller_id}")
async def drop_from_queue(caller_id: str):
    """Drop a caller from the queue"""
    call_sid = caller_service.get_call_sid(caller_id)
    caller_service.remove_from_queue(caller_id)
    if call_sid:
        await _signalwire_end_call(call_sid)
    return {"status": "dropped"}
```

**Step 4: Update `hangup_real_caller`**

End the phone call when hanging up:

```python
@app.post("/api/hangup/real")
async def hangup_real_caller():
    """Hang up on real caller — disconnect immediately, summarize in background"""
    if not session.active_real_caller:
        raise HTTPException(400, "No active real caller")

    caller_id = session.active_real_caller["caller_id"]
    caller_phone = session.active_real_caller["phone"]
    conversation_snapshot = list(session.conversation)
    auto_followup_enabled = session.auto_followup

    # End the phone call via SignalWire
    call_sid = caller_service.get_call_sid(caller_id)
    caller_service.hangup(caller_id)
    if call_sid:
        asyncio.create_task(_signalwire_end_call(call_sid))

    # Stop host streaming if no more active callers
    if len(caller_service.active_calls) == 0:
        audio_service.stop_host_stream()

    session.active_real_caller = None

    # Play hangup sound in background
    import threading
    hangup_sound = settings.sounds_dir / "hangup.wav"
    if hangup_sound.exists():
        threading.Thread(target=audio_service.play_sfx, args=(str(hangup_sound),), daemon=True).start()

    # Summarize and store history in background
    asyncio.create_task(
        _summarize_real_call(caller_phone, conversation_snapshot, auto_followup_enabled)
    )

    return {
        "status": "disconnected",
        "caller": caller_phone,
    }
```

**Step 5: Update `_handle_real_caller_transcription`**

Change `caller_name` to `caller_phone`:

```python
async def _handle_real_caller_transcription(caller_id: str, pcm_data: bytes, sample_rate: int):
    """Transcribe a chunk of real caller audio and add to conversation"""
    call_info = caller_service.active_calls.get(caller_id)
    if not call_info:
        return

    text = await transcribe_audio(pcm_data, source_sample_rate=sample_rate)
    if not text or not text.strip():
        return

    caller_phone = call_info["phone"]
    print(f"[Real Caller] {caller_phone}: {text}")

    session.add_message(f"real_caller:{caller_phone}", text)

    if session.ai_respond_mode == "auto" and session.current_caller_key:
        asyncio.create_task(_check_ai_auto_respond(text, caller_phone))
```

**Step 6: Update `_summarize_real_call`**

Change `caller_name` parameter to `caller_phone`:

```python
async def _summarize_real_call(caller_phone: str, conversation: list, auto_followup_enabled: bool):
    """Background task: summarize call and store in history"""
    summary = ""
    if conversation:
        transcript_text = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in conversation
        )
        summary = await llm_service.generate(
            messages=[{"role": "user", "content": f"Summarize this radio show call in 1-2 sentences:\n{transcript_text}"}],
            system_prompt="You summarize radio show conversations concisely. Focus on what the caller talked about and any emotional moments.",
        )

    session.call_history.append(CallRecord(
        caller_type="real",
        caller_name=caller_phone,
        summary=summary,
        transcript=conversation,
    ))
    print(f"[Real Caller] {caller_phone} call summarized: {summary[:80]}...")

    if auto_followup_enabled:
        await _auto_followup(summary)
```

**Step 7: Update `_check_ai_auto_respond`**

Change parameter name from `real_caller_name` to `real_caller_phone`:

```python
async def _check_ai_auto_respond(real_caller_text: str, real_caller_phone: str):
```

(The body doesn't use the name/phone parameter in any way that needs changing.)

**Step 8: Update TTS streaming references**

In `text_to_speech` endpoint and `_check_ai_auto_respond`, the `session.active_real_caller` dict now uses `phone` instead of `name`. No code change needed for the TTS streaming since it only uses `caller_id`.

**Step 9: Verify server starts**

```bash
cd /Users/lukemacneil/ai-podcast && python -c "from backend.main import app; print('OK')"
```

**Step 10: Commit**

```bash
git add backend/main.py
git commit -m "Update hangup and queue endpoints for SignalWire REST API"
```

---

## Task 6: Update Frontend for Phone Callers

**Files:**
- Modify: `frontend/js/app.js`
- Modify: `frontend/index.html`

**Step 1: Update queue rendering in app.js**

In `renderQueue` function (around line 875), change `caller.name` to `caller.phone`:

```javascript
    el.innerHTML = queue.map(caller => {
        const mins = Math.floor(caller.wait_time / 60);
        const secs = caller.wait_time % 60;
        const waitStr = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
        return `
            <div class="queue-item">
                <span class="queue-name">${caller.phone}</span>
                <span class="queue-wait">waiting ${waitStr}</span>
                <button class="queue-take-btn" onclick="takeCall('${caller.caller_id}')">Take Call</button>
                <button class="queue-drop-btn" onclick="dropCall('${caller.caller_id}')">Drop</button>
            </div>
        `;
    }).join('');
```

**Step 2: Update `takeCall` log message**

In `takeCall` function (around line 896), change `data.caller.name` to `data.caller.phone`:

```javascript
        if (data.status === 'on_air') {
            showRealCaller(data.caller);
            log(`${data.caller.phone} is on air — Channel ${data.caller.channel}`);
        }
```

**Step 3: Update `showRealCaller` to use phone**

In `showRealCaller` function (around line 939):

```javascript
function showRealCaller(callerInfo) {
    const nameEl = document.getElementById('real-caller-name');
    const chEl = document.getElementById('real-caller-channel');
    if (nameEl) nameEl.textContent = callerInfo.phone;
    if (chEl) chEl.textContent = `Ch ${callerInfo.channel}`;
```

**Step 4: Update index.html queue section header**

In `frontend/index.html`, change the queue section header (line 56) — remove the call-in page link:

```html
            <section class="queue-section">
                <h2>Incoming Calls</h2>
                <div id="call-queue" class="call-queue">
```

**Step 5: Bump cache version in index.html**

Find the app.js script tag and bump the version:

```html
<script src="/js/app.js?v=13"></script>
```

**Step 6: Commit**

```bash
git add frontend/js/app.js frontend/index.html
git commit -m "Update frontend for phone caller display"
```

---

## Task 7: Remove Browser Call-In Files

**Files:**
- Delete: `frontend/call-in.html`
- Delete: `frontend/js/call-in.js`

**Step 1: Delete files**

```bash
cd /Users/lukemacneil/ai-podcast && rm frontend/call-in.html frontend/js/call-in.js
```

**Step 2: Commit**

```bash
git add frontend/call-in.html frontend/js/call-in.js
git commit -m "Remove browser call-in page"
```

---

## Task 8: Update Tests

**Files:**
- Modify: `tests/test_caller_service.py`

**Step 1: Update tests for `name` → `phone` rename**

Throughout `test_caller_service.py`, change:
- `add_to_queue(caller_id, "TestName")` → `add_to_queue(caller_id, "+15551234567")`
- `caller["name"]` → `caller["phone"]`
- `call_info["name"]` → `call_info["phone"]`

Also remove any tests for `notify_caller` or `disconnect_caller` if they exist, since those methods were removed.

**Step 2: Run all tests**

```bash
cd /Users/lukemacneil/ai-podcast && python -m pytest tests/ -v
```

Expected: All pass.

**Step 3: Commit**

```bash
git add tests/
git commit -m "Update tests for SignalWire phone caller format"
```

---

## Task 9: Configure SignalWire Webhook and End-to-End Test

**Step 1: Start the server**

```bash
cd /Users/lukemacneil/ai-podcast && python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Step 2: Verify webhook endpoint responds**

```bash
curl -X POST http://localhost:8000/api/signalwire/voice \
  -d "From=+15551234567&CallSid=test123" \
  -H "Content-Type: application/x-www-form-urlencoded"
```

Expected: XML response with `<Connect><Stream>` containing the WebSocket URL.

**Step 3: Verify Cloudflare tunnel is running**

```bash
curl -s https://radioshow.macneilmediagroup.com/api/server/status
```

Expected: JSON response with `"status": "running"`.

**Step 4: Configure SignalWire webhook**

In the SignalWire dashboard:
1. Go to Phone Numbers → 208-439-5853
2. Set "When a call comes in" to: `https://radioshow.macneilmediagroup.com/api/signalwire/voice`
3. Method: POST
4. Handler type: LaML Webhooks

**Step 5: Test with a real call**

Call 208-439-5853 from a phone. Expected:
1. Call connects (no ringing/hold — goes straight to stream)
2. Caller appears in queue on host dashboard with phone number
3. Host clicks "Take Call" → audio flows bidirectionally
4. Host clicks "Hang Up" → phone call ends

**Step 6: Commit any fixes needed**

```bash
git add -A
git commit -m "Final SignalWire integration fixes"
```

---

## Summary

| Task | What | Key Files |
|------|------|-----------|
| 1 | SignalWire config | `config.py`, `.env` |
| 2 | CallerService protocol update | `caller_service.py` |
| 3 | Voice webhook endpoint | `main.py` |
| 4 | WebSocket stream handler | `main.py` |
| 5 | Hangup/queue via REST API | `main.py` |
| 6 | Frontend phone display | `app.js`, `index.html` |
| 7 | Remove browser call-in | `call-in.html`, `call-in.js` |
| 8 | Update tests | `tests/` |
| 9 | Configure & test | SignalWire dashboard |

Tasks 1-5 are sequential backend. Task 6-7 are frontend (can parallel after task 5). Task 8 after task 2. Task 9 is final integration test.
