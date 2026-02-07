# Real Callers + AI Follow-Up Design

## Overview

Add real phone callers to the AI Radio Show via Twilio, alongside existing AI callers. Real callers dial a phone number, wait in a hold queue, and get taken on air by the host. Three-way conversations between host, real caller, and AI caller are supported. AI follow-up callers automatically reference what real callers said.

## Requirements

- Real callers connect via Twilio phone number
- Full-duplex audio — host and caller talk simultaneously, talk over each other
- Each real caller gets their own dedicated audio channel for recording
- Three-way calls: host + real caller + AI caller all live at once
- AI caller can respond manually (host-triggered) or automatically (listens and decides when to jump in)
- AI follow-up callers reference real caller conversations via show history
- Auto follow-up mode: system picks an AI caller and connects them after a real call
- Simple hold queue — callers wait with hold music, host sees list and picks who goes on air
- Twilio webhooks exposed via Cloudflare tunnel

## Architecture

### Audio Routing (Loopback Channels)

```
Ch 1:  Host mic (existing)
Ch 2:  AI callers / TTS (existing)
Ch 3+: Real callers (dynamically assigned per call)
Ch N-1: Music (existing)
Ch N:   SFX (existing)
```

### Call Flow — Real Caller

```
Caller dials Twilio number
  → Twilio POST /api/twilio/voice
  → TwiML response: greeting + enqueue with hold music
  → Caller waits in hold queue
  → Host sees caller in dashboard queue panel
  → Host clicks "Take Call"
  → POST /api/queue/take/{call_sid}
  → Twilio opens WebSocket to /api/twilio/stream
  → Bidirectional audio:
      Caller audio → decode mulaw → dedicated Loopback channel
      Host audio + AI TTS → encode mulaw → Twilio → caller hears both
  → Real-time Whisper transcription of caller audio
  → Host hangs up → call summarized → stored in show history
```

### Three-Way Call Flow

```
Host mic ──────→ Ch 1 (recording)
               → Twilio outbound (real caller hears you)
               → Whisper transcription (AI gets your words)

Real caller ──→ Ch 3+ (recording, dedicated channel)
               → Whisper transcription (AI gets their words)
               → Host headphones

AI TTS ───────→ Ch 2 (recording)
               → Twilio outbound (real caller hears AI)
               → Host headphones (already works)
```

Conversation history becomes three-party with role labels: `host`, `real_caller`, `ai_caller`.

### AI Auto-Respond Mode

When toggled on, after each real caller transcription chunk:

1. Lightweight LLM call ("should I respond?" — use fast model like Haiku)
2. If YES → full response generated → TTS → plays on AI channel + streams to Twilio
3. Cooldown (~10s) prevents rapid-fire
4. Host can override with mute button

### AI Follow-Up System

After a real caller hangs up:

1. Full transcript (host + real caller + any AI) summarized by LLM
2. Summary stored in `session.call_history`
3. Next AI caller's system prompt includes show history:
   ```
   EARLIER IN THE SHOW:
   - Dave (real caller) called about his wife leaving after 12 years.
     He got emotional about his kids.
   - Jasmine called about her boss hitting on her at work.
   You can reference these if it feels natural. Don't force it.
   ```

**Host-triggered (default):** Click any AI caller as normal. They already have show context.

**Auto mode:** After real caller hangs up, system waits ~5-10s, picks a fitting AI caller via short LLM call, biases their background generation toward the topic, auto-connects.

## Backend Changes

### New Module: `backend/services/twilio_service.py`

Manages Twilio integration:
- WebSocket handler for Media Streams (decode/encode mulaw 8kHz ↔ PCM)
- Call queue state (waiting callers, SIDs, timestamps, assigned channels)
- Channel pool management (allocate/release Loopback channels for real callers)
- Outbound audio mixing (host + AI TTS → mulaw → Twilio)
- Methods: `take_call()`, `hangup_real_caller()`, `get_queue()`, `send_audio_to_caller()`

### New Endpoints

```python
# Twilio webhooks
POST /api/twilio/voice            # Incoming call → TwiML (greet + enqueue)
POST /api/twilio/hold-music       # Hold music TwiML for waiting callers
WS   /api/twilio/stream           # Media Streams WebSocket (bidirectional audio)

# Host controls
GET  /api/queue                    # List waiting callers (number, wait time)
POST /api/queue/take/{call_sid}    # Dequeue caller → start media stream
POST /api/queue/drop/{call_sid}    # Drop caller from queue

# AI follow-up
POST /api/followup/generate        # Summarize last real call, trigger AI follow-up
```

### Session Model Changes

```python
class CallRecord:
    caller_type: str          # "ai" or "real"
    caller_name: str          # "Tony" or "Caller #3"
    summary: str              # LLM-generated summary after hangup
    transcript: list[dict]    # Full conversation [{role, content}]

class Session:
    # Existing fields...
    call_history: list[CallRecord]     # All calls this episode
    active_real_caller: dict | None    # {call_sid, phone, channel, name}
    active_ai_caller: str | None       # Caller key
    ai_respond_mode: str               # "manual" or "auto"
    auto_followup: bool                # Auto-generate AI follow-up after real calls
```

Three-party conversation history uses roles: `host`, `real_caller:{name}`, `ai_caller:{name}`.

### AI Caller Prompt Changes

`get_caller_prompt()` extended to include:
- Show history from `session.call_history`
- Current real caller context (if three-way call active)
- Instructions for referencing real callers naturally

## Frontend Changes

### New: Call Queue Panel

Between callers section and chat. Shows waiting real callers with phone number and wait time. "Take Call" and "Drop" buttons per caller. Polls `/api/queue` every few seconds.

### Modified: Active Call Indicator

Shows real caller and AI caller simultaneously when both active:
- Real caller: name, channel number, call duration, hang up button
- AI caller: name, Manual/Auto toggle, "Let [name] respond" button (manual mode)
- Auto Follow-Up checkbox

### Modified: Chat Log

Three-party with visual distinction:
- Host messages: existing style
- Real caller: labeled "Dave (caller)", distinct color
- AI caller: labeled "Tony (AI)", distinct color

### Modified: Caller Grid

When real caller is active, clicking an AI caller adds them as third party instead of starting fresh call. Indicator shows which AI callers have been on the show this session.

## Dependencies

- `twilio` Python package (for TwiML generation, REST API)
- Twilio account with phone number (~$1.15/mo + per-minute)
- Cloudflare tunnel for exposing webhook endpoints
- `audioop` or equivalent for mulaw encode/decode (stdlib in Python 3.11)

## Configuration

New env vars in `.env`:
```
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1...
TWILIO_WEBHOOK_BASE_URL=https://your-tunnel.cloudflare.com
```
