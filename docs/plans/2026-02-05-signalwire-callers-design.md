# SignalWire Phone Call-In Design

## Goal

Replace browser-based WebSocket call-in with real phone calls via SignalWire. Callers dial 208-439-5853 and enter the show queue.

## Architecture

SignalWire handles PSTN connectivity. When a call comes in, SignalWire hits our webhook, we return XML telling it to open a bidirectional WebSocket stream with L16@16kHz audio. The audio flows through our existing pipeline — same queue, channel allocation, transcription, host mic streaming, and TTS streaming.

## Call Flow

1. Caller dials 208-439-5853
2. SignalWire hits `POST /api/signalwire/voice` (via Cloudflare tunnel)
3. We return `<Connect><Stream codec="L16@16000h">` XML
4. SignalWire opens WebSocket to `/api/signalwire/stream`
5. Caller enters queue — host sees phone number on dashboard
6. Host takes call — audio flows bidirectionally
7. Host hangs up — we call SignalWire REST API to end the phone call

## Audio Path

```
Phone → PSTN → SignalWire → WebSocket (base64 L16 JSON) → Our server
Our server → WebSocket (base64 L16 JSON) → SignalWire → PSTN → Phone
```

## SignalWire WebSocket Protocol

Incoming: `{"event": "media", "media": {"payload": "<base64 L16 PCM 16kHz>"}}`
Outgoing: `{"event": "media", "media": {"payload": "<base64 L16 PCM 16kHz>"}}`
Start: `{"event": "start", "start": {"streamSid": "...", "callSid": "..."}}`
Stop: `{"event": "stop"}`

## What Changes

- Remove: browser call-in page, browser WebSocket handler
- Add: SignalWire webhook + WebSocket handler, hangup via REST API
- Modify: CallerService (name→phone, base64 JSON encoding for send), dashboard (show phone number)
- Unchanged: AudioService, queue logic, transcription, TTS streaming, three-way calls

## Config

```
SIGNALWIRE_PROJECT_ID=8eb54732-ade3-4487-8b40-ecd2cd680df7
SIGNALWIRE_SPACE=macneil-media-group-llc.signalwire.com
SIGNALWIRE_TOKEN=PT9c9b61f44ee49914c614fed32aa5c3d7b9372b5199d81dec
SIGNALWIRE_PHONE=+12084395853
```

Webhook URL: `https://radioshow.macneilmediagroup.com/api/signalwire/voice`
No SDK needed — httpx for the one REST call (hangup).
