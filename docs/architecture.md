# Luke at the Roost — Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BROWSER (Control Panel)                          │
│                                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ ┌───────────────┐  │
│  │ Caller   │ │  Chat    │ │  Music/  │ │Settings│ │  Server Log   │  │
│  │ Buttons  │ │  Window  │ │  Ads/SFX │ │ Modal  │ │  (live tail)  │  │
│  │ (0-9)    │ │          │ │          │ │        │ │               │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ └───────┬───────┘  │
│       │            │            │            │              │           │
│  ┌────┴────────────┴────────────┴────────────┴──────────────┴───────┐  │
│  │                    frontend/js/app.js                             │  │
│  │  Polling: queue (3s), chat updates (real-time), logs (1s)        │  │
│  │  Push-to-talk: record/stop → transcribe → chat → TTS → play     │  │
│  └──────────────────────────┬───────────────────────────────────────┘  │
└─────────────────────────────┼───────────────────────────────────────────┘
                              │ REST API + WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (main.py)                            │
│                     uvicorn :8000                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Caller Generation Pipeline

```
Session Reset / First Access to Caller Slot
    │
    ▼
_randomize_callers()
    │  Assigns unique names (from 24M/24F pool) and voices (5M/5F) to 10 slots
    │
    ▼
generate_caller_background(base)
    │
    ├─ Demographics: age (from range), job (gendered pool), location
    │                                                        │
    │                              ┌─────────────────────────┘
    │                              ▼
    │                     pick_location()
    │                     80% LOCATIONS_LOCAL (weighted: Animas, Lordsburg)
    │                     20% LOCATIONS_OUT_OF_STATE
    │                              │
    │                              ▼
    │                     _get_town_from_location()
    │                     └─ TOWN_KNOWLEDGE[town]
    │                        32 towns with real facts
    │                        "Only reference real places..."
    │
    ├─ 70% → PROBLEMS (100+ templates)
    │        Fill {affair_person}, {fantasy_subject}, etc. from PROBLEM_FILLS
    │
    ├─ 30% → TOPIC_CALLIN (61 entries)
    │        Prestige TV, science, poker, photography, physics, US news
    │
    ├─ 2x random INTERESTS (86 entries: TV shows, science, tech, poker, etc.)
    │
    └─ 2x random QUIRKS (conversational style traits)
    │
    ▼
Result: "43, works IT for the city in Lordsburg. Just finished Severance
        season 2... Follows JWST discoveries... Deflects with humor...
        ABOUT WHERE THEY LIVE (Lordsburg): Small town on I-10, about 2,500
        people... Only reference real places..."
```

### News Enrichment (at pickup time)

```
POST /api/call/{key}
    │
    ▼
enrich_caller_background(background)     ← 5s timeout, fails silently
    │
    ├─ _extract_search_query(background)
    │   ├─ Check _TOPIC_SEARCH_MAP (50+ keyword→query mappings)
    │   │   "severance" → "Severance TV show"
    │   │   "quantum"   → "quantum physics research"
    │   │   "poker"     → "poker tournament"
    │   │
    │   └─ Fallback: extract keywords from problem sentence
    │
    ▼
SearXNG (localhost:8888)
    │  /search?q=...&format=json&categories=news
    │
    ▼
LLM summarizes headline+snippet → natural one-liner
    │  "Recently read about how Severance ties up the Lumon mystery"
    │
    ▼
Appended to background: "..., and it's been on their mind."
```

---

## AI Caller Conversation Flow

```
    Host speaks (push-to-talk or type)
        │
        ▼
POST /api/record/start → record from input device
POST /api/record/stop  → transcribe (Whisper @ 16kHz)
        │
        ▼
POST /api/chat { text }
        │
        ├─ session.add_message("user", text)
        │
        ├─ Build system prompt: get_caller_prompt()
        │   ├─ Caller identity + background + town knowledge
        │   ├─ Show history (summaries of previous callers)
        │   ├─ Conversation summary (last 6 messages)
        │   └─ HOW TO TALK rules (varied length, no rehashing, etc.)
        │
        ├─ Last 10 messages → _normalize_messages_for_llm()
        │
        ▼
LLMService.generate(messages, system_prompt)
        │
        ├─ OpenRouter: primary model (15s timeout)
        ├─ Fallback 1: gemini-flash-1.5 (10s)
        ├─ Fallback 2: gpt-4o-mini (10s)
        ├─ Fallback 3: llama-3.1-8b (10s)
        └─ Last resort: "Sorry, I totally blanked out..."
        │
        ▼
clean_for_tts()              → strip (actions), *gestures*, fix phonetics
ensure_complete_thought()    → trim to last complete sentence
        │
        ▼
Response returned to frontend
        │
        ▼
POST /api/tts { text, voice_id }
        │
        ▼
generate_speech(text, voice_id)
        │
        ├─ Inworld (default cloud)     ─┐
        ├─ ElevenLabs (cloud)           │
        ├─ F5-TTS (local, cloned)       ├─→ PCM audio bytes (24kHz)
        ├─ Kokoro MLX (local, fast)     │
        ├─ ChatTTS / StyleTTS2 / etc.  ─┘
        │
        ▼
AudioService.play_caller_audio(bytes, 24000)
        │
        └─→ Output Device Channel 1 (caller TTS)
```

---

## Real Caller (Phone) Flow

```
Caller dials 208-439-LUKE
        │
        ▼
SignalWire routes to webhook
        │
        ▼
POST /api/signalwire/voice
        │
        ├─ If OFF AIR → play message + hangup
        │
        └─ If ON AIR → return BXML:
           <Stream url="wss://.../api/signalwire/stream" codec="L16@16000h">
        │
        ▼
WebSocket /api/signalwire/stream connects
        │
        ├─ "start" event → add to queue, play ring SFX
        │                   broadcast_event("caller_queued")
        │
        │   [Caller waits in queue until host takes them]
        │
        ├─ Host clicks "Take Call" in UI
        │   POST /api/queue/take/{caller_id}
        │   └─ CallerService.take_call() → allocate channel
        │   └─ Start host mic streaming → _host_audio_sender()
        │
        ├─ "media" events (continuous) ← caller's voice
        │   │
        │   ├─ route_real_caller_audio(pcm) → Ch 9 (host monitoring)
        │   │
        │   └─ Buffer 3s chunks → transcribe (Whisper)
        │       │
        │       └─ broadcast_chat() → appears in chat window
        │
        │   Host mic audio → _host_audio_sync_callback()
        │   │
        │   └─ _host_audio_sender() → CallerService.send_audio_to_caller()
        │       └─ base64 encode → WebSocket → SignalWire → caller's phone
        │
        │   If AI caller also active (auto-respond mode):
        │   │
        │   └─ _debounced_auto_respond() (4s silence)
        │       └─ LLM → TTS → play on Ch 1 + stream to real caller
        │
        ├─ Host hangs up
        │   POST /api/hangup/real
        │   └─ _signalwire_end_call(call_sid) → end phone call
        │   └─ _summarize_real_call() → LLM summary → call_history
        │   └─ Optional: _auto_followup() → pick AI caller to continue
        │
        └─ "stop" event or disconnect → cleanup
```

---

## Audio Routing (Multi-Channel Output)

```
All audio goes to ONE physical output device (Loopback/interface)
Each content type on a separate channel for mixing in DAW/OBS

┌─────────────────────────────────────────────────────────────┐
│                   Output Device (e.g. Loopback 16ch)        │
│                                                             │
│   Ch 1  ◄── Caller TTS (AI voices)          play_caller_audio()
│   Ch 2  ◄── Music (loops)                   play_music()
│   Ch 3  ◄── Sound Effects (one-shots)       play_sfx()
│   Ch 9  ◄── Live Caller Audio (monitoring)  route_real_caller_audio()
│   Ch 11 ◄── Ads (one-shots, no loop)        play_ad()
│                                                             │
│   All channels configurable via Settings panel              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Input Device (mic/interface)               │
│                                                             │
│   Ch N  ──► Host mic recording (push-to-talk)               │
│         ──► Host mic streaming (to real callers via WS)     │
└─────────────────────────────────────────────────────────────┘
```

---

## External Services

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  SignalWire   │     │  OpenRouter   │     │   SearXNG    │
│              │     │              │     │  (local)     │
│  Phone calls │     │  LLM API     │     │  News search │
│  REST + WS   │     │  Claude,GPT  │     │  :8888       │
│  Bidirectional│     │  Gemini,Llama│     │              │
│  audio stream│     │  Fallback    │     │              │
└──────────────┘     └──────────────┘     └──────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Inworld    │     │  ElevenLabs  │     │  Local TTS   │
│              │     │              │     │              │
│  TTS (cloud) │     │  TTS (cloud) │     │  Kokoro MLX  │
│  Default     │     │  Premium     │     │  F5-TTS      │
│  provider    │     │              │     │  ChatTTS     │
│              │     │              │     │  + others    │
└──────────────┘     └──────────────┘     └──────────────┘

┌──────────────┐
│  Castopod    │
│              │
│  Podcast     │
│  publishing  │
│  (NAS)       │
└──────────────┘
```

---

## Session Lifecycle

```
New Session (reset)
    │
    ├─ Randomize all 10 caller names + voices
    ├─ Clear conversation, call history, research
    ├─ New session ID
    │
    ▼
Show goes ON AIR (toggle)
    │
    ├─ SignalWire starts accepting calls
    │
    ▼
Caller interactions (loop)
    │
    ├─ Pick AI caller (click button 0-9)
    │   ├─ Generate background (if first time this session)
    │   ├─ Enrich with news (SearXNG → LLM summary)
    │   ├─ Conversation loop (chat/respond/auto-respond)
    │   └─ Hangup → summarize → add to call_history
    │
    ├─ Take real caller from queue
    │   ├─ Route audio both directions
    │   ├─ Transcribe caller speech in real-time
    │   ├─ Optional: AI caller auto-responds to real caller
    │   └─ Hangup → summarize → add to call_history
    │
    ├─ Play music / ads / SFX between calls
    │
    └─ Each new caller sees show_history (summaries of all previous calls)
        "EARLIER IN THE SHOW: Tony talked about... Carmen discussed..."
    │
    ▼
Show goes OFF AIR
    │
    └─ Incoming calls get off-air message + hangup
```

---

## Key Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Epoch-based staleness** | `_session_epoch` in main.py | Prevents stale LLM/TTS responses from playing after hangup |
| **Fallback chain** | LLMService | Guarantees a response even if primary model times out |
| **Debounced auto-respond** | `_debounced_auto_respond()` | Waits 4s for real caller to stop talking before AI jumps in |
| **Silent failure** | News enrichment | If search/LLM fails, caller just doesn't have news context |
| **Threading for audio** | `play_caller_audio()` | Audio playback can't block the async event loop |
| **Ring buffer** | `route_real_caller_audio()` | Absorbs jitter in real caller audio stream |
| **Lock contention guard** | `_ai_response_lock` | Only one AI response generates at a time |
| **Town knowledge injection** | `TOWN_KNOWLEDGE` dict | Prevents LLM from inventing fake local businesses |

---

## File Map

```
ai-podcast/
├── backend/
│   ├── main.py              ← FastAPI app, all endpoints, caller generation, session
│   ├── config.py            ← Settings (env vars, paths)
│   └── services/
│       ├── audio.py         ← Multi-channel audio I/O (sounddevice)
│       ├── caller_service.py← Phone queue, WebSocket registry, audio routing
│       ├── llm.py           ← OpenRouter/Ollama with fallback chain
│       ├── news.py          ← SearXNG search + caching
│       ├── tts.py           ← 8 TTS providers (cloud + local)
│       └── transcription.py ← Whisper speech-to-text
├── frontend/
│   ├── index.html           ← Control panel layout
│   ├── js/app.js            ← UI logic, polling, event handlers
│   └── css/style.css        ← Dark theme styling
├── sounds/                  ← SFX files (ring, hangup, busy, etc.)
├── music/                   ← Background music tracks
├── ads/                     ← Ad audio files
├── website/                 ← Landing page (lukeattheroost.com)
├── publish_episode.py       ← Castopod episode publisher
└── run.sh                   ← Server launcher with restart support
```
