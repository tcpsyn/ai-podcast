# AI Podcast - Project Instructions

## Castopod (Podcast Publishing)
- **URL**: https://podcast.macneilmediagroup.com
- **Podcast handle**: `@LukeAtTheRoost`
- **API Auth**: Basic auth (credentials in .env: CASTOPOD_USERNAME, CASTOPOD_PASSWORD)
- **Container**: `castopod-castopod-1`
- **Database**: `castopod-mariadb-1` (user: castopod, db: castopod)

## Running the App
```bash
# Start backend — ALWAYS use --reload-dir to avoid CPU thrashing from file watchers
python -m uvicorn backend.main:app --reload --reload-dir backend --host 0.0.0.0 --port 8000

# Or use run.sh
./run.sh
```

## Publishing Episodes
```bash
python publish_episode.py ~/Desktop/episode.mp3
```

## Environment Variables
Required in `.env`:
- OPENROUTER_API_KEY
- ELEVENLABS_API_KEY (optional)
- INWORLD_API_KEY (for Inworld TTS)

## Post-Production Pipeline
- **Stem Recorder** (`backend/services/stem_recorder.py`): Records 5 WAV stems (host, caller, music, sfx, ads) during live shows. Uses lock-free deque architecture — audio callbacks just append to deques, a background writer thread drains to disk. `write()` for continuous streams (host mic, music, ads), `write_sporadic()` for burst sources (caller TTS, SFX) with time-aligned silence padding.
- **Audio hooks** in `backend/services/audio.py`: 7 tap points guarded by `if self.stem_recorder:`. Persistent mic stream (`start_stem_mic`/`stop_stem_mic`) runs during recording to capture host voice continuously, not just during push-to-talk.
- **API endpoints**: `POST /api/recording/start`, `POST /api/recording/stop` (auto-runs postprod in background thread), `POST /api/recording/process`
- **Frontend**: REC button in header with red pulse animation when recording
- **Post-prod script** (`postprod.py`): 6-step pipeline — load stems → gap removal → voice compression (ffmpeg acompressor) → music ducking → stereo mix → EBU R128 loudness normalization to -16 LUFS. All steps skippable via CLI flags.
- **Known issues resolved**: Lock-free recorder (old version used threading.Lock in audio callbacks causing crashes), scipy.signal.resample replaced with nearest-neighbor (was producing artifacts on small chunks), sys import bug in auto-postprod, host mic not captured without persistent stream

## LLM Settings
- `_pick_response_budget()` in main.py controls caller dialog token limits (150-450 tokens). MiniMax respects limits strictly — if responses seem short, check these values.
- Default max_tokens in llm.py is 300 (for non-caller uses)
- Grok (`x-ai/grok-4.3`) works well for natural dialog; MiniMax tends toward terse responses. Note `grok-4`, `grok-4-fast` and `grok-4.1-fast` were retired from OpenRouter — a retired id 404s, and `llm.py` swallows the error and returns empty text, so callers go silent with nothing in the logs. `tests/test_model_config.py` guards against reintroducing them.
- `generate_with_tools()` in llm.py supports OpenRouter function calling for the intern feature

## Caller Generation System
- **Two-stage pipeline**: (1) batch identity pregen via Sonnet 4.6 at session start, (2) live dialog via Sonnet 4.6 per turn. Cost ~$3-4/show.
- **Slim caller dict**: Populated once at `Session._pregenerate_backgrounds()` via `caller_gen.generate_batch()`. Keys: `name`, `age`, `voice`, `location`, `identity`, `situation`, `reason_calling`, `opening_line`, `secret_want`, `specific_details`, `emotional_register`. Stored in `session.caller_backgrounds[caller_key]`.
- **Dialog model**: Always Sonnet 4.6 via the `caller_dialog` category in `config.category_models`. No per-caller model routing — deleted in Phase 5B.
- **Prompt builder**: `get_caller_prompt(caller)` in main.py builds the slim system prompt from the dict; see `tests/test_caller_prompt.py` for the contract.
- **Regulars**: `backend/services/regulars_v2.py` loads lore from Obsidian markdown files for named recurring callers (e.g. Silas). The batch prompt optionally includes 2-3 active regulars per session.
- **Inter-caller awareness**: `get_show_history()` scores previous callers by keyword overlap with the current caller's `situation`/`reason_calling`. Reaction frequency scales with match strength (60%/35%/15%).
- **Caller memory**: Returning callers auto-promote from first-timers at ~5% probability after 8+ exchanges. `RegularCallerService` tracks summaries, relationships, arc state.
- **Call quality signals**: `_assess_call_quality()` captures exchange count, response length, host engagement, caller depth, natural ending.

## Devon (Intern Character)
- **Service**: `backend/services/intern.py` — persistent show character, not a caller
- **Personality**: 23-year-old NMSU grad, eager, slightly incompetent, gets yelled at. Voice: "Nate" (Inworld), no phone filter.
- **Tools**: web_search (SearXNG), get_headlines, fetch_webpage, wikipedia_lookup — via `generate_with_tools()` function calling
- **SearXNG**: runs on the NAS (`http://mmgnas:8888`), deployed via `deploy_searxng.sh`. URL is `settings.searxng_url` (env `SEARXNG_URL`). If Devon's web_search returns "Search failed", check the `searxng` container on mmgnas is Up. The config enables the JSON API + disables the bot limiter — both required for programmatic queries.
- **Endpoints**: `POST /api/intern/ask`, `/interject`, `/monitor`, `GET /api/intern/suggestion`, `POST /api/intern/suggestion/play`, `/dismiss`
- **Auto-monitoring**: Watches conversation every 15s during calls, buffers suggestions for host approval
- **Persistence**: `data/intern.json` stores lookup history
- **Frontend**: Ask Devon input (D key), Interject button, monitor toggle, suggestion indicator with Play/Dismiss

## Frontend Control Panel
- **Keyboard shortcuts**: 1-0 (callers), H (hangup), W (wrap up), M (music toggle), D (ask Devon), Escape (close modals)
- **Wrap It Up**: Amber button that signals callers to wind down gracefully. Reduces response budget, injects wrap-up signals, forces goodbye after 2 exchanges.
- **Caller info panel**: Shows identity, situation, signature detail, secret want during active calls
- **Caller buttons**: Populated from the slim caller background dicts
- **Pinned SFX**: Cheer/Applause/Boo always visible, rest collapsible
- **Visual polish**: Thinking pulse, call glow, compact media row, smoother transitions

## Website
- **Domain**: lukeattheroost.com (behind Cloudflare)
- **Analytics**: Cloudflare Web Analytics (enable in Cloudflare dashboard, no code changes needed)
- **Deploy**: `npx wrangler pages deploy website/ --project-name=lukeattheroost --branch=main`

## Podcast Workflow
- Publishing pipeline: episodes go through Castopod, CDN, website, YouTube, and social
- Always check Python venv is active and packages are installed before running publish scripts
- Episode numbering: check Castopod for the latest episode number, don't hardcode

## Scripts
- `publish_episode.py` — Transcribes audio, generates metadata (title, description, cover art), publishes to Castopod. Usage: `python publish_episode.py ~/Desktop/episode.mp3`
- `make_clips.py` — Two-pass clip extraction: fast Whisper transcription → LLM selects best moments → quality Whisper re-transcription for precise timestamps. Usage: `python make_clips.py ~/Desktop/episode.mp3 --count 3`
- `generate_milestone_images.py` — Generates social milestone images via Gemini Flash (requires GOOGLE_API_KEY)
- `post_milestone.py` — Posts milestone announcements to social platforms via Postiz
- `make_x_launch_assets.py` — Generates branded visual assets for X/Twitter (header, quote cards, intro/review graphics)
- `schedule_x_launch.py` — Schedules X/Twitter launch campaign posts via Postiz API

## Reaper Scripts
- `reaper/dialog_regions.lua` — Background script that polls `/tmp/reaper_state.txt` and creates colored regions (green=DIALOG, red=AD, blue=IDENT) as the backend writes state changes during recording
- `reaper/strip_silence_dialog.lua` — Post-production script: strips long silences from dialog regions, normalizes AD/IDENT/music volume, trims music to voice length with fade-out, mutes music during AD/IDENT regions

## Cost Dashboard
- **Route**: `/costs` — standalone analytics page, linked from control panel header
- **Database**: `data/costs.db` (SQLite) — aggregates all session cost data for cross-session queries
- **Data layer**: `backend/services/cost_db.py` — schema, JSON import, all query functions
- **Dual-write**: `cost_tracker.py` writes to both JSON (`data/cost_reports/`) and SQLite on every LLM/TTS call
- **API**: 8 endpoints under `/api/costs/` — summary, timeline, models, categories, sessions, session detail, expensive calls, TTS providers
- **Frontend**: `frontend/costs.html`, `frontend/css/costs.css`, `frontend/js/costs.js` — Chart.js for visualizations
- **Pricing**: Hardcoded in `cost_tracker.py` (`OPENROUTER_PRICING`, `TTS_PRICING`) — update when provider prices change
- **Not tracked yet**: SignalWire call costs

## Data Directory
State files (not config — these are written at runtime):
- `regulars.json` — Returning caller profiles (backgrounds, key moments, arc status, relationships)
- `used_topics_history.json` — Previously used caller topics to avoid repeats
- `session_checkpoint.json` — Current show session state (call history, caller queue)
- `publish_state.json` — Publishing pipeline progress per episode
- `intern.json` — Devon's lookup history
- `emails.json` — Listener email submissions
- `voicemails.json` — Listener voicemail submissions

## Personal
- Don't build anything until you have 95% clarity on what I want you to do. Ask clarifying questions until you reach 95% understanding of what I'm asking
- When working as a team, propose the plan before executing — don't just start building
- Flag trade-offs that affect show quality or listener experience rather than silently resolving them
