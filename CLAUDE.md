# AI Podcast - Project Instructions

## Git Remote (Gitea)
- **Repo**: `git@gitea-nas:luke/ai-podcast.git`
- **Web**: http://mmgnas:3000/luke/ai-podcast
- **SSH Host**: `gitea-nas` (configured in ~/.ssh/config)
  - HostName: `mmgnas` (use `mmgnas-10g` if wired connection issues)
  - Port: `2222`
  - User: `git`
  - IdentityFile: `~/.ssh/gitea_mmgnas`

## NAS Access
- **Hostname**: `mmgnas` (wireless) or `mmgnas-10g` (wired/10G)
- **SSH Port**: 8001
- **User**: luke
- **Docker path**: `/share/CACHEDEV1_DATA/.qpkg/container-station/bin/docker`

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

## Post-Production Pipeline (added Feb 2026)
- **Branch**: `feature/real-callers` — all current work is here, pushed to gitea
- **Stem Recorder** (`backend/services/stem_recorder.py`): Records 5 WAV stems (host, caller, music, sfx, ads) during live shows. Uses lock-free deque architecture — audio callbacks just append to deques, a background writer thread drains to disk. `write()` for continuous streams (host mic, music, ads), `write_sporadic()` for burst sources (caller TTS, SFX) with time-aligned silence padding.
- **Audio hooks** in `backend/services/audio.py`: 7 tap points guarded by `if self.stem_recorder:`. Persistent mic stream (`start_stem_mic`/`stop_stem_mic`) runs during recording to capture host voice continuously, not just during push-to-talk.
- **API endpoints**: `POST /api/recording/start`, `POST /api/recording/stop` (auto-runs postprod in background thread), `POST /api/recording/process`
- **Frontend**: REC button in header with red pulse animation when recording
- **Post-prod script** (`postprod.py`): 6-step pipeline — load stems → gap removal → voice compression (ffmpeg acompressor) → music ducking → stereo mix → EBU R128 loudness normalization to -16 LUFS. All steps skippable via CLI flags.
- **Known issues resolved**: Lock-free recorder (old version used threading.Lock in audio callbacks causing crashes), scipy.signal.resample replaced with nearest-neighbor (was producing artifacts on small chunks), sys import bug in auto-postprod, host mic not captured without persistent stream

## LLM Settings
- `_pick_response_budget()` in main.py controls caller dialog token limits (150-450 tokens). MiniMax respects limits strictly — if responses seem short, check these values.
- Default max_tokens in llm.py is 300 (for non-caller uses)
- Grok (`x-ai/grok-4-fast`) works well for natural dialog; MiniMax tends toward terse responses
- `generate_with_tools()` in llm.py supports OpenRouter function calling for the intern feature

## Caller Generation System
- **Two-stage pipeline**: (1) batch identity pregen via Sonnet 4.6 at session start, (2) live dialog via Haiku 4.5 per turn. Cost ~$1/show.
- **Slim caller dict**: Populated once at `Session._pregenerate_backgrounds()` via `caller_gen.generate_batch()`. Keys: `name`, `age`, `voice`, `location`, `identity`, `situation`, `reason_calling`, `opening_line`, `secret_want`, `specific_details`, `emotional_register`. Stored in `session.caller_backgrounds[caller_key]`.
- **Dialog model**: Always Haiku 4.5 via the `caller_dialog` category in `config.category_models`. No per-caller model routing — deleted in Phase 5B.
- **Prompt builder**: `get_caller_prompt(caller)` in main.py builds the slim system prompt from the dict; see `tests/test_caller_prompt.py` for the contract.
- **Regulars**: `backend/services/regulars_v2.py` loads lore from Obsidian markdown files for named recurring callers (e.g. Silas). The batch prompt optionally includes 2-3 active regulars per session.
- **Inter-caller awareness**: `get_show_history()` scores previous callers by keyword overlap with the current caller's `situation`/`reason_calling`. Reaction frequency scales with match strength (60%/35%/15%).
- **Caller memory**: Returning callers auto-promote from first-timers at ~5% probability after 8+ exchanges. `RegularCallerService` tracks summaries, relationships, arc state.
- **Call quality signals**: `_assess_call_quality()` captures exchange count, response length, host engagement, caller depth, natural ending.

## Devon (Intern Character)
- **Service**: `backend/services/intern.py` — persistent show character, not a caller
- **Personality**: 23-year-old NMSU grad, eager, slightly incompetent, gets yelled at. Voice: "Nate" (Inworld), no phone filter.
- **Tools**: web_search (SearXNG), get_headlines, fetch_webpage, wikipedia_lookup — via `generate_with_tools()` function calling
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

## Git Push
- If `mmgnas` times out, use the 10g hostname:
  ```bash
  GIT_SSH_COMMAND="ssh -o HostName=mmgnas-10g -p 2222 -i ~/.ssh/gitea_mmgnas" git push origin main
  ```

## Hetzner VPS
- **IP**: `REDACTED_VPS_IP`
- **SSH**: `ssh root@REDACTED_VPS_IP` (uses default key `~/.ssh/id_rsa`)
- **Specs**: 2 CPU, 4GB RAM, 38GB disk (~33GB free)
- **Mail**: `docker-mailserver` at `/opt/mailserver/`
- **Manage accounts**: `docker exec mailserver setup email add/del/list`
- **Available for future services** — has headroom for lightweight containers. Not suitable for storage-heavy services (e.g. Castopod with daily episodes) without a disk upgrade or attached volume.

## Podcast Workflow
- Publishing pipeline: episodes go through Castopod, CDN, website, YouTube, and social
- Always check Python venv is active and packages are installed before running publish scripts
- Episode numbering must be verified against existing episodes

## Episodes Published
- Episode 6 published 2026-02-08 (podcast6.mp3, ~31 min)
