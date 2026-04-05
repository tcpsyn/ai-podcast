# Caller Generation Redesign

## Overview

Replace the current caller generation pipeline (static pools + weights + scoring + 9-model routing) with a two-stage architecture: **rich identities pre-generated in one sonnet batch call per show**, then **live dialog through a single haiku-4.5 model**. The goal is callers that feel genuinely dynamic across a wide emotional range (Stern / Coast to Coast / Loveline / Delilah / O&A), with Silas preserved as a canonical character and a hard-gated, arc-driven regulars tier replacing the current low-bar promotion system.

**Target cost:** ~$0.95-1.05 per show (down from $4.32 in the sonnet-only era; roughly flat vs. the current $0.94 multi-model average, but with materially better caller quality).

## Problem

Callers currently feel static, repetitive, and homogenized. Audit findings:

1. **Every caller's reason-for-calling is a static pick** from ~200 hand-written PROBLEMS entries (or sibling pools). The LLM only writes *around* a pre-chosen seed — "dynamic generation" is an illusion.
2. **~2000 tokens of uniform prompt chorus** sit on top of every caller's dialog (GO WITH THE HOST, REACT TO LUKE, BANNED PHRASES, shape directives, etc.). This drowns out whatever personality the style block or model was trying to inject.
3. **9 dialog models × per-model param tuning × style→model map** with no empirical feedback loop. The "style diversity" layer re-introduces variety AFTER static pools and prompt chorus already homogenized it. Layers cancel each other out.
4. **Redundant style + shape systems** — 18 styles map nearly 1:1 with affinity-weighted shapes, with SHAPE_DIRECTIVES reinforcing the same thing in different words.
5. **Voice matching scoring is noisy** — 4-dim soft matching rarely produces strong discrimination; most callers score "medium" on every axis.
6. **5% random promotion to regular** produces too many recurring callers with no arc progression (e.g., "potato salad guy" calling weekly to complain about the same thing).
7. **~400 lines of dead template fallback code** only triggered on sanity-check failure.

Root cause: the system constrains creativity at both ends (static picks at input, uniform chorus at output), then tries to re-introduce variety with model/style/shape layering that cancels itself out.

## Architecture

Two stages, two models.

### Stage 1 — Pre-generation batch (sonnet-4.6, one call per show)

Runs before the show starts. One LLM call generates all caller identities for tonight in a single structured output.

**Inputs to the batch prompt:**
- Today's date, day of week, weather, time of year
- 3-5 news headlines (grounding, not forced topicality)
- Local NM/SW cultural context
- Last 2 episodes' caller summaries (anti-repeat directive)
- Active regulars' lore files (Silas always if he's in tonight's lineup; optional 1-2 tier-2 regulars)
- Creative north-star directive: "Generate 12 callers spanning earnest weirdo → chaotic character → vulnerable confession → advice-seeker → absurd. Maximum character distance between callers. No two callers should feel like siblings. Do not default to sitcom plots."

**Output per caller (JSON):**
- `name`, `age`, `voice_suggestion`, `location`
- `identity` — rich paragraph describing who this person actually is
- `situation` — what's specifically happening in their life RIGHT NOW
- `reason_calling` — why they picked up the phone tonight (a moment, not a category)
- `opening_line` — verbatim first line they'll say
- `secret_want` — what they actually want from the call (not always what they say)
- `specific_details` — 2-3 concrete details to drop in dialog
- `emotional_register` — tone this caller will bring

**No static pools. No weights. No scoring. Sonnet invents everything from the rich context.**

### Stage 2 — Live dialog (claude-haiku-4.5, one model for all callers)

When a caller is selected during the show, haiku receives a ~400-token prompt:

```
You are [name]. [identity paragraph from pre-gen.]

You're calling Luke's show because [situation + reason_calling].
What you secretly want from this call: [secret_want].
Specific details you'll drop if it feels natural: [specific_details].

Speak as this person. React to what Luke says. Stay in character.
Don't narrate. No stage directions. Just talk.
```

**No style block. No shape directive. No banned-phrases list. No per-model param tuning.** Identity from Stage 1 carries the weight; haiku plays the character.

## Regulars System

Three tiers.

### Tier 1 — Silas (canonical)

Silas's lore file lives at `~/code/dotfiles/silas/silas.md` (user's Obsidian vault). The file contains:
- **Frozen identity**: voice, age, core personality, relationship to host, canonical facts
- **Arc log**: append-only log of what's happened in past episodes
- **Current arc threads**: active storylines (e.g., "cult splintering," "rival prophet")

When Silas is in tonight's lineup, the batch prompt receives his full lore file verbatim, with the instruction: *"Invent a fresh reason Silas is calling tonight — a new cult development, grievance, or specific recent event. DO NOT alter his voice, personality, or core traits. Write a new scene for an existing character."*

**Protection from drift:** git-tracked lore file + pre-ship sample-call validation (see below).

### Tier 2 — Arc regulars (max 2-3 active at a time)

Lore files at `~/code/dotfiles/regulars/<name>.md`. Each file contains:
- **Frozen identity** (same shape as Silas)
- **Arc plan** written at promotion time: "3-5 episode arc, here's how it progresses, here's what might resolve it"
- **Arc state**: which call in the arc we're on, what's happened so far

**Return cadence:** every 3-4 episodes, when the arc has something new to advance. Not on a fixed schedule.

### Tier 3 — Walk-ins

Everyone else. Fresh each episode. No memory, no file, no callback.

### Hard promotion gate

Replaces the current 5% random roll. After a call ends, sonnet evaluates the call and answers: *"Does this character have a 3-5 episode arc in them? If yes, write the arc plan. If no, say why not."* Only promotes if sonnet produces a credible arc plan. Expected promotion rate: ~1 in 20 calls.

### Arc retirement

When sonnet judges the arc resolved (or after 5 calls without resolution), the character is archived — no more callbacks. Lore file is kept for reference but moved to `~/code/dotfiles/regulars/archived/`.

### Migration from current regulars

All current regulars except Silas are archived at cutover. `data/regulars.json` is kept as historical record. Going forward, only NEW characters can earn tier-2 status through the promotion gate.

## Model Stack

| Purpose | Model | Frequency | Cost/show (est.) |
|---------|-------|-----------|------------------|
| Caller identity pre-gen | claude-sonnet-4.6 | 1 call | ~$0.15 |
| Live dialog (all callers) | claude-haiku-4.5 | ~500 turns | ~$0.30-0.40 |
| Post-call summary | gemini-2.5-flash | 1/caller | ~$0.01 |
| Promotion evaluation | claude-sonnet-4.6 | 1/caller post-call | ~$0.05 |
| Devon monitor | gemini-2.5-flash | every 15s | ~$0.08 |
| Devon ask | gemini-2.5-flash | on demand | ~$0.01 |
| News summary | gemini-2.5-flash | 1/show | <$0.01 |
| **LLM total** | | | **~$0.60-0.70** |
| TTS (Inworld, unchanged) | | | ~$0.39 |
| **Total** | | | **~$1.00-1.10** |

## Code to Delete (~2000 lines)

From `backend/main.py`:
- `PROBLEMS`, `STORIES`, `GOSSIP`, `ADVICE`, `TOPIC_CALLIN`, `CELEBRATIONS`, `WEIRD`, `HOT_TAKES` content pools
- `INTERESTS`, `QUIRKS`, `RELATIONSHIP_STATUS`, `VEHICLES`, `BEFORE_CALLING`, `CALLING_FROM`, `MEMORIES`, `HAVING_RIGHT_NOW`, `STRONG_OPINIONS`, `CONTRADICTIONS`, `VERBAL_TICS`, `EMOTIONAL_ARCS`, `SHOW_RELATIONSHIP`, `LATE_NIGHT_REASONS`, `DRIFT_TENDENCIES`, `ROAD_CONTEXT`, `PHONE_SITUATION`, `BACKGROUND_MUSIC`, `RECENT_ERRAND`, `TV_TONIGHT`, `LOCAL_FOOD_OPINIONS`, `NOSTALGIA` color pools
- `CALLER_STYLES` (18 style paragraphs), `CALLER_STYLE_KEYS`
- `STYLE_VOICE_PREFERENCES`, `STYLE_SPEED_MODIFIERS`, `STYLE_PHONE_QUALITY`
- `CALL_SHAPES`, `SHAPE_STYLE_AFFINITIES`, `SHAPE_DIRECTIVES`, `_LATE_SHOW_SHAPES`
- `_SPICY_KEYWORDS`, `_ABSURD_KEYWORDS`, `_HEAVY_POOLS`, `_LIGHT_POOLS`, `_HEAVY_STYLES`, `_LIGHT_STYLES`, `_EVASIVE_STYLES`
- `caller_model_pool`, `caller_model_map`, `_CALLER_DIALOG_MODEL_PARAMS`
- `_generate_pool_weights`, `_pick_unique_reason`, `_pick_caller_style`, `_pick_call_shape`, `_assign_call_shape`
- `_match_voices_to_styles`, voice-scoring logic
- `_sort_caller_queue` (greedy placement scoring)
- `generate_caller_background` (template fallback, ~400 lines)
- `_build_relationship_context` (inter-caller thematic scoring)
- `SHOW_HISTORY_REACTIONS`, adaptive reaction frequency logic

Kept:
- `CALLER_BASES` (10 slots with gender/age — still drives voice/name selection)
- `MALE_NAMES`, `FEMALE_NAMES` (used as naming hints for sonnet, not picks)
- Voice rosters (`INWORLD_MALE`, etc.) — sonnet suggests, we validate against roster
- `_pick_response_budget`, `_retry_if_too_short`, `_has_repetition` (dialog-side post-processing)
- `_assess_call_quality`, `_summarize_ai_call` (post-call hooks)

## New Code (~300 lines)

- **`backend/services/caller_gen.py`** — batch pre-gen orchestration, prompt assembly, JSON parsing, voice-roster validation
- **`backend/services/regulars_v2.py`** — lore file loader, arc state management, promotion gate, arc retirement
- **Modified `get_caller_prompt`** in `main.py` — slimmed ~400-token prompt assembly
- **Modified `Session.get_caller_model`** — returns haiku-4.5 unconditionally
- **Migration script** — archive current regulars except Silas, bootstrap Silas lore file

## Validation Gate (before ship)

Before deploying to live shows:

1. Generate 10 sample calls with the new system: 5 featuring Silas, 5 walk-ins
2. User listens to all 10
3. **Approve essence → ship.** Silas still sounds like Silas; walk-ins span the emotional range; callers feel distinct and specific.
4. **Drift or bland → iterate the prompt.** Tune batch prompt directives, re-run, re-validate.

## Non-Goals

- Not redesigning Devon (intern) — stays on gemini-flash as-is
- Not redesigning voice rosters or TTS provider — Inworld stays
- Not building a quality feedback loop that updates prompts from call outcomes (future work)
- Not pre-generating AUDIO for caller opening lines (future work, could cut latency)
- Not rebuilding the frontend caller panel — style badges etc. can stay or be simplified later
