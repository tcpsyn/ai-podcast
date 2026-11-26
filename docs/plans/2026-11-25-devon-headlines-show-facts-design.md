# Devon: Opening Headlines + Show Facts Memory

Date: 2026-11-25

## Problems

1. **Devon forgets things mid-show.** Ep62: Luke opened by saying the show was in West Virginia; by the end Devon thought they were in Alpine.
   - `ask()` only sees the last 6 conversation lines + Devon's last 10 messages; `interject()` sees 8 lines + 6 of Devon's.
   - `session.conversation` is cleared at every call start/end, and `/api/intern/ask` never logs the host's question — only Devon's answer.
   - `INTERN_SYSTEM_PROMPT` hardcodes "You live in Alpine, Texas — you moved out to the Big Bend with the show", which wins once the contrary fact scrolls out.
   - Callers have the same problem: `caller_gen.BATCH_SYSTEM_PROMPT` says the show "broadcasts out of Alpine".
2. **Headlines are stale.** `news_service.get_headlines()` searches SearXNG for the literal word "news" with no time range. Half the results are from Wikinews, which shut down May 2026.

## Decisions

| Question | Decision |
|---|---|
| Headline mix | 2 big stories + 3 offbeat/weird ones |
| Headline window | Since last episode's release date (Castopod), min 1 day, max 7 days |
| When Devon raises them | Only during the show open (before the first call). After that he knows them but doesn't initiate |
| Who gets show facts | Devon **and** callers |
| Fact capture | Automatic (Haiku) + editable Show Facts panel; manual edits are pinned |
| Cross-show memory | Out of scope — facts reset each show |

## 1. Curated opening headlines

**Build** (background task on session reset, alongside caller pregen):
1. Determine window: days since last Castopod episode `published_at`, clamped to [1, 7]. `time_range=day` if ≤1 day, else `week`. Castopod failure → 1 day.
2. Parallel SearXNG `categories=news` queries: top-news queries (e.g. "top news", "breaking news") and offbeat queries (e.g. "weird news", "strange but true", "offbeat news").
3. Drop `wikinews` results and anything with `publishedDate` before the window start. Dedupe by normalized title.
4. Haiku (`news_summary` category) picks 5: 2 `big`, 3 `offbeat`. Each: `title`, `kind`, `summary` (1 line), `angle` (why it's good for the show). JSON out; invalid → empty list.
5. Store on session as `opening_headlines`; persist in `session_checkpoint.json`.

**Devon usage:**
- While no call has started this show: `ask()` (and interject/monitor) include an **OPENING HEADLINES** block — weave in at most one per reply, only when it fits, in his own voice, never as a list, never repeat one he already raised (checked against his current-show history).
- After the first call starts: block becomes **STORIES YOU KNOW ABOUT** — background knowledge only, don't initiate.

**Callers:** the 5 headline titles passed to `caller_gen.generate_batch()` come from the curated list instead of the old raw fetch.

**Failure:** any error → no headlines block; show proceeds; reason logged.

## 2. Show facts

**Fact:** short durable statement true for tonight's show ("Luke is broadcasting from his mom's house in West Virginia tonight, not Alpine"). Not passing remarks or caller stories.

**Capture:**
- New show-wide host-line log, independent of `session.conversation`. Fed from `/api/chat`, the AI-respond path, and `/api/intern/ask` (which also starts logging the host's question).
- Extraction trigger: every 3 new host lines, or 60s idle with unprocessed lines. Runs in background; never blocks a reply.
- Haiku (`show_facts` category, reasoning off) sees recent host lines + current facts, returns `add` / `replace` / `remove` ops. Replacements handle corrections ("actually we're back in Alpine").
- Pinned facts (manually added/edited) are never changed by the extractor.

**UI:** Show Facts panel in the control panel — list, edit, delete, add. Usable pre-show.

**Consumers:**
- Devon: SHOW FACTS block near the top of every ask/interject/monitor prompt — "true right now, overrides anything else you assume". Background line softened to "the show normally broadcasts from Alpine…".
- Callers: compact facts block in `get_caller_prompt()`.
- Devon's current-show history window raised from 10 to 20 messages.

**Cost:** ~1 Haiku call per 3 host lines — cents per show.

## 3. Components

- **New `backend/services/show_memory.py`:** host-line log, fact store (`text`, `pinned`, `updated_at`), fact extractor + merge, curated headline builder. Serializable for the checkpoint.
- **`backend/services/intern.py`:** facts + headlines blocks in prompts; soften Alpine line; history 10 → 20.
- **`backend/main.py`:** log host lines; kick off headline build on reset; checkpoint save/load; `GET/POST/PATCH/DELETE /api/show/facts`; facts into `get_caller_prompt()`; curated headlines into caller batch context.
- **`backend/services/news.py`:** time-ranged multi-query fetch, Wikinews filter.
- **`backend/config.py`:** `show_facts` → Haiku 5.5 in `category_models`; `REASONING_EFFORT` none.
- **Frontend:** Show Facts panel.

## 4. Error handling

Everything degrades softly: extractor error/bad JSON → facts unchanged; headline failure → no block; Castopod down → 1-day window. Nothing blocks a call or a Devon reply.

## 5. Tests

- Fact merge: correction replaces; pinned survives extraction; dedupe; bad JSON → no change.
- Devon prompt: facts block present and ahead of background; headlines block only pre-first-call, then "stories you know about".
- Caller prompt includes facts (extends `tests/test_caller_prompt.py`).
- Headlines: Wikinews + pre-window dropped; window clamp; curation failure → `[]`.
- Model config: `show_facts` registered and covered by `tests/test_model_config.py`.

## Out of scope

Cross-show fact memory; Devon raising headlines during between-call lulls.
