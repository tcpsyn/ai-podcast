# Caller Variety, Movable Callers, and Devon Search Fix

**Date:** 2026-06-02
**Status:** Approved, implementing

## Problems

1. **Callers feel formulaic** — too many deep moral dilemmas; the roster prompt
   *enforces* it ("at least half moral dilemmas" + "NEVER generate callers who
   are just enthusiastic about a hobby").
2. **Callers circle forever** — they don't take advice, restating the same moral
   issue until the host cuts them off. The live dialog prompt gives a `situation`
   + `secret_want` but no mechanic to be moved, persuaded, or reach resolution.
3. **Devon's search fails** — SearXNG at `localhost:8888` is unreachable whenever
   the laptop's Docker Desktop isn't running. Fragile.

## Changes

### A. Roster variety — `backend/services/caller_gen.py` (`BATCH_SYSTEM_PROMPT`)

- Remove the absolute ban on hobby/job/story callers.
- Replace "at least half moral dilemmas" with an explicit distribution for a
  10-caller roster (dilemmas still lead):
  - **4–5** moral dilemmas / confessions / betrayals — keep the STAKES language.
  - **2–3** storytellers & enthusiasts — a wild thing that happened, a fascinating
    niche obsession or fact, a vivid slice-of-life. No deep dilemma required.
  - **1–2** believers / chaos — earnest UFO/cryptid/conspiracy callers
    (Coast-to-Coast sincerity) or a big eccentric personality on a rant.
- Keep the anti-collision rule, "reason they HAD to call tonight," and
  "even lighter callers need energy and specificity."

### B. Movable callers — `backend/main.py` (`get_caller_prompt`)

Add a concise conversational-arc block for every caller:
- You have a position/want, but you're a real person — if Luke makes a good
  point, genuinely react: agree, change your mind, soften, dig in, or decide.
- Don't restate a point/dilemma you've already made; the conversation must move.
  You can reach a resolution, a decision, or an emotional turn. You are not
  required to stay stuck.

Stacks on the existing "don't restate facts / move the story forward" rule.
Must keep `get_caller_prompt` output under the 3500-char test cap.

### C. SearXNG → NAS — `backend/config.py`, `backend/services/news.py`

- Deploy SearXNG to **mmgnas** as an always-on container (deploy-nas-docker).
- Add `searxng_url` to `config.py` (env-overridable), default to the NAS URL.
- `news.py` reads `settings.searxng_url` instead of the hardcoded constant;
  `intern.py` imports `SEARXNG_URL` from `news.py`, so Devon, headlines, and
  caller news-grounding all follow.

## Verification

- Run test suite (note: 2 pre-existing stale `test_caller_gen.py` failures).
- Live caller-gen batch → review the roster mix.
- Devon end-to-end: a `web_search` against the NAS SearXNG returns real results.
