# Wellspring Shared Lore + Continuity — Design

## Problem
- Wellspring members only saw their own lore file; nothing shared. Junie didn't know where the Wellspring was.
- The live dialog prompt never saw lore at all — only pregen fields + a one-line arc_state.
- Nothing wrote back after calls. Silas's arc_state was frozen at 2026-05-31 ("the move is fracturing the community"), so every call replayed people leaving.
- Silas's legacy `regulars.json` history was polluted (a misattributed Imogen summary, Alpine vs Terlingua).

## Canon decisions
- The move is done: ~30 members live on raw desert outside Terlingua. Deming land is in foreclosure.
- Some members hear each other's calls (Silas listens; others catch it in town). Calls ripple.
- Continuity auto-updates after every lore-regular call; Luke edits/deletes in Obsidian to override.

## Files (Obsidian vault, `~/code/dotfiles`)
- `wellspring/wellspring.md` — the bible. Hand-written: The Place, The Current, The People.
  Machine-maintained: `## Storylines` (one `### slug` block each with status / last_advanced / involves / current state),
  `## Established Facts`, `## Timeline`.
- Each member file gains `## The Wellspring, to me` — their view of the place, Silas, other members, what they know and don't.

## Storylines (initial 7)
Water + the neighbor, the seeker with money, someone from Silas's past (his son), the first desert Unbinding,
the consent stones, the heretic couple, Silas's celibacy.

## Prompt wiring
- Pregen (batch + single-regular): Wellspring members get the condensed bible + their perspective. Silas is assigned
  one storyline — the open one advanced least recently.
- Live dialog (`get_caller_prompt`): Wellspring members get the same condensed bible + perspective + instruction to stay
  consistent with shared facts and filter through their own point of view.

## Continuity pass
Hooked into `_summarize_ai_call()` for any lore regular. One Sonnet 5.5 call returns JSON:
arc_log entry, new arc_state, timeline entry, storyline updates (advance/close/new), established facts, ripples.
Writes are atomic and touch only machine-maintained sections + frontmatter `arc_state` + `## Arc Log`.
Failure logs and never blocks hangup.

## Testing
Unit: bible parse, storyline pick, continuity writes only marked sections, failure isolation, Wellspring-only prompt injection.
Live: generate Junie + Silas same night; Junie knows Terlingua and Doyle; no contradictions with Silas.
