# Relocate the Show to Alpine, TX / Big Bend

**Date:** 2026-05-31
**Status:** Approved, implementing

## Goal

Luke moved to Alpine, Texas. Move the show's world from southern New Mexico
(Deming/Lordsburg) to the Big Bend region of far West Texas. Callers should come
from and be familiar with Alpine, Marfa, Marathon, Terlingua, Fort Stockton, and
the Big Bend area. Devon moves with the show. Silas gets an arc beat that
relocates The Wellspring to Terlingua.

## Changes

### 1. Caller generation — `backend/services/caller_gen.py`

- `BATCH_SYSTEM_PROMPT` opener: "…radio show in New Mexico" → "…in Alpine, Texas,
  in the Big Bend country of far West Texas."
- Add a **geographic knowledge block** (real, accurate facts) so callers ground
  themselves in real places instead of generic desert. Towns:
  - **Alpine** — hub (~6k), Brewster County seat, Sul Ross State University,
    mile-high ranching/railroad town, small arts scene.
  - **Marfa** — minimalist-art tourist town (Chinati/Donald Judd, Prada Marfa),
    the Marfa Lights, old ranching families vs. hipster influx.
  - **Marathon** — tiny, the Gage Hotel, east gateway to Big Bend NP.
  - **Terlingua** — quicksilver-mining ghost town, famous chili cookoff,
    river-rafting outfitters, off-grid desert eccentrics, the Starlight Theatre,
    near the Rio Grande / Mexico border.
  - **Fort Stockton** — oilfield/I-10 town to the north, Pecos County, Paisano
    Pete, working-class.
  - **Big Bend** — Chisos Mountains, Rio Grande, dark skies (McDonald Observatory
    near Fort Davis), remote, border proximity, Permian Basin oil money north.
  - Rule: only reference real places/facts; don't invent businesses or landmarks.

### 2. Whisper prompt — `backend/services/transcription.py`

Update locale and seed proper nouns: "…a late-night radio talk show in Alpine,
Texas, in the Big Bend region. Callers reference Alpine, Marfa, Marathon,
Terlingua, Fort Stockton, and Big Bend."

### 3. Devon — `backend/services/intern.py`

Moved with the show. Keep "Communications degree from NMSU" (alma mater);
"You live in a studio in Deming" → "studio in Alpine."

### 4. Silas — relocate The Wellspring to Terlingua

- `~/code/dotfiles/silas/silas.md`: identity location Deming → Terlingua badlands;
  update `arc_state` frontmatter to the relocation; add an Arc Log entry
  (2026-05-31) — post-reckoning fresh start, moving the commune to desert outside
  Terlingua, the move straining the forty members.
- `data/regulars.json`: update Silas's `job`, `location`, and the
  `stable_seeds.style` "commune outside Deming" → Terlingua. Leave the 6 past
  `call_history` summaries untouched — they happened in Deming; the move is the
  new development.

### 5. Weather/town-news enrichment — `backend/main.py` (now a real feature)

The block in `enrich_caller_background()` called `_get_town_from_location` /
`_get_weather_for_town`, which never existed — silently NameError-ing. Implement:

- `BIG_BEND_TOWNS`: dict of town → (lat, lon) for Alpine, Marfa, Marathon,
  Terlingua, Fort Stockton, Big Bend (Chisos Basin), Fort Davis, Presidio.
- `_get_town_from_location(text)`: scan lowercased text for a known town,
  return canonical key (handles "ft stockton"/"fort stockton").
- `_get_weather_for_town(town)`: Open-Meteo current weather (free, no key),
  WMO weather_code → phrase, returns e.g. "58°F, clear skies".
- Update town-news query: NM/AZ branch → `f"{town.title()} Texas"`.
- Wire session base_ctx `weather` (was hardcoded "cool desert night") to live
  Alpine weather, graceful fallback on failure.

All network calls degrade gracefully (existing `asyncio.timeout` + try/except).

## Out of scope / left as-is

- Past `call_history` summaries (genuinely happened in Deming).
- `main.py:3124` state-abbreviation map (generic, not show-locale).
