# Show Quality Fixes — Episode 47 Post-Mortem

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 5 bugs that ruined tonight's show: theme ignored by callers, wrong LLM models assigned, phonetic pronunciation mangling, voice-age mismatch, and low minimum response threshold.

**Architecture:** All fixes are in `backend/main.py` except voice-age matching which also touches `backend/services/tts.py` voice matching logic. Each fix is independent — no ordering dependencies between tasks.

**Tech Stack:** Python, FastAPI

---

### Task 1: Regenerate caller backgrounds when theme is set

**Problem:** `_pregenerate_backgrounds()` runs on startup when `session.show_theme` is still `""`. Setting theme via `POST /api/show-theme` only stores the string — doesn't regenerate. Callers have zero theme connection.

**Files:**
- Modify: `backend/main.py:9891-9900` (`set_show_theme` endpoint)
- Modify: `backend/main.py:5899-5927` (`_pregenerate_backgrounds`)

**Step 1: Modify `set_show_theme` to regenerate unused caller backgrounds**

In `backend/main.py`, replace the `set_show_theme` endpoint (lines 9891-9900):

```python
@app.post("/api/show-theme")
async def set_show_theme(data: dict):
    theme = data.get("theme", "").strip()[:100]
    old_theme = session.show_theme
    session.show_theme = theme
    if theme:
        print(f"[Theme] Show theme set: {theme}")
    elif old_theme:
        print(f"[Theme] Show theme cleared (was: {old_theme})")

    # Regenerate backgrounds for callers that haven't been on air yet
    if theme != old_theme:
        unused_keys = [k for k in CALLER_BASES if k not in session.used_callers]
        if unused_keys:
            print(f"[Theme] Regenerating {len(unused_keys)} unused caller backgrounds for theme: {theme or '(none)'}")
            asyncio.create_task(_regenerate_backgrounds_for_keys(unused_keys))

    return {"theme": session.show_theme}
```

**Step 2: Add `_regenerate_backgrounds_for_keys` helper**

Add this right after `_pregenerate_backgrounds()` (after line 5927):

```python
async def _regenerate_backgrounds_for_keys(keys: list[str]):
    """Regenerate backgrounds for specific caller keys (e.g. after theme change)."""
    tasks = []
    for key in keys:
        base = CALLER_BASES.get(key)
        if base and not base.get("returning"):
            tasks.append((key, _generate_caller_background_llm(base)))

    if not tasks:
        return

    results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)
    for (key, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            print(f"[Theme] Regen failed for caller {key}: {result}")
        else:
            session.caller_backgrounds[key] = result
            # Clear cached model so it re-evaluates with new style
            session.caller_models.pop(key, None)

    print(f"[Theme] Regenerated {sum(1 for r in results if not isinstance(r, Exception))}/{len(tasks)} backgrounds")
    _match_voices_to_styles()
    _sort_caller_queue()
```

**Step 3: Verify `used_callers` exists on session**

Check that `session.used_callers` tracks which callers have already been on air. If it doesn't exist, use `session.call_history` caller keys instead.

**Step 4: Test manually**

```bash
# Start server
python -m uvicorn backend.main:app --reload --reload-dir backend --host 0.0.0.0 --port 8000
# Set theme and check logs for "[Theme] Regenerating..." messages
curl -X POST http://localhost:8000/api/show-theme -H "Content-Type: application/json" -d '{"theme": "Road Stories"}'
```

**Step 5: Commit**

```bash
git add backend/main.py
git commit -m "Regenerate caller backgrounds when show theme is set"
```

---

### Task 2: Fix style-to-model matching race condition

**Problem:** `get_caller_model()` is called before `caller_styles` is populated. `caller_styles.get(key)` returns `""`, `_normalize_style_key("")` returns `""`, no match in `caller_model_map` → falls through to `caller_model_pool[0]` (grok-4.1-fast) for everyone.

**Files:**
- Modify: `backend/main.py:6848-6875` (`get_caller_model`)

**Step 1: Fix `get_caller_model` to defer assignment when style is unknown**

Replace `get_caller_model` (lines 6848-6875):

```python
    def get_caller_model(self, caller_key: str) -> str | None:
        """Get the assigned model for a caller, or assign one based on strategy.
        Returns None to use default category routing."""
        if self.caller_model_strategy == "single":
            return None  # use default category_models["caller_dialog"]

        # Already assigned — keep consistent for the whole call
        if caller_key in self.caller_models:
            return self.caller_models[caller_key]

        model = None
        if self.caller_model_strategy == "cycle":
            if self.caller_model_pool:
                model = self.caller_model_pool[self._caller_model_cycle_idx % len(self.caller_model_pool)]
                self._caller_model_cycle_idx += 1
        elif self.caller_model_strategy == "style_matched":
            raw_style = self.caller_styles.get(caller_key, "")
            style_key = _normalize_style_key(raw_style) if raw_style else ""
            if style_key:
                model = self.caller_model_map.get(style_key)
            if not model:
                # Style not yet populated or no mapping — use fallback, not pool[0]
                model = self.caller_model_fallback

        if model:
            self.caller_models[caller_key] = model
            caller_name = CALLER_BASES.get(caller_key, {}).get("name", caller_key)
            style_info = self.caller_styles.get(caller_key, "unknown")
            print(f"[CallerModel] Assigned {model} to {caller_name} (style={_normalize_style_key(style_info) if style_info else 'none'}, strategy={self.caller_model_strategy})")

        return model
```

The key change: when `style_key` is empty (style not yet populated) or has no mapping, use `caller_model_fallback` (claude-sonnet-4.6) instead of `caller_model_pool[0]` (grok-4.1-fast). Claude Sonnet is a much safer default — empathetic, verbose, coherent.

**Step 2: Commit**

```bash
git add backend/main.py
git commit -m "Fix style-to-model race condition — use fallback instead of pool[0]"
```

---

### Task 3: Fix pronunciation fixes producing literal phonetic text

**Problem:** `_PRONUNCIATION_FIXES` replaces "Animas" with "Ah nee mahs" as literal text. TTS reads each word separately ("Ah" "nee" "mahs") instead of blending into the intended pronunciation.

**Files:**
- Modify: `backend/main.py:9141-9152` (`_PRONUNCIATION_FIXES`)
- Modify: `backend/main.py:9212-9216` (`_apply_pronunciation_fixes`)

**Step 1: Remove pronunciation fixes that sound worse than originals**

The Inworld TTS actually handles most proper nouns fine. The fixes were added speculatively and cause more harm than good. Remove the place names that TTS can handle, keep only abbreviations:

Replace `_PRONUNCIATION_FIXES` (lines 9141-9152):

```python
_PRONUNCIATION_FIXES = {
    "Castopod": "Casto pod",
    "vs": "versus",
    "govt": "government",
    "dept": "department",
}
```

Remove `Lordsburg`, `Hachita`, `Deming`, `Bootheel`, `Animas`, and `Rodeo`. These place names either sound fine through TTS or the phonetic replacement sounds worse.

**Step 2: Commit**

```bash
git add backend/main.py
git commit -m "Remove pronunciation fixes that produce worse TTS output"
```

---

### Task 4: Add age-awareness to voice matching

**Problem:** Brandy (55 years old) got "Kayla" (young-sounding voice). `_match_voices_to_styles()` scores on style dimensions (weight, energy, warmth, age_feel) but the `age_feel` preference comes from the communication style, not the character's actual age. A "confrontational" style prefers `age_feel: None` (no preference), so a 55-year-old can get a young voice.

**Files:**
- Modify: `backend/main.py:6106-6156` (`_match_voices_to_styles`)

**Step 1: Add character age to voice scoring**

In `_match_voices_to_styles`, after getting the style preferences, override `age_feel` based on the caller's actual age from their background:

```python
def _match_voices_to_styles():
    """Re-assign voices to match caller communication styles after backgrounds are generated."""
    from .services.tts import VOICE_PROFILES

    for key, base in CALLER_BASES.items():
        if base.get("returning"):
            continue

        style_raw = session.caller_styles.get(key, "")
        if not style_raw:
            continue

        style_key = _normalize_style_key(style_raw)
        prefs = STYLE_VOICE_PREFERENCES.get(style_key)
        if not prefs:
            continue

        # Copy prefs so we don't mutate the shared dict
        prefs = dict(prefs)

        # Override age_feel based on character's actual age
        bg = session.caller_backgrounds.get(key)
        if isinstance(bg, CallerBackground) and bg.age:
            if bg.age >= 50:
                prefs["age_feel"] = "mature"
            elif bg.age >= 35:
                prefs["age_feel"] = "middle"
            elif bg.age < 25:
                prefs["age_feel"] = "young"
            # 25-34: keep style preference or None

        gender = base["gender"]
        pool = INWORLD_MALE_VOICES if gender == "male" else INWORLD_FEMALE_VOICES
        voice_pool = [v for v in pool if v not in BLACKLISTED_VOICES]

        scored = []
        for voice_name in voice_pool:
            profile = VOICE_PROFILES.get(voice_name)
            if not profile:
                scored.append((voice_name, 0))
                continue
            score = 0
            for dim in ["weight", "energy", "warmth", "age_feel"]:
                pref_val = prefs.get(dim)
                if pref_val and profile.get(dim) == pref_val:
                    score += 1
            scored.append((voice_name, score))

        if scored:
            names = [s[0] for s in scored]
            weights = [max(1, s[1] * 3) for s in scored]
            chosen = random.choices(names, weights=weights, k=1)[0]

            used_voices = {CALLER_BASES[k]["voice"] for k in CALLER_BASES if k != key and "voice" in CALLER_BASES[k]}
            if chosen in used_voices:
                alternatives = [(n, w) for n, w in zip(names, weights) if n not in used_voices]
                if alternatives:
                    alt_names, alt_weights = zip(*alternatives)
                    chosen = random.choices(alt_names, weights=alt_weights, k=1)[0]

            old_voice = base.get("voice", "")
            base["voice"] = chosen
            if old_voice != chosen:
                print(f"[VoiceMatch] {base.get('name', key)}: {old_voice} → {chosen} (style: {style_key}, age: {bg.age if isinstance(bg, CallerBackground) else '?'})")
```

**Step 2: Commit**

```bash
git add backend/main.py
git commit -m "Add age-awareness to voice matching — 55yo won't get young voices"
```

---

### Task 5: Raise minimum response word count

**Problem:** `MIN_RESPONSE_WORDS = 30` lets through fragmented, telegram-style responses that are technically 30+ words but terrible radio.

**Files:**
- Modify: `backend/main.py:8844` (`MIN_RESPONSE_WORDS`)

**Step 1: Raise the minimum**

Change line 8844:

```python
MIN_RESPONSE_WORDS = 50  # Retry if response is shorter than this
```

50 words is roughly 2-3 spoken sentences — enough to be a coherent radio response without being overly demanding for short-form exchanges.

**Step 2: Commit**

```bash
git add backend/main.py
git commit -m "Raise MIN_RESPONSE_WORDS from 30 to 50"
```
