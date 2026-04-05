# Caller Generation Redesign — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current static-pool + 9-model caller generation with a two-stage architecture: rich identities pre-generated in one sonnet-4.6 batch call per show, plus live dialog through a single claude-haiku-4.5 model. Tiered regulars system with Silas canonical in the Obsidian vault.

**Architecture:** Two new service modules (`caller_gen.py`, `regulars_v2.py`). Phase 1-3 are ADDITIVE — no old code removed until user validates sample calls. Phase 5 deletes ~2000 lines after cutover.

**Tech Stack:** Python 3.11, FastAPI, OpenRouter (`anthropic/claude-sonnet-4.6`, `anthropic/claude-haiku-4.5`, `google/gemini-2.5-flash`), pytest, pydantic.

**Design doc:** `docs/plans/2026-04-05-caller-generation-redesign.md` (committed at `fa491c1`)

---

## Phase 1 — New `caller_gen` service (additive)

### Task 1: Create CallerIdentity dataclass + JSON schema

**Files:**
- Create: `backend/services/caller_gen.py`
- Test: `tests/test_caller_gen.py`

**Step 1: Write failing test for dataclass parsing**

```python
# tests/test_caller_gen.py
import pytest
from backend.services.caller_gen import CallerIdentity, parse_batch_response

SAMPLE_JSON = """
{
  "callers": [
    {
      "name": "Danny Ortega",
      "age": 47,
      "voice_suggestion": "Marcus",
      "location": "Silver City, NM",
      "identity": "A plumber who inherited his uncle's taxidermy shop...",
      "situation": "He's been getting calls from people...",
      "reason_calling": "Someone left a note in his mailbox tonight...",
      "opening_line": "Luke, I need to ask you something weird.",
      "secret_want": "Permission to just throw it all away",
      "specific_details": ["the elk head in the basement", "the note said 'she forgot'", "his uncle's Rolodex"],
      "emotional_register": "quietly unsettled, trying to sound casual"
    }
  ]
}
"""

def test_parse_batch_response_returns_caller_list():
    callers = parse_batch_response(SAMPLE_JSON)
    assert len(callers) == 1
    assert callers[0].name == "Danny Ortega"
    assert callers[0].age == 47
    assert "taxidermy" in callers[0].identity
    assert len(callers[0].specific_details) == 3

def test_parse_batch_response_rejects_missing_fields():
    bad = '{"callers": [{"name": "Jim"}]}'
    with pytest.raises(ValueError, match="missing"):
        parse_batch_response(bad)
```

**Step 2: Run test, verify it fails**
Run: `/Users/lukemacneil/code/ai-podcast/venv/bin/python -m pytest tests/test_caller_gen.py -v`
Expected: ImportError / ModuleNotFoundError

**Step 3: Implement the dataclass and parser**

```python
# backend/services/caller_gen.py
from dataclasses import dataclass, field
from typing import Optional
import json

REQUIRED_FIELDS = {
    "name", "age", "voice_suggestion", "location", "identity",
    "situation", "reason_calling", "opening_line", "secret_want",
    "specific_details", "emotional_register"
}

@dataclass
class CallerIdentity:
    name: str
    age: int
    voice_suggestion: str
    location: str
    identity: str
    situation: str
    reason_calling: str
    opening_line: str
    secret_want: str
    specific_details: list[str]
    emotional_register: str
    # set after voice validation
    voice_resolved: Optional[str] = None

def parse_batch_response(raw: str) -> list[CallerIdentity]:
    data = json.loads(raw)
    callers = data.get("callers", [])
    result = []
    for c in callers:
        missing = REQUIRED_FIELDS - set(c.keys())
        if missing:
            raise ValueError(f"CallerIdentity missing fields: {missing}")
        result.append(CallerIdentity(**{k: c[k] for k in REQUIRED_FIELDS}))
    return result
```

**Step 4: Run test, verify pass**
Run: `/Users/lukemacneil/code/ai-podcast/venv/bin/python -m pytest tests/test_caller_gen.py -v`
Expected: 2 passed

**Step 5: Commit**
```bash
git add backend/services/caller_gen.py tests/test_caller_gen.py
git commit -m "Add CallerIdentity dataclass and batch JSON parser"
```

---

### Task 2: Voice roster validator

**Files:**
- Modify: `backend/services/caller_gen.py`
- Modify: `tests/test_caller_gen.py`

**Step 1: Write failing test**

```python
def test_resolve_voice_matches_exact():
    from backend.services.caller_gen import resolve_voice
    roster = ["Marcus", "Dennis", "Priya", "Edward"]
    assert resolve_voice("Marcus", roster) == "Marcus"

def test_resolve_voice_case_insensitive():
    from backend.services.caller_gen import resolve_voice
    roster = ["Marcus", "Dennis"]
    assert resolve_voice("marcus", roster) == "Marcus"

def test_resolve_voice_falls_back_when_no_match():
    from backend.services.caller_gen import resolve_voice
    roster = ["Marcus", "Dennis"]
    # Deterministic fallback: return first from roster
    assert resolve_voice("Santiago", roster) == "Marcus"

def test_resolve_voice_empty_suggestion_falls_back():
    from backend.services.caller_gen import resolve_voice
    assert resolve_voice("", ["Marcus"]) == "Marcus"
```

**Step 2: Run test, verify it fails (ImportError)**

**Step 3: Implement**

```python
# Add to backend/services/caller_gen.py
def resolve_voice(suggestion: str, roster: list[str]) -> str:
    """Map sonnet's voice suggestion to a real voice in the roster.
    Case-insensitive exact match; deterministic fallback to first roster entry."""
    if not suggestion or not roster:
        return roster[0] if roster else ""
    lower_map = {v.lower(): v for v in roster}
    return lower_map.get(suggestion.lower(), roster[0])
```

**Step 4: Run test, verify 4 pass**

**Step 5: Commit**
```bash
git add backend/services/caller_gen.py tests/test_caller_gen.py
git commit -m "Add voice roster validator for caller_gen"
```

---

### Task 3: Batch prompt builder

**Files:**
- Modify: `backend/services/caller_gen.py`
- Modify: `tests/test_caller_gen.py`

**Step 1: Write failing test**

```python
def test_build_batch_prompt_includes_context():
    from backend.services.caller_gen import build_batch_prompt
    ctx = {
        "date": "Saturday, April 5, 2026",
        "weather": "cool desert night, 48°F",
        "headlines": ["New Mexico legislature approves water bill"],
        "recent_caller_summaries": ["Jerry called about his neighbor's goat"],
        "regulars_included": [],
        "caller_count": 12,
        "voice_roster": ["Marcus", "Dennis", "Priya"],
    }
    prompt = build_batch_prompt(ctx)
    assert "Saturday, April 5, 2026" in prompt
    assert "water bill" in prompt
    assert "Jerry called about his neighbor's goat" in prompt
    assert "12 callers" in prompt
    assert "Marcus" in prompt  # voice roster listed
    assert "Stern" in prompt
    assert "Coast to Coast" in prompt
    assert "Loveline" in prompt
    assert "Delilah" in prompt
    assert "Opie and Anthony" in prompt

def test_build_batch_prompt_includes_silas_lore_when_present():
    from backend.services.caller_gen import build_batch_prompt
    ctx = {
        "date": "...",
        "weather": "...",
        "headlines": [],
        "recent_caller_summaries": [],
        "regulars_included": [{"name": "Silas", "lore": "Silas leads a small desert cult...", "arc_state": "seeking new members"}],
        "caller_count": 12,
        "voice_roster": ["Marcus"],
    }
    prompt = build_batch_prompt(ctx)
    assert "Silas" in prompt
    assert "desert cult" in prompt
    assert "seeking new members" in prompt
    assert "DO NOT alter his voice, personality, or core traits" in prompt
```

**Step 2: Run test, verify it fails**

**Step 3: Implement `build_batch_prompt`**

```python
# Add to backend/services/caller_gen.py
BATCH_SYSTEM_PROMPT = """You are writing a roster of callers for Luke's late-night radio show in New Mexico.

CREATIVE RANGE: Your callers must span the emotional range of Howard Stern (chaos, strong characters), Coast to Coast AM (earnest weirdos, sincere believers), Loveline (real problems, real advice-seeking), Delilah (emotional vulnerability, connection), and Opie and Anthony (sharp, irreverent, specific people).

Maximum character distance between callers. No two callers should feel like siblings. Do not default to sitcom plots. Real humans are specific and strange. Give each caller details that could only belong to them.

You will output strict JSON with a "callers" array. Each caller has exactly these fields: name, age, voice_suggestion, location, identity, situation, reason_calling, opening_line, secret_want, specific_details (array of 2-3 strings), emotional_register."""

def build_batch_prompt(ctx: dict) -> str:
    lines = [
        f"Tonight is {ctx['date']}. {ctx['weather']}.",
        "",
        "Today's news headlines (ground callers in real context, but do not force topicality):",
    ]
    for h in ctx["headlines"]:
        lines.append(f"- {h}")
    lines.append("")

    if ctx["recent_caller_summaries"]:
        lines.append("Recent callers (DO NOT repeat these archetypes or situations):")
        for s in ctx["recent_caller_summaries"]:
            lines.append(f"- {s}")
        lines.append("")

    if ctx["regulars_included"]:
        lines.append("RECURRING CHARACTERS IN TONIGHT'S LINEUP:")
        lines.append("")
        for r in ctx["regulars_included"]:
            lines.append(f"### {r['name']}")
            lines.append(r["lore"])
            lines.append(f"Current arc state: {r['arc_state']}")
            lines.append("")
            lines.append(f"For {r['name']}: invent a fresh reason he is calling tonight — a new development, grievance, or specific recent event. DO NOT alter his voice, personality, or core traits. Write a new scene for an existing character.")
            lines.append("")

    lines.append(f"Available voices (voice_suggestion must match one of these exactly):")
    lines.append(", ".join(ctx["voice_roster"]))
    lines.append("")
    lines.append(f"Generate {ctx['caller_count']} callers. Output JSON only, no prose.")
    return BATCH_SYSTEM_PROMPT + "\n\n" + "\n".join(lines)
```

**Step 4: Run test, verify pass**

**Step 5: Commit**
```bash
git add backend/services/caller_gen.py tests/test_caller_gen.py
git commit -m "Add batch prompt builder for caller_gen"
```

---

### Task 4: Batch generation function (LLM call)

**Files:**
- Modify: `backend/services/caller_gen.py`

**Step 1: Add generation function**

```python
# Add to backend/services/caller_gen.py
import httpx
from ..config import settings
from .cost_tracker import cost_tracker

BATCH_MODEL = "anthropic/claude-sonnet-4.6"

async def generate_batch(ctx: dict) -> list[CallerIdentity]:
    """Call sonnet-4.6 with the batch prompt, parse + voice-resolve the response."""
    prompt = build_batch_prompt(ctx)
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            json={
                "model": BATCH_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "max_tokens": 8000,
                "temperature": 0.9,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    cost_tracker.record_llm_call(
        category="background_gen",
        model=BATCH_MODEL,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        caller_name=None,
    )

    callers = parse_batch_response(content)
    for c in callers:
        c.voice_resolved = resolve_voice(c.voice_suggestion, ctx["voice_roster"])
    return callers
```

**Step 2: Manual smoke test** (network call, requires OPENROUTER_API_KEY)

Create `scratch/test_batch_gen.py`:
```python
import asyncio
from backend.services.caller_gen import generate_batch

async def main():
    ctx = {
        "date": "Saturday, April 5, 2026",
        "weather": "cool desert night, 48°F",
        "headlines": ["NM legislature approves water bill"],
        "recent_caller_summaries": [],
        "regulars_included": [],
        "caller_count": 4,
        "voice_roster": ["Marcus", "Dennis", "Priya", "Edward"],
    }
    callers = await generate_batch(ctx)
    for c in callers:
        print(f"\n=== {c.name} ({c.age}, {c.location}) voice={c.voice_resolved}")
        print(f"identity: {c.identity}")
        print(f"situation: {c.situation}")
        print(f"reason_calling: {c.reason_calling}")
        print(f"opening_line: {c.opening_line}")

asyncio.run(main())
```

Run: `/Users/lukemacneil/code/ai-podcast/venv/bin/python scratch/test_batch_gen.py`
Expected: 4 distinct, specific caller identities printed. Manually verify they feel distinct + grounded.

**Step 3: Commit (keep scratch file out of git)**
```bash
echo "scratch/" >> .gitignore
git add backend/services/caller_gen.py .gitignore
git commit -m "Add batch generation function using sonnet-4.6"
```

---

## Phase 2 — Regulars v2

### Task 5: Lore file loader (Obsidian markdown)

**Files:**
- Create: `backend/services/regulars_v2.py`
- Test: `tests/test_regulars_v2.py`

**Step 1: Write failing tests**

```python
# tests/test_regulars_v2.py
import tempfile
from pathlib import Path
from backend.services.regulars_v2 import Regular, load_regular, REGULARS_DIR, SILAS_DIR

def test_load_regular_parses_frontmatter_and_body(tmp_path):
    lore_file = tmp_path / "silas.md"
    lore_file.write_text("""---
name: Silas
voice: Dennis
age: 54
arc_state: Cult is splintering after the eclipse failure
---

# Silas

Silas runs a small desert cult outside Truth or Consequences...

## Arc Log

- 2026-03-01: First call, introduced the cult
- 2026-03-20: Prophesied the eclipse
""")
    reg = load_regular(lore_file)
    assert reg.name == "Silas"
    assert reg.voice == "Dennis"
    assert reg.age == 54
    assert "splintering" in reg.arc_state
    assert "Silas runs a small desert cult" in reg.lore_body
```

**Step 2: Run test, verify fail**

**Step 3: Implement**

```python
# backend/services/regulars_v2.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re

HOME = Path.home()
VAULT = HOME / "code" / "dotfiles"
SILAS_DIR = VAULT / "silas"
REGULARS_DIR = VAULT / "regulars"
ARCHIVED_DIR = REGULARS_DIR / "archived"

@dataclass
class Regular:
    name: str
    voice: str
    age: int
    arc_state: str
    lore_body: str
    file_path: Path

def load_regular(path: Path) -> Regular:
    text = path.read_text()
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not m:
        raise ValueError(f"No frontmatter in {path}")
    fm_raw, body = m.group(1), m.group(2).strip()
    fm = {}
    for line in fm_raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    return Regular(
        name=fm["name"],
        voice=fm["voice"],
        age=int(fm["age"]),
        arc_state=fm.get("arc_state", ""),
        lore_body=body,
        file_path=path,
    )

def load_all_active_regulars() -> list[Regular]:
    out = []
    if SILAS_DIR.exists():
        for f in SILAS_DIR.glob("*.md"):
            out.append(load_regular(f))
    if REGULARS_DIR.exists():
        for f in REGULARS_DIR.glob("*.md"):
            out.append(load_regular(f))
    return out
```

**Step 4: Run test, verify pass**

**Step 5: Commit**
```bash
git add backend/services/regulars_v2.py tests/test_regulars_v2.py
git commit -m "Add Regular dataclass + lore file loader"
```

---

### Task 6: Bootstrap Silas lore file in Obsidian vault

**Files:**
- Create: `~/code/dotfiles/silas/silas.md`

**Step 1: Inspect current Silas data**
Run: `/Users/lukemacneil/code/ai-podcast/venv/bin/python -c "import json; d=json.load(open('data/regulars.json')); silas=[r for r in d if r.get('name','').lower()=='silas']; print(json.dumps(silas, indent=2))"`

**Step 2: Create the lore file by hand-crafting from existing data**

File contents (template — adjust based on step 1 findings):
```markdown
---
name: Silas
voice: [fill from existing data]
age: [fill from existing data]
arc_state: [current ongoing thread]
---

# Silas

[Paragraph describing who Silas is — cult leader, charismatic weirdness, his relationship to Luke, his core essence. Draw from existing regulars.json key_moments + background.]

## Canonical Traits (frozen)

- [trait 1]
- [trait 2]
- [trait 3]

## Arc Log

- YYYY-MM-DD: [event from history]
- YYYY-MM-DD: [event from history]
```

**Step 3: Commit to the Obsidian vault (dotfiles repo)**
```bash
cd ~/code/dotfiles
git add silas/silas.md
git commit -m "Add Silas canonical lore file for ai-podcast caller generation"
```

**Step 4: Back in caller-redesign worktree, verify loader picks it up**
```bash
cd /Users/lukemacneil/code/ai-podcast/.worktrees/caller-redesign
/Users/lukemacneil/code/ai-podcast/venv/bin/python -c "from backend.services.regulars_v2 import load_all_active_regulars; print([r.name for r in load_all_active_regulars()])"
```
Expected: `['Silas']`

---

### Task 7: Archive current regulars (except Silas)

**Files:**
- Modify: `data/regulars.json` (filter to Silas only)
- Create: `data/regulars.archived.json` (full backup)

**Step 1: Write migration script**

Create `scripts/archive_regulars.py`:
```python
import json
from pathlib import Path

src = Path("data/regulars.json")
data = json.loads(src.read_text())

Path("data/regulars.archived.json").write_text(json.dumps(data, indent=2))

silas_only = [r for r in data if r.get("name", "").lower() == "silas"]
src.write_text(json.dumps(silas_only, indent=2))
print(f"Archived {len(data)} regulars. Kept {len(silas_only)} (Silas).")
```

**Step 2: Run migration**
```bash
cd /Users/lukemacneil/code/ai-podcast/.worktrees/caller-redesign
/Users/lukemacneil/code/ai-podcast/venv/bin/python scripts/archive_regulars.py
```

**Step 3: Commit**
```bash
git add data/regulars.json data/regulars.archived.json scripts/archive_regulars.py
git commit -m "Archive non-Silas regulars for redesign cutover"
```

---

### Task 8: Promotion gate (post-call LLM evaluation)

**Files:**
- Modify: `backend/services/regulars_v2.py`
- Modify: `tests/test_regulars_v2.py`

**Step 1: Write failing test with mocked LLM**

```python
# Add to tests/test_regulars_v2.py
import pytest
from unittest.mock import AsyncMock, patch
from backend.services.regulars_v2 import evaluate_promotion

@pytest.mark.asyncio
async def test_evaluate_promotion_returns_arc_plan_when_worthy():
    fake_response = {
        "promote": True,
        "arc_plan": "3 episodes. He'll start distant, then reveal he's actually the one who damaged the car, then resolve with an apology.",
        "reason": "Has clear internal conflict with room to grow",
    }
    with patch("backend.services.regulars_v2._call_sonnet", new=AsyncMock(return_value=fake_response)):
        result = await evaluate_promotion(caller_name="Bobby", call_transcript="...")
    assert result["promote"] is True
    assert "3 episodes" in result["arc_plan"]

@pytest.mark.asyncio
async def test_evaluate_promotion_rejects_when_no_arc():
    fake_response = {"promote": False, "arc_plan": None, "reason": "One-note complaint, no growth"}
    with patch("backend.services.regulars_v2._call_sonnet", new=AsyncMock(return_value=fake_response)):
        result = await evaluate_promotion(caller_name="Carl", call_transcript="...")
    assert result["promote"] is False
```

**Step 2: Run test, verify fail**

**Step 3: Implement**

```python
# Add to backend/services/regulars_v2.py
import httpx
import json
from ..config import settings
from .cost_tracker import cost_tracker

PROMOTION_MODEL = "anthropic/claude-sonnet-4.6"

PROMOTION_PROMPT = """You are evaluating whether a one-time caller should become a recurring character.

CALLER: {name}
TRANSCRIPT:
{transcript}

A recurring character must have a 3-5 episode arc with genuine progression — not just "calls weekly to complain about the same thing." The arc must have a possible resolution.

Output JSON:
{{"promote": true|false, "arc_plan": "...", "reason": "..."}}

Bar is HIGH. Only promote if the character has real internal conflict, growth potential, and a believable resolution trajectory."""

async def _call_sonnet(prompt: str) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            json={
                "model": PROMOTION_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "max_tokens": 500,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    usage = data.get("usage", {})
    cost_tracker.record_llm_call(
        category="promotion_eval",
        model=PROMOTION_MODEL,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        caller_name=None,
    )
    return json.loads(data["choices"][0]["message"]["content"])

async def evaluate_promotion(caller_name: str, call_transcript: str) -> dict:
    prompt = PROMOTION_PROMPT.format(name=caller_name, transcript=call_transcript)
    return await _call_sonnet(prompt)
```

**Step 4: Run test, verify pass**

**Step 5: Commit**
```bash
git add backend/services/regulars_v2.py tests/test_regulars_v2.py
git commit -m "Add promotion gate for tier-2 regulars"
```

---

### Task 9: Create promoted-regular writer

**Files:**
- Modify: `backend/services/regulars_v2.py`
- Modify: `tests/test_regulars_v2.py`

**Step 1: Write failing test**

```python
def test_write_new_regular_creates_lore_file(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.services.regulars_v2.REGULARS_DIR", tmp_path)
    from backend.services.regulars_v2 import write_new_regular

    write_new_regular(
        name="Bobby",
        voice="Marcus",
        age=34,
        identity_paragraph="A landscaper in Las Cruces who...",
        arc_plan="3 episodes: distant → reveal → apology",
        first_call_summary="Called about damaged car",
    )
    f = tmp_path / "bobby.md"
    assert f.exists()
    body = f.read_text()
    assert "name: Bobby" in body
    assert "voice: Marcus" in body
    assert "A landscaper in Las Cruces" in body
    assert "3 episodes: distant → reveal → apology" in body
```

**Step 2: Run test, verify fail**

**Step 3: Implement**

```python
# Add to backend/services/regulars_v2.py
from datetime import date

def write_new_regular(name: str, voice: str, age: int, identity_paragraph: str,
                       arc_plan: str, first_call_summary: str) -> Path:
    REGULARS_DIR.mkdir(parents=True, exist_ok=True)
    slug = name.lower().replace(" ", "-")
    path = REGULARS_DIR / f"{slug}.md"
    today = date.today().isoformat()
    content = f"""---
name: {name}
voice: {voice}
age: {age}
arc_state: {arc_plan}
promoted_on: {today}
---

# {name}

{identity_paragraph}

## Arc Plan

{arc_plan}

## Arc Log

- {today}: {first_call_summary}
"""
    path.write_text(content)
    return path
```

**Step 4: Run test, verify pass**

**Step 5: Commit**
```bash
git add backend/services/regulars_v2.py tests/test_regulars_v2.py
git commit -m "Add writer for promoted tier-2 regulars"
```

---

## Phase 3 — Integration into main.py

### Task 10: Slim caller prompt builder

**Files:**
- Modify: `backend/main.py` (add new `get_caller_prompt_slim` function alongside existing)
- Create: `tests/test_caller_prompt_slim.py`

**Step 1: Write failing test**

```python
# tests/test_caller_prompt_slim.py
from backend.main import get_caller_prompt_slim

def test_slim_prompt_includes_identity_and_situation():
    caller = {
        "name": "Danny",
        "identity": "A plumber who inherited a taxidermy shop",
        "situation": "Getting strange calls about taxidermy",
        "reason_calling": "Someone left a note",
        "secret_want": "Permission to throw it all away",
        "specific_details": ["elk head in basement", "note said she forgot"],
    }
    prompt = get_caller_prompt_slim(caller)
    assert "Danny" in prompt
    assert "taxidermy shop" in prompt
    assert "strange calls" in prompt
    assert "elk head" in prompt
    assert "she forgot" in prompt
    assert "Permission to throw it all away" in prompt
    assert "React to what Luke says" in prompt
    assert "Stay in character" in prompt
    # Assert it's under 800 tokens (roughly 3200 chars) — should be ~400 tokens
    assert len(prompt) < 3200
```

**Step 2: Run test, verify fail**

**Step 3: Implement in `backend/main.py`**

Add near existing `get_caller_prompt` (around line 6556):

```python
def get_caller_prompt_slim(caller: dict) -> str:
    """Slim caller system prompt. Identity carries the weight."""
    name = caller.get("name", "")
    identity = caller.get("identity", "")
    situation = caller.get("situation", "")
    reason = caller.get("reason_calling", "")
    want = caller.get("secret_want", "")
    details = caller.get("specific_details", []) or []
    detail_str = " | ".join(f"- {d}" for d in details)

    return f"""You are {name}. {identity}

You're calling Luke's late-night radio show because: {situation} — specifically, {reason}.

What you secretly want from this call: {want}

Specific details you'll drop if it feels natural:
{detail_str}

Speak as this person. React to what Luke says. Stay in character.
Don't narrate. No stage directions. Just talk.
Keep responses natural — 1-3 sentences most of the time. Real callers don't monologue."""
```

**Step 4: Run test, verify pass**

**Step 5: Commit**
```bash
git add backend/main.py tests/test_caller_prompt_slim.py
git commit -m "Add slim caller prompt builder (~400 tokens)"
```

---

### Task 11: Switch dialog model to haiku-4.5 (behind flag)

**Files:**
- Modify: `backend/main.py` around `Session.get_caller_model` (~line 6894)

**Step 1: Read current implementation**
Read `backend/main.py:6800-6900` to find `Session.get_caller_model` and the `caller_model_map`.

**Step 2: Add feature flag check**

Modify `Session.get_caller_model` to accept a flag, returning haiku when enabled:

```python
# Add class-level attribute on Session:
use_slim_caller_path: bool = False  # flipped by env var or settings

# Modify get_caller_model:
def get_caller_model(self, caller_key: str) -> str:
    if self.use_slim_caller_path:
        return "anthropic/claude-haiku-4.5"
    # ... existing style-matched logic unchanged ...
```

**Step 3: Add env var reader in Session.__init__**

```python
import os
self.use_slim_caller_path = os.environ.get("CALLER_REDESIGN", "0") == "1"
```

**Step 4: Commit**
```bash
git add backend/main.py
git commit -m "Add feature flag for slim caller path (haiku-4.5)"
```

---

### Task 12: Wire batch gen into Session.reset (behind flag)

**Files:**
- Modify: `backend/main.py` (Session.reset + caller setup)

**Step 1: Read current `_pregenerate_backgrounds`** at `backend/main.py:5908`.

**Step 2: Add new batch path**

Create an async method on Session (new code, not replacing):

```python
async def _pregenerate_backgrounds_slim(self):
    """New path: single sonnet-4.6 batch call generates all caller identities."""
    from .services import caller_gen, regulars_v2
    from datetime import datetime

    voice_roster = [name for name in INWORLD_MALE + INWORLD_FEMALE
                    if name not in BLACKLISTED_VOICES]

    active_regulars = regulars_v2.load_all_active_regulars()
    # Include Silas always if he exists; tier-2 based on arc state (implement later)
    regulars_for_tonight = [
        {"name": r.name, "lore": r.lore_body, "arc_state": r.arc_state}
        for r in active_regulars
    ][:3]  # cap at 3 regulars max

    ctx = {
        "date": datetime.now().strftime("%A, %B %d, %Y"),
        "weather": "cool desert night",  # TODO: real weather feed
        "headlines": self.news_headlines[:5] if self.news_headlines else [],
        "recent_caller_summaries": self._get_recent_summaries(),
        "regulars_included": regulars_for_tonight,
        "caller_count": 12,
        "voice_roster": voice_roster,
    }

    identities = await caller_gen.generate_batch(ctx)

    # Map identities into CALLER_BASES slots
    for i, identity in enumerate(identities[:10]):
        key = str(i + 1)
        self.caller_backgrounds[key] = {
            "name": identity.name,
            "age": identity.age,
            "voice": identity.voice_resolved,
            "location": identity.location,
            "identity": identity.identity,
            "situation": identity.situation,
            "reason_calling": identity.reason_calling,
            "opening_line": identity.opening_line,
            "secret_want": identity.secret_want,
            "specific_details": identity.specific_details,
            "emotional_register": identity.emotional_register,
        }

def _get_recent_summaries(self) -> list[str]:
    # Return last 2 shows' caller summaries — stub for now, can wire into cost_db
    return []
```

**Step 3: Call it from Session.reset when flag is on**

Find `_pregenerate_backgrounds` call in Session.reset, replace with:

```python
if self.use_slim_caller_path:
    asyncio.create_task(self._pregenerate_backgrounds_slim())
else:
    asyncio.create_task(self._pregenerate_backgrounds())
```

**Step 4: Route caller prompt to slim builder when flag is on**

In the caller-prompt assembly code (wherever `get_caller_prompt` is called), add:

```python
if session.use_slim_caller_path:
    system_prompt = get_caller_prompt_slim(caller)
else:
    system_prompt = get_caller_prompt(caller, ...)
```

**Step 5: Smoke test** — start server with flag on
```bash
cd /Users/lukemacneil/code/ai-podcast/.worktrees/caller-redesign
CALLER_REDESIGN=1 /Users/lukemacneil/code/ai-podcast/venv/bin/python -m uvicorn backend.main:app --reload-dir backend --host 0.0.0.0 --port 8000
```
In browser, click "Reset session" or similar, verify logs show batch gen happening.

**Step 6: Commit**
```bash
git add backend/main.py
git commit -m "Wire batch gen into Session.reset behind CALLER_REDESIGN flag"
```

---

## Phase 4 — Validation Gate

### Task 13: Sample call generation script

**Files:**
- Create: `scripts/generate_sample_calls.py`

**Step 1: Implement**

```python
# scripts/generate_sample_calls.py
"""Generate 10 sample caller dialogues for user validation before cutover.

5 with Silas (if lore exists), 5 walk-ins. Writes transcripts to docs/samples/.
"""
import asyncio
import json
from pathlib import Path
from datetime import datetime
import httpx

from backend.services import caller_gen, regulars_v2
from backend.services.tts import INWORLD_MALE, INWORLD_FEMALE
from backend.main import get_caller_prompt_slim
from backend.config import settings

DIALOG_MODEL = "anthropic/claude-haiku-4.5"
HOST_PROMPTS = [
    "Hey, what's going on tonight?",
    "So what's the story?",
    "Tell me more about that.",
    "Wait — really? When did that happen?",
    "Okay, and what did you do?",
]

async def dialog_turn(system_prompt: str, conversation: list) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            json={
                "model": DIALOG_MODEL,
                "messages": [{"role": "system", "content": system_prompt}] + conversation,
                "max_tokens": 300,
                "temperature": 0.9,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

async def main():
    out_dir = Path("docs/samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    voice_roster = [n for n in INWORLD_MALE + INWORLD_FEMALE]

    # Batch 1: with Silas
    regulars = regulars_v2.load_all_active_regulars()
    if regulars:
        ctx_with_silas = {
            "date": datetime.now().strftime("%A, %B %d, %Y"),
            "weather": "cool desert night",
            "headlines": [],
            "recent_caller_summaries": [],
            "regulars_included": [{"name": r.name, "lore": r.lore_body, "arc_state": r.arc_state} for r in regulars],
            "caller_count": 5,
            "voice_roster": voice_roster,
        }
        silas_batch = await caller_gen.generate_batch(ctx_with_silas)
    else:
        silas_batch = []

    # Batch 2: walk-ins
    ctx_walkins = {
        "date": datetime.now().strftime("%A, %B %d, %Y"),
        "weather": "cool desert night",
        "headlines": [],
        "recent_caller_summaries": [],
        "regulars_included": [],
        "caller_count": 5,
        "voice_roster": voice_roster,
    }
    walkin_batch = await caller_gen.generate_batch(ctx_walkins)

    all_callers = silas_batch + walkin_batch
    for caller in all_callers:
        cdict = {
            "name": caller.name,
            "identity": caller.identity,
            "situation": caller.situation,
            "reason_calling": caller.reason_calling,
            "secret_want": caller.secret_want,
            "specific_details": caller.specific_details,
        }
        system_prompt = get_caller_prompt_slim(cdict)
        conversation = [{"role": "assistant", "content": caller.opening_line}]

        for host_line in HOST_PROMPTS:
            conversation.append({"role": "user", "content": host_line})
            reply = await dialog_turn(system_prompt, conversation)
            conversation.append({"role": "assistant", "content": reply})

        transcript = [f"CALLER ({caller.name}, {caller.age}, {caller.location}): {conversation[0]['content']}"]
        for i in range(1, len(conversation), 2):
            transcript.append(f"LUKE: {conversation[i]['content']}")
            if i + 1 < len(conversation):
                transcript.append(f"CALLER: {conversation[i+1]['content']}")

        fname = out_dir / f"sample_{caller.name.replace(' ', '_').lower()}.txt"
        fname.write_text(
            f"=== {caller.name} ({caller.age}, {caller.location}) ===\n"
            f"voice: {caller.voice_resolved}\n"
            f"emotional_register: {caller.emotional_register}\n"
            f"secret_want: {caller.secret_want}\n\n"
            + "\n".join(transcript)
        )
        print(f"Wrote {fname}")

asyncio.run(main())
```

**Step 2: Run it**
```bash
cd /Users/lukemacneil/code/ai-podcast/.worktrees/caller-redesign
/Users/lukemacneil/code/ai-podcast/venv/bin/python scripts/generate_sample_calls.py
```

Expected: 10 files in `docs/samples/`.

**Step 3: Commit**
```bash
git add scripts/generate_sample_calls.py docs/samples/
git commit -m "Add sample call generator for validation gate"
```

---

### Task 14: **USER VALIDATION CHECKPOINT** 🛑

**STOP HERE. User must:**

1. Read all 10 sample transcripts in `docs/samples/`
2. Verify Silas (if present) still sounds like Silas — voice, personality, essence intact
3. Verify walk-ins span the emotional range (some earnest, some chaotic, some vulnerable, some absurd)
4. Verify callers feel distinct, not sibling-like
5. Verify dialog responses are terse, specific, and reactive to the host

**Decision:**
- ✅ Approve → proceed to Phase 5 (deletion)
- ❌ Drift/bland → iterate prompts in `caller_gen.py` `BATCH_SYSTEM_PROMPT` + `get_caller_prompt_slim`, regenerate, re-validate

---

## Phase 5 — Deletion (only after user approval)

### Task 15: Delete static content pools

**Files:**
- Modify: `backend/main.py`

**Step 1: Remove these constants** (grep to find their definitions):
- `PROBLEMS`, `STORIES`, `GOSSIP`, `ADVICE`, `TOPIC_CALLIN`, `CELEBRATIONS`, `WEIRD`, `HOT_TAKES`

**Step 2: Remove functions that use them:**
- `_generate_pool_weights`
- `_pick_unique_reason`

**Step 3: Remove references** (grep for each pool name, delete lines that use them).

**Step 4: Run tests**
```bash
/Users/lukemacneil/code/ai-podcast/venv/bin/python -m pytest tests/ -v
```
Fix any test failures caused by removals.

**Step 5: Commit**
```bash
git add backend/main.py
git commit -m "Delete static content pools (PROBLEMS, STORIES, etc.)"
```

---

### Task 16: Delete color-detail pools

**Files:**
- Modify: `backend/main.py`

**Step 1: Remove these constants:**
`INTERESTS`, `QUIRKS`, `RELATIONSHIP_STATUS`, `VEHICLES`, `BEFORE_CALLING`, `CALLING_FROM`, `MEMORIES`, `HAVING_RIGHT_NOW`, `STRONG_OPINIONS`, `CONTRADICTIONS`, `VERBAL_TICS`, `EMOTIONAL_ARCS`, `SHOW_RELATIONSHIP`, `LATE_NIGHT_REASONS`, `DRIFT_TENDENCIES`, `ROAD_CONTEXT`, `PHONE_SITUATION`, `BACKGROUND_MUSIC`, `RECENT_ERRAND`, `TV_TONIGHT`, `LOCAL_FOOD_OPINIONS`, `NOSTALGIA`

**Step 2: Remove keyword filter lists:**
`_SPICY_KEYWORDS`, `_ABSURD_KEYWORDS`, `_HEAVY_POOLS`, `_LIGHT_POOLS`, `_HEAVY_STYLES`, `_LIGHT_STYLES`, `_EVASIVE_STYLES`

**Step 3: Run tests, fix breakages, commit**
```bash
git add backend/main.py
git commit -m "Delete color-detail pools and keyword filters"
```

---

### Task 17: Delete style system

**Files:**
- Modify: `backend/main.py`

**Remove:**
- `CALLER_STYLES` (18-style dict)
- `CALLER_STYLE_KEYS`
- `STYLE_VOICE_PREFERENCES`
- `STYLE_SPEED_MODIFIERS`
- `STYLE_PHONE_QUALITY`
- `_pick_caller_style` function
- `caller_model_map`, `_CALLER_DIALOG_MODEL_PARAMS`
- style-matching branches in `Session.get_caller_model` (should return haiku unconditionally now)

**Commit:**
```bash
git add backend/main.py
git commit -m "Delete style system (CALLER_STYLES, style-to-model map)"
```

---

### Task 18: Delete shape system

**Files:**
- Modify: `backend/main.py`

**Remove:**
- `CALL_SHAPES`
- `SHAPE_STYLE_AFFINITIES`
- `SHAPE_DIRECTIVES` (big dict)
- `_LATE_SHOW_SHAPES`
- `_pick_call_shape`
- `_assign_call_shape`
- All `shape`-related fields from CallerBackground and downstream uses

**Commit:**
```bash
git add backend/main.py
git commit -m "Delete call-shape system"
```

---

### Task 19: Delete voice matching and queue sort

**Files:**
- Modify: `backend/main.py`

**Remove:**
- `_match_voices_to_styles` function
- Voice-scoring logic (uses VOICE_PROFILES dimensions)
- `_sort_caller_queue` (greedy placement scoring)
- `SHOW_HISTORY_REACTIONS` + adaptive reaction frequency
- `_build_relationship_context` (inter-caller thematic scoring)

**Commit:**
```bash
git add backend/main.py
git commit -m "Delete voice matching, queue sort, thematic scoring"
```

---

### Task 20: Delete template fallback + remove flag

**Files:**
- Modify: `backend/main.py`

**Remove:**
- `generate_caller_background` (template-based fallback, ~400 lines)
- `_pregenerate_backgrounds` (old path — keep only `_pregenerate_backgrounds_slim` and rename it to drop `_slim`)
- Old `get_caller_prompt` (keep `get_caller_prompt_slim`, rename to `get_caller_prompt`)
- `CALLER_REDESIGN` env var + `use_slim_caller_path` flag (always-on now)

**Commit:**
```bash
git add backend/main.py
git commit -m "Remove template fallback, old generation path, feature flag"
```

---

### Task 21: Final test sweep + reload documentation

**Step 1: Run all tests**
```bash
/Users/lukemacneil/code/ai-podcast/venv/bin/python -m pytest tests/ -v
```
Expected: All pass.

**Step 2: Line count diff**
```bash
git diff main --stat backend/main.py
```
Expected: ~2000 lines removed, ~300 added.

**Step 3: Update CLAUDE.md**

Update the "Caller Generation System" section in `CLAUDE.md` to describe the new architecture (replace old description of CallerBackground dataclass, voice-personality matching, SHAPE_STYLE_AFFINITIES, etc.).

**Step 4: Commit**
```bash
git add CLAUDE.md
git commit -m "Update CLAUDE.md to document new caller generation architecture"
```

---

### Task 22: Ship decision

Stop here. Merge strategy is user's call:
- PR on GitHub via `/push` flow
- Direct merge to main
- More test shows on the feature branch first

---

## Post-Ship Work (out of scope for this plan)

- Real weather feed integration (currently "cool desert night" hardcoded)
- Recent-shows summary feed from `data/costs.db` for anti-repeat context
- Arc retirement automation (detect resolved arcs, move to `regulars/archived/`)
- Post-call promotion trigger wired into hangup flow
- Pre-generated opening-line audio (cut latency when caller clicked)
