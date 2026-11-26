# Devon Opening Headlines + Show Facts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Devon opens each show knowing a curated mix of fresh headlines, and Devon + callers never lose host-established facts (e.g. "we're in West Virginia tonight").

**Architecture:** A new `show_memory` service singleton (same pattern as `news_service`/`intern_service`) owns a show-wide host-line log, a fact list maintained by a Haiku extractor + manual edits, and the curated opening headlines. Devon's system prompt is prefixed with its context block; callers get a compact facts block in `get_caller_prompt()`. State is reset per show and persisted in `session_checkpoint.json`.

**Tech Stack:** FastAPI, httpx, OpenRouter via `llm_service.generate`, SearXNG, pytest (`asyncio.run` style, no pytest-asyncio), vanilla JS.

Design: `docs/plans/2026-11-25-devon-headlines-show-facts-design.md`

**Git:** the working tree on `main` already has ~25 unrelated modified files (config.py, main.py, intern.py, llm.py, tests…). Never use `git commit -a`; stage only this task's files/hunks. Only commit if the user has asked for commits.

Run tests with: `venv/bin/python -m pytest -q` (baseline: 193 passed).

**Known pre-existing facts the engineer needs:**
- `session.conversation` is cleared at every `start_call`/`end_call` (`backend/main.py:783-793`) — never rely on it for show-wide memory.
- `/api/intern/ask` (`backend/main.py:~5063`) logs only Devon's answer, not the host's question.
- `_fetch_session_headlines()` (`backend/main.py:1584`) is never called; `session.news_headlines` is always empty. Don't delete it (out of scope); just stop depending on it for callers.
- Caller backgrounds may be pre-warmed before the show (`Session.populate_backgrounds`, `backend/main.py:1163`). Callers in a pre-warmed lineup get whatever curated headlines existed when the batch was built.

---

### Task 1: Route a `show_facts` LLM category to Haiku

**Files:**
- Modify: `backend/config.py` (the `category_models` dict, ~line 41)
- Test: `tests/test_model_config.py`

**Step 1: Write the failing test** — append to `tests/test_model_config.py`:

```python
def test_show_facts_routes_to_cheap_model_with_reasoning_off():
    from backend.services.llm import REASONING_EFFORT
    model = settings.category_models.get("show_facts")
    assert model == "anthropic/claude-haiku-5.5"
    assert REASONING_EFFORT.get(model) == "none"
```

**Step 2:** `venv/bin/python -m pytest tests/test_model_config.py -q` → FAIL (`None == 'anthropic/claude-haiku-5.5'`).

**Step 3:** In `category_models` add after `"topic_gen"`:

```python
        "show_facts": "anthropic/claude-haiku-5.5",          # extracts durable show facts from host lines ($0.10/$0.50)
```

**Step 4:** Re-run → PASS.

**Step 5:** `git add -p backend/config.py tests/test_model_config.py && git commit -m "Route show_facts category to Haiku"` (both files have unrelated uncommitted edits — stage only this task's hunks)

---

### Task 2: Fact store — pure merge logic

**Files:**
- Create: `backend/services/show_memory.py`
- Test: `tests/test_show_memory.py`

**Step 1: Write failing tests** — `tests/test_show_memory.py`:

```python
from backend.services.show_memory import ShowMemory


def _mem(*texts, pinned=False):
    m = ShowMemory()
    for t in texts:
        m.add_fact(t, pinned=pinned)
    return m


def test_extractor_correction_replaces_fact():
    m = _mem("Luke is broadcasting from West Virginia tonight")
    fid = m.facts[0]["id"]
    m.apply_ops([{"op": "replace", "id": fid, "text": "Luke is back in Alpine tonight"}])
    assert [f["text"] for f in m.facts] == ["Luke is back in Alpine tonight"]


def test_pinned_fact_survives_extractor_ops():
    m = _mem("Luke's back is out", pinned=True)
    fid = m.facts[0]["id"]
    m.apply_ops([{"op": "remove", "id": fid}, {"op": "replace", "id": fid, "text": "x"}])
    assert m.facts[0]["text"] == "Luke's back is out"


def test_duplicate_add_is_skipped():
    m = _mem("Luke is in West Virginia")
    m.apply_ops([{"op": "add", "text": "  luke is in west virginia. "}])
    assert len(m.facts) == 1


def test_deleted_fact_is_not_re_added_by_extractor():
    m = _mem("Devon owes Luke $20")
    m.delete_fact(m.facts[0]["id"])
    m.apply_ops([{"op": "add", "text": "Devon owes Luke $20"}])
    assert m.facts == []


def test_malformed_ops_change_nothing():
    m = _mem("Luke is in West Virginia")
    before = [dict(f) for f in m.facts]
    m.apply_ops([{"op": "explode"}, {"nope": 1}, "garbage", {"op": "replace", "id": 999, "text": "x"}])
    assert m.facts == before


def test_fact_list_is_capped():
    m = ShowMemory()
    m.apply_ops([{"op": "add", "text": f"fact number {i}"} for i in range(30)])
    assert len(m.facts) == ShowMemory.MAX_FACTS


def test_round_trips_through_dict():
    m = _mem("Luke is in West Virginia", pinned=True)
    m.mark_call_started()
    m2 = ShowMemory()
    m2.load_dict(m.to_dict())
    assert m2.facts == m.facts and m2.calls_started is True
```

**Step 2:** `venv/bin/python -m pytest tests/test_show_memory.py -q` → FAIL (module not found).

**Step 3: Implement** — `backend/services/show_memory.py`:

```python
"""Show-wide memory: host-established facts + Devon's curated opening headlines.

Lives for one show. session.conversation is wiped every call, so anything that
must survive the whole show (where Luke is tonight, running bits) lives here."""

import re
import time


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


class ShowMemory:
    MAX_FACTS = 12

    def __init__(self):
        self.reset()

    def reset(self):
        self.facts: list[dict] = []
        self.host_lines: list[str] = []
        self.processed_lines = 0
        self.dismissed: set[str] = set()
        self.headlines: list[dict] = []
        self.calls_started = False
        self._next_id = 1

    # --- facts ---
    def add_fact(self, text: str, pinned: bool = False) -> dict | None:
        text = text.strip()
        if not text or len(self.facts) >= self.MAX_FACTS:
            return None
        if any(_norm(f["text"]) == _norm(text) for f in self.facts):
            return None
        fact = {"id": self._next_id, "text": text, "pinned": pinned, "updated_at": time.time()}
        self._next_id += 1
        self.facts.append(fact)
        self.dismissed.discard(_norm(text))
        return fact

    def edit_fact(self, fact_id: int, text: str) -> dict | None:
        fact = self._get(fact_id)
        if not fact or not text.strip():
            return None
        fact.update(text=text.strip(), pinned=True, updated_at=time.time())
        return fact

    def delete_fact(self, fact_id: int) -> bool:
        fact = self._get(fact_id)
        if not fact:
            return False
        self.facts.remove(fact)
        self.dismissed.add(_norm(fact["text"]))
        return True

    def apply_ops(self, ops: list) -> None:
        """Apply extractor output. Never touches pinned facts or re-adds dismissed ones."""
        for op in ops if isinstance(ops, list) else []:
            if not isinstance(op, dict):
                continue
            kind, text = op.get("op"), (op.get("text") or "").strip()
            if kind == "add" and text and _norm(text) not in self.dismissed:
                self.add_fact(text)
                continue
            fact = self._get(op.get("id"))
            if not fact or fact["pinned"]:
                continue
            if kind == "replace" and text:
                fact.update(text=text, updated_at=time.time())
            elif kind == "remove":
                self.facts.remove(fact)

    def _get(self, fact_id) -> dict | None:
        return next((f for f in self.facts if f["id"] == fact_id), None)

    # --- show state ---
    def mark_call_started(self):
        self.calls_started = True

    # --- persistence ---
    def to_dict(self) -> dict:
        return {
            "facts": self.facts, "host_lines": self.host_lines[-60:],
            "processed_lines": min(self.processed_lines, len(self.host_lines[-60:])),
            "dismissed": sorted(self.dismissed), "headlines": self.headlines,
            "calls_started": self.calls_started, "next_id": self._next_id,
        }

    def load_dict(self, data: dict):
        self.reset()
        self.facts = list(data.get("facts", []))
        self.host_lines = list(data.get("host_lines", []))
        self.processed_lines = data.get("processed_lines", len(self.host_lines))
        self.dismissed = set(data.get("dismissed", []))
        self.headlines = list(data.get("headlines", []))
        self.calls_started = bool(data.get("calls_started", False))
        self._next_id = data.get("next_id", max((f["id"] for f in self.facts), default=0) + 1)


show_memory = ShowMemory()
```

**Step 4:** Re-run → all PASS.

**Step 5:** `git add backend/services/show_memory.py tests/test_show_memory.py && git commit -m "Add show memory fact store"`

---

### Task 3: Prompt blocks for Devon and callers

**Files:**
- Modify: `backend/services/show_memory.py`
- Test: `tests/test_show_memory.py`

**Step 1: Failing tests** — append:

```python
HEADLINES = [
    {"title": "Senate passes budget", "kind": "big", "summary": "Shutdown averted", "angle": "everyone's talking"},
    {"title": "Man marries his truck", "kind": "offbeat", "summary": "Ohio ceremony", "angle": "pure Devon bait"},
]


def test_devon_block_has_facts_and_opening_headlines_before_first_call():
    m = _mem("Luke is in West Virginia tonight")
    m.headlines = HEADLINES
    block = m.devon_block()
    assert "SHOW FACTS" in block and "West Virginia" in block
    assert "OPENING HEADLINES" in block and "Man marries his truck" in block


def test_devon_block_downgrades_headlines_after_first_call():
    m = ShowMemory()
    m.headlines = HEADLINES
    m.mark_call_started()
    block = m.devon_block()
    assert "OPENING HEADLINES" not in block
    assert "STORIES YOU KNOW ABOUT" in block and "Senate passes budget" in block


def test_empty_memory_gives_empty_blocks():
    m = ShowMemory()
    assert m.devon_block() == "" and m.caller_block() == ""


def test_caller_block_lists_facts_only():
    m = _mem("Luke is in West Virginia tonight")
    m.headlines = HEADLINES
    block = m.caller_block()
    assert "West Virginia" in block and "truck" not in block
```

**Step 2:** Run → FAIL (`AttributeError: devon_block`).

**Step 3: Implement** — add methods to `ShowMemory`:

```python
    # --- prompt blocks ---
    def _facts_lines(self) -> str:
        return "\n".join(f"- {f['text']}" for f in self.facts)

    def devon_block(self) -> str:
        parts = []
        if self.facts:
            parts.append(
                "SHOW FACTS — established on air tonight. These are true RIGHT NOW and "
                "override anything else you assume (including where the show is):\n"
                + self._facts_lines()
            )
        if self.headlines:
            lines = "\n".join(
                f"- [{h.get('kind', '')}] {h['title']} — {h.get('summary', '')} ({h.get('angle', '')})"
                for h in self.headlines
            )
            if not self.calls_started:
                parts.append(
                    "OPENING HEADLINES — stories you came in knowing about today. The show just "
                    "started: work ONE of these into a reply when it fits naturally, in your own "
                    "voice (\"dude, did you see...\"). Never list them. Never bring up one you've "
                    "already mentioned this show. If nothing fits this reply, skip it.\n" + lines
                )
            else:
                parts.append(
                    "STORIES YOU KNOW ABOUT — background only. Don't bring these up yourself; "
                    "talk about one only if Luke or a caller raises it.\n" + lines
                )
        return "\n\n".join(parts)

    def caller_block(self) -> str:
        if not self.facts:
            return ""
        return (
            "\n\nTONIGHT'S SHOW — things Luke has established on air. They are true right now; "
            "never contradict them (e.g. don't place the show somewhere else):\n"
            + self._facts_lines() + "\n"
        )

    def headline_titles(self) -> list[str]:
        return [h["title"] for h in self.headlines]
```

**Step 4:** Run → PASS. **Step 5:** `git add backend/services/show_memory.py tests/test_show_memory.py && git commit -m "Add show memory prompt blocks"`

---

### Task 4: Fact extractor (Haiku) with triggers

**Files:**
- Modify: `backend/services/show_memory.py`
- Test: `tests/test_show_memory.py`

**Step 1: Failing tests** — append:

```python
import asyncio
import json


def test_extract_applies_llm_ops(monkeypatch):
    from backend.services import show_memory as sm
    m = ShowMemory()
    m.add_host_line("We drove out to my mom's place in West Virginia this week", schedule=False)

    async def fake_generate(**kw):
        assert "West Virginia" in kw["messages"][0]["content"]
        return json.dumps({"ops": [{"op": "add", "text": "Luke is broadcasting from his mom's house in West Virginia"}]})

    monkeypatch.setattr(sm.llm_service, "generate", fake_generate)
    asyncio.run(m.extract_facts())
    assert "West Virginia" in m.facts[0]["text"]
    assert m.processed_lines == 1


def test_extract_bad_json_leaves_facts(monkeypatch):
    from backend.services import show_memory as sm
    m = _mem("Luke is in West Virginia")
    m.add_host_line("blah", schedule=False)

    async def fake_generate(**kw):
        return "not json at all"

    monkeypatch.setattr(sm.llm_service, "generate", fake_generate)
    asyncio.run(m.extract_facts())
    assert [f["text"] for f in m.facts] == ["Luke is in West Virginia"]


def test_extract_noop_without_new_lines(monkeypatch):
    from backend.services import show_memory as sm
    called = []

    async def fake_generate(**kw):
        called.append(1)
        return "{}"

    monkeypatch.setattr(sm.llm_service, "generate", fake_generate)
    asyncio.run(ShowMemory().extract_facts())
    assert called == []
```

**Step 2:** Run → FAIL.

**Step 3: Implement.** Add imports at top of `show_memory.py`:

```python
import asyncio
import json

from .llm import llm_service
```

Add constants + methods:

```python
EXTRACT_EVERY = 3
EXTRACT_IDLE_SECONDS = 60

FACT_EXTRACTOR_PROMPT = """You maintain the list of SHOW FACTS for tonight's episode of a late-night radio call-in show hosted by Luke.
A show fact is a short, durable statement that stays true for the rest of tonight's show and that the intern and callers must not contradict:
where Luke is broadcasting from tonight, his physical state, who's in the studio, running bits, promises, things he's told the audience about his life this week.
NOT facts: passing remarks, jokes that don't stick, anything a caller said about themselves, opinions.

Given the current facts (with ids) and Luke's newest lines, return JSON:
{"ops": [{"op": "add", "text": "..."}, {"op": "replace", "id": 3, "text": "..."}, {"op": "remove", "id": 4}]}
Use "replace" when Luke corrects or updates an existing fact (e.g. he's back in Alpine now). Write facts in third person, one short sentence each.
Return {"ops": []} if nothing new is durable."""
```

```python
    def add_host_line(self, text: str, schedule: bool = True):
        text = (text or "").strip()
        if not text:
            return
        self.host_lines.append(text)
        if schedule:
            self._schedule_extraction()

    def _schedule_extraction(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        pending = len(self.host_lines) - self.processed_lines
        if getattr(self, "_idle_task", None) and not self._idle_task.done():
            self._idle_task.cancel()
        if pending >= EXTRACT_EVERY:
            loop.create_task(self.extract_facts())
        else:
            self._idle_task = loop.create_task(self._extract_after_idle())

    async def _extract_after_idle(self):
        try:
            await asyncio.sleep(EXTRACT_IDLE_SECONDS)
            await self.extract_facts()
        except asyncio.CancelledError:
            pass

    async def extract_facts(self):
        new_lines = self.host_lines[self.processed_lines:]
        if not new_lines:
            return
        upto = len(self.host_lines)
        facts = "\n".join(f"[{f['id']}] {f['text']}" for f in self.facts) or "(none yet)"
        content = (f"CURRENT FACTS:\n{facts}\n\nLUKE'S NEWEST LINES:\n"
                   + "\n".join(f"- {line}" for line in new_lines[-12:]))
        try:
            raw = await llm_service.generate(
                messages=[{"role": "user", "content": content}],
                system_prompt=FACT_EXTRACTOR_PROMPT,
                max_tokens=400,
                response_format={"type": "json_object"},
                category="show_facts",
            )
            ops = json.loads(raw).get("ops", [])
        except Exception as e:
            print(f"[ShowMemory] Fact extraction failed: {e}")
            return
        self.processed_lines = upto
        self.apply_ops(ops)
        if ops:
            print(f"[ShowMemory] Facts now: {[f['text'] for f in self.facts]}")
```

Also add `self._idle_task = None` in `reset()`.

**Step 4:** Run → PASS. **Step 5:** `git add backend/services/show_memory.py tests/test_show_memory.py && git commit -m "Add show fact extractor"`

---

### Task 5: Curated opening headlines

**Files:**
- Modify: `backend/services/news.py` (add `search_recent`)
- Modify: `backend/services/show_memory.py`
- Test: `tests/test_show_memory.py`

**Step 1: Failing tests** — append:

```python
from datetime import datetime, timedelta, timezone

from backend.services.news import NewsItem
from backend.services.show_memory import filter_items, window_days


def test_window_days_clamped():
    now = datetime(2026, 11, 25, tzinfo=timezone.utc)
    assert window_days(None, now) == 1
    assert window_days(now - timedelta(hours=5), now) == 1
    assert window_days(now - timedelta(days=3, hours=2), now) == 4
    assert window_days(now - timedelta(days=40), now) == 7


def test_filter_drops_wikinews_old_and_duplicates():
    since = datetime(2026, 11, 20, tzinfo=timezone.utc)
    items = [
        NewsItem("Old blockade story", "wikinews", "2026-11-22T00:00:00"),
        NewsItem("Stale story", "brave.news", "2026-05-04T00:00:00"),
        NewsItem("Fresh story", "brave.news", "2026-11-24T10:00:00"),
        NewsItem("Fresh Story!", "bing news", ""),
        NewsItem("Undated story", "bing news", ""),
    ]
    assert [i.title for i in filter_items(items, since)] == ["Fresh story", "Undated story"]


def test_curation_failure_returns_empty(monkeypatch):
    from backend.services import show_memory as sm

    async def fake_recent(queries, time_range):
        return [NewsItem("Fresh story", "brave.news", "")]

    async def fake_generate(**kw):
        raise RuntimeError("openrouter down")

    async def fake_last():
        return None

    monkeypatch.setattr(sm.news_service, "search_recent", fake_recent)
    monkeypatch.setattr(sm.llm_service, "generate", fake_generate)
    monkeypatch.setattr(sm, "last_episode_date", fake_last)
    m = ShowMemory()
    asyncio.run(m.build_headlines())
    assert m.headlines == []


def test_curation_keeps_only_valid_picks(monkeypatch):
    from backend.services import show_memory as sm

    async def fake_recent(queries, time_range):
        return [NewsItem("A", "brave.news", ""), NewsItem("B", "bing news", "")]

    async def fake_generate(**kw):
        return json.dumps({"headlines": [
            {"title": "A", "kind": "big", "summary": "s", "angle": "a"},
            {"title": "", "kind": "offbeat"},
            {"kind": "offbeat", "summary": "no title"},
        ]})

    async def fake_last():
        return None

    monkeypatch.setattr(sm.news_service, "search_recent", fake_recent)
    monkeypatch.setattr(sm.llm_service, "generate", fake_generate)
    monkeypatch.setattr(sm, "last_episode_date", fake_last)
    m = ShowMemory()
    asyncio.run(m.build_headlines())
    assert [h["title"] for h in m.headlines] == ["A"]
```

**Step 2:** Run → FAIL.

**Step 3a: `news.py`** — add to `NewsService`:

```python
    async def search_recent(self, queries: list[str], time_range: str) -> list[NewsItem]:
        """Parallel news searches limited to a SearXNG time_range ('day'/'week')."""
        async def one(q):
            try:
                resp = await self.client.get(
                    f"{SEARXNG_URL}/search",
                    params={"q": q, "format": "json", "categories": "news", "time_range": time_range},
                )
                resp.raise_for_status()
                return self._parse_searxng(resp.json(), max_items=15)
            except Exception as e:
                print(f"[News] Recent search failed for '{q}': {e}")
                return []
        results = await asyncio.gather(*(one(q) for q in queries))
        return [item for batch in results for item in batch]
```

**Step 3b: `show_memory.py`** — imports:

```python
import math
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import httpx

from .news import news_service
```

Module-level:

```python
FEED_URL = "https://podcast.macneilmediagroup.com/@LukeAtTheRoost/feed.xml"
BIG_QUERIES = ["top news", "breaking news"]
OFFBEAT_QUERIES = ["weird news", "strange but true", "offbeat news"]

HEADLINE_CURATOR_PROMPT = """You pick stories for Devon, the 23-year-old intern on a late-night comedy call-in radio show, to bring up in the opening banter.
From the candidate stories, pick exactly 5: 2 "big" stories (major national/world news most listeners have heard about) and 3 "offbeat" ones (weird, funny, can-you-believe-this stories that are great to riff on).
Skip tragedies with deaths, anything too grim to joke near, sports scores, stock tickers, and duplicates.
Return JSON: {"headlines": [{"title": "...", "kind": "big|offbeat", "summary": "one line, plain facts", "angle": "why it's fun to talk about"}]}"""


def window_days(last: datetime | None, now: datetime) -> int:
    if last is None:
        return 1
    days = math.ceil((now - last).total_seconds() / 86400)
    return max(1, min(7, days))


def _parse_date(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None


def filter_items(items: list, since: datetime) -> list:
    seen, out = set(), []
    for item in items:
        if "wikinews" in (item.source or "").lower():
            continue
        published = _parse_date(item.published) if item.published else None
        if published and published < since:
            continue
        key = _norm(item.title)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


async def last_episode_date() -> datetime | None:
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(FEED_URL)
            resp.raise_for_status()
        match = re.search(r"<item>.*?<pubDate>([^<]+)</pubDate>", resp.text, re.S)
        return parsedate_to_datetime(match.group(1)) if match else None
    except Exception as e:
        print(f"[ShowMemory] Couldn't read last episode date: {e}")
        return None
```

Method on `ShowMemory`:

```python
    async def build_headlines(self):
        now = datetime.now(timezone.utc)
        days = window_days(await last_episode_date(), now)
        since = now - timedelta(days=days)
        time_range = "day" if days <= 1 else "week"
        try:
            items = await news_service.search_recent(BIG_QUERIES + OFFBEAT_QUERIES, time_range)
            items = filter_items(items, since)[:40]
            if not items:
                print("[ShowMemory] No fresh headlines found")
                return
            listing = "\n".join(f"- {i.title}: {i.content[:160]}" for i in items)
            raw = await llm_service.generate(
                messages=[{"role": "user", "content": f"CANDIDATE STORIES (last {days} day(s)):\n{listing}"}],
                system_prompt=HEADLINE_CURATOR_PROMPT,
                max_tokens=700,
                response_format={"type": "json_object"},
                category="news_summary",
            )
            picks = json.loads(raw).get("headlines", [])
        except Exception as e:
            print(f"[ShowMemory] Headline curation failed: {e}")
            return
        self.headlines = [
            {k: str(h.get(k, "")).strip() for k in ("title", "kind", "summary", "angle")}
            for h in picks if isinstance(h, dict) and str(h.get("title", "")).strip()
        ][:5]
        print(f"[ShowMemory] Opening headlines ({days}d window): {self.headline_titles()}")
```

**Step 4:** Run → PASS. **Step 5:** `git add backend/services/news.py backend/services/show_memory.py tests/test_show_memory.py && git commit -m "Add curated opening headlines"`

---

### Task 6: Devon uses show memory

**Files:**
- Modify: `backend/services/intern.py` (line 24 background; `ask()` ~372-412; `interject()` ~500-506)
- Test: `tests/test_intern_prompt.py`

**Step 1: Failing test** — create `tests/test_intern_prompt.py`:

```python
from backend.services.intern import INTERN_SYSTEM_PROMPT, intern_service
from backend.services.show_memory import show_memory


def setup_function():
    show_memory.reset()


def test_system_prompt_is_plain_without_memory():
    assert intern_service._system_prompt() == INTERN_SYSTEM_PROMPT


def test_show_facts_come_before_background():
    show_memory.add_fact("Luke is broadcasting from West Virginia tonight")
    prompt = intern_service._system_prompt()
    assert prompt.index("West Virginia") < prompt.index("YOUR BACKGROUND")


def test_background_no_longer_hardcodes_show_location():
    assert "moved out to the Big Bend with the show" not in INTERN_SYSTEM_PROMPT
    assert "SHOW FACTS" in INTERN_SYSTEM_PROMPT
```

**Step 2:** Run → FAIL.

**Step 3: Implement** in `intern.py`:
- Import: `from .show_memory import show_memory`
- Line 24: replace `You live in a studio in Alpine, Texas — you moved out to the Big Bend with the show.` with
  `You live in a studio in Alpine, Texas. The show normally broadcasts from the Big Bend — but if the SHOW FACTS say Luke is somewhere else tonight, that's where the show is.`
- Add method to the intern class:

```python
    def _system_prompt(self) -> str:
        block = show_memory.devon_block()
        return f"{block}\n\n{INTERN_SYSTEM_PROMPT}" if block else INTERN_SYSTEM_PROMPT
```

- In `ask()`: `messages.extend(relevant[-10:])` → `messages.extend(relevant[-20:])`; `system_prompt=INTERN_SYSTEM_PROMPT` → `system_prompt=self._system_prompt()`.
- In `interject()`: same `system_prompt=self._system_prompt()` swap.

**Step 4:** Run full suite `venv/bin/python -m pytest -q` → all PASS.

**Step 5:** `git add backend/services/intern.py tests/test_intern_prompt.py && git commit -m "Give Devon show facts and opening headlines"` (intern.py already has unrelated uncommitted edits — use `git add -p` to stage only this task's hunks)

---

### Task 7: Callers get show facts + curated headlines

**Files:**
- Modify: `backend/main.py` — `get_caller_prompt` (line 554), its 3 call sites (3718, 4695, 4817), headlines in `_build_backgrounds` (~1006-1009)
- Test: `tests/test_caller_prompt.py`

**Step 1: Failing test** — append to `tests/test_caller_prompt.py`:

```python
def test_prompt_includes_show_facts_when_given():
    prompt = get_caller_prompt(
        {"name": "Danny", "identity": "A plumber", "situation": "Leaky pipes"},
        facts="\n\nTONIGHT'S SHOW — ...\n- Luke is in West Virginia tonight\n",
    )
    assert "West Virginia" in prompt


def test_prompt_has_no_facts_block_by_default():
    prompt = get_caller_prompt({"name": "Danny", "identity": "A plumber", "situation": "Leaky pipes"})
    assert "TONIGHT'S SHOW" not in prompt
```

**Step 2:** Run → FAIL (`unexpected keyword argument 'facts'`).

**Step 3: Implement:**
- Signature: `def get_caller_prompt(caller: dict, theme: str = "", facts: str = "") -> str:`
- In the return f-string, change `{regular_block}{world_block}{theme_block}` → `{regular_block}{world_block}{theme_block}{facts}`.
- Import near other service imports: `from .services.show_memory import show_memory`
- All 3 call sites: `get_caller_prompt(slim_caller, theme=session.show_theme, facts=show_memory.caller_block())`
- In `_build_backgrounds` replace the `headlines` block (`if self.news_headlines: ...`) with:

```python
        headlines: list[str] = show_memory.headline_titles()
```

**Step 4:** Full suite → PASS. **Step 5:** `git add -p backend/main.py && git add tests/test_caller_prompt.py && git commit -m "Give callers show facts and curated headlines"` (main.py has unrelated uncommitted edits — stage only this task's hunks)

---

### Task 8: Wire lifecycle — reset, first call, host lines, checkpoint

**Files:**
- Modify: `backend/main.py`

No new unit test (glue code); verified by the full suite plus Task 10's live check.

**Step 1:** `Session.reset()` (~line 1218, next to `intern_service.new_show()`): add `show_memory.reset()`.

**Step 2:** `reset_session()` endpoint (~line 2689): after `session.reset()` and **before** `await session.populate_backgrounds()`:

```python
    try:
        async with asyncio.timeout(25):
            await show_memory.build_headlines()
    except Exception as e:
        print(f"[ShowMemory] Headline build skipped: {e}")
```

(Awaited so freshly generated lineups get the curated titles; capped so a slow SearXNG can't stall show start.)

**Step 3:** `Session.start_call()` (line 783): add `show_memory.mark_call_started()`.

**Step 4:** `Session.add_message()` (line 795): after the append, add

```python
        if role == "user":
            show_memory.add_host_line(content)
```

**Step 5:** `intern_ask()` (~line 5063): before calling `intern_service.ask(...)`, add `show_memory.add_host_line(question)`.

**Step 6:** Checkpoint save dict (~line 1420): add `"show_memory": show_memory.to_dict(),`. Checkpoint load (~line 1478, after `news_headlines`): add `show_memory.load_dict(data.get("show_memory", {}))`.

**Step 7:** `venv/bin/python -m pytest -q` → all PASS. `git add -p backend/main.py && git commit -m "Wire show memory into session lifecycle"`

---

### Task 9: Show Facts API + control panel

**Files:**
- Modify: `backend/main.py` (new endpoints next to `/api/show-theme`, ~line 4134)
- Modify: `frontend/index.html` (below the `.theme-bar` at line 21-26)
- Modify: `frontend/js/app.js` (next to `loadShowTheme`, ~line 1349)
- Modify: `frontend/css/style.css` (after `.theme-btn` rules, ~line 190)
- Test: `tests/test_show_facts_api.py`

**Step 1: Failing test** — `tests/test_show_facts_api.py`:

```python
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.show_memory import show_memory

client = TestClient(app)


def setup_function():
    show_memory.reset()


def test_add_edit_delete_fact():
    r = client.post("/api/show/facts", json={"text": "Luke is in West Virginia"})
    fact = r.json()["fact"]
    assert fact["pinned"] is True
    r = client.patch(f"/api/show/facts/{fact['id']}", json={"text": "Luke is in Charleston, WV"})
    assert r.json()["fact"]["text"] == "Luke is in Charleston, WV"
    assert [f["text"] for f in client.get("/api/show/facts").json()["facts"]] == ["Luke is in Charleston, WV"]
    assert client.delete(f"/api/show/facts/{fact['id']}").status_code == 200
    assert client.get("/api/show/facts").json()["facts"] == []


def test_missing_fact_404s():
    assert client.patch("/api/show/facts/999", json={"text": "x"}).status_code == 404
    assert client.delete("/api/show/facts/999").status_code == 404
```

(If `TestClient(app)` triggers heavy startup in this codebase, construct it without the context manager as above — startup events only run inside `with TestClient(app)`.)

**Step 2:** Run → FAIL (404 on POST).

**Step 3: Endpoints** in `main.py`:

```python
@app.get("/api/show/facts")
async def get_show_facts():
    return {"facts": show_memory.facts}


@app.post("/api/show/facts")
async def add_show_fact(data: dict):
    fact = show_memory.add_fact(data.get("text", ""), pinned=True)
    if not fact:
        raise HTTPException(400, "Empty, duplicate, or too many facts")
    return {"fact": fact}


@app.patch("/api/show/facts/{fact_id}")
async def edit_show_fact(fact_id: int, data: dict):
    fact = show_memory.edit_fact(fact_id, data.get("text", ""))
    if not fact:
        raise HTTPException(404, "Fact not found")
    return {"fact": fact}


@app.delete("/api/show/facts/{fact_id}")
async def delete_show_fact(fact_id: int):
    if not show_memory.delete_fact(fact_id):
        raise HTTPException(404, "Fact not found")
    return {"status": "deleted"}
```

**Step 4:** Run → PASS.

**Step 5: UI.** `index.html`, directly after the closing `</div>` of `.theme-bar`:

```html
            <div class="facts-bar">
                <span class="theme-label">Show facts:</span>
                <ul id="show-facts-list" class="facts-list"></ul>
                <input type="text" id="show-fact-input" class="theme-input" placeholder="e.g. Broadcasting from West Virginia tonight" maxlength="200">
                <button id="add-fact-btn" class="theme-btn set" title="Add show fact">Add</button>
            </div>
```

`app.js` (no `alert`/`prompt` dialogs — inline inputs only):

```javascript
async function loadShowFacts() {
    try {
        const { facts } = await (await fetch('/api/show/facts')).json();
        const list = document.getElementById('show-facts-list');
        if (list.contains(document.activeElement)) return; // don't clobber an in-progress edit
        list.innerHTML = '';
        for (const f of facts) {
            const li = document.createElement('li');
            li.className = 'fact-item' + (f.pinned ? ' pinned' : '');
            const input = document.createElement('input');
            input.className = 'fact-text';
            input.value = f.text;
            input.addEventListener('change', () => editShowFact(f.id, input.value));
            const del = document.createElement('button');
            del.className = 'theme-btn clear';
            del.textContent = '✕';
            del.title = 'Delete fact';
            del.addEventListener('click', () => deleteShowFact(f.id));
            li.append(input, del);
            list.append(li);
        }
    } catch (e) {
        console.error('Failed to load show facts:', e);
    }
}

async function addShowFact() {
    const input = document.getElementById('show-fact-input');
    const text = input.value.trim();
    if (!text) return;
    await fetch('/api/show/facts', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) });
    input.value = '';
    loadShowFacts();
}

async function editShowFact(id, text) {
    await fetch(`/api/show/facts/${id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) });
    loadShowFacts();
}

async function deleteShowFact(id) {
    await fetch(`/api/show/facts/${id}`, { method: 'DELETE' });
    loadShowFacts();
}
```

Wire it where `loadShowTheme()` is first called and its button listeners are attached (search `set-theme-btn` in `app.js`):

```javascript
    document.getElementById('add-fact-btn')?.addEventListener('click', addShowFact);
    document.getElementById('show-fact-input')?.addEventListener('keydown', e => { if (e.key === 'Enter') addShowFact(); });
    loadShowFacts();
    setInterval(loadShowFacts, 10000);
```

Also call `loadShowFacts()` after a session reset completes (search `/api/session/reset` in `app.js`). Make sure the existing keyboard shortcuts (1-0, H, W, M, D) don't fire while typing in `.fact-text`/`#show-fact-input` — check how the shortcut handler skips `#devon-input` and extend it to these inputs.

`style.css`:

```css
.facts-bar { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; padding: 4px 0; }
.facts-list { display: flex; flex-wrap: wrap; gap: 4px; list-style: none; margin: 0; padding: 0; }
.fact-item { display: flex; align-items: center; gap: 2px; }
.fact-text { background: transparent; border: 1px solid var(--border, #444); border-radius: 4px; color: inherit; padding: 2px 6px; min-width: 14ch; font-size: 0.85em; }
.fact-item.pinned .fact-text { border-color: var(--accent, #d4a017); }
```

(Match variable names to those used by `.theme-input` in `style.css`.)

**Step 6:** `git add -p backend/main.py && git add frontend/index.html frontend/js/app.js frontend/css/style.css tests/test_show_facts_api.py && git commit -m "Add Show Facts panel and API"`

---

### Task 10: Live verification

1. `venv/bin/python -m pytest -q` → all pass (193 + new).
2. Start backend: `./run.sh`. Requires Tailscale up (SearXNG via `SEARXNG_URL=http://mmg-nas:8888`).
3. `curl -s -X POST localhost:8000/api/session/reset` → logs show `[ShowMemory] Opening headlines (Nd window): [...]` with 5 fresh titles (no Wikinews).
4. `curl -s -X POST localhost:8000/api/intern/ask -H 'Content-Type: application/json' -d '{"question":"Devon, we drove out to my mom'"'"'s place in West Virginia this week, we are broadcasting from here tonight"}'` → Devon replies, likely weaving in a headline.
5. Send two more host lines via `/api/intern/ask`; after the 3rd, logs show `[ShowMemory] Facts now: [...West Virginia...]`. `curl localhost:8000/api/show/facts` shows it.
6. Ask Devon "where are we broadcasting from tonight?" → answers West Virginia.
7. Open the control panel; confirm the Show Facts bar lists the fact, edit/delete/add work, and shortcuts don't fire while typing.
8. `curl -s localhost:8000/api/costs/summary` (or the /costs page) shows `show_facts` + `news_summary` calls costing cents.
