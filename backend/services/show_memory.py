"""Show-wide memory: host-established facts + Devon's curated opening headlines.

Lives for one show. session.conversation is wiped every call, so anything that
must survive the whole show (where Luke is tonight, running bits) lives here."""

import asyncio
import json
import math
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import httpx

from .llm import llm_service
from .news import news_service

HOST_LINES_KEPT = 60
EXTRACT_EVERY = 3
EXTRACT_IDLE_SECONDS = 60
HEADLINES_REUSE_SECONDS = 30 * 60
EXTRACT_MAX_LINES = 12  # older pending lines are skipped (and marked processed) after a long backlog

FACT_EXTRACTOR_PROMPT = """You maintain the list of SHOW FACTS for tonight's episode of a late-night radio call-in show hosted by Luke.
A show fact is a short, durable statement that stays true for the rest of tonight's show and that the intern and callers must not contradict:
where Luke is broadcasting from tonight, his physical state, who's in the studio, running bits, promises, things he's told the audience about his life this week.
NOT facts: passing remarks, jokes that don't stick, anything a caller said about themselves, opinions.

Given the current facts (with ids) and Luke's newest lines, return JSON:
{"ops": [{"op": "add", "text": "..."}, {"op": "replace", "id": 3, "text": "..."}, {"op": "remove", "id": 4}]}
Use "replace" when Luke corrects or updates an existing fact (e.g. he's back in Alpine now). Write facts in third person, one short sentence each.
Return {"ops": []} if nothing new is durable."""

FEED_URL = "https://podcast.macneilmediagroup.com/@LukeAtTheRoost/feed.xml"
BIG_QUERIES = ["top news", "breaking news"]
OFFBEAT_QUERIES = ["weird news", "strange but true", "offbeat news"]

HEADLINE_CURATOR_PROMPT = """You pick stories for Devon, the 23-year-old intern on a late-night comedy call-in radio show, to bring up in the opening banter.
From the candidate stories, pick exactly 5: 2 "big" stories (major national/world news most listeners have heard about) and 3 "offbeat" ones (weird, funny, can-you-believe-this stories that are great to riff on).
Skip tragedies with deaths, anything too grim to joke near, sports scores, stock tickers, and duplicates.
Return JSON: {"headlines": [{"title": "...", "kind": "big|offbeat", "summary": "one line, plain facts", "angle": "why it's fun to talk about"}]}"""


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


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
        if not match:
            return None
        dt = parsedate_to_datetime(match.group(1))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception as e:
        print(f"[ShowMemory] Couldn't read last episode date: {e}")
        return None


class ShowMemory:
    MAX_FACTS = 12

    def __init__(self):
        self.reset()

    def reset(self):
        for task in (getattr(self, "_idle_task", None), getattr(self, "_extract_task", None)):
            if task and not task.done():
                task.cancel()
        self._generation = getattr(self, "_generation", 0) + 1
        self.facts: list[dict] = []
        self.host_lines: list[str] = []
        self.processed_lines = 0
        self.dismissed: set[str] = set()
        self.headlines: list[dict] = []
        self.headlines_built_at = 0.0
        self.calls_started = False
        self._next_id = 1
        self._idle_task = None
        self._extract_task = None
        self._extracting = False

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
        if any(f is not fact and _norm(f["text"]) == _norm(text) for f in self.facts):
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
            kind, text = op.get("op"), str(op.get("text") or "").strip()
            if kind == "add" and text and _norm(text) not in self.dismissed:
                self.add_fact(text)
                continue
            try:
                fact = self._get(int(op.get("id")))
            except (TypeError, ValueError):
                continue
            if not fact or fact["pinned"]:
                continue
            if kind == "replace" and text:
                fact.update(text=text, updated_at=time.time())
            elif kind == "remove":
                self.facts.remove(fact)

    def _get(self, fact_id) -> dict | None:
        return next((f for f in self.facts if f["id"] == fact_id), None)

    # --- extraction ---
    def add_host_line(self, text: str, schedule: bool = True):
        text = (text or "").strip()
        if not text:
            return
        self.host_lines.append(text)
        if schedule:
            self._schedule_extraction()

    def _schedule_extraction(self, idle_only: bool = False):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        pending = len(self.host_lines) - self.processed_lines
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        if self._extracting:
            return
        if pending >= EXTRACT_EVERY and not idle_only:
            self._extract_task = loop.create_task(self.extract_facts())
        else:
            self._idle_task = loop.create_task(self._extract_after_idle())

    async def _extract_after_idle(self):
        try:
            await asyncio.sleep(EXTRACT_IDLE_SECONDS)
        except asyncio.CancelledError:
            return
        self._idle_task = None
        self._extract_task = asyncio.current_task()
        await self.extract_facts()

    async def extract_facts(self):
        new_lines = self.host_lines[self.processed_lines:]
        if not new_lines or self._extracting:
            return
        generation = self._generation
        self._extracting = True
        upto = len(self.host_lines)
        facts = "\n".join(f"[{f['id']}] {f['text']}" for f in self.facts) or "(none yet)"
        content = (f"CURRENT FACTS:\n{facts}\n\nLUKE'S NEWEST LINES:\n"
                   + "\n".join(f"- {line}" for line in new_lines[-EXTRACT_MAX_LINES:]))
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
            ops = None
        finally:
            if generation == self._generation:
                self._extracting = False
        if generation != self._generation:
            return
        if ops is not None:
            self.processed_lines = upto
            self.apply_ops(ops)
            if ops:
                print(f"[ShowMemory] Facts now: {[f['text'] for f in self.facts]}")
        if len(self.host_lines) > self.processed_lines:
            self._schedule_extraction(idle_only=ops is None)

    # --- opening headlines ---
    async def build_headlines(self):
        generation = self._generation
        try:
            now = datetime.now(timezone.utc)
            days = window_days(await last_episode_date(), now)
            since = now - timedelta(days=days)
            time_range = "day" if days <= 1 else "week"
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
            if not isinstance(picks, list):
                picks = []
        except Exception as e:
            print(f"[ShowMemory] Headline curation failed: {e}")
            return
        if generation != self._generation:
            return
        self.headlines = [
            {k: str(h.get(k, "")).strip() for k in ("title", "kind", "summary", "angle")}
            for h in picks if isinstance(h, dict) and str(h.get("title", "")).strip()
        ][:5]
        self.headlines_built_at = time.time()
        print(f"[ShowMemory] Opening headlines ({days}d window): {self.headline_titles()}")

    def headlines_fresh(self, now: float | None = None) -> bool:
        age = (time.time() if now is None else now) - self.headlines_built_at
        return bool(self.headlines) and 0 <= age < HEADLINES_REUSE_SECONDS

    # --- show state ---
    def mark_call_started(self):
        self.calls_started = True

    # --- prompt blocks ---
    def _facts_lines(self) -> str:
        return "\n".join(f"- {f['text']}" for f in self.facts)

    def devon_block(self, opening: bool = True) -> str:
        parts = []
        if self.facts:
            parts.append(
                "SHOW FACTS — established on air tonight. These are true RIGHT NOW and "
                "override anything else you assume (including where the show is):\n"
                + self._facts_lines()
            )
        headlines = [h for h in self.headlines if h.get("title")]
        if headlines:
            lines = "\n".join(
                f"- [{h.get('kind', '')}] {h.get('title', '')} — {h.get('summary', '')} ({h.get('angle', '')})"
                for h in headlines
            )
            if opening and not self.calls_started:
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
        return [h["title"] for h in self.headlines if h.get("title")]

    # --- persistence ---
    def to_dict(self) -> dict:
        kept = self.host_lines[-HOST_LINES_KEPT:]
        dropped = len(self.host_lines) - len(kept)
        return {
            "facts": self.facts, "host_lines": kept,
            "processed_lines": max(0, self.processed_lines - dropped),
            "dismissed": sorted(self.dismissed), "headlines": self.headlines,
            "headlines_built_at": self.headlines_built_at,
            "calls_started": self.calls_started, "next_id": self._next_id,
        }

    def load_dict(self, data: dict):
        self.reset()
        self.facts = list(data.get("facts", []))
        self.host_lines = list(data.get("host_lines", []))
        self.processed_lines = data.get("processed_lines", len(self.host_lines))
        self.dismissed = set(data.get("dismissed", []))
        self.headlines = list(data.get("headlines", []))
        self.headlines_built_at = float(data.get("headlines_built_at", 0.0) or 0.0)
        self.calls_started = bool(data.get("calls_started", False))
        self._next_id = data.get("next_id", max((f["id"] for f in self.facts), default=0) + 1)


show_memory = ShowMemory()
