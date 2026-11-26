"""Shared Wellspring lore — the bible every Wellspring caller draws on, and the
post-call continuity pass that keeps the bible and each regular's arc current.

The bible lives in the Obsidian vault. Hand-written sections are never touched;
only Storylines, Established Facts and Timeline are rewritten by code."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import json
import os
import re
import tempfile

import httpx

from ..config import settings
from .cost_tracker import cost_tracker
from .regulars_v2 import VAULT, Regular, load_all_active_regulars

BIBLE_PATH = VAULT / "wellspring" / "wellspring.md"
GROUP = "wellspring"
CONTINUITY_MODEL = "anthropic/claude-sonnet-5.5"
TIMELINE_IN_PROMPT = 8
ARC_LOG_IN_PROMPT = 4
PERSPECTIVE_HEADING = "## The Wellspring, to me"
NOT_AIRED = "(not yet aired)"


@dataclass
class Storyline:
    slug: str
    title: str = ""
    status: str = "open"
    last_advanced: str = ""
    involves: list[str] = field(default_factory=list)
    state: str = ""


@dataclass
class Bible:
    preamble: str
    sections: list[tuple[str, str]]  # (heading, body) in file order
    storylines: list[Storyline]
    facts: list[str]
    timeline: list[str]

    def section(self, heading: str) -> str:
        return next((body for h, body in self.sections if h == heading), "")

    @property
    def open_storylines(self) -> list[Storyline]:
        return [s for s in self.storylines if s.status == "open"]


# --- Parsing / rendering ---

def _bullets(body: str) -> list[str]:
    return [line[2:].strip() for line in body.splitlines() if line.startswith("- ")]


def _parse_storylines(body: str) -> list[Storyline]:
    out = []
    for chunk in re.split(r"^### ", body, flags=re.M)[1:]:
        lines = chunk.splitlines()
        s = Storyline(slug=lines[0].strip())
        for line in lines[1:]:
            m = re.match(r"^- (\w+): (.*)$", line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()
            if key == "involves":
                s.involves = [n.strip() for n in val.split(",") if n.strip()]
            elif hasattr(s, key):
                setattr(s, key, val)
        out.append(s)
    return out


def parse_bible(text: str) -> Bible:
    parts = re.split(r"^## ", text, flags=re.M)
    preamble = parts[0]
    sections = []
    for part in parts[1:]:
        heading, _, body = part.partition("\n")
        sections.append((heading.strip(), body.strip()))
    body_of = dict(sections)
    return Bible(
        preamble=preamble,
        sections=sections,
        storylines=_parse_storylines(body_of.get("Storylines", "")),
        facts=_bullets(body_of.get("Established Facts", "")),
        timeline=_bullets(body_of.get("Timeline", "")),
    )


def _render_storylines(storylines: list[Storyline]) -> str:
    blocks = []
    for s in storylines:
        blocks.append("\n".join([
            f"### {s.slug}",
            f"- title: {s.title}",
            f"- status: {s.status}",
            f"- last_advanced: {s.last_advanced}",
            f"- involves: {', '.join(s.involves)}",
            f"- state: {s.state}",
        ]))
    return "\n\n".join(blocks)


def render_bible(bible: Bible) -> str:
    generated = {
        "Storylines": _render_storylines(bible.storylines),
        "Established Facts": "\n".join(f"- {f}" for f in bible.facts),
        "Timeline": "\n".join(f"- {t}" for t in bible.timeline),
    }
    out = [bible.preamble.rstrip() + "\n"]
    for heading, body in bible.sections:
        out.append(f"\n## {heading}\n\n{generated.get(heading, body).strip()}\n")
    return "".join(out)


def _atomic_write(path: Path, text: str):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def load_bible(path: Path = BIBLE_PATH) -> Bible | None:
    if not path.exists():
        return None
    return parse_bible(path.read_text())


def save_bible(bible: Bible, path: Path = BIBLE_PATH):
    _atomic_write(path, render_bible(bible))


# --- Lore-file sections ---

def _section(body: str, heading: str) -> str:
    m = re.search(rf"^{re.escape(heading)}\s*$(.*?)(?=^## |\Z)", body, flags=re.M | re.S)
    return m.group(1).strip() if m else ""


def perspective(lore_body: str) -> str:
    return _section(lore_body, PERSPECTIVE_HEADING)


def recent_arc_log(lore_body: str, n: int = ARC_LOG_IN_PROMPT) -> list[str]:
    entries = [e for e in _bullets(_section(lore_body, "## Arc Log")) if NOT_AIRED not in e]
    return entries[-n:]


def find_regular(name: str) -> Regular | None:
    if not name:
        return None
    for r in load_all_active_regulars():
        if r.name.lower() == name.lower():
            return r
    return None


def is_member(regular: Regular | None) -> bool:
    return bool(regular) and regular.group == GROUP


# --- Prompt context ---

def pick_storyline(bible: Bible, member: str) -> Storyline | None:
    """The open storyline involving this member that has gone longest untouched."""
    pool = [s for s in bible.open_storylines if member in s.involves] or bible.open_storylines
    return min(pool, key=lambda s: s.last_advanced or "", default=None)


def world_context(bible: Bible, member: str, tonight: Storyline | None = None) -> str:
    lines = [
        "THE WELLSPRING — SHARED WORLD. Every Wellspring caller lives in this same world. "
        "These facts are canon: never contradict them, never invent a different location, "
        "layout, or history for the Wellspring.",
        "",
        bible.section("The Place"),
        "",
        bible.section("The Current"),
        "",
        bible.section("The People"),
        "",
        "WHAT'S GOING ON RIGHT NOW (open storylines — mention any you'd plausibly know about, "
        "from your own angle; don't explain things you wouldn't know):",
    ]
    lines += [f"- {s.title}: {s.state}" for s in bible.open_storylines]
    if bible.facts:
        lines += ["", "ESTABLISHED FACTS (said on air — canon):"]
        lines += [f"- {f}" for f in bible.facts]
    if bible.timeline:
        lines += ["", "RECENT EVENTS (most recent last):"]
        lines += [f"- {t}" for t in bible.timeline[-TIMELINE_IN_PROMPT:]]
    if tonight:
        lines += [
            "",
            f"TONIGHT {member.upper()} IS CALLING MAINLY ABOUT: {tonight.title}. Where it stood: "
            f"{tonight.state} Something new and concrete has happened since — move this story "
            f"forward one real step. Don't rehash where it stood.",
        ]
    return "\n".join(lines).strip()


def perspective_block(regular: Regular) -> str:
    view = perspective(regular.lore_body)
    if not view:
        return ""
    return (
        f"HOW {regular.name.upper()} SEES IT — your own perspective. Filter the shared world "
        f"through this. You only know what you'd plausibly know; other members see the same "
        f"events differently.\n{view}"
    )


def prompt_context(regular: Regular | None, assign_storyline: bool = False) -> str:
    """World + perspective block for a Wellspring member, '' for anyone else.
    With assign_storyline, the group lead gets tonight's storyline (pregen only)."""
    if not is_member(regular):
        return ""
    bible = load_bible()
    if not bible:
        return ""
    tonight = pick_storyline(bible, regular.name) if assign_storyline and regular.group_lead else None
    return "\n\n".join(p for p in (world_context(bible, regular.name, tonight),
                                    perspective_block(regular)) if p)


# --- Continuity pass ---

CONTINUITY_PROMPT = """You keep continuity for a recurring character on Luke's late-night radio show. {name} just called. Record what happened so the next call picks up from here instead of repeating itself.

{name}'s lore and arc so far:
{lore}

{world}

THE CALL THAT JUST AIRED:
{transcript}

Output strict JSON:
{{
  "arc_log": "1-2 concrete sentences, no date: what {name} revealed, decided or changed on this call",
  "arc_state": "1-2 short sentences: where {name}'s story stands going into the next call",
  "timeline": "one short line for the shared timeline, from the community's point of view, no date, or empty",
  "storylines": [{{"slug": "existing slug", "state": "the storyline's full current state, 2-4 sentences, keeping still-true details", "status": "open or closed"}}],
  "new_storylines": [{{"slug": "kebab-case", "title": "...", "involves": ["Name"], "state": "..."}}],
  "facts": ["0-3 durable details other members must not contradict"],
  "ripples": "one short line: who at the Wellspring heard this call or will hear about it, and how it lands, or empty"
}}

Rules:
- Record only what was actually said on the call plus its immediate, obvious consequences. Do not invent new plot.
- Only list storylines the call actually touched. Close one only if the call resolved it. A storyline state replaces the old one, so carry forward anything still true.
- Add a new storyline only if the call clearly opened one, or if fewer than 4 would remain open.
- Facts are not a recap. Only durable world details another member could plausibly reference (who paid for what, a name, a number, a decision). Skip anything already covered by a storyline state or the arc log. Most calls add 0-2.{solo_rule}"""

SOLO_RULE = "\n- This character is not part of a shared world: leave timeline, storylines, new_storylines, facts and ripples empty."


async def _call_continuity_model(prompt: str) -> dict:
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            json={
                "model": CONTINUITY_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "max_tokens": 2000,
                "reasoning": {"effort": "low"},
            },
        )
        resp.raise_for_status()
        data = resp.json()
    cost_tracker.record_llm_call(category="call_summary", model=CONTINUITY_MODEL,
                                 usage_data=data.get("usage", {}))
    content = data["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = "\n".join(content.splitlines()[1:-1])
    return json.loads(content)


def apply_to_regular(path: Path, arc_log: str, arc_state: str, today: str):
    text = path.read_text()
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if m and arc_state:
        fm = m.group(1)
        line = f"arc_state: {arc_state}"
        fm = re.sub(r"^arc_state:.*$", lambda _: line, fm, flags=re.M) if re.search(r"^arc_state:", fm, re.M) else fm + "\n" + line
        text = f"---\n{fm}\n---\n" + text[m.end():]
    if arc_log:
        entry = f"- {today}: {arc_log}"
        text = re.sub(rf"^- {re.escape(NOT_AIRED)}.*\n?", "", text, flags=re.M)
        log = re.search(r"^## Arc Log[ \t]*\n(.*?)(?=^## |\Z)", text, flags=re.M | re.S)
        if log:
            body = log.group(1).rstrip()
            new_body = (body + "\n" if body else "\n") + entry + "\n"
            if log.end() < len(text):
                new_body += "\n"
            text = text[:log.start(1)] + new_body + text[log.end():]
        else:
            text = text.rstrip() + f"\n\n## Arc Log\n\n{entry}\n"
    _atomic_write(path, text)


def apply_to_bible(bible: Bible, result: dict, today: str) -> Bible:
    by_slug = {s.slug: s for s in bible.storylines}
    for upd in result.get("storylines") or []:
        s = by_slug.get(upd.get("slug", ""))
        if not s:
            continue
        s.state = upd.get("state") or s.state
        s.status = "closed" if upd.get("status") == "closed" else "open"
        s.last_advanced = today
    for new in result.get("new_storylines") or []:
        slug = re.sub(r"[^a-z0-9-]", "", (new.get("slug") or "").lower())
        if not slug or slug in by_slug:
            continue
        s = Storyline(slug=slug, title=new.get("title", slug), last_advanced=today,
                      involves=list(new.get("involves") or []), state=new.get("state", ""))
        bible.storylines.append(s)
        by_slug[slug] = s
    for fact in result.get("facts") or []:
        if fact and fact not in bible.facts:
            bible.facts.append(fact)
    for line in (result.get("timeline"), result.get("ripples")):
        if line:
            bible.timeline.append(f"{today}: {line}")
    return bible


async def run_continuity(caller_name: str, transcript_text: str, today: str | None = None) -> bool:
    """Update a lore regular's arc (and the bible, for Wellspring members) after a call.
    Returns False when the caller isn't a lore regular."""
    regular = find_regular(caller_name)
    if not regular or not transcript_text.strip():
        return False
    today = today or date.today().isoformat()
    member = is_member(regular)
    bible = load_bible() if member else None
    prompt = CONTINUITY_PROMPT.format(
        name=regular.name,
        lore=regular.lore_body,
        world=(world_context(bible, regular.name) + "\n\nStoryline slugs: "
               + ", ".join(s.slug for s in bible.storylines)) if bible else "",
        transcript=transcript_text,
        solo_rule="" if bible else SOLO_RULE,
    )
    result = await _call_continuity_model(prompt)
    apply_to_regular(regular.file_path, result.get("arc_log", ""), result.get("arc_state", ""), today)
    if bible:
        save_bible(apply_to_bible(bible, result, today))
    touched = [s.get("slug") for s in result.get("storylines") or []]
    print(f"[Continuity] {regular.name}: arc updated; storylines touched={touched}; "
          f"facts+={len(result.get('facts') or [])}")
    return True
