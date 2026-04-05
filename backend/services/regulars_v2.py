from dataclasses import dataclass
from pathlib import Path
import json
import re

import httpx

from ..config import settings
from .cost_tracker import cost_tracker

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
        usage_data=usage,
    )
    return json.loads(data["choices"][0]["message"]["content"])


async def evaluate_promotion(caller_name: str, call_transcript: str) -> dict:
    prompt = PROMOTION_PROMPT.format(name=caller_name, transcript=call_transcript)
    return await _call_sonnet(prompt)
