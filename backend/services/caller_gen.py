from dataclasses import dataclass
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


def resolve_voice(suggestion: str, roster: list[str]) -> str:
    """Map sonnet's voice suggestion to a real voice in the roster.
    Case-insensitive exact match; deterministic fallback to first roster entry."""
    if not suggestion or not roster:
        return roster[0] if roster else ""
    lower_map = {v.lower(): v for v in roster}
    return lower_map.get(suggestion.lower(), roster[0])


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
