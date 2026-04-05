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
