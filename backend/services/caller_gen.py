from dataclasses import dataclass
from typing import Optional
import json
import random

import httpx

from ..config import settings
from .cost_tracker import cost_tracker

BATCH_MODEL = "anthropic/claude-sonnet-4.6"

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
    stripped = raw.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1])
    data = json.loads(stripped)
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
    Case-insensitive exact match; random fallback if unmatched (so LLM
    hallucinations don't all collapse onto roster[0])."""
    if not roster:
        return ""
    if suggestion:
        match = {v.lower(): v for v in roster}.get(suggestion.lower())
        if match:
            return match
        fallback = random.choice(roster)
        print(f"[caller_gen] voice '{suggestion}' not in roster — random fallback to '{fallback}'")
        return fallback
    return random.choice(roster)


BATCH_SYSTEM_PROMPT = """You are writing a roster of callers for Luke's late-night radio show, broadcast out of Alpine, Texas, in the Big Bend country of far West Texas.

WHERE THESE CALLERS LIVE: The show's world is the Big Bend region of far West Texas — high desert, remote, dark skies, ranching country up against the Mexico border. Most callers are from or know this area. Ground them in REAL places and facts only — never invent businesses or landmarks that don't exist. The towns:
- Alpine — the hub of the region (about 6,000 people), Brewster County seat, home of Sul Ross State University, a mile-high old ranching and railroad town with a small arts scene. The show broadcasts from here.
- Marfa — minimalist-art tourist town (the Chinati Foundation / Donald Judd, the Prada Marfa installation), famous for the unexplained Marfa Lights. Old ranching families chafing against an influx of artists and out-of-towners.
- Marathon — tiny, known for the historic Gage Hotel, the eastern gateway to Big Bend National Park.
- Terlingua — a quicksilver-mining ghost town turned off-grid haven for desert eccentrics; famous for its chili cookoff, river-rafting outfitters on the Rio Grande, and the Starlight Theatre. Right up against the Mexico border (Boquillas crossing).
- Fort Stockton — an oilfield and I-10 town to the north in Pecos County (Paisano Pete the roadrunner), more working-class and blue-collar than the artsy towns.
- The Big Bend itself — the Chisos Mountains, the Rio Grande, some of the darkest night skies in the country (the McDonald Observatory is near Fort Davis), brutally remote, with Permian Basin oil money booming to the north.
Callers can reference real ranches, the heat and wind, the drive times (everything is hours apart), border life, Sul Ross students, oilfield work, tourists, and the desert weirdos — but keep it real and specific to this place.

CREATIVE RANGE: Your callers must span the emotional range of Howard Stern (chaos, strong characters), Coast to Coast AM (earnest weirdos, sincere believers), Loveline (real problems, real advice-seeking), Delilah (emotional vulnerability, connection), and Opie and Anthony (sharp, irreverent, specific people).

ROSTER MIX — VARY THE CALL TYPES. A great show is not ten people in moral crisis. Build the roster of 10 callers roughly like this:
- 4-5 DILEMMA CALLS (the dramatic spine): real human conflict with STAKES — moral dilemmas, confessions, betrayals, impossible choices, or something the caller did and can't take back. Think: "I found out my dad has a second family," "I got someone fired and they deserved it but now their kid is sick," "my best friend's wife hit on me and I didn't say no." These need genuine emotional weight.
- 2-3 STORY / ENTHUSIAST CALLS (the relief): callers with a wild thing that happened to them, a fascinating obsession or piece of knowledge, a strange-but-true story, or a vivid slice of life. NO deep moral dilemma required — they call because the story is great, the fact is amazing, or they're bursting to talk about the thing they love. Still specific, still a reason they called TONIGHT, but the energy is delight or wonder or a great yarn, not anguish.
- 1-2 TRUE BELIEVER / CHAOS CALLS (the spice): an earnest, sincere weirdo — UFOs, cryptids, a government conspiracy, a pattern only they can see, a paranormal experience (Coast to Coast AM energy, dead serious about it) — OR a big eccentric personality on a tear about something trivial. Played straight, never winking.

Every caller still needs SOMETHING that makes the audience lean in — a problem, a secret, a story, a wild belief, or an irresistible enthusiasm. But not every caller carries grief. Let the show breathe.

Maximum character distance between callers. No two callers should feel like siblings.

ANTI-COLLISION RULE — THIS IS NON-NEGOTIABLE: All callers in this roster must be clearly differentiated. No two callers may share the same hobby, obsession, profession archetype, or story theme. Specifically forbidden within a single roster: two BBQ competitors, two taxidermists, two amateur-radio or mystery-signal callers, two callers with ex-spouse drama, two believers chasing the SAME phenomenon (e.g. two UFO callers, or two cryptid callers — if you have two believer calls they must be about completely different things), two retired military, two grandmothers-of-many, two callers calling about a weird neighbor, two callers with religious-object stories. Each caller's defining "thing" — their hook, their obsession, the specific topic they're calling about — must appear exactly once in the roster. Before finalizing, scan your output and swap any collisions.

Do not default to sitcom plots. Real humans are specific and strange. Give each caller details that could only belong to them.

OPENING LINE RULES: Each caller's opening_line must be unique and specific to their situation. NEVER write "I've been listening for X years" or "long-time listener, first-time caller" or any variant. The caller should jump into their story or problem immediately — nervous, excited, angry, whatever fits. The opening line is the hook that makes the audience lean in.

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

    theme = (ctx.get("theme") or "").strip()
    if theme:
        lines.append(f'TONIGHT\'S SHOW THEME: "{theme}"')
        lines.append(
            f'Roughly 2/3 of callers MUST be calling BECAUSE OF this theme — the theme '
            f'should be woven directly into their reason_calling and situation, not '
            f'just acknowledged. Make the connection specific and personal (a story, a '
            f'conflict, a moment) not abstract. The remaining 1/3 of the roster can be '
            f'unrelated walk-ins for variety. Do NOT make every caller theme-connected — '
            f'variety still matters.'
        )
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
            lines.append(f"CRITICAL — this caller MUST appear in the output JSON with EXACTLY these fields locked: name=\"{r['name']}\", voice_suggestion=\"{r['voice']}\", age={r['age']}. Do NOT rename this caller. Do NOT give him a different voice. Do NOT change his age. Only invent: location, identity, situation, reason_calling, opening_line, secret_want, specific_details, emotional_register.")
            lines.append("")

    lines.append(f"Available voices (voice_suggestion must match one of these exactly):")
    lines.append(", ".join(ctx["voice_roster"]))
    lines.append("")
    lines.append(f"Generate {ctx['caller_count']} callers. Output JSON only, no prose.")
    return BATCH_SYSTEM_PROMPT + "\n\n" + "\n".join(lines)


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
                "max_tokens": 16000,
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
        usage_data=usage,
    )

    callers = parse_batch_response(content)
    for c in callers:
        c.voice_resolved = resolve_voice(c.voice_suggestion, ctx["voice_roster"])
    return callers


REGULAR_SYSTEM_PROMPT = """You are writing a single caller — a recurring character on Luke's late-night radio show. The character's identity is fixed (name, voice, age, core lore). You only invent a fresh reason he's calling tonight, grounded in his current arc state.

Output strict JSON with these fields only: location, identity, situation, reason_calling, opening_line, secret_want, specific_details (array of 2-3 strings), emotional_register. Do NOT include name, voice_suggestion, or age — those are locked."""


async def generate_regular_situation(regular: dict, ctx: dict) -> dict:
    """Generate a fresh situation for a locked regular. Returns a dict matching
    the CallerIdentity schema. One small LLM call (~$0.01)."""
    lines = [
        f"Tonight is {ctx['date']}. {ctx['weather']}.",
        "",
        f"### {regular['name']}",
        regular["lore"],
        f"Current arc state: {regular['arc_state']}",
        "",
        f"Invent a fresh reason {regular['name']} is calling tonight — a new development, grievance, or specific recent event. DO NOT alter his voice, personality, or core traits.",
        "",
        "Output JSON only, no prose.",
    ]
    prompt = REGULAR_SYSTEM_PROMPT + "\n\n" + "\n".join(lines)

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            json={
                "model": BATCH_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "max_tokens": 1500,
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
        usage_data=usage,
    )

    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1])
    return json.loads(stripped)
