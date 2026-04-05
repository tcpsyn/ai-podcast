"""Generate 10 sample caller dialogues for user validation before cutover.

5 with Silas batch (if lore exists), 5 walk-ins. Writes transcripts to docs/samples/.
"""
import asyncio
import random
from datetime import datetime
from pathlib import Path

import httpx

from backend.config import settings
from backend.main import (
    BLACKLISTED_VOICES,
    INWORLD_FEMALE_VOICES,
    INWORLD_MALE_VOICES,
    get_caller_prompt,
)
from backend.services import caller_gen, regulars_v2

DIALOG_MODEL = "anthropic/claude-haiku-4.5"

HOST_REACTIONS = [
    "Yeah?",
    "Go on.",
    "Mm-hmm.",
    "Wait — really?",
    "Hold on, back up.",
    "Is that right?",
    "How'd you end up there?",
    "And then what?",
    "What'd she say?",
    "So what are you gonna do about it?",
    "That's — okay, keep going.",
    "I mean, what do you want me to tell you?",
]


async def dialog_turn(client: httpx.AsyncClient, system_prompt: str, conversation: list) -> str:
    last_err = None
    for attempt in range(3):
        try:
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
            return resp.json()["choices"][0]["message"]["content"].strip()
        except (httpx.ReadTimeout, httpx.HTTPStatusError) as e:
            last_err = e
            print(f"  [retry {attempt+1}/3] dialog_turn failed: {type(e).__name__}")
    raise last_err


async def main():
    out_dir = Path("docs/samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    voice_roster = [
        n for n in INWORLD_MALE_VOICES + INWORLD_FEMALE_VOICES
        if n not in BLACKLISTED_VOICES
    ]

    date_str = datetime.now().strftime("%A, %B %d, %Y")

    # Batch 1: include Silas (if lore exists)
    regulars = regulars_v2.load_all_active_regulars()
    if regulars:
        ctx_with_silas = {
            "date": date_str,
            "weather": "cool desert night",
            "headlines": [],
            "recent_caller_summaries": [],
            "regulars_included": [
                {"name": r.name, "lore": r.lore_body, "arc_state": r.arc_state}
                for r in regulars
            ],
            "caller_count": 5,
            "voice_roster": voice_roster,
        }
        print("[Batch 1] Generating 5 callers including Silas...")
        silas_batch = await caller_gen.generate_batch(ctx_with_silas)
        print(f"[Batch 1] Got {len(silas_batch)} callers: {[c.name for c in silas_batch]}")
    else:
        print("[Batch 1] No regulars found — skipping")
        silas_batch = []

    # Batch 2: walk-ins
    ctx_walkins = {
        "date": date_str,
        "weather": "cool desert night",
        "headlines": [],
        "recent_caller_summaries": [],
        "regulars_included": [],
        "caller_count": 5,
        "voice_roster": voice_roster,
    }
    print("[Batch 2] Generating 5 walk-in callers...")
    walkin_batch = await caller_gen.generate_batch(ctx_walkins)
    print(f"[Batch 2] Got {len(walkin_batch)} callers: {[c.name for c in walkin_batch]}")

    all_callers = silas_batch + walkin_batch

    async with httpx.AsyncClient(timeout=120.0) as client:
        for idx, caller in enumerate(all_callers, 1):
            print(f"[Dialog {idx}/{len(all_callers)}] {caller.name}...")
            cdict = {
                "name": caller.name,
                "identity": caller.identity,
                "situation": caller.situation,
                "reason_calling": caller.reason_calling,
                "secret_want": caller.secret_want,
                "specific_details": caller.specific_details,
            }
            system_prompt = get_caller_prompt(cdict)
            conversation = [{"role": "assistant", "content": caller.opening_line}]

            for _ in range(4):
                host_line = random.choice(HOST_REACTIONS)
                conversation.append({"role": "user", "content": host_line})
                reply = await dialog_turn(client, system_prompt, conversation)
                conversation.append({"role": "assistant", "content": reply})

            transcript_lines = [
                f"CALLER ({caller.name}, {caller.age}, {caller.location}): "
                f"{conversation[0]['content']}"
            ]
            for i in range(1, len(conversation), 2):
                transcript_lines.append(f"LUKE: {conversation[i]['content']}")
                if i + 1 < len(conversation):
                    transcript_lines.append(f"CALLER: {conversation[i+1]['content']}")

            slug = caller.name.replace(" ", "_").lower()
            fname = out_dir / f"sample_{slug}.txt"
            fname.write_text(
                f"=== {caller.name} ({caller.age}, {caller.location}) ===\n"
                f"voice: {caller.voice_resolved}\n"
                f"emotional_register: {caller.emotional_register}\n"
                f"secret_want: {caller.secret_want}\n\n"
                + "\n".join(transcript_lines)
                + "\n"
            )
            print(f"Wrote {fname}")


if __name__ == "__main__":
    asyncio.run(main())
