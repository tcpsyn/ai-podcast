"""The intern is Devon, not Devin.

Whisper heard "Devin" in 21 of 58 transcripts because the initial prompt in
backend/services/transcription.py never seeded his name as a proper noun. That
was invisible while transcripts were plain .txt nobody read; now they render as
indexed text on every episode page, so a regression would be public.

The durable fix is seeding "Devon" in the Whisper initial prompt. This test
catches it if that regresses or a new transcript slips through.
"""

import re
from pathlib import Path

TRANSCRIPTS = Path(__file__).resolve().parent.parent / "website" / "transcripts"


def test_no_transcript_spells_the_intern_devin():
    offenders = []
    for f in sorted(TRANSCRIPTS.glob("*.txt")):
        hits = len(re.findall(r"\bdevin\b", f.read_text(errors="replace"), re.IGNORECASE))
        if hits:
            offenders.append(f"{f.name} ({hits})")
    assert not offenders, f"transcripts spell the intern 'Devin': {offenders}"


def test_devon_is_present_as_a_speaker_label():
    """Guards against a rename that accidentally removed him entirely."""
    labelled = [f.name for f in TRANSCRIPTS.glob("*.txt")
                if re.search(r"^DEVON:", f.read_text(errors="replace"), re.MULTILINE)]
    assert len(labelled) >= 20, f"expected Devon labelled in many episodes, got {len(labelled)}"
