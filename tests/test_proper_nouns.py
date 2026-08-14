"""Whisper mishears recurring show names; correct them after transcription.

publish_episode.py transcribes with LightningWhisperMLX, whose transcribe()
signature is (audio_path, language) — it accepts no initial_prompt, so there is
no way to condition it on proper nouns the way make_clips.py does with
mlx_whisper. The intern came out as "Devin" in 21 of 58 published transcripts
because of this.

A deterministic pass over the finished text fixes the known names without
swapping the transcription engine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from publish_episode import fix_proper_nouns


def test_corrects_the_intern_name_in_all_casings():
    assert fix_proper_nouns("Devin, where's my coffee?") == "Devon, where's my coffee?"
    assert fix_proper_nouns("DEVIN: Hey Luke.") == "DEVON: Hey Luke."
    assert fix_proper_nouns("devin said so") == "devon said so"


def test_leaves_correct_spelling_alone():
    text = "Devon is the intern. DEVON: Hi."
    assert fix_proper_nouns(text) == text


def test_only_matches_whole_words():
    """Must not corrupt a longer word that happens to contain the name."""
    assert fix_proper_nouns("Devinshire") == "Devinshire"
    assert fix_proper_nouns("mcdevins") == "mcdevins"


def test_preserves_surrounding_text_exactly():
    src = "LUKE: Alright. Let's check in with Devin and see how he's doing.\n\nDEVIN: Hey!"
    out = fix_proper_nouns(src)
    assert out == "LUKE: Alright. Let's check in with Devon and see how he's doing.\n\nDEVON: Hey!"
    assert len(out) == len(src)


def test_empty_and_none_safe():
    assert fix_proper_nouns("") == ""
    assert fix_proper_nouns(None) == ""


def test_word_count_is_never_changed():
    src = "Devin talked to Devin about Devin. " * 20
    assert len(fix_proper_nouns(src).split()) == len(src.split())
