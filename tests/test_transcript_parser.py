import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from website_gen.transcript import parse_transcript


def test_splits_speaker_turns():
    raw = "LUKE: Welcome back.\n\nSLIM: Hey Luke, thanks for taking my call."
    turns = parse_transcript(raw)
    assert turns == [("LUKE", "Welcome back."),
                     ("SLIM", "Hey Luke, thanks for taking my call.")]


def test_unlabeled_paragraph_carries_previous_speaker():
    raw = "LUKE: First thing.\n\nStill Luke talking."
    assert parse_transcript(raw) == [("LUKE", "First thing."),
                                     ("LUKE", "Still Luke talking.")]


def test_ignores_blank_and_whitespace_paragraphs():
    raw = "LUKE: One.\n\n   \n\nSLIM: Two."
    assert len(parse_transcript(raw)) == 2


def test_speaker_name_with_spaces_is_not_treated_as_label():
    """A colon mid-sentence must not be mistaken for a speaker label."""
    raw = "LUKE: Here's the thing: it was the alternator."
    turns = parse_transcript(raw)
    assert len(turns) == 1
    assert turns[0][0] == "LUKE"
    assert "the thing: it was" in turns[0][1]


def test_empty_input_returns_empty_list():
    assert parse_transcript("") == []
    assert parse_transcript("   \n\n  ") == []
