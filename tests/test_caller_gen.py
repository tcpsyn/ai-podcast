import pytest
from backend.services.caller_gen import CallerIdentity, parse_batch_response

SAMPLE_JSON = """
{
  "callers": [
    {
      "name": "Danny Ortega",
      "age": 47,
      "voice_suggestion": "Marcus",
      "location": "Silver City, NM",
      "identity": "A plumber who inherited his uncle's taxidermy shop...",
      "situation": "He's been getting calls from people...",
      "reason_calling": "Someone left a note in his mailbox tonight...",
      "opening_line": "Luke, I need to ask you something weird.",
      "secret_want": "Permission to just throw it all away",
      "specific_details": ["the elk head in the basement", "the note said 'she forgot'", "his uncle's Rolodex"],
      "emotional_register": "quietly unsettled, trying to sound casual"
    }
  ]
}
"""


def test_parse_batch_response_returns_caller_list():
    callers = parse_batch_response(SAMPLE_JSON)
    assert len(callers) == 1
    assert callers[0].name == "Danny Ortega"
    assert callers[0].age == 47
    assert "taxidermy" in callers[0].identity
    assert len(callers[0].specific_details) == 3


def test_parse_batch_response_rejects_missing_fields():
    bad = '{"callers": [{"name": "Jim"}]}'
    with pytest.raises(ValueError, match="missing"):
        parse_batch_response(bad)


FENCED_JSON = '''```json
{"callers": [{"name": "Terrence", "age": 61, "voice_suggestion": "Marcus", "location": "Tucumcari, NM", "identity": "Retired irrigation engineer", "situation": "Reading water bill", "reason_calling": "Found clause", "opening_line": "Luke.", "secret_want": "Vindication", "specific_details": ["a", "b"], "emotional_register": "intense"}]}
```'''


def test_parse_batch_response_strips_markdown_fences():
    callers = parse_batch_response(FENCED_JSON)
    assert len(callers) == 1
    assert callers[0].name == "Terrence"


def test_resolve_voice_matches_exact():
    from backend.services.caller_gen import resolve_voice
    roster = ["Marcus", "Dennis", "Priya", "Edward"]
    assert resolve_voice("Marcus", roster) == "Marcus"


def test_resolve_voice_case_insensitive():
    from backend.services.caller_gen import resolve_voice
    roster = ["Marcus", "Dennis"]
    assert resolve_voice("marcus", roster) == "Marcus"


def test_resolve_voice_falls_back_when_no_match():
    from backend.services.caller_gen import resolve_voice
    roster = ["Marcus", "Dennis"]
    # Deterministic fallback: return first from roster
    assert resolve_voice("Santiago", roster) == "Marcus"


def test_resolve_voice_empty_suggestion_falls_back():
    from backend.services.caller_gen import resolve_voice
    assert resolve_voice("", ["Marcus"]) == "Marcus"


def test_build_batch_prompt_includes_context():
    from backend.services.caller_gen import build_batch_prompt
    ctx = {
        "date": "Saturday, April 5, 2026",
        "weather": "cool desert night, 48°F",
        "headlines": ["New Mexico legislature approves water bill"],
        "recent_caller_summaries": ["Jerry called about his neighbor's goat"],
        "regulars_included": [],
        "caller_count": 12,
        "voice_roster": ["Marcus", "Dennis", "Priya"],
    }
    prompt = build_batch_prompt(ctx)
    assert "Saturday, April 5, 2026" in prompt
    assert "water bill" in prompt
    assert "Jerry called about his neighbor's goat" in prompt
    assert "12 callers" in prompt
    assert "Marcus" in prompt  # voice roster listed
    assert "Stern" in prompt
    assert "Coast to Coast" in prompt
    assert "Loveline" in prompt
    assert "Delilah" in prompt
    assert "Opie and Anthony" in prompt


def test_build_batch_prompt_includes_silas_lore_when_present():
    from backend.services.caller_gen import build_batch_prompt
    ctx = {
        "date": "...",
        "weather": "...",
        "headlines": [],
        "recent_caller_summaries": [],
        "regulars_included": [{"name": "Silas", "lore": "Silas leads a small desert cult...", "arc_state": "seeking new members"}],
        "caller_count": 12,
        "voice_roster": ["Marcus"],
    }
    prompt = build_batch_prompt(ctx)
    assert "Silas" in prompt
    assert "desert cult" in prompt
    assert "seeking new members" in prompt
    assert "DO NOT alter his voice, personality, or core traits" in prompt
