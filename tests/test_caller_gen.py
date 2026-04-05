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
