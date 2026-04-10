from backend.main import get_caller_prompt


def test_prompt_includes_identity_and_situation():
    caller = {
        "name": "Danny",
        "identity": "A plumber who inherited a taxidermy shop",
        "situation": "Getting strange calls about taxidermy",
        "reason_calling": "Someone left a note",
        "secret_want": "Permission to throw it all away",
        "specific_details": ["elk head in basement", "note said she forgot"],
    }
    prompt = get_caller_prompt(caller)
    assert "Danny" in prompt
    assert "taxidermy shop" in prompt
    assert "strange calls" in prompt
    assert "elk head" in prompt
    assert "she forgot" in prompt
    assert "Permission to throw it all away" in prompt
    assert "React to what Luke says" in prompt
    assert "Stay in character" in prompt
    assert "NEVER use asterisks" in prompt
    assert "NEVER use parenthetical stage directions" in prompt
    assert "Mix short punchy replies with longer ones" in prompt
    assert len(prompt) < 3500
