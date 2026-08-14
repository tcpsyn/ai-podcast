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
    assert "YOU CAN BE MOVED" in prompt
    assert "NEVER restate the same dilemma" in prompt
    assert len(prompt) < 3500


def test_prompt_includes_opening_line():
    caller = {
        "name": "Tina",
        "identity": "A nurse who just got off a 16-hour shift",
        "situation": "Found her ex's wedding invitation in her mailbox",
        "reason_calling": "She's not sure if she should go",
        "secret_want": "Wants someone to tell her she's over him",
        "opening_line": "Luke, I literally just got home from work and there's this gold envelope sitting on my kitchen counter.",
        "specific_details": ["invitation was hand-addressed", "wedding is in two weeks"],
    }
    prompt = get_caller_prompt(caller)
    assert "gold envelope" in prompt
    assert "FIRST message" in prompt
    assert "listening for" in prompt.lower() or "been listening" in prompt.lower()


def test_prompt_without_opening_line():
    caller = {
        "name": "Ray",
        "identity": "A retired mechanic",
        "situation": "Neighbor's dog dug up something weird",
        "reason_calling": "He thinks it might be human bones",
        "secret_want": "Doesn't want to call the cops on his neighbor",
        "specific_details": ["bones were wrapped in a tarp"],
    }
    prompt = get_caller_prompt(caller)
    assert "FIRST message" not in prompt
    assert "planned opening" not in prompt.lower()
