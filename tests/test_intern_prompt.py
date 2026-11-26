from backend.services.intern import DEVON_MONITOR_PROMPT, INTERN_SYSTEM_PROMPT, intern_service
from backend.services.show_memory import show_memory


def setup_function():
    show_memory.reset()


def teardown_function():
    show_memory.reset()


def test_system_prompt_is_plain_without_memory():
    assert intern_service._system_prompt() == INTERN_SYSTEM_PROMPT


def test_show_facts_come_before_background():
    show_memory.add_fact("Luke is broadcasting from West Virginia tonight")
    prompt = intern_service._system_prompt()
    assert prompt.index("West Virginia") < prompt.index("YOUR BACKGROUND")


def test_monitor_prompt_gets_show_facts_too():
    assert intern_service._system_prompt(DEVON_MONITOR_PROMPT) == DEVON_MONITOR_PROMPT
    show_memory.add_fact("Luke is broadcasting from West Virginia tonight")
    prompt = intern_service._system_prompt(DEVON_MONITOR_PROMPT)
    assert prompt.startswith("SHOW FACTS")
    assert prompt.endswith(DEVON_MONITOR_PROMPT)


HEADLINE = {"title": "Man marries his truck", "kind": "offbeat", "summary": "Ohio", "angle": "bait"}


def test_monitor_prompt_never_pushes_opening_headlines():
    show_memory.headlines = [HEADLINE]
    prompt = intern_service._system_prompt(DEVON_MONITOR_PROMPT, opening=False)
    assert "STORIES YOU KNOW ABOUT" in prompt
    assert "OPENING HEADLINES" not in prompt


def test_ask_prompt_offers_opening_headlines_before_first_call():
    show_memory.headlines = [HEADLINE]
    prompt = intern_service._system_prompt()
    assert "OPENING HEADLINES" in prompt


def test_interject_uses_non_opening_prompt(monkeypatch):
    import asyncio
    from backend.services import intern as intern_mod

    show_memory.headlines = [HEADLINE]
    seen = {}

    async def fake_generate_with_tools(**kwargs):
        seen["system_prompt"] = kwargs["system_prompt"]
        return "NOTHING_TO_ADD", []

    monkeypatch.setattr(intern_mod.llm_service, "generate_with_tools", fake_generate_with_tools)
    conversation = [{"role": "user", "content": "hey"}, {"role": "assistant", "content": "hi Luke"}]
    asyncio.run(intern_service.interject(conversation))
    assert "STORIES YOU KNOW ABOUT" in seen["system_prompt"]
    assert "OPENING HEADLINES" not in seen["system_prompt"]


def test_background_no_longer_hardcodes_show_location():
    assert "moved out to the Big Bend with the show" not in INTERN_SYSTEM_PROMPT
    assert "SHOW FACTS" in INTERN_SYSTEM_PROMPT
