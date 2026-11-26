import asyncio

import pytest

from backend import main
from backend.main import _retry_if_stale, _stale_phrases, get_caller_prompt
from backend.services.intern import INTERN_SYSTEM_PROMPT


@pytest.mark.parametrize("text,label", [
    ("Okay. That lands harder than I wanted it to.", "that lands"),
    ("You just said something that lands, Luke.", "that lands"),
    ("But here's what's keeping me up at night.", "what's keeping me up"),
    ("That’s the part I can’t get past.", "that's the part"),
    ("I keep coming back to the truck.", "I keep coming back to"),
    ("I hear you, Luke.", "I hear you"),
    ("Here's the thing about Marfa.", "here's the thing"),
])
def test_detects_stale_phrases(text, label):
    assert label in _stale_phrases(text)


@pytest.mark.parametrize("text", [
    "The plane landed in Midland around nine.",
    "I can't hear you, Luke, you're breaking up.",
    "That's fair game at the county fair.",
    "We sat with the dog till the vet came.",
])
def test_ignores_ordinary_speech(text):
    assert _stale_phrases(text) == []


def test_caller_prompt_bans_phrases_and_drops_let_it_land():
    prompt = get_caller_prompt({"name": "Concho", "identity": "a welder", "situation": "x"})
    assert "let it land" not in prompt
    for phrase in ('"that lands"', '"what\'s keeping me up"', '"that\'s the part"'):
        assert phrase in prompt


def test_devon_prompt_bans_phrases():
    assert '"that\'s the part"' in INTERN_SYSTEM_PROMPT


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def generate(self, messages, **kw):
        self.calls.append(messages)
        return self.replies.pop(0)


def run_retry(monkeypatch, response, replies, wrapping_up=False, conversation=()):
    fake = FakeLLM(replies)
    monkeypatch.setattr(main, "llm_service", fake)
    out = asyncio.run(_retry_if_stale(
        response, [{"role": "user", "content": "hi"}], "sys", 300, "Concho", None,
        list(conversation), wrapping_up=wrapping_up, tag="Test"))
    return out, fake


def test_stale_reply_is_regenerated(monkeypatch):
    out, fake = run_retry(monkeypatch, "That lands, Luke.", ["Okay, yeah. You got me there."])
    assert out == "Okay, yeah. You got me there."
    assert '"that lands"' in fake.calls[0][-1]["content"]


def test_keeps_original_when_retry_still_stale(monkeypatch):
    out, _ = run_retry(monkeypatch, "That lands, Luke.", ["I hear you, Luke."])
    assert out == "That lands, Luke."


def test_clean_reply_makes_no_extra_call(monkeypatch):
    out, fake = run_retry(monkeypatch, "My truck's on blocks again.", [])
    assert out == "My truck's on blocks again." and fake.calls == []


def test_stale_phrase_checked_during_wrap_up(monkeypatch):
    out, _ = run_retry(monkeypatch, "That's fair. Thanks, Luke.", ["Alright. Thanks, Luke."], wrapping_up=True)
    assert out == "Alright. Thanks, Luke."


def test_hangup_is_never_retried(monkeypatch):
    out, fake = run_retry(monkeypatch, "That lands. [HANGUP]", [])
    assert fake.calls == []
