"""Regression guard: a show once ran with session.caller_backgrounds empty because
populate_backgrounds() was reachable only via POST /api/session/reset. Every caller
got a hollow prompt and collapsed into generic relationship filler.
"""
import asyncio

import pytest

import backend.main as m


@pytest.fixture
def quiet_call(monkeypatch):
    monkeypatch.setattr(m.audio_service, "stop_caller_audio", lambda *a, **k: None)
    monkeypatch.setattr(m, "_maybe_generate_callback", lambda: None)
    monkeypatch.setattr(m, "_enrich_background_async", lambda key: asyncio.sleep(0))
    monkeypatch.setattr(m.session, "intern_monitoring", False)
    m.session.current_caller_key = None
    yield
    # asyncio.run() closes the loop it creates and clears the thread's current
    # loop. tests/test_caller_service.py still calls asyncio.get_event_loop(),
    # which then raises — and this module sorts ahead of it. Leave a usable loop.
    asyncio.set_event_loop(asyncio.new_event_loop())


def test_start_call_populates_empty_lineup(monkeypatch, quiet_call):
    key = next(iter(m.CALLER_BASES))
    m.session.caller_backgrounds = {}
    called = []

    async def fake_populate():
        called.append(True)
        m.session.caller_backgrounds = {
            key: {"name": "Silas", "voice": "Sebastian", "identity": "commune founder",
                  "situation": "the convoy stopped", "specific_details": []}
        }

    monkeypatch.setattr(m.session, "populate_backgrounds", fake_populate)
    asyncio.run(m.start_call(key))

    assert called, "start_call must populate backgrounds when the lineup is empty"
    assert m.session.caller_backgrounds, "lineup should be populated after the call starts"


def test_start_call_does_not_repopulate_existing_lineup(monkeypatch, quiet_call):
    key = next(iter(m.CALLER_BASES))
    m.session.caller_backgrounds = {
        key: {"name": "Silas", "voice": "Sebastian", "identity": "commune founder",
              "situation": "the convoy stopped", "specific_details": []}
    }
    called = []

    async def fake_populate():
        called.append(True)

    monkeypatch.setattr(m.session, "populate_backgrounds", fake_populate)
    asyncio.run(m.start_call(key))

    assert not called, "an existing lineup must not be regenerated mid-show"


def test_prompt_from_populated_background_is_not_hollow():
    prompt = m.get_caller_prompt({
        "name": "Silas", "identity": "founder of The Wellspring",
        "situation": "the convoy stopped on the shoulder of 118",
        "reason_calling": "Priscilla will not come out of the trailer",
        "secret_want": "to hear he is not the reason she broke",
        "specific_details": ["livestock trailer", "nine years in"],
    })
    assert "Silas" in prompt
    assert "Wellspring" in prompt
    assert "You are . " not in prompt
