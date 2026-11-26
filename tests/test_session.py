import sys
sys.path.insert(0, "/Users/lukemacneil/ai-podcast")

import pytest

from backend.main import Session, CallRecord


@pytest.fixture(autouse=True)
def _no_live_data_writes(monkeypatch):
    import backend.main as main
    monkeypatch.setattr(main.intern_service, "new_show", lambda: None)
    monkeypatch.setattr(main.intern_service, "_save", lambda: None)


def test_call_record_creation():
    record = CallRecord(
        caller_type="real",
        caller_name="Dave",
        summary="Called about his wife leaving",
        transcript=[{"role": "host", "content": "What happened?"}],
    )
    assert record.caller_type == "real"
    assert record.caller_name == "Dave"


def test_session_call_history():
    s = Session()
    assert s.call_history == []
    record = CallRecord(
        caller_type="ai", caller_name="Tony",
        summary="Talked about gambling", transcript=[],
    )
    s.call_history.append(record)
    assert len(s.call_history) == 1


def test_session_active_real_caller():
    s = Session()
    assert s.active_real_caller is None
    s.active_real_caller = {
        "caller_id": "abc123",
        "channel": 3, "name": "Dave",
    }
    assert s.active_real_caller["channel"] == 3


def test_session_three_party_conversation():
    s = Session()
    s.start_call("1")  # AI caller Tony
    s.add_message("host", "Hey Tony")
    s.add_message("ai_caller:Tony", "What's up man")
    s.add_message("real_caller:Dave", "Yeah I agree with Tony")
    assert len(s.conversation) == 3
    assert s.conversation[2]["role"] == "real_caller:Dave"


def test_session_get_show_history_summary():
    s = Session()
    s.call_history.append(CallRecord(
        caller_type="real", caller_name="Dave",
        summary="Called about his wife leaving after 12 years",
        transcript=[],
    ))
    s.call_history.append(CallRecord(
        caller_type="ai", caller_name="Jasmine",
        summary="Talked about her boss hitting on her",
        transcript=[],
    ))
    summary = s.get_show_history()
    assert "Dave" in summary
    assert "Jasmine" in summary
    assert "EARLIER IN THE SHOW" in summary


def test_show_history_reactions_constant_is_usable():
    from backend.main import SHOW_HISTORY_REACTIONS

    assert SHOW_HISTORY_REACTIONS, "generic reaction pool must not be empty"
    assert all(isinstance(r, str) and r.strip() for r in SHOW_HISTORY_REACTIONS)
    # Each one is interpolated as "...and you {reaction}." so it must not
    # carry its own leading/trailing punctuation or an unfilled placeholder.
    for r in SHOW_HISTORY_REACTIONS:
        assert not r.endswith("."), r
        assert "{" not in r, r


def test_build_specific_reaction_falls_back_when_record_has_no_details():
    """A CallRecord with neither key_details nor situation_summary hits the
    generic branch — this used to raise NameError mid-show."""
    from backend.main import SHOW_HISTORY_REACTIONS

    s = Session()
    bare = CallRecord(
        caller_type="ai", caller_name="Jasmine",
        summary="Talked about her boss", transcript=[],
    )
    reaction = s._build_specific_reaction({}, bare)
    assert reaction in SHOW_HISTORY_REACTIONS


def test_get_show_history_never_raises_when_reaction_branch_fires(monkeypatch):
    """Force the reaction branch every time so the fallback path is covered
    deterministically rather than at its ~15% random rate."""
    import backend.main as main

    monkeypatch.setattr(main.random, "random", lambda: 0.0)

    s = Session()
    s.call_history.append(CallRecord(
        caller_type="real", caller_name="Dave",
        summary="Called about his wife leaving", transcript=[],
    ))
    summary = s.get_show_history()
    assert "DAVE" in summary
    assert "and you " in summary


def test_session_reset_clears_history():
    s = Session()
    s.call_history.append(CallRecord(
        caller_type="real", caller_name="Dave",
        summary="test", transcript=[],
    ))
    s.active_real_caller = {"caller_id": "abc123"}
    s.ai_respond_mode = "auto"
    s.reset()
    assert s.call_history == []
    assert s.active_real_caller is None
    assert s.ai_respond_mode == "manual"


def test_session_conversation_summary_three_party():
    s = Session()
    s.start_call("1")
    s.add_message("host", "Tell me what happened")
    s.add_message("real_caller:Dave", "She just left man")
    s.add_message("ai_caller:Tony", "Same thing happened to me")
    summary = s.get_conversation_summary()
    assert "Dave" in summary
    assert "Tony" in summary


def test_recent_summaries_uses_wider_dedup_window(monkeypatch):
    """Phase 5B deleted cross-episode topic dedup, leaving only a 2-show window.
    The batch generator should now see LINEUP_DEDUP_SHOWS worth of history."""
    import backend.main as m

    history = [
        {"lineup": [{"name": f"Caller{i}", "situation": f"situation number {i}"}]}
        for i in range(m.LINEUP_DEDUP_SHOWS + 5)
    ]
    monkeypatch.setattr(m, "_load_lineup_history", lambda: history)

    summaries = Session()._get_recent_summaries()
    assert len(summaries) == m.LINEUP_DEDUP_SHOWS
    assert m.LINEUP_DEDUP_SHOWS > 2
    # Keeps the most recent shows, drops the oldest
    assert "situation number 4" not in " ".join(summaries)
    assert f"situation number {m.LINEUP_DEDUP_SHOWS + 4}" in " ".join(summaries)


def test_lineup_history_retains_at_least_the_dedup_window():
    """Truncating the file below the dedup window would silently shrink it."""
    import backend.main as m
    assert m.LINEUP_HISTORY_MAX >= m.LINEUP_DEDUP_SHOWS


def test_fresh_session_reports_no_lineup():
    s = Session()
    assert s.caller_backgrounds == {}
    assert bool(s.caller_backgrounds) is False


def test_session_feeds_show_memory():
    from backend.main import show_memory
    show_memory.reset()
    try:
        s = Session()
        s.add_message("user", "We're in West Virginia tonight")
        s.add_message("assistant", "Oh nice")
        s.add_message("real_caller:555", "Hi Luke")
        assert show_memory.host_lines == ["We're in West Virginia tonight"]
        assert not show_memory.calls_started
        s.start_call("1")
        assert show_memory.calls_started
        s.reset()
        assert show_memory.host_lines == []
        assert not show_memory.calls_started
    finally:
        show_memory.reset()


def _patch_reset_deps(monkeypatch, built):
    import backend.main as main

    async def fake_populate():
        built.append("populate")

    async def fake_build():
        built.append("build")

    monkeypatch.setattr(main.session, "populate_backgrounds", fake_populate)
    monkeypatch.setattr(main.show_memory, "build_headlines", fake_build)
    return main


def test_reset_session_carries_fresh_headlines(monkeypatch):
    import asyncio
    import time
    built = []
    main = _patch_reset_deps(monkeypatch, built)
    main.show_memory.reset()
    try:
        main.show_memory.headlines = [{"title": "A", "kind": "big", "summary": "", "angle": ""}]
        main.show_memory.headlines_built_at = time.time() - 60
        main.show_memory.add_fact("Luke is in West Virginia")
        asyncio.run(main.reset_session())
        assert built == ["populate"]
        assert main.show_memory.headline_titles() == ["A"]
        assert main.show_memory.facts == []
    finally:
        main.show_memory.reset()


def test_reset_session_rebuilds_stale_headlines_before_fresh_lineup(monkeypatch):
    import asyncio
    import time
    built = []
    main = _patch_reset_deps(monkeypatch, built)
    main.show_memory.reset()
    monkeypatch.setattr(main.session, "pending_backgrounds", None)
    monkeypatch.setattr(main.session, "_prewarm_task", None)
    try:
        main.show_memory.headlines = [{"title": "Old", "kind": "big", "summary": "", "angle": ""}]
        main.show_memory.headlines_built_at = time.time() - 3600
        asyncio.run(main.reset_session())
        assert built == ["build", "populate"]
        assert main.show_memory.headlines == []
    finally:
        main.show_memory.reset()


def test_reset_session_mid_show_does_not_carry_headlines(monkeypatch):
    import asyncio
    import time
    built = []
    main = _patch_reset_deps(monkeypatch, built)
    main.show_memory.reset()
    monkeypatch.setattr(main.session, "pending_backgrounds", None)
    monkeypatch.setattr(main.session, "_prewarm_task", None)
    try:
        main.show_memory.headlines = [{"title": "A", "kind": "big", "summary": "", "angle": ""}]
        main.show_memory.headlines_built_at = time.time() - 60
        main.show_memory.mark_call_started()
        asyncio.run(main.reset_session())
        assert built == ["build", "populate"]
        assert main.show_memory.headlines == []
    finally:
        main.show_memory.reset()
