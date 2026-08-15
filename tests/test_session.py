import sys
sys.path.insert(0, "/Users/lukemacneil/ai-podcast")

from backend.main import Session, CallRecord


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
