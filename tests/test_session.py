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
