import sys
sys.path.insert(0, "/Users/lukemacneil/ai-podcast")

from backend.main import Session, CallRecord, get_caller_prompt


def test_caller_prompt_includes_show_history():
    s = Session()
    s.call_history.append(CallRecord(
        caller_type="real", caller_name="Dave",
        summary="Called about his wife leaving after 12 years",
        transcript=[],
    ))

    s.start_call("1")  # Tony
    caller = s.caller
    show_history = s.get_show_history()
    prompt = get_caller_prompt(caller, "", show_history)
    assert "Dave" in prompt
    assert "wife leaving" in prompt
    assert "EARLIER IN THE SHOW" in prompt


def test_caller_prompt_without_history():
    s = Session()
    s.start_call("1")
    caller = s.caller
    prompt = get_caller_prompt(caller, "")
    assert "EARLIER IN THE SHOW" not in prompt
    assert caller["name"] in prompt


def test_caller_prompt_backward_compatible():
    """Verify get_caller_prompt works with just 2 args (no show_history)"""
    s = Session()
    s.start_call("1")
    caller = s.caller
    prompt = get_caller_prompt(caller, "Host: hello")
    assert "hello" in prompt
