import sys
sys.path.insert(0, "/Users/lukemacneil/ai-podcast")

from backend.services.twilio_service import TwilioService


def test_queue_starts_empty():
    svc = TwilioService()
    assert svc.get_queue() == []


def test_add_caller_to_queue():
    svc = TwilioService()
    svc.add_to_queue("CA123", "+15125550142")
    q = svc.get_queue()
    assert len(q) == 1
    assert q[0]["call_sid"] == "CA123"
    assert q[0]["phone"] == "+15125550142"
    assert "wait_time" in q[0]


def test_remove_caller_from_queue():
    svc = TwilioService()
    svc.add_to_queue("CA123", "+15125550142")
    svc.remove_from_queue("CA123")
    assert svc.get_queue() == []


def test_allocate_channel():
    svc = TwilioService()
    ch1 = svc.allocate_channel()
    ch2 = svc.allocate_channel()
    assert ch1 == 3
    assert ch2 == 4
    svc.release_channel(ch1)
    ch3 = svc.allocate_channel()
    assert ch3 == 3


def test_take_call():
    svc = TwilioService()
    svc.add_to_queue("CA123", "+15125550142")
    result = svc.take_call("CA123")
    assert result["call_sid"] == "CA123"
    assert result["channel"] >= 3
    assert svc.get_queue() == []
    assert svc.active_calls["CA123"]["channel"] == result["channel"]


def test_hangup_real_caller():
    svc = TwilioService()
    svc.add_to_queue("CA123", "+15125550142")
    svc.take_call("CA123")
    ch = svc.active_calls["CA123"]["channel"]
    svc.hangup("CA123")
    assert "CA123" not in svc.active_calls
    assert ch not in svc._allocated_channels


def test_caller_counter_increments():
    svc = TwilioService()
    svc.add_to_queue("CA1", "+15125550001")
    svc.add_to_queue("CA2", "+15125550002")
    r1 = svc.take_call("CA1")
    r2 = svc.take_call("CA2")
    assert r1["name"] == "Caller #1"
    assert r2["name"] == "Caller #2"
