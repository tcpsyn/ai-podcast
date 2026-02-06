import sys
sys.path.insert(0, "/Users/lukemacneil/ai-podcast")

from backend.services.caller_service import CallerService


def test_queue_starts_empty():
    svc = CallerService()
    assert svc.get_queue() == []


def test_add_caller_to_queue():
    svc = CallerService()
    svc.add_to_queue("abc123", "+15551234567")
    q = svc.get_queue()
    assert len(q) == 1
    assert q[0]["caller_id"] == "abc123"
    assert q[0]["phone"] == "+15551234567"
    assert "wait_time" in q[0]


def test_remove_caller_from_queue():
    svc = CallerService()
    svc.add_to_queue("abc123", "+15551234567")
    svc.remove_from_queue("abc123")
    assert svc.get_queue() == []


def test_allocate_channel():
    svc = CallerService()
    ch1 = svc.allocate_channel()
    ch2 = svc.allocate_channel()
    assert ch1 == 3
    assert ch2 == 4
    svc.release_channel(ch1)
    ch3 = svc.allocate_channel()
    assert ch3 == 3


def test_take_call():
    svc = CallerService()
    svc.add_to_queue("abc123", "+15551234567")
    result = svc.take_call("abc123")
    assert result["caller_id"] == "abc123"
    assert result["channel"] >= 3
    assert svc.get_queue() == []
    assert svc.active_calls["abc123"]["channel"] == result["channel"]


def test_hangup_real_caller():
    svc = CallerService()
    svc.add_to_queue("abc123", "+15551234567")
    svc.take_call("abc123")
    ch = svc.active_calls["abc123"]["channel"]
    svc.hangup("abc123")
    assert "abc123" not in svc.active_calls
    assert ch not in svc._allocated_channels


def test_caller_counter_increments():
    svc = CallerService()
    svc.add_to_queue("id1", "+15551234567")
    svc.add_to_queue("id2", "+15559876543")
    r1 = svc.take_call("id1")
    r2 = svc.take_call("id2")
    assert r1["phone"] == "+15551234567"
    assert r2["phone"] == "+15559876543"


def test_register_and_unregister_websocket():
    svc = CallerService()
    fake_ws = object()
    svc.register_websocket("abc123", fake_ws)
    assert svc._websockets["abc123"] is fake_ws
    svc.unregister_websocket("abc123")
    assert "abc123" not in svc._websockets


def test_hangup_clears_websocket():
    svc = CallerService()
    svc.add_to_queue("abc123", "+15551234567")
    svc.take_call("abc123")
    svc.register_websocket("abc123", object())
    svc.hangup("abc123")
    assert "abc123" not in svc._websockets


def test_reset_clears_websockets():
    svc = CallerService()
    svc.register_websocket("id1", object())
    svc.register_websocket("id2", object())
    svc.reset()
    assert svc._websockets == {}


def test_send_audio_no_websocket():
    """send_audio_to_caller returns silently when no WS registered"""
    import asyncio
    svc = CallerService()
    asyncio.get_event_loop().run_until_complete(
        svc.send_audio_to_caller("NONE", b"\x00" * 100, 16000)
    )


def test_send_audio_json():
    """send_audio_to_caller sends base64 JSON via SignalWire protocol"""
    import asyncio
    import json
    import base64

    class FakeWS:
        def __init__(self):
            self.sent_text = []
        async def send_text(self, data):
            self.sent_text.append(data)

    svc = CallerService()
    ws = FakeWS()
    svc.register_websocket("abc123", ws)
    pcm = b"\x00\x01" * 100
    asyncio.get_event_loop().run_until_complete(
        svc.send_audio_to_caller("abc123", pcm, 16000)
    )
    assert len(ws.sent_text) == 1
    msg = json.loads(ws.sent_text[0])
    assert msg["event"] == "media"
    assert base64.b64decode(msg["media"]["payload"]) == pcm


def test_take_call_preserves_caller_phone():
    """take_call uses the phone from the queue"""
    svc = CallerService()
    svc.add_to_queue("abc123", "+15551234567")
    result = svc.take_call("abc123")
    assert result["phone"] == "+15551234567"
