import asyncio
import json

import pytest

from backend.main import (
    Voicemail,
    _load_voicemails,
    _save_voicemails,
    play_voicemail_on_air,
)
import backend.main as main


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Point the voicemail store at a temp dir and start from a clean slate."""
    meta = tmp_path / "voicemails.json"
    monkeypatch.setattr(main, "VOICEMAILS_META", meta)
    monkeypatch.setattr(main, "_voicemails", [])
    monkeypatch.setattr(main, "_deleted_vm_timestamps", set())
    return meta


def test_voicemail_defaults_to_empty_transcript():
    vm = Voicemail(id="a1", phone="+15551234567", timestamp=1.0, duration=30, file_path="/x.wav")
    assert vm.transcript == ""


def test_transcript_round_trips_through_save_and_load(isolated_store):
    main._voicemails.append(
        Voicemail(
            id="a1", phone="+15551234567", timestamp=1.0, duration=30,
            file_path="/x.wav", transcript="Hey Luke, it's Susan out in Deming.",
        )
    )
    _save_voicemails()

    on_disk = json.loads(isolated_store.read_text())
    assert on_disk["voicemails"][0]["transcript"] == "Hey Luke, it's Susan out in Deming."

    main._voicemails.clear()
    _load_voicemails()
    assert main._voicemails[0].transcript == "Hey Luke, it's Susan out in Deming."


def test_load_tolerates_legacy_entries_without_transcript(isolated_store):
    isolated_store.write_text(json.dumps({
        "voicemails": [{
            "id": "old", "phone": "+15551234567", "timestamp": 1.0,
            "duration": 30, "file_path": "/x.wav", "listened": True,
        }],
        "deleted_timestamps": [],
    }))
    _load_voicemails()
    assert main._voicemails[0].transcript == ""


def test_play_on_air_puts_transcript_into_conversation(isolated_store, tmp_path, monkeypatch):
    """The actual fix: Devon reads session.conversation, so playing a
    voicemail on air must write its transcript there."""
    wav = tmp_path / "vm.wav"
    wav.write_bytes(b"fake wav bytes")

    main._voicemails.append(
        Voicemail(
            id="vm1", phone="+15753424105", timestamp=1.0, duration=77,
            file_path=str(wav), transcript="My neighbor keeps moving my fence posts.",
        )
    )

    monkeypatch.setattr(main.session, "conversation", [])
    monkeypatch.setattr(main, "_save_voicemails", lambda: None)

    played = {}
    monkeypatch.setattr(
        main.audio_service, "play_caller_audio",
        lambda data, sr: played.update(bytes=len(data), rate=sr),
    )

    class FakeLibrosa:
        @staticmethod
        def load(path, sr=None, mono=True):
            import numpy as np
            return np.zeros(sr or 24000, dtype="float32"), sr or 24000

    monkeypatch.setitem(__import__("sys").modules, "librosa", FakeLibrosa)

    result = asyncio.run(play_voicemail_on_air("vm1"))
    assert result["status"] == "playing"

    roles = [m["role"] for m in main.session.conversation]
    contents = [m["content"] for m in main.session.conversation]
    assert any("voicemail" in r for r in roles), f"no voicemail role in {roles}"
    assert any("fence posts" in c for c in contents), f"transcript missing from {contents}"


def test_voicemail_role_is_normalized_to_a_valid_llm_role():
    """A raw 'voicemail:+1555...' role would reach the LLM API as an invalid
    role; llm.py swallows the resulting error and returns empty text, so
    callers would silently go mute."""
    out = main._normalize_messages_for_llm([
        {"role": "voicemail:+15753424105", "content": "My name is Sandra."},
    ])
    assert out[0]["role"] in ("user", "assistant", "system"), out[0]["role"]
    assert "Sandra" in out[0]["content"]
    assert "+15753424105" in out[0]["content"], "caller identity should survive"


def test_play_on_air_without_transcript_still_announces_the_voicemail(isolated_store, tmp_path, monkeypatch):
    """An untranscribed voicemail should still tell Devon something happened,
    rather than silently playing audio he can't perceive."""
    wav = tmp_path / "vm.wav"
    wav.write_bytes(b"fake wav bytes")

    main._voicemails.append(
        Voicemail(
            id="vm2", phone="+15753424105", timestamp=1.0, duration=77,
            file_path=str(wav), transcript="",
        )
    )

    monkeypatch.setattr(main.session, "conversation", [])
    monkeypatch.setattr(main, "_save_voicemails", lambda: None)
    monkeypatch.setattr(main.audio_service, "play_caller_audio", lambda data, sr: None)

    class FakeLibrosa:
        @staticmethod
        def load(path, sr=None, mono=True):
            import numpy as np
            return np.zeros(sr or 24000, dtype="float32"), sr or 24000

    monkeypatch.setitem(__import__("sys").modules, "librosa", FakeLibrosa)

    asyncio.run(play_voicemail_on_air("vm2"))

    assert main.session.conversation, "nothing was added to the conversation"
    assert any("voicemail" in m["role"] for m in main.session.conversation)
