"""A --resume run must not invent the episode title.

Episode 58's first run published to Castopod, then died on the YouTube upload
(see test_youtube_tags.py). The re-run with --resume rebuilt the title from the
URL slug, which is lowercase and punctuation-free, so
"Episode 58: Rayfield's Nephew, the Marfa Lights, and Why Nobody Believes Concho"
went to YouTube as
"Episode 58: Rayfield S Nephew The Marfa Lights And Why Nobody Believes Concho"
with the description replaced by a placeholder.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from publish_episode import recover_metadata, save_metadata, _decode_db_row

EP58_TITLE = "Episode 58: Rayfield's Nephew, the Marfa Lights, and Why Nobody Believes Concho"
EP58_SLUG = "episode-58-rayfield-s-nephew-the-marfa-lights-and-why-nobody-believes-concho"
EP58_DESC = "Rayfield calls about his nephew and the catalytic converters."
CHAPTERS = [{"startTime": 0, "title": "Intro"}]


def test_metadata_file_roundtrips_losslessly(tmp_path):
    path = tmp_path / "ep58.metadata.json"
    save_metadata({"title": EP58_TITLE, "description": EP58_DESC,
                   "chapters": CHAPTERS, "thumbnail_text": "RAYFIELD"}, str(path))

    meta = recover_metadata(58, EP58_SLUG, path, CHAPTERS)

    assert meta["title"] == EP58_TITLE
    assert meta["description"] == EP58_DESC


def test_metadata_file_wins_over_slug(tmp_path):
    """The exact ep58 regression: apostrophes, commas and case must survive."""
    path = tmp_path / "ep58.metadata.json"
    save_metadata({"title": EP58_TITLE, "description": EP58_DESC,
                   "chapters": CHAPTERS, "thumbnail_text": "RAYFIELD"}, str(path))

    meta = recover_metadata(58, EP58_SLUG, path, CHAPTERS)

    assert "Rayfield S Nephew" not in meta["title"]
    assert "Rayfield's Nephew" in meta["title"]
    assert "the Marfa Lights" in meta["title"]
    assert meta["description"] != "Episode 58 of Luke at the Roost."


def test_falls_back_to_castopod_db_when_file_missing(tmp_path):
    path = tmp_path / "missing.metadata.json"
    calls = []

    def db_lookup(episode_number):
        calls.append(episode_number)
        return {"title": EP58_TITLE, "description": EP58_DESC}

    meta = recover_metadata(58, EP58_SLUG, path, CHAPTERS, db_lookup=db_lookup)

    assert calls == [58]
    assert meta["title"] == EP58_TITLE
    assert meta["description"] == EP58_DESC


def test_slug_fallback_only_when_file_and_db_unavailable(tmp_path, capsys):
    path = tmp_path / "missing.metadata.json"

    meta = recover_metadata(58, EP58_SLUG, path, CHAPTERS,
                            db_lookup=lambda n: None)

    assert meta["title"].startswith("Episode 58: ")
    warning = capsys.readouterr().out
    assert "lossy" in warning.lower() or "warning" in warning.lower()


def test_slug_fallback_does_not_title_case_away_real_words(tmp_path):
    """Even degraded, the fallback should not be silently trusted."""
    meta = recover_metadata(58, EP58_SLUG, tmp_path / "nope.json", CHAPTERS,
                            db_lookup=lambda n: None)
    assert meta.get("title_is_reconstructed") is True


def test_episode_prefix_stripped_only_at_start(tmp_path):
    """A global replace would also eat the phrase inside the title."""
    slug = "episode-5-the-episode-5-mixup"
    meta = recover_metadata(5, slug, tmp_path / "nope.json", CHAPTERS,
                            db_lookup=lambda n: None)
    assert meta["title"].lower().count("episode 5") == 2


def test_chapters_come_from_chapters_json_not_metadata_file(tmp_path):
    path = tmp_path / "ep58.metadata.json"
    save_metadata({"title": EP58_TITLE, "description": EP58_DESC,
                   "chapters": [{"startTime": 999, "title": "Stale"}],
                   "thumbnail_text": "X"}, str(path))

    meta = recover_metadata(58, EP58_SLUG, path, CHAPTERS)

    assert meta["chapters"] == CHAPTERS


def test_decodes_base64_wrapped_by_mariadb():
    """TO_BASE64 wraps at 76 chars and mysql renders those breaks as a literal
    backslash-n, which b64decode rejects outright."""
    import base64 as b64
    payload = json.dumps({"title": EP58_TITLE, "description": EP58_DESC})
    encoded = b64.b64encode(payload.encode()).decode()
    wrapped = "\\n".join(encoded[i:i + 76] for i in range(0, len(encoded), 76))

    assert "\\n" in wrapped, "test needs a payload long enough to wrap"
    row = _decode_db_row(wrapped)

    assert row["title"] == EP58_TITLE
    assert row["description"] == EP58_DESC


def test_decodes_base64_with_real_newlines():
    import base64 as b64
    payload = json.dumps({"title": EP58_TITLE, "description": EP58_DESC})
    encoded = b64.b64encode(payload.encode()).decode()
    wrapped = "\n".join(encoded[i:i + 76] for i in range(0, len(encoded), 76))

    assert _decode_db_row(wrapped)["title"] == EP58_TITLE


def test_decode_db_row_returns_none_on_garbage():
    assert _decode_db_row("") is None
    assert _decode_db_row("not base64 at all !!!") is None


def test_save_metadata_keeps_only_publishable_fields(tmp_path):
    path = tmp_path / "m.json"
    save_metadata({"title": EP58_TITLE, "description": EP58_DESC,
                   "chapters": CHAPTERS, "thumbnail_text": "RAYFIELD",
                   "transcript": "huge blob that should not be persisted"}, str(path))

    saved = json.loads(path.read_text())
    assert "transcript" not in saved
    assert saved["title"] == EP58_TITLE
