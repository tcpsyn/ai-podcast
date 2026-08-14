import shutil
from pathlib import Path

import pytest

from generate_episode_pages import generate

FIXTURE_FEED = Path(__file__).parent / "fixtures" / "feed_sample.xml"

EP58_SLUG = "episode-58-rayfield-s-nephew-the-marfa-lights-and-why-nobody-believes-concho"
EP57_SLUG = "episode-57-trace-s-box-of-family-secrets"


@pytest.fixture
def feed_file(tmp_path):
    dest = tmp_path / "feed.xml"
    shutil.copyfile(FIXTURE_FEED, dest)
    return dest


@pytest.fixture
def transcripts_dir(tmp_path):
    d = tmp_path / "transcripts"
    d.mkdir()
    (d / f"{EP58_SLUG}.txt").write_text(
        "LUKE: Marfa Lights, line one.\n\nCONCHO: Nobody believes me.\n"
    )
    (d / f"{EP57_SLUG}.txt").write_text("LUKE: Trace, what's in the box?\n\nTRACE: Letters.\n")
    return d


@pytest.fixture
def out_root(tmp_path):
    return tmp_path / "site"


def test_writes_one_index_html_per_feed_episode(feed_file, transcripts_dir, out_root):
    n = generate(feed_file, transcripts_dir, out_root)
    assert n == 2
    assert (out_root / "episode" / EP58_SLUG / "index.html").exists()
    assert (out_root / "episode" / EP57_SLUG / "index.html").exists()


def test_orphan_transcript_without_feed_item_is_skipped(feed_file, transcripts_dir, out_root):
    """episode-32 has a transcript but was never published to the feed."""
    (transcripts_dir / "episode-32-tacos-taxes-and-tall-tales.txt").write_text("LUKE: Hi.")
    generate(feed_file, transcripts_dir, out_root)
    assert not (out_root / "episode" / "episode-32-tacos-taxes-and-tall-tales").exists()


def test_missing_transcript_still_produces_a_page(feed_file, transcripts_dir, out_root):
    """An episode published before its transcript lands must not break the build."""
    for f in transcripts_dir.glob("*.txt"):
        f.unlink()
    n = generate(feed_file, transcripts_dir, out_root)
    assert n == 2
    html = next((out_root / "episode").rglob("index.html")).read_text()
    assert "Transcript not yet available" in html


def test_dry_run_writes_nothing(feed_file, transcripts_dir, out_root):
    n = generate(feed_file, transcripts_dir, out_root, dry_run=True)
    assert n == 2
    assert not out_root.exists()


def test_transcript_content_lands_in_the_page(feed_file, transcripts_dir, out_root):
    generate(feed_file, transcripts_dir, out_root)
    html = (out_root / "episode" / EP58_SLUG / "index.html").read_text()
    assert "transcript-turn" in html
    assert "Nobody believes me." in html


def test_pages_link_to_each_other(feed_file, transcripts_dir, out_root):
    """Prev/next links are how a crawler reaches all 57 pages."""
    generate(feed_file, transcripts_dir, out_root)
    pages = list((out_root / "episode").rglob("index.html"))
    combined = "\n".join(p.read_text() for p in pages)
    assert combined.count("/episode/") >= len(pages)


def test_prev_points_older_and_next_points_newer(feed_file, transcripts_dir, out_root):
    generate(feed_file, transcripts_dir, out_root)
    newest = (out_root / "episode" / EP58_SLUG / "index.html").read_text()
    oldest = (out_root / "episode" / EP57_SLUG / "index.html").read_text()
    assert f'rel="prev" href="/episode/{EP57_SLUG}/"' in newest
    assert 'rel="next"' not in newest
    assert f'rel="next" href="/episode/{EP58_SLUG}/"' in oldest
    assert 'rel="prev"' not in oldest


def test_accepts_string_paths(feed_file, transcripts_dir, out_root):
    n = generate(str(feed_file), str(transcripts_dir), str(out_root))
    assert n == 2
