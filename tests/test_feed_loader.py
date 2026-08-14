from pathlib import Path

import pytest

from website_gen.feed import parse_feed

FIXTURE = Path(__file__).parent / "fixtures" / "feed_sample.xml"


@pytest.fixture
def feed_xml():
    return FIXTURE.read_text(encoding="utf-8")


def test_parses_core_fields(feed_xml):
    eps = parse_feed(feed_xml)
    ep = next(e for e in eps if e.number == 58)
    assert ep.slug == "episode-58-rayfield-s-nephew-the-marfa-lights-and-why-nobody-believes-concho"
    assert ep.title.startswith("Episode 58: Rayfield's Nephew")
    assert ep.duration_seconds == 4770
    assert ep.audio_url.startswith("https://")


def test_description_is_stripped_of_cdata_and_html(feed_xml):
    ep = parse_feed(feed_xml)[0]
    assert "<![CDATA[" not in ep.description
    assert "<p>" not in ep.description


def test_slug_comes_from_link_and_drops_trailing_slash(feed_xml):
    for ep in parse_feed(feed_xml):
        assert not ep.slug.endswith("/")
        assert "/" not in ep.slug


def test_pubdate_parses_to_iso_date(feed_xml):
    ep = next(e for e in parse_feed(feed_xml) if e.number == 58)
    assert ep.published_iso.startswith("2026-08-04")


def test_returns_all_items_in_fixture(feed_xml):
    assert len(parse_feed(feed_xml)) == 2
