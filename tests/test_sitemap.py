import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from generate_episode_pages import build_sitemap
from website_gen.feed import parse_feed

FIXTURE_FEED = Path(__file__).parent / "fixtures" / "feed_sample.xml"


@pytest.fixture
def episodes():
    return parse_feed(FIXTURE_FEED.read_text(encoding="utf-8"))


@pytest.fixture
def generated_sitemap(episodes):
    return build_sitemap(episodes)


def test_sitemap_contains_no_query_param_urls(generated_sitemap):
    assert "episode.html?slug=" not in generated_sitemap


def test_sitemap_has_one_entry_per_episode(generated_sitemap, episodes):
    assert generated_sitemap.count("<loc>https://lukeattheroost.com/episode/") == len(episodes)


def test_static_pages_survive_regeneration(generated_sitemap):
    for path in ["", "/how-it-works", "/clips", "/stats", "/privacy", "/terms", "/llms.txt"]:
        assert f"<loc>https://lukeattheroost.com{path}</loc>" in generated_sitemap


def test_lastmod_is_date_only_not_a_timestamp(generated_sitemap):
    for m in re.findall(r"<lastmod>([^<]*)</lastmod>", generated_sitemap):
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", m), f"bad lastmod: {m}"


def test_sitemap_is_valid_xml(generated_sitemap):
    root = ET.fromstring(generated_sitemap)
    assert root.tag.endswith("urlset")


def test_episode_urls_have_trailing_slash(generated_sitemap):
    for loc in re.findall(r"<loc>(https://lukeattheroost\.com/episode/[^<]*)</loc>", generated_sitemap):
        assert loc.endswith("/"), loc


def test_404_page_is_not_listed(generated_sitemap):
    assert "/404" not in generated_sitemap


def test_static_pages_come_before_episodes(generated_sitemap):
    first_episode = generated_sitemap.index("<loc>https://lukeattheroost.com/episode/")
    last_static = generated_sitemap.index("<loc>https://lukeattheroost.com/privacy</loc>")
    assert last_static < first_episode


def test_episodes_are_newest_first(generated_sitemap):
    locs = re.findall(r"<loc>https://lukeattheroost\.com/episode/([^<]*)/</loc>", generated_sitemap)
    assert locs[0].startswith("episode-58-")
    assert locs[1].startswith("episode-57-")


def test_declares_xml_and_sitemap_namespace(generated_sitemap):
    assert generated_sitemap.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"' in generated_sitemap
