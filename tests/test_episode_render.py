import json
import re

import pytest

from website_gen.feed import Episode
from website_gen.render import render_episode_page


@pytest.fixture
def sample_episode():
    return Episode(
        number=58,
        slug="episode-58-rayfield-s-nephew-the-marfa-lights-and-why-nobody-believes-concho",
        title="Episode 58: Rayfield's Nephew, the Marfa Lights, and Why Nobody Believes Concho",
        description=(
            "Luke fields calls about stolen catalytic converters, unexplained phenomena over "
            "Mitchell Flat, a sheriff's deputy's moral dilemma, and a fence builder convinced "
            "someone is mapping West Texas water sources."
        ),
        published_iso="2026-08-04T09:22:38+00:00",
        duration_seconds=4770,
        audio_url=(
            "https://podcast.macneilmediagroup.com/audio/@LukeAtTheRoost/"
            "episode-58-rayfield-s-nephew-the-marfa-lights-and-why-nobody-believes-concho.mp3"
        ),
    )


@pytest.fixture
def other_episode():
    return Episode(
        number=57,
        slug="episode-57-trace-s-box-of-family-secrets",
        title="Episode 57: Trace's Box of Family Secrets",
        description="Letters in a shoebox turn a family story inside out.",
        published_iso="2026-06-02T11:32:58+00:00",
        duration_seconds=4066,
        audio_url=(
            "https://podcast.macneilmediagroup.com/audio/@LukeAtTheRoost/"
            "episode-57-trace-s-box-of-family-secrets.mp3"
        ),
    )


def test_title_and_canonical_are_episode_specific(sample_episode):
    html = render_episode_page(sample_episode, turns=[("LUKE", "Hello.")])
    assert "<title>Episode 58: Rayfield" in html
    assert '<link rel="canonical" href="https://lukeattheroost.com/episode/episode-58-' in html


def test_transcript_is_in_the_html_not_fetched_by_js(sample_episode):
    html = render_episode_page(sample_episode, turns=[("LUKE", "The Marfa Lights are real.")])
    assert "The Marfa Lights are real." in html
    assert "fetch(" not in html


def test_emits_valid_podcastepisode_schema(sample_episode):
    html = render_episode_page(sample_episode, turns=[("LUKE", "Hi.")])
    block = re.search(r'<script type="application/ld\+json">(.*?)</script>', html, re.S).group(1)
    data = json.loads(block)
    types = {o["@type"] for o in (data if isinstance(data, list) else [data])}
    assert "PodcastEpisode" in types


def test_schema_carries_big_bend_content_location(sample_episode):
    html = render_episode_page(sample_episode, turns=[])
    block = re.search(r'<script type="application/ld\+json">(.*?)</script>', html, re.S).group(1)
    data = json.loads(block)
    ep = next(o for o in data if o["@type"] == "PodcastEpisode")
    assert ep["contentLocation"]["address"]["addressLocality"] == "Alpine"


def test_escapes_html_in_transcript_text(sample_episode):
    html = render_episode_page(sample_episode, turns=[("LUKE", "5 < 6 & <script>alert(1)</script>")])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_escapes_quotes_in_title_meta(sample_episode):
    sample_episode.title = 'Episode 1: The "Best" Show'
    html = render_episode_page(sample_episode, turns=[])
    assert 'content="Episode 1: The "Best"' not in html


def test_schema_json_is_valid_even_with_quotes_in_title(sample_episode):
    """JSON-LD must stay parseable when the title contains quotes and apostrophes."""
    sample_episode.title = 'Ep "quoted" and Rayfield\'s'
    html = render_episode_page(sample_episode, turns=[])
    block = re.search(r'<script type="application/ld\+json">(.*?)</script>', html, re.S).group(1)
    json.loads(block)  # must not raise


def test_speaker_labels_get_semantic_markup(sample_episode):
    html = render_episode_page(sample_episode, turns=[("LUKE", "Hi."), ("SLIM", "Hey.")])
    assert html.count('class="transcript-turn"') == 2
    assert "LUKE" in html and "SLIM" in html


def test_all_asset_paths_are_root_absolute(sample_episode):
    """Pages live at /episode/<slug>/ so relative asset paths would 404."""
    html = render_episode_page(sample_episode, turns=[])
    assert 'href="css/' not in html
    assert 'src="js/' not in html
    assert re.search(r'href="/css/style\.css\?v=\d+"', html)
    assert 'src="/js/footer.js"' in html


def test_prev_next_links_render_when_given(sample_episode, other_episode):
    html = render_episode_page(sample_episode, turns=[], prev_ep=other_episode)
    assert f'/episode/{other_episode.slug}/' in html


def test_missing_transcript_renders_placeholder(sample_episode):
    html = render_episode_page(sample_episode, turns=[])
    assert "Transcript not yet available" in html


def test_audio_element_present_for_no_js_playback(sample_episode):
    html = render_episode_page(sample_episode, turns=[])
    assert "<audio" in html and sample_episode.audio_url in html
