"""YouTube rejects the whole upload with `invalidTags` if the tag list busts
its 500-character budget. Any tag containing a space gets wrapped in quotes and
those quotes count, so the naive sum of tag lengths understates the real cost.

Episode 58 failed here after a 284 MB upload had already completed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from publish_episode import _extract_youtube_tags, YOUTUBE_TAG_BUDGET


def tag_cost(tags):
    """Mirror YouTube's accounting: quotes around multi-word tags, plus commas."""
    return sum(len(t) + (2 if " " in t else 0) for t in tags) + max(0, len(tags) - 1)


# The real chapter titles from episode 58, which produced a 534-char tag list.
EP58_CHAPTERS = [
    {"title": "Intro & Election Fraud Voicemail"},
    {"title": "Rayfield: Nephew Stealing Catalytic Converters"},
    {"title": "Suki & the Marfa Lights"},
    {"title": "Aurora Toothbrush Sponsor"},
    {"title": "Gus: Partner's Unlogged Stop at Meth House"},
    {"title": "Merritt: Son Stealing from Family Store"},
    {"title": "Concho: Mysterious Water Source Surveyors"},
    {"title": "Desmond: Colleague's Research Misconduct"},
    {"title": "Iron Heart Survival School Sponsor"},
    {"title": "Fern: Donald Judd's Aluminum Boxes in Marfa"},
    {"title": "Outro & Show Reflection"},
]


def test_episode_58_tags_fit_the_budget():
    """The exact input that broke the ep58 upload."""
    tags = _extract_youtube_tags({"chapters": EP58_CHAPTERS})
    assert tag_cost(tags) <= YOUTUBE_TAG_BUDGET, (
        f"{tag_cost(tags)} chars > {YOUTUBE_TAG_BUDGET} budget: {tags}"
    )


def test_pathological_long_titles_still_fit():
    chapters = [{"title": "A" * 49 + f" {i}"} for i in range(40)]
    tags = _extract_youtube_tags({"chapters": chapters})
    assert tag_cost(tags) <= YOUTUBE_TAG_BUDGET, tag_cost(tags)
    assert len(tags) <= 25


def test_no_chapters_still_returns_base_tags():
    tags = _extract_youtube_tags({"chapters": []})
    assert tags, "should still return the base SEO tags"
    assert tag_cost(tags) <= YOUTUBE_TAG_BUDGET


def test_base_tags_are_prioritised_over_chapter_titles():
    """Base SEO tags matter more than chapter titles; they must survive trimming."""
    chapters = [{"title": "B" * 48 + f" {i}"} for i in range(30)]
    tags = _extract_youtube_tags({"chapters": chapters})
    assert "podcast" in tags
    assert "Luke at the Roost" in tags


def test_angle_brackets_are_stripped():
    """YouTube rejects tags containing < or >."""
    chapters = [{"title": "Weird <script> Chapter"}]
    tags = _extract_youtube_tags({"chapters": chapters})
    assert not any("<" in t or ">" in t for t in tags), tags


def test_skips_intro_outro_and_short_titles():
    chapters = [{"title": "Intro"}, {"title": "Outro"}, {"title": "ab"},
                {"title": "A Real Chapter Title"}]
    tags = _extract_youtube_tags({"chapters": chapters})
    assert "Intro" not in tags and "Outro" not in tags and "ab" not in tags
    assert "A Real Chapter Title" in tags
