import pytest

from publish_episode import MASTODON_LIMIT, THREADS_LIMIT, _build_platform_content

LONG_DESC = ("Luke takes calls from Darnell Pruitt, a Pecos County oilfield hauler deciding whether to let "
             "his 15-year-old daughter move in, and Noreen, a nurse who watched a coworker pocket leftover "
             "opioids. Other callers include Hollis Beckett, who climbed a ladder to read a mystery date on an "
             "old church bell, Junie, who is planning her escape from the Wellspring, Wendell, who has been "
             "losing 11 minutes off his dashboard clock, and Sloan Whitaker, who sat in a casino parking lot "
             "with a $2,200 raffle windfall. Tune in to hear how each caller's dilemma gets resolved.")
META = {"title": "Episode 62: Darnell's Daughter, a Room Full of Tire Rims, and a Nurse Who Saw Too Much",
        "description": LONG_DESC}
URL = "https://lukeattheroost.com/episode/episode-62-darnell-s-daughter-a-room-full-of-tire-rims-and-a-nurse-who-saw-too-much/"
YT = "https://youtube.com/watch?v=bWUSBbmjnZY"


@pytest.mark.parametrize("platform,limit", [("threads", THREADS_LIMIT), ("mastodon", MASTODON_LIMIT)])
def test_long_episode_fits_limit_and_keeps_link(platform, limit):
    content = _build_platform_content(META, URL, YT, platform)
    assert len(content) <= limit
    assert content.startswith(META["title"])
    assert ("lukeattheroost.com" in content) if platform == "threads" else (URL in content and YT in content)


def test_trim_ends_cleanly():
    content = _build_platform_content(META, URL, YT, "mastodon")
    desc_part = content.split("\n\n")[1]
    assert desc_part.endswith((".", "…"))


def test_short_description_untouched():
    meta = {"title": "Ep 1", "description": "Short and sweet."}
    assert "Short and sweet." in _build_platform_content(meta, URL, None, "threads")
