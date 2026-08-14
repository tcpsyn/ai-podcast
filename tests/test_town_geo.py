from backend.main import _get_town_from_location, BIG_BEND_TOWNS


def test_matches_simple_town():
    assert _get_town_from_location("Alpine, Texas") == "alpine"
    assert _get_town_from_location("Marfa") == "marfa"
    assert _get_town_from_location("outside Terlingua") == "terlingua"


def test_matches_two_word_town():
    assert _get_town_from_location("Fort Stockton, TX") == "fort stockton"
    assert _get_town_from_location("ft stockton") == "fort stockton"


def test_case_insensitive():
    assert _get_town_from_location("MARATHON, texas") == "marathon"


def test_no_match_returns_none():
    assert _get_town_from_location("Albuquerque, New Mexico") is None
    assert _get_town_from_location("") is None


def test_every_town_has_coords():
    for town, coords in BIG_BEND_TOWNS.items():
        assert town == town.lower()
        lat, lon = coords
        assert 28.0 < lat < 32.0
        assert -105.0 < lon < -102.0
