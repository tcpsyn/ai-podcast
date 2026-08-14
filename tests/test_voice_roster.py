from backend.main import (
    INWORLD_MALE_VOICES,
    INWORLD_FEMALE_VOICES,
    BLACKLISTED_VOICES,
)

NEWLY_ADDED = ["Arthur", "Daniel", "Brooke", "Joy", "Selene", "Zadie"]

# Child / adolescent voices — deliberately kept off the roster for an
# adult late-night call-in show.
EXCLUDED_CHILD_VOICES = ["Abby", "Mia", "Riley"]


def test_rosters_are_sorted_and_deduped():
    for roster in (INWORLD_MALE_VOICES, INWORLD_FEMALE_VOICES):
        assert roster == sorted(roster), "roster must stay alphabetical"
        assert len(roster) == len(set(roster)), "duplicate voice in roster"


def test_no_voice_appears_in_both_genders():
    overlap = set(INWORLD_MALE_VOICES) & set(INWORLD_FEMALE_VOICES)
    assert not overlap, f"voice in both rosters: {overlap}"


def test_newly_added_voices_are_present():
    roster = set(INWORLD_MALE_VOICES) | set(INWORLD_FEMALE_VOICES)
    missing = [v for v in NEWLY_ADDED if v not in roster]
    assert not missing, f"missing: {missing}"


def test_child_voices_stay_off_the_roster():
    roster = set(INWORLD_MALE_VOICES) | set(INWORLD_FEMALE_VOICES)
    present = [v for v in EXCLUDED_CHILD_VOICES if v in roster]
    assert not present, f"child/adolescent voice on roster: {present}"


def test_blacklist_only_references_real_roster_voices():
    roster = set(INWORLD_MALE_VOICES) | set(INWORLD_FEMALE_VOICES)
    orphans = sorted(set(BLACKLISTED_VOICES) - roster)
    assert not orphans, f"blacklist names voices not in the roster: {orphans}"


def test_effective_pool_is_large_enough_for_session_sampling():
    """main.py samples 25 voices per session from the non-blacklisted pool."""
    roster = set(INWORLD_MALE_VOICES) | set(INWORLD_FEMALE_VOICES)
    effective = roster - set(BLACKLISTED_VOICES)
    assert len(effective) >= 25
