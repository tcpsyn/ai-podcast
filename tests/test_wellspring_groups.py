"""Faction grouping for regulars.

Eight Wellspring characters share one lore world. Without a group cap they would
crowd out every other regular and turn the show into the cult hour.
"""
import random

import pytest

from backend.main import (
    GROUP_TWO_UP_CHANCE,
    MAX_REGULARS_PER_SHOW,
    _select_regulars_for_tonight,
)
from backend.services.regulars_v2 import Regular, load_regular


def make_regular(name, group="", group_lead=False, explicitness="low", register=""):
    return Regular(
        name=name, voice="Grant", age=47, arc_state="", lore_body="lore",
        file_path=None, group=group, group_lead=group_lead,
        register=register, explicitness=explicitness,
    )


def wellspring_roster():
    return [
        make_regular("Silas", group="wellspring", group_lead=True),
        make_regular("Doyle", group="wellspring"),
        make_regular("Renata", group="wellspring"),
        make_regular("Alvin", group="wellspring"),
        make_regular("Junie", group="wellspring"),
        make_regular("Ford", group="wellspring"),
        make_regular("Marguerite", group="wellspring"),
        make_regular("Cyrus", group="wellspring"),
    ]


ALWAYS = lambda name: 999  # never appeared -> probability 1.0, always a candidate


def test_group_never_exceeds_two():
    for seed in range(300):
        picked = _select_regulars_for_tonight(
            wellspring_roster(), ALWAYS, rng=random.Random(seed)
        )
        ws = [r for r, _, _ in picked if r.group == "wellspring"]
        assert len(ws) <= 2, f"seed {seed} produced {len(ws)} Wellspring callers"


def test_two_up_rate_is_near_configured_chance():
    two_ups = 0
    trials = 2000
    for seed in range(trials):
        picked = _select_regulars_for_tonight(
            wellspring_roster(), ALWAYS, rng=random.Random(seed)
        )
        ws = [r for r, _, _ in picked if r.group == "wellspring"]
        if len(ws) == 2:
            two_ups += 1
    rate = two_ups / trials
    assert abs(rate - GROUP_TWO_UP_CHANCE) < 0.05, f"two-up rate was {rate:.3f}"


def test_two_up_pairs_the_lead_with_one_other():
    """'Both sides tonight' means Silas plus a member, never two pool members."""
    seen_pair = False
    for seed in range(500):
        picked = _select_regulars_for_tonight(
            wellspring_roster(), ALWAYS, rng=random.Random(seed)
        )
        ws = [r for r, _, _ in picked if r.group == "wellspring"]
        if len(ws) == 2:
            seen_pair = True
            names = {r.name for r in ws}
            assert "Silas" in names, f"two-up without the lead: {names}"
            assert len(names) == 2
    assert seen_pair, "no two-up occurred in 500 seeds — test is not exercising the path"


def test_ungrouped_regulars_are_not_capped():
    roster = [make_regular(f"Solo{i}") for i in range(5)]
    picked = _select_regulars_for_tonight(roster, ALWAYS, rng=random.Random(1))
    assert len(picked) == MAX_REGULARS_PER_SHOW


def test_group_does_not_crowd_out_other_regulars():
    """The whole point: an ungrouped regular still gets slots on most nights."""
    roster = wellspring_roster() + [make_regular("Nadine"), make_regular("Boyd")]
    appearances = 0
    trials = 200
    for seed in range(trials):
        picked = _select_regulars_for_tonight(roster, ALWAYS, rng=random.Random(seed))
        if any(r.name in ("Nadine", "Boyd") for r, _, _ in picked):
            appearances += 1
    assert appearances / trials > 0.9


def test_never_exceeds_global_regular_cap():
    roster = wellspring_roster() + [make_regular(f"Solo{i}") for i in range(6)]
    for seed in range(200):
        picked = _select_regulars_for_tonight(roster, ALWAYS, rng=random.Random(seed))
        assert len(picked) <= MAX_REGULARS_PER_SHOW


def test_low_probability_regulars_can_be_skipped():
    """A regular seen last show should not be guaranteed."""
    roster = [make_regular("Silas", group="wellspring", group_lead=True)]
    skipped = any(
        not _select_regulars_for_tonight(roster, lambda n: 0, rng=random.Random(s))
        for s in range(50)
    )
    assert skipped


# --- frontmatter parsing ---

FRONTMATTER = """---
name: Doyle
voice: Grant
age: 47
group: wellspring
register: dry, understated, factual
explicitness: low
arc_state: Eleven months of telling himself he could leave tomorrow
---

# Doyle

He runs the soap kettles.
"""


def test_load_regular_parses_new_fields(tmp_path):
    path = tmp_path / "doyle.md"
    path.write_text(FRONTMATTER)
    r = load_regular(path)
    assert r.group == "wellspring"
    assert r.group_lead is False
    assert r.register == "dry, understated, factual"
    assert r.explicitness == "low"


def test_load_regular_defaults_when_fields_absent(tmp_path):
    """Existing regular files predate these fields and must keep loading."""
    path = tmp_path / "old.md"
    path.write_text("---\nname: Silas\nvoice: Sebastian\nage: 52\n---\n\n# Silas\n\nlore\n")
    r = load_regular(path)
    assert r.group == ""
    assert r.group_lead is False
    assert r.register == ""
    assert r.explicitness == "low"


def test_group_lead_parses_truthy(tmp_path):
    path = tmp_path / "silas.md"
    path.write_text(
        "---\nname: Silas\nvoice: Sebastian\nage: 52\ngroup: wellspring\n"
        "group_lead: true\nexplicitness: medium\n---\n\n# Silas\n\nlore\n"
    )
    r = load_regular(path)
    assert r.group_lead is True
    assert r.explicitness == "medium"


def test_invalid_explicitness_falls_back_to_low(tmp_path):
    path = tmp_path / "bad.md"
    path.write_text(
        "---\nname: X\nvoice: Grant\nage: 30\nexplicitness: extremely\n---\n\n# X\n\nlore\n"
    )
    assert load_regular(path).explicitness == "low"


# --- anchor weighting ---

def weighted_roster():
    return [
        make_regular("Silas", group="wellspring", group_lead=True),
        make_regular("Doyle", group="wellspring"),
        make_regular("Renata", group="wellspring"),
        make_regular("Alvin", group="wellspring"),
    ]


def test_group_weight_defaults_to_one(tmp_path):
    path = tmp_path / "x.md"
    path.write_text("---\nname: X\nvoice: Grant\nage: 30\n---\n\n# X\n\nlore\n")
    assert load_regular(path).group_weight == 1.0


def test_group_weight_parses(tmp_path):
    path = tmp_path / "doyle.md"
    path.write_text(
        "---\nname: Doyle\nvoice: Grant\nage: 47\ngroup: wellspring\n"
        "group_weight: 6\n---\n\n# Doyle\n\nlore\n"
    )
    assert load_regular(path).group_weight == 6.0


def test_invalid_group_weight_falls_back_to_one(tmp_path):
    for bad in ("heavy", "-3", "0"):
        path = tmp_path / f"bad-{bad}.md"
        path.write_text(
            f"---\nname: B\nvoice: Grant\nage: 30\ngroup_weight: {bad}\n---\n\n# B\n\nlore\n"
        )
        assert load_regular(path).group_weight == 1.0


def test_weighted_anchor_appears_far_more_than_pool_members():
    """The anchor carries an arc; without weighting it lands 1 show in 8 and the
    arc takes 30+ episodes to resolve."""
    import collections
    roster = weighted_roster()
    for r in roster:
        r.group_weight = 6.0 if r.name == "Doyle" else 1.0
    counts = collections.Counter()
    trials = 3000
    for seed in range(trials):
        for r, _, _ in _select_regulars_for_tonight(roster, ALWAYS, rng=random.Random(seed)):
            counts[r.name] += 1
    assert counts["Doyle"] > 4 * counts["Renata"]
    assert counts["Doyle"] > 4 * counts["Alvin"]


def test_weighting_does_not_break_the_group_cap():
    roster = wellspring_roster()
    for r in roster:
        r.group_weight = 6.0 if r.name in ("Doyle", "Silas") else 1.0
    for seed in range(300):
        picked = _select_regulars_for_tonight(roster, ALWAYS, rng=random.Random(seed))
        ws = [r for r, _, _ in picked if r.group == "wellspring"]
        assert len(ws) <= 2


def test_every_pool_member_remains_reachable():
    """Weighting must not starve a pool member to zero — they are the variety."""
    import collections
    roster = wellspring_roster()
    for r in roster:
        r.group_weight = 6.0 if r.name in ("Doyle", "Silas") else 1.0
    counts = collections.Counter()
    for seed in range(4000):
        for r, _, _ in _select_regulars_for_tonight(roster, ALWAYS, rng=random.Random(seed)):
            counts[r.name] += 1
    for r in roster:
        assert counts[r.name] > 0, f"{r.name} never appeared"
