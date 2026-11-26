"""Shared Wellspring lore: bible parsing, storyline rotation, the post-call
continuity pass, and prompt injection for members only."""
import asyncio
from unittest.mock import AsyncMock, patch

from backend.services import wellspring
from backend.services.regulars_v2 import load_regular

BIBLE = """# The Wellspring

Header prose stays untouched.

## The Place

Raw desert off FM 170 outside Terlingua.

## The Current

Static, the Unbinding.

## The People

- Silas (52) — founder.

## Storylines

### water
- title: Water and the neighbor
- status: open
- last_advanced: 2026-08-14
- involves: Silas, Doyle
- state: No well yet.

### stones
- title: The consent stones
- status: open
- last_advanced: 2026-09-20
- involves: Renata, Silas
- state: Stones are causing a hierarchy.

### old
- title: Closed thing
- status: closed
- last_advanced: 2026-01-01
- involves: Silas
- state: Done.

## Established Facts

- No signal on the land.

## Timeline

- 2026-03-02: First call.
"""

MEMBER = """---
name: Doyle
voice: Grant
age: 47
group: wellspring
arc_state: Has started writing things down.
---

# Doyle

Runs the kettles.

## The Wellspring, to me

A good soap room with bad paperwork.

## Arc Log

- (not yet aired)
"""

LONER = """---
name: Crispin
voice: Evan
age: 40
arc_state: Panicking about the journal.
---

# Crispin

A fiction writer.

## Arc Log

- 2026-09-01: First call.
"""


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_bible_round_trips_unchanged():
    assert wellspring.render_bible(wellspring.parse_bible(BIBLE)) == BIBLE


def test_bible_parses_machine_sections():
    b = wellspring.parse_bible(BIBLE)
    assert [s.slug for s in b.storylines] == ["water", "stones", "old"]
    assert b.storylines[0].involves == ["Silas", "Doyle"]
    assert [s.slug for s in b.open_storylines] == ["water", "stones"]
    assert b.facts == ["No signal on the land."]
    assert b.timeline == ["2026-03-02: First call."]
    assert "Terlingua" in b.section("The Place")


def test_pick_storyline_takes_stalest_open_thread_for_member():
    b = wellspring.parse_bible(BIBLE)
    assert wellspring.pick_storyline(b, "Silas").slug == "water"
    assert wellspring.pick_storyline(b, "Renata").slug == "stones"
    assert wellspring.pick_storyline(b, "Nobody").slug == "water"


def test_world_context_includes_shared_facts_and_assigned_storyline():
    b = wellspring.parse_bible(BIBLE)
    ctx = wellspring.world_context(b, "Silas", b.storylines[0])
    assert "Terlingua" in ctx
    assert "No signal on the land." in ctx
    assert "Water and the neighbor: No well yet." in ctx
    assert "Closed thing" not in ctx
    assert "TONIGHT SILAS IS CALLING MAINLY ABOUT: Water and the neighbor" in ctx


def test_apply_to_bible_updates_only_machine_sections():
    b = wellspring.parse_bible(BIBLE)
    result = {
        "storylines": [{"slug": "water", "state": "Dub dropped the permit fight.", "status": "closed"},
                       {"slug": "unknown", "state": "ignored"}],
        "new_storylines": [{"slug": "The Well!", "title": "Digging", "involves": ["Gerald"], "state": "Rig arrives."}],
        "facts": ["No signal on the land.", "Gerald found water at 300 feet."],
        "timeline": "Dub backs off.",
        "ripples": "Cyrus heard the call in town.",
    }
    text = wellspring.render_bible(wellspring.apply_to_bible(b, result, "2026-10-25"))
    out = wellspring.parse_bible(text)
    water = next(s for s in out.storylines if s.slug == "water")
    assert (water.status, water.state, water.last_advanced) == ("closed", "Dub dropped the permit fight.", "2026-10-25")
    assert any(s.slug == "thewell" and s.involves == ["Gerald"] for s in out.storylines)
    assert out.facts == ["No signal on the land.", "Gerald found water at 300 feet."]
    assert out.timeline[-2:] == ["2026-10-25: Dub backs off.", "2026-10-25: Cyrus heard the call in town."]
    assert "Header prose stays untouched." in text
    assert out.section("The Place") == "Raw desert off FM 170 outside Terlingua."


def test_apply_to_regular_replaces_arc_state_and_appends_log(tmp_path):
    p = write(tmp_path, "doyle.md", MEMBER)
    wellspring.apply_to_regular(p, "Asked Silas for his statement.", "Waiting on the statement.", "2026-10-25")
    text = p.read_text()
    assert "arc_state: Waiting on the statement." in text
    assert "Has started writing things down." not in text
    assert "(not yet aired)" not in text
    assert text.rstrip().endswith("- 2026-10-25: Asked Silas for his statement.")
    assert "A good soap room with bad paperwork." in text
    reg = load_regular(p)
    assert reg.group == "wellspring"
    assert wellspring.recent_arc_log(reg.lore_body) == ["2026-10-25: Asked Silas for his statement."]


def test_apply_to_regular_inserts_before_following_section(tmp_path):
    p = write(tmp_path, "x.md", LONER.replace("## Arc Log\n\n- 2026-09-01: First call.\n",
                                             "## Arc Log\n\n- 2026-09-01: First call.\n\n## Notes\n\nkeep\n"))
    wellspring.apply_to_regular(p, "Second call.", "", "2026-10-25")
    text = p.read_text()
    assert "- 2026-09-01: First call.\n- 2026-10-25: Second call.\n\n## Notes\n\nkeep" in text
    assert "arc_state: Panicking about the journal." in text


def test_perspective_and_prompt_context_only_for_members(tmp_path, monkeypatch):
    bible_path = write(tmp_path, "wellspring.md", BIBLE)
    monkeypatch.setattr(wellspring, "BIBLE_PATH", bible_path)
    monkeypatch.setattr(wellspring.load_bible, "__defaults__", (bible_path,))
    member = load_regular(write(tmp_path, "doyle.md", MEMBER))
    loner = load_regular(write(tmp_path, "crispin.md", LONER))
    ctx = wellspring.prompt_context(member)
    assert "Terlingua" in ctx and "A good soap room with bad paperwork." in ctx
    assert "TONIGHT" not in ctx  # only the group lead gets an assigned storyline
    assert wellspring.prompt_context(loner) == ""


def test_run_continuity_ignores_non_regulars():
    with patch.object(wellspring, "find_regular", return_value=None):
        assert asyncio.run(wellspring.run_continuity("Walk-in", "LUKE: hi")) is False


def test_run_continuity_writes_member_and_bible(tmp_path, monkeypatch):
    bible_path = write(tmp_path, "wellspring.md", BIBLE)
    monkeypatch.setattr(wellspring.load_bible, "__defaults__", (bible_path,))
    monkeypatch.setattr(wellspring.save_bible, "__defaults__", (bible_path,))
    member = load_regular(write(tmp_path, "doyle.md", MEMBER))
    fake = {"arc_log": "Called about the well.", "arc_state": "Hauling water.",
            "timeline": "Doyle called about the well.",
            "storylines": [{"slug": "water", "state": "Still hauling.", "status": "open"}],
            "new_storylines": [], "facts": [], "ripples": ""}
    with patch.object(wellspring, "find_regular", return_value=member), \
         patch.object(wellspring, "_call_continuity_model", new=AsyncMock(return_value=fake)) as model:
        assert asyncio.run(wellspring.run_continuity("Doyle", "LUKE: hi\nDOYLE: hey", today="2026-10-25"))
    assert "THE WELLSPRING — SHARED WORLD" in model.call_args.args[0]
    assert "arc_state: Hauling water." in member.file_path.read_text()
    out = wellspring.parse_bible(bible_path.read_text())
    assert out.timeline[-1] == "2026-10-25: Doyle called about the well."
    assert next(s for s in out.storylines if s.slug == "water").last_advanced == "2026-10-25"


def test_caller_prompt_injects_world_for_member_only(monkeypatch):
    from backend.main import get_caller_prompt
    caller ={"name": "Doyle", "identity": "soap guy", "situation": "the well", "reason_calling": "water"}
    fake_member = type("R", (), {"name": "Doyle", "arc_state": "Hauling water.", "group": "wellspring",
                                 "lore_body": "## Arc Log\n\n- 2026-10-01: Asked for the statement.\n"})()
    monkeypatch.setattr(wellspring, "find_regular", lambda n: fake_member if n == "Doyle" else None)
    monkeypatch.setattr(wellspring, "prompt_context", lambda r, **_: "THE WELLSPRING — SHARED WORLD")
    prompt = get_caller_prompt(caller)
    assert "THE WELLSPRING — SHARED WORLD" in prompt
    assert "Hauling water." in prompt
    assert "Asked for the statement." in prompt

    walk_in = get_caller_prompt({**caller, "name": "Rhonda"})
    assert "SHARED WORLD" not in walk_in
