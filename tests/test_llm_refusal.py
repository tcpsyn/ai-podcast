"""Refusal detection.

A content refusal arrives as an ordinary 200, not an exception. Undetected it
either reaches air as broken-character text or reads as dead silence, and there
is no signal telling the host which knob caused it.
"""
from backend.services.llm import _looks_like_refusal


def test_detects_meta_refusals():
    for text in [
        "I can't help with that request.",
        "As an AI, I don't feel comfortable writing this.",
        "I can't generate that kind of content.",
        "I won't write explicit sexual content.",
        "I'm not able to generate that content.",
        "I can't create that content.",
        "I'm not comfortable writing this material.",
        "That would go against my guidelines.",
    ]:
        assert _looks_like_refusal(text), f"missed refusal: {text!r}"


def test_in_character_dialog_is_not_a_refusal():
    """Detection is tuned for precision. Silencing a working caller is worse than
    missing a refusal, and callers on this show talk like this constantly."""
    for text in [
        "Luke, I can't help with that — she's my sister, what am I supposed to do?",
        "I can't assist him anymore, that's the whole problem.",
        "He said he can't write the check until Friday.",
        "I cannot believe he said that to me on the air.",
        "I can't continue pretending everything out here is fine, Luke.",
        "I won't write her another letter, Luke. I've written six.",
        "I can't help with the kettles anymore, my back's gone.",
    ]:
        assert not _looks_like_refusal(text), f"false positive: {text!r}"


def test_only_the_opening_is_inspected():
    """A refusal marker buried deep in real dialog must not trip detection."""
    dialog = (
        "So I get out to the property and the gate's chained, which it never is, "
        "and Cyrus is standing there like he's been waiting on me all morning. "
        "And I'm thinking, I can't assist with this anymore, I really can't."
    )
    assert not _looks_like_refusal(dialog)


def test_case_and_whitespace_insensitive():
    assert _looks_like_refusal("   I CAN'T HELP WITH THAT REQUEST.  ")


def test_bare_refusal_verb_without_meta_object_is_allowed_through():
    """Deliberate: 'I can't help with that' is ambiguous with dialog, so it stays
    on the air. Missing a refusal costs one odd line; a false positive kills a call."""
    assert not _looks_like_refusal("I can't help with that.")


def test_empty_text_is_not_a_refusal():
    assert not _looks_like_refusal("")
    assert not _looks_like_refusal("   ")
