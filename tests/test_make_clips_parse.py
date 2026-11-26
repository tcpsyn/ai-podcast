import json

import pytest

from make_clips import _parse_json_array


def test_takes_corrected_array_after_draft():
    content = (
        '[{"title":"x","start_time":3.0,"end_time":4.0,"caption_text":"placeholder"}]\n\n'
        "Correction: the response above is malformed.\n\n"
        '[{"title":"a","start_time":184.0,"end_time":231.0,"caption_text":"q"}]'
    )
    assert _parse_json_array(content)[0]["start_time"] == 184.0


def test_strips_code_fences():
    assert _parse_json_array('```json\n[{"a": 1}]\n```') == [{"a": 1}]


def test_ignores_brackets_inside_strings():
    content = '[{"caption_text": "he said [laughs] fine"}]'
    assert _parse_json_array(content)[0]["caption_text"] == "he said [laughs] fine"


def test_raises_when_no_array():
    with pytest.raises(json.JSONDecodeError):
        _parse_json_array("sorry, I can't do that")
