import asyncio
from unittest.mock import AsyncMock, patch

from backend.services.regulars_v2 import Regular, load_regular, REGULARS_DIR, SILAS_DIR


def test_load_regular_parses_frontmatter_and_body(tmp_path):
    lore_file = tmp_path / "silas.md"
    lore_file.write_text("""---
name: Silas
voice: Dennis
age: 54
arc_state: Cult is splintering after the eclipse failure
---

# Silas

Silas runs a small desert cult outside Truth or Consequences...

## Arc Log

- 2026-03-01: First call, introduced the cult
- 2026-03-20: Prophesied the eclipse
""")
    reg = load_regular(lore_file)
    assert reg.name == "Silas"
    assert reg.voice == "Dennis"
    assert reg.age == 54
    assert "splintering" in reg.arc_state
    assert "Silas runs a small desert cult" in reg.lore_body


def test_evaluate_promotion_returns_arc_plan_when_worthy():
    from backend.services.regulars_v2 import evaluate_promotion
    fake_response = {
        "promote": True,
        "arc_plan": "3 episodes. He'll start distant, then reveal he's actually the one who damaged the car, then resolve with an apology.",
        "reason": "Has clear internal conflict with room to grow",
    }
    with patch("backend.services.regulars_v2._call_sonnet", new=AsyncMock(return_value=fake_response)):
        result = asyncio.run(evaluate_promotion(caller_name="Bobby", call_transcript="..."))
    assert result["promote"] is True
    assert "3 episodes" in result["arc_plan"]


def test_evaluate_promotion_rejects_when_no_arc():
    from backend.services.regulars_v2 import evaluate_promotion
    fake_response = {"promote": False, "arc_plan": None, "reason": "One-note complaint, no growth"}
    with patch("backend.services.regulars_v2._call_sonnet", new=AsyncMock(return_value=fake_response)):
        result = asyncio.run(evaluate_promotion(caller_name="Carl", call_transcript="..."))
    assert result["promote"] is False
