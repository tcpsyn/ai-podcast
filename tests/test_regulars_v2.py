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
