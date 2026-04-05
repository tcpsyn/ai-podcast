from dataclasses import dataclass
from pathlib import Path
import re

HOME = Path.home()
VAULT = HOME / "code" / "dotfiles"
SILAS_DIR = VAULT / "silas"
REGULARS_DIR = VAULT / "regulars"
ARCHIVED_DIR = REGULARS_DIR / "archived"


@dataclass
class Regular:
    name: str
    voice: str
    age: int
    arc_state: str
    lore_body: str
    file_path: Path


def load_regular(path: Path) -> Regular:
    text = path.read_text()
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not m:
        raise ValueError(f"No frontmatter in {path}")
    fm_raw, body = m.group(1), m.group(2).strip()
    fm = {}
    for line in fm_raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    return Regular(
        name=fm["name"],
        voice=fm["voice"],
        age=int(fm["age"]),
        arc_state=fm.get("arc_state", ""),
        lore_body=body,
        file_path=path,
    )


def load_all_active_regulars() -> list[Regular]:
    out = []
    if SILAS_DIR.exists():
        for f in SILAS_DIR.glob("*.md"):
            out.append(load_regular(f))
    if REGULARS_DIR.exists():
        for f in REGULARS_DIR.glob("*.md"):
            out.append(load_regular(f))
    return out
