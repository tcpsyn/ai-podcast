import json
from pathlib import Path

src = Path("data/regulars.json")
data = json.loads(src.read_text())

Path("data/regulars.archived.json").write_text(json.dumps(data, indent=2))

regulars = data.get("regulars", [])
silas_only = [r for r in regulars if r.get("name", "").lower() == "silas"]
data["regulars"] = silas_only
src.write_text(json.dumps(data, indent=2))
print(f"Archived {len(regulars)} regulars. Kept {len(silas_only)} (Silas).")
