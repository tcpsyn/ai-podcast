"""Returning caller persistence service"""

import json
import time
import uuid
from pathlib import Path
from typing import Optional

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "regulars.json"
MAX_REGULARS = 12


class RegularCallerService:
    """Manages persistent 'regular' callers who return across sessions"""

    def __init__(self):
        self._regulars: list[dict] = []
        self._load()

    def _load(self):
        if DATA_FILE.exists():
            try:
                with open(DATA_FILE) as f:
                    data = json.load(f)
                self._regulars = data.get("regulars", [])
                print(f"[Regulars] Loaded {len(self._regulars)} regular callers")
            except Exception as e:
                print(f"[Regulars] Failed to load: {e}")
                self._regulars = []

    def _save(self):
        try:
            DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(DATA_FILE, "w") as f:
                json.dump({"regulars": self._regulars}, f, indent=2)
        except Exception as e:
            print(f"[Regulars] Failed to save: {e}")

    def get_regulars(self) -> list[dict]:
        return list(self._regulars)

    def get_returning_callers(self, count: int = 2) -> list[dict]:
        """Get up to `count` regulars for returning caller slots"""
        import random
        if not self._regulars:
            return []
        available = [r for r in self._regulars if len(r.get("call_history", [])) > 0]
        if not available:
            return []
        return random.sample(available, min(count, len(available)))

    def add_regular(self, name: str, gender: str, age: int, job: str,
                    location: str, personality_traits: list[str],
                    first_call_summary: str, voice: str = None,
                    stable_seeds: dict = None) -> dict:
        """Promote a first-time caller to regular"""
        # Retire oldest if at cap
        if len(self._regulars) >= MAX_REGULARS:
            self._regulars.sort(key=lambda r: r.get("last_call", 0))
            retired = self._regulars.pop(0)
            print(f"[Regulars] Retired {retired['name']} to make room")

        regular = {
            "id": str(uuid.uuid4())[:8],
            "name": name,
            "gender": gender,
            "age": age,
            "job": job,
            "location": location,
            "personality_traits": personality_traits,
            "voice": voice,
            "stable_seeds": stable_seeds or {},
            "call_history": [
                {"summary": first_call_summary, "timestamp": time.time()}
            ],
            "last_call": time.time(),
            "created_at": time.time(),
        }
        self._regulars.append(regular)
        self._save()
        print(f"[Regulars] Promoted {name} to regular (total: {len(self._regulars)})")
        return regular

    def update_after_call(self, regular_id: str, call_summary: str):
        """Update a regular's history after a returning call"""
        for regular in self._regulars:
            if regular["id"] == regular_id:
                regular.setdefault("call_history", []).append(
                    {"summary": call_summary, "timestamp": time.time()}
                )
                regular["last_call"] = time.time()
                self._save()
                print(f"[Regulars] Updated {regular['name']} call history ({len(regular['call_history'])} calls)")
                return
        print(f"[Regulars] Regular {regular_id} not found for update")


regular_caller_service = RegularCallerService()
