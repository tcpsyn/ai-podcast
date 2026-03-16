"""Avatar service — fetches deterministic face photos from randomuser.me"""

import asyncio
from pathlib import Path

import httpx

AVATAR_DIR = Path(__file__).parent.parent.parent / "data" / "avatars"


class AvatarService:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        AVATAR_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    def get_path(self, name: str) -> Path | None:
        path = AVATAR_DIR / f"{name}.jpg"
        return path if path.exists() else None

    async def get_or_fetch(self, name: str, gender: str = "male") -> Path:
        """Get cached avatar or fetch from randomuser.me. Returns file path."""
        g = "female" if gender.lower().startswith("f") else "male"
        path = AVATAR_DIR / f"{name}.jpg"
        # Check for gender mismatch marker — re-fetch if gender changed
        marker = AVATAR_DIR / f"{name}.gender"
        if path.exists():
            cached_gender = marker.read_text().strip() if marker.exists() else None
            if cached_gender == g:
                return path
            # Gender mismatch or no marker — re-fetch
            path.unlink(missing_ok=True)

        try:
            seed = f"{name.lower().replace(' ', '_')}_{g}"
            resp = await self.client.get(
                "https://randomuser.me/api/",
                params={"gender": g, "seed": seed},
                timeout=8.0,
            )
            resp.raise_for_status()
            data = resp.json()
            photo_url = data["results"][0]["picture"]["large"]

            photo_resp = await self.client.get(photo_url, timeout=8.0)
            photo_resp.raise_for_status()

            path.write_bytes(photo_resp.content)
            marker.write_text(g)
            print(f"[Avatar] Fetched avatar for {name} ({g})")
            return path
        except Exception as e:
            print(f"[Avatar] Failed to fetch for {name}: {e}")
            raise

    async def prefetch_batch(self, callers: list[dict]):
        """Fetch avatars for multiple callers in parallel.
        Each dict should have 'name' and 'gender' keys."""
        tasks = []
        for caller in callers:
            name = caller.get("name", "")
            gender = caller.get("gender", "male")
            if name and not (AVATAR_DIR / f"{name}.jpg").exists():
                tasks.append(self.get_or_fetch(name, gender))

        if not tasks:
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)
        fetched = sum(1 for r in results if not isinstance(r, Exception))
        failed = sum(1 for r in results if isinstance(r, Exception))
        if fetched:
            print(f"[Avatar] Pre-fetched {fetched} avatars{f', {failed} failed' if failed else ''}")

    async def ensure_devon(self):
        """Pre-fetch Devon's avatar on startup."""
        try:
            await self.get_or_fetch("Devon", "male")
        except Exception:
            pass


avatar_service = AvatarService()
