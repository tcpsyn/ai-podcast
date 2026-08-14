import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEGACY = "episode.html?slug="


def _files():
    for pattern in ("website/*.html", "website/js/*.js", "website/llms.txt", "*.py"):
        yield from ROOT.glob(pattern)


def test_no_source_file_builds_a_legacy_episode_url():
    offenders = []
    for f in _files():
        if f.name == "_worker.js":
            continue  # the worker's reference is the redirect itself
        if LEGACY in f.read_text(errors="replace"):
            offenders.append(str(f.relative_to(ROOT)))
    assert not offenders, f"legacy episode URLs still present: {offenders}"


def test_client_rendered_episode_page_is_gone():
    assert not (ROOT / "website" / "episode.html").exists()
    assert not (ROOT / "website" / "js" / "episode.js").exists()


def test_worker_still_redirects_legacy_urls():
    worker = (ROOT / "website" / "_worker.js").read_text()
    assert "/episode.html" in worker and "301" in worker


def test_redirects_file_has_no_dead_episode_target():
    redirects = (ROOT / "website" / "_redirects").read_text()
    assert "/episode 302" not in redirects
