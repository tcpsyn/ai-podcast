"""Publishing must rebuild the static episode pages and the sitemap.

Before static pages existed, publish_episode.py appended one entry to
sitemap.xml itself. That appender is gone — generate_episode_pages.py owns the
sitemap now — so the publish has to invoke it, or a newly published episode
would have no page and never reach the sitemap.

A generator failure must never abort a publish: by the time this runs the audio
is already live on Castopod and the RSS feed has been rebuilt.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import publish_episode


class _Result:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def test_invokes_the_generator_with_sitemap(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd) or _Result())

    assert publish_episode.regenerate_website_pages() is True

    assert len(calls) == 1
    cmd = calls[0]
    assert "generate_episode_pages.py" in " ".join(cmd)
    assert "--sitemap" in cmd


def test_uses_the_running_interpreter(monkeypatch):
    """Must not shell out to a bare 'python' that may not have the venv."""
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd) or _Result())

    publish_episode.regenerate_website_pages()

    assert calls[0][0] == sys.executable


def test_generator_failure_is_not_fatal(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: _Result(returncode=1, stderr="boom"))

    assert publish_episode.regenerate_website_pages() is False


def test_generator_timeout_is_not_fatal(monkeypatch):
    def _boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 300)

    monkeypatch.setattr(subprocess, "run", _boom)

    assert publish_episode.regenerate_website_pages() is False


def test_missing_generator_is_not_fatal(monkeypatch):
    def _boom(cmd, **kw):
        raise OSError("no such file")

    monkeypatch.setattr(subprocess, "run", _boom)

    assert publish_episode.regenerate_website_pages() is False


def test_publish_flow_calls_it_after_copying_the_transcript():
    """Guards the wiring, not just the helper."""
    source = Path(publish_episode.__file__).read_text()
    copy_at = source.index("Transcript copied to website/transcripts/")
    call_at = source.index("regenerate_website_pages()", copy_at)
    assert call_at > copy_at


def test_generator_runs_after_the_publish_call():
    """Ordering regression guard.

    generate_episode_pages.py is driven entirely by the RSS feed, so it only
    sees an episode once Castopod has published it. For 58 episodes the
    regenerate call sat at step 3.7 — before the step-4 publish — so the
    episode being published was silently absent from its own run and its page
    only appeared on the NEXT publish. Social posts link to /episode/<slug>/,
    so every launch-day link 404'd.

    Asserted on source order because the surrounding publish flow does network
    and filesystem work that isn't practical to drive end to end here.
    """
    src = Path(publish_episode.__file__).read_text()
    body = src[src.index("def main("):]

    publish_call = body.index("published = publish_episode(")
    regen_call = body.index("regenerate_website_pages()")
    deploy_call = body.index("wrangler")

    assert publish_call < regen_call, (
        "regenerate_website_pages() must run AFTER publish_episode(), or the new "
        "episode is missing from the feed the generator reads"
    )
    assert regen_call < deploy_call, (
        "regenerate_website_pages() must run BEFORE the wrangler deploy, or the "
        "freshly built page never ships"
    )


def test_regenerate_is_called_exactly_once_in_main():
    src = Path(publish_episode.__file__).read_text()
    body = src[src.index("def main("):]
    assert body.count("regenerate_website_pages()") == 1
