#!/usr/bin/env python3
"""Generate a static, crawlable page for every episode in the podcast feed."""

import argparse
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from xml.sax.saxutils import escape

from website_gen.feed import parse_feed
from website_gen.render import FEED_URL, render_episode_page
from website_gen.transcript import parse_transcript

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = REPO_ROOT / "website"
DEFAULT_TRANSCRIPTS = DEFAULT_OUTPUT / "transcripts"
USER_AGENT = "lukeattheroost-site-generator/1.0"

SITE_URL = "https://lukeattheroost.com"

# (path, lastmod, changefreq, priority) — /404 is deliberately excluded.
STATIC_PAGES = (
    ("", "2026-03-15", "weekly", "1.0"),
    ("/llms.txt", "2026-03-15", "weekly", "0.5"),
    ("/how-it-works", "2026-03-10", "monthly", "0.8"),
    ("/clips", "2026-03-10", "weekly", "0.8"),
    ("/terms", "2026-02-25", "yearly", "0.3"),
    ("/stats", "2026-02-15", "daily", "0.6"),
    ("/privacy", "2026-02-15", "yearly", "0.3"),
)


def build_sitemap(episodes, static_pages=None) -> str:
    """Render the full sitemap: static pages first, then episodes newest first."""
    if static_pages is None:
        static_pages = STATIC_PAGES

    entries = [(f"{SITE_URL}{path}", lastmod, freq, pri) for path, lastmod, freq, pri in static_pages]

    ordered = sorted(
        episodes,
        key=lambda ep: (ep.published_iso, ep.number if ep.number is not None else -1),
        reverse=True,
    )
    for episode in ordered:
        entries.append(
            (
                f"{SITE_URL}/episode/{episode.slug}/",
                episode.published_iso[:10],
                "monthly",
                "0.7",
            )
        )

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for loc, lastmod, freq, pri in entries:
        lines.append("  <url>")
        lines.append(f"    <loc>{escape(loc)}</loc>")
        if lastmod:
            lines.append(f"    <lastmod>{lastmod}</lastmod>")
        lines.append(f"    <changefreq>{freq}</changefreq>")
        lines.append(f"    <priority>{pri}</priority>")
        lines.append("  </url>")
    lines.append("</urlset>")
    return "\n".join(lines) + "\n"


def generate(feed_path, transcripts_dir, output_root, dry_run=False, sitemap=False) -> int:
    """Write <output_root>/episode/<slug>/index.html for each feed episode.

    Driven entirely by the feed: a transcript with no feed item is ignored, and a
    feed item with no transcript still gets a page. Never touches the network.
    """
    feed_path = Path(feed_path)
    transcripts_dir = Path(transcripts_dir)
    output_root = Path(output_root)

    episodes = parse_feed(feed_path.read_text(encoding="utf-8"))
    episodes.sort(key=lambda ep: ep.number if ep.number is not None else -1, reverse=True)

    written = 0
    for i, episode in enumerate(episodes):
        transcript_file = transcripts_dir / f"{episode.slug}.txt"
        turns = (
            parse_transcript(transcript_file.read_text(encoding="utf-8"))
            if transcript_file.is_file()
            else []
        )
        html = render_episode_page(
            episode,
            turns,
            prev_ep=episodes[i + 1] if i + 1 < len(episodes) else None,
            next_ep=episodes[i - 1] if i > 0 else None,
        )
        page = output_root / "episode" / episode.slug / "index.html"
        if not dry_run:
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_text(html, encoding="utf-8")
        written += 1
        print(f"{'would write' if dry_run else 'wrote'} {page} ({len(turns)} turns)")

    if sitemap:
        sitemap_file = output_root / "sitemap.xml"
        if not dry_run:
            sitemap_file.parent.mkdir(parents=True, exist_ok=True)
            sitemap_file.write_text(build_sitemap(episodes), encoding="utf-8")
        print(f"{'would write' if dry_run else 'wrote'} {sitemap_file}")

    return written


def _fetch_feed(url: str) -> str:
    # Cloudflare 403s the default Python-urllib user agent.
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feed", help="local feed XML file (default: fetch the live feed)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="output directory")
    parser.add_argument("--transcripts", default=DEFAULT_TRANSCRIPTS, help="transcript directory")
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    parser.add_argument("--sitemap", action="store_true", help="also write <output>/sitemap.xml")
    args = parser.parse_args()

    feed_path = args.feed
    tmp_feed = None
    if not feed_path:
        try:
            xml_text = _fetch_feed(FEED_URL)
        except Exception as exc:
            print(f"Failed to fetch {FEED_URL}: {exc}", file=sys.stderr)
            sys.exit(1)
        tmp_feed = Path(tempfile.mkdtemp(prefix="episode-feed-")) / "feed.xml"
        tmp_feed.write_text(xml_text, encoding="utf-8")
        feed_path = tmp_feed

    try:
        count = generate(
            feed_path,
            args.transcripts,
            args.output,
            dry_run=args.dry_run,
            sitemap=args.sitemap,
        )
    finally:
        if tmp_feed is not None:
            shutil.rmtree(tmp_feed.parent, ignore_errors=True)

    print(f"{count} episode pages {'planned' if args.dry_run else 'written'}")


if __name__ == "__main__":
    main()
