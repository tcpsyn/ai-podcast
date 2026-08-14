# Crawlable Episode Pages Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn 57 client-rendered `?slug=` episode pages into 57 static, crawlable HTML pages at `/episode/<slug>/` containing the full transcript, so the Big Bend content the show already produces becomes indexable.

**Architecture:** A build-time Python generator reads the Castopod RSS feed and the existing `website/transcripts/*.txt` files, and writes one fully static page per episode into `website/episode/<slug>/index.html`. No runtime feed dependency, no JS required to see content. The Cloudflare worker gains a 301 from the legacy `?slug=` URL, and loses its user-agent gate. `publish_episode.py` calls the generator so new episodes get a page automatically.

**Tech Stack:** Python 3.11 (stdlib `xml.etree`, `html`, `json`, `pathlib`), pytest, Cloudflare Pages + `_worker.js`, wrangler.

---

## Why This Matters (context for the implementer)

Current state, verified 2026-08-14:

- `website/episode.html` is a JS shell. It has **zero** server-rendered episode content — it fetches `/feed`, finds the matching `<item>` client-side, then fetches `/transcripts/<slug>.txt`.
- URLs are query params: `/episode.html?slug=episode-58-...`. 54 of the 64 sitemap entries are this shape.
- `website/transcripts/` holds **58 `.txt` files, 3.0 MB total**, none of which appear in `sitemap.xml` and which are linked only from JavaScript.
- `_worker.js:107-165` injects real `<title>`/OG tags — but **only when the User-Agent matches `facebookexternalhit|twitterbot|linkedinbot|slackbot|discordbot|telegrambot|whatsapp|pinterest|redditbot`**. Googlebot, Bingbot, GPTBot, ClaudeBot and PerplexityBot are all absent, so search and answer engines get the generic shell.

That UA gate is also a **cloaking risk**: serving different HTML to crawlers than to users is against Google's guidelines. Google deprecated "dynamic rendering" as a workaround. Task 6 removes the gate rather than extending it — once pages are static, nothing needs UA sniffing.

### Data facts confirmed before writing this plan

- RSS feed: `https://podcast.macneilmediagroup.com/@LukeAtTheRoost/feed.xml`, HTTP 200, 124 KB, **57 `<item>` elements**.
- Each item has: `<title>`, `<link>`, `<pubDate>`, `<guid>`, `<description>` (CDATA-wrapped HTML), `<enclosure url=...>`, `<itunes:duration>` (seconds, e.g. `4770`), `<itunes:episode>` (e.g. `58`).
- Slug is derived from `<link>`: everything after `/episodes/`, trailing slash stripped.
- **All 57 feed episodes have a matching transcript file.** Zero missing.
- One orphan transcript, `episode-32-tacos-taxes-and-tall-tales.txt`, has no feed item — ep 32 never finished publishing (`data/publish_state.json` shows a castopod step and nothing else). **The generator must skip orphans, not crash on them.**
- Transcript format is plain text, blank-line separated, `SPEAKER: text` per paragraph — e.g. `LUKE: Alright, welcome back...` / `SLIM: Hey Luke, yeah thanks...`.

---

## Task 1: Transcript parser

**Files:**
- Create: `website_gen/transcript.py`
- Test: `tests/test_transcript_parser.py`

**Step 1: Write the failing test**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from website_gen.transcript import parse_transcript


def test_splits_speaker_turns():
    raw = "LUKE: Welcome back.\n\nSLIM: Hey Luke, thanks for taking my call."
    turns = parse_transcript(raw)
    assert turns == [("LUKE", "Welcome back."),
                     ("SLIM", "Hey Luke, thanks for taking my call.")]


def test_unlabeled_paragraph_carries_previous_speaker():
    raw = "LUKE: First thing.\n\nStill Luke talking."
    assert parse_transcript(raw) == [("LUKE", "First thing."),
                                     ("LUKE", "Still Luke talking.")]


def test_ignores_blank_and_whitespace_paragraphs():
    raw = "LUKE: One.\n\n   \n\nSLIM: Two."
    assert len(parse_transcript(raw)) == 2


def test_speaker_name_with_spaces_is_not_treated_as_label():
    """A colon mid-sentence must not be mistaken for a speaker label."""
    raw = "LUKE: Here's the thing: it was the alternator."
    turns = parse_transcript(raw)
    assert len(turns) == 1
    assert turns[0][0] == "LUKE"
    assert "the thing: it was" in turns[0][1]


def test_empty_input_returns_empty_list():
    assert parse_transcript("") == []
    assert parse_transcript("   \n\n  ") == []
```

**Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_transcript_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'website_gen'`

**Step 3: Write minimal implementation**

```python
import re

SPEAKER_RE = re.compile(r"^([A-Z][A-Z0-9 .'-]{0,30}):\s*(.*)$", re.DOTALL)


def parse_transcript(raw: str) -> list[tuple[str, str]]:
    """Split a transcript into (speaker, text) turns.

    Paragraphs are blank-line separated. A paragraph that does not open with a
    SPEAKER: label is attributed to whoever spoke last, which is how the
    transcriber emits long turns that wrap.
    """
    turns: list[tuple[str, str]] = []
    current = None
    for para in re.split(r"\n\s*\n", raw or ""):
        para = para.strip()
        if not para:
            continue
        m = SPEAKER_RE.match(para)
        if m:
            current = m.group(1).strip()
            text = m.group(2).strip()
        else:
            text = para
        if current is None:
            current = "LUKE"
        if text:
            turns.append((current, text))
    return turns
```

**Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/test_transcript_parser.py -v`
Expected: PASS, 5 tests

**Step 5: Commit**

```bash
git add website_gen/transcript.py tests/test_transcript_parser.py
git commit -m "Add transcript parser for episode page generation"
```

---

## Task 2: RSS feed loader

**Files:**
- Create: `website_gen/feed.py`
- Test: `tests/test_feed_loader.py`
- Fixture: `tests/fixtures/feed_sample.xml` (hand-trim two `<item>` blocks out of the live feed)

**Step 1: Write the failing test**

```python
from website_gen.feed import parse_feed, Episode


def test_parses_core_fields(feed_xml):
    eps = parse_feed(feed_xml)
    ep = next(e for e in eps if e.number == 58)
    assert ep.slug == "episode-58-rayfield-s-nephew-the-marfa-lights-and-why-nobody-believes-concho"
    assert ep.title.startswith("Episode 58: Rayfield's Nephew")
    assert ep.duration_seconds == 4770
    assert ep.audio_url.startswith("https://")


def test_description_is_stripped_of_cdata_and_html(feed_xml):
    ep = parse_feed(feed_xml)[0]
    assert "<![CDATA[" not in ep.description
    assert "<p>" not in ep.description


def test_slug_comes_from_link_and_drops_trailing_slash(feed_xml):
    for ep in parse_feed(feed_xml):
        assert not ep.slug.endswith("/")
        assert "/" not in ep.slug


def test_pubdate_parses_to_iso_date(feed_xml):
    ep = next(e for e in parse_feed(feed_xml) if e.number == 58)
    assert ep.published_iso.startswith("2026-08-04")
```

**Step 2: Run to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_feed_loader.py -v`
Expected: FAIL — module missing

**Step 3: Implement**

Use `xml.etree.ElementTree` with the itunes namespace `http://www.itunes.com/dtds/podcast-1.0.dtd`. Dataclass:

```python
@dataclass
class Episode:
    number: int | None
    slug: str
    title: str
    description: str
    published_iso: str
    duration_seconds: int | None
    audio_url: str
```

Parse `pubDate` with `email.utils.parsedate_to_datetime`. Strip CDATA and tags from `<description>` with a regex, then `html.unescape`.

**Step 4: Verify passes.** **Step 5: Commit.**

```bash
git commit -m "Add RSS feed loader for episode page generation"
```

---

## Task 3: Page renderer

**Files:**
- Create: `website_gen/render.py`, `website_gen/templates/episode.html`
- Test: `tests/test_episode_render.py`

The template is a full standalone page matching the existing site chrome (copy the `<nav>`, footer markup and `css/style.css?v=6` link from `website/how-it-works.html` so it looks native).

**Step 1: Write the failing test**

```python
import json, re
from website_gen.render import render_episode_page


def test_title_and_canonical_are_episode_specific(sample_episode):
    html = render_episode_page(sample_episode, turns=[("LUKE", "Hello.")])
    assert "<title>Episode 58: Rayfield" in html
    assert '<link rel="canonical" href="https://lukeattheroost.com/episode/episode-58-' in html


def test_transcript_is_in_the_html_not_fetched_by_js(sample_episode):
    html = render_episode_page(sample_episode, turns=[("LUKE", "The Marfa Lights are real.")])
    assert "The Marfa Lights are real." in html
    assert "fetch(" not in html


def test_emits_valid_podcastepisode_schema(sample_episode):
    html = render_episode_page(sample_episode, turns=[("LUKE", "Hi.")])
    block = re.search(r'<script type="application/ld\+json">(.*?)</script>', html, re.S).group(1)
    data = json.loads(block)
    types = {o["@type"] for o in (data if isinstance(data, list) else [data])}
    assert "PodcastEpisode" in types


def test_escapes_html_in_transcript_text(sample_episode):
    html = render_episode_page(sample_episode, turns=[("LUKE", "5 < 6 & <script>alert(1)</script>")])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_escapes_quotes_in_title_meta(sample_episode):
    sample_episode.title = 'Episode 1: The "Best" Show'
    html = render_episode_page(sample_episode, turns=[])
    assert 'content="Episode 1: The "Best"' not in html  # must be escaped


def test_speaker_labels_get_semantic_markup(sample_episode):
    html = render_episode_page(sample_episode, turns=[("LUKE", "Hi."), ("SLIM", "Hey.")])
    assert html.count('class="transcript-turn"') == 2
    assert "LUKE" in html and "SLIM" in html
```

**Step 3: Implementation notes**

- Escape every interpolated value with `html.escape(value, quote=True)`. The transcript is user-facing text from Whisper; treat it as untrusted.
- Schema block is a JSON array containing `PodcastEpisode` and `BreadcrumbList`:

```python
{
  "@context": "https://schema.org",
  "@type": "PodcastEpisode",
  "url": f"https://lukeattheroost.com/episode/{ep.slug}/",
  "name": ep.title,
  "description": ep.description,
  "datePublished": ep.published_iso,
  "timeRequired": f"PT{ep.duration_seconds}S",
  "associatedMedia": {"@type": "MediaObject", "contentUrl": ep.audio_url},
  "partOfSeries": {
      "@type": "PodcastSeries",
      "name": "Luke at the Roost",
      "url": "https://lukeattheroost.com",
  },
  "contentLocation": {
      "@type": "Place",
      "name": "Big Bend, West Texas",
      "address": {"@type": "PostalAddress", "addressLocality": "Alpine",
                  "addressRegion": "TX", "addressCountry": "US"},
  },
}
```

The `contentLocation` on every episode is the point of the whole exercise — it is what ties 57 pages of West Texas conversation to the region geographically.

- Include a native `<audio controls preload="none" src="{audio_url}">` so the page is useful without JS.
- Add prev/next episode links. Internal linking is what gets 57 pages crawled instead of 3.

**Step 5: Commit**

```bash
git commit -m "Add episode page renderer with PodcastEpisode schema"
```

---

## Task 4: Generator CLI

**Files:**
- Create: `generate_episode_pages.py` (repo root, matching the existing script convention)
- Test: `tests/test_episode_generator.py`

**Behaviour:**

```
python generate_episode_pages.py              # fetch live feed, write all pages
python generate_episode_pages.py --feed FILE  # use a local feed (tests, offline)
python generate_episode_pages.py --dry-run    # report what would be written
```

**Step 1: Failing tests**

```python
def test_writes_one_index_html_per_feed_episode(tmp_path, feed_file, transcripts_dir):
    n = generate(feed_file, transcripts_dir, tmp_path)
    assert n == 2
    assert (tmp_path / "episode" / "episode-58-rayfield-s-nephew-the-marfa-lights-and-why-nobody-believes-concho" / "index.html").exists()


def test_orphan_transcript_without_feed_item_is_skipped(tmp_path, feed_file, transcripts_dir):
    """episode-32 has a transcript but was never published to the feed."""
    generate(feed_file, transcripts_dir, tmp_path)
    assert not (tmp_path / "episode" / "episode-32-tacos-taxes-and-tall-tales").exists()


def test_missing_transcript_still_produces_a_page(tmp_path, feed_file, transcripts_dir):
    """An episode published before its transcript lands must not break the build."""
    (transcripts_dir / "episode-58-....txt").unlink()
    generate(feed_file, transcripts_dir, tmp_path)
    html = (tmp_path / "episode" / "episode-58-..." / "index.html").read_text()
    assert "Transcript not yet available" in html


def test_dry_run_writes_nothing(tmp_path, feed_file, transcripts_dir):
    generate(feed_file, transcripts_dir, tmp_path, dry_run=True)
    assert not (tmp_path / "episode").exists()
```

**Step 5: Commit**

```bash
git commit -m "Add episode page generator CLI"
```

---

## Task 5: Sitemap regeneration

**Files:**
- Modify: `generate_episode_pages.py` (add `--sitemap`)
- Modify: `website/sitemap.xml` (regenerated output)
- Test: `tests/test_sitemap.py`

The current sitemap has 64 `<url>` entries, 54 of them `?slug=` URLs. Those must be **replaced**, not supplemented — leaving both forms invites duplicate-content dilution even with a canonical.

**Tests:**

```python
def test_sitemap_contains_no_query_param_urls(generated_sitemap):
    assert "episode.html?slug=" not in generated_sitemap


def test_sitemap_has_one_entry_per_episode(generated_sitemap):
    assert generated_sitemap.count("<loc>https://lukeattheroost.com/episode/") == 57


def test_static_pages_survive_regeneration(generated_sitemap):
    for path in ["", "/how-it-works", "/clips", "/stats", "/privacy", "/terms", "/llms.txt"]:
        assert f"<loc>https://lukeattheroost.com{path}</loc>" in generated_sitemap


def test_lastmod_matches_episode_publish_date(generated_sitemap):
    assert "<lastmod>2026-08-04</lastmod>" in generated_sitemap
```

**Commit:** `git commit -m "Generate sitemap from feed with clean episode URLs"`

---

## Task 6: Worker — 301 the old URL, remove the UA gate

**Files:**
- Modify: `website/_worker.js:107-165`

Replace the entire social-crawler injection block with a redirect. Once pages are static, the injection is dead code and the UA gate is a liability.

```javascript
    // Legacy query-param episode URLs -> clean paths.
    // Published social posts and YouTube descriptions still point at the old
    // form, so this 301 has to stay indefinitely.
    if (url.pathname === "/episode.html" && url.searchParams.get("slug")) {
      const slug = url.searchParams.get("slug").replace(/[^a-z0-9-]/gi, "");
      if (slug) {
        return Response.redirect(`https://lukeattheroost.com/episode/${slug}/`, 301);
      }
    }
```

**Why the slug is sanitised:** it lands in a `Location:` header. Stripping to `[a-z0-9-]` prevents CRLF injection and open-redirect via a crafted `?slug=`.

**Verify manually after deploy:**

```bash
curl -sI "https://lukeattheroost.com/episode.html?slug=episode-58-rayfield-s-nephew-the-marfa-lights-and-why-nobody-believes-concho" | head -3
# Expect: HTTP/2 301  +  location: https://lukeattheroost.com/episode/episode-58-.../
```

**Commit:** `git commit -m "Redirect legacy episode URLs and drop crawler UA gate"`

---

## Task 7: Retire the client-rendered page

**Files:**
- Delete: `website/episode.html`, `website/js/episode.js`
- Modify: `website/js/app.js` — episode links must point at `/episode/<slug>/`
- Modify: `website/_redirects` — keep `/episodes.html /episode 302`

Check every internal link first:

```bash
grep -rn "episode.html?slug=" website/ --include=*.html --include=*.js
```

All must become `/episode/<slug>/`. Also fix `publish_episode.py:1168`, which builds the `episode_url` used in social posts and YouTube descriptions:

```python
episode_url = f"https://lukeattheroost.com/episode/{episode_slug}/"
```

**Commit:** `git commit -m "Point internal links at clean episode URLs"`

---

## Task 8: Wire into the publish pipeline

**Files:**
- Modify: `publish_episode.py` — after the transcript is written to `website/transcripts/`

```python
subprocess.run([sys.executable, "generate_episode_pages.py", "--sitemap"], check=False)
```

`check=False` on purpose: a generator failure must not abort a publish that has already pushed audio to Castopod. Log loudly instead.

**Test:** `tests/test_publish_generates_page.py` — assert the generator is invoked with `--sitemap` after a successful publish (monkeypatch `subprocess.run`).

**Commit:** `git commit -m "Regenerate episode pages during publish"`

---

## Task 9: Deploy and verify

```bash
./venv/bin/python -m pytest tests/ -q          # full suite green
python generate_episode_pages.py --sitemap     # 57 pages + sitemap
npx wrangler pages deploy website/ --project-name=lukeattheroost --branch=main
```

**Post-deploy checks — all must pass:**

```bash
# 1. Page is static: transcript present with JS disabled
curl -s https://lukeattheroost.com/episode/episode-58-.../ | grep -c "Mitchell Flat"     # >= 1

# 2. Real title in raw HTML, no UA spoofing
curl -s https://lukeattheroost.com/episode/episode-58-.../ | grep -o "<title>.*</title>"

# 3. Same bytes for Googlebot as for a browser (no cloaking)
diff <(curl -s -A "Mozilla/5.0" URL) <(curl -s -A "Googlebot/2.1" URL) && echo "no cloaking"

# 4. Legacy URL 301s
curl -sI "https://lukeattheroost.com/episode.html?slug=episode-58-..." | grep -i "^location"

# 5. Schema validates
# paste page source into https://validator.schema.org/
```

Then: submit the new `sitemap.xml` in Google Search Console and request indexing on three episode pages as a sample.

---

## Risks

| Risk | Mitigation |
|---|---|
| URL change loses existing ranking | 301 (permanent) from Task 6 passes equity; the old form stays supported forever |
| Published YouTube/social links break | Same 301. Ep 58's YouTube description contains a `?slug=` link — verify it after deploy |
| Transcripts contain explicit content | Show is already rated explicit; consider `<meta name="rating" content="adult">` on episode pages |
| Whisper errors become permanent indexed text | Transcripts are already public at `/transcripts/*.txt`; this changes exposure, not accuracy |
| 3 MB of new HTML on every deploy | Static assets, Cloudflare-cached; wrangler uploads only changed files |
| Feed unreachable at build time | `--feed` flag lets the build use a cached copy; generator exits non-zero rather than writing empty pages |

## Out of scope

Tier 3 — the "Big Bend episodes" topic hub, and off-site work (Alpine Avalanche, Marfa/Big Bend groups, Sul Ross). Worth doing after these pages exist to link to.
