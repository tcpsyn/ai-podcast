import html
import json
from datetime import datetime

SITE_URL = "https://lukeattheroost.com"
COVER_IMAGE = "https://cdn.lukeattheroost.com/media/podcasts/LukeAtTheRoost/cover_feed.png?v=3"
FEED_URL = "https://podcast.macneilmediagroup.com/@LukeAtTheRoost/feed.xml"

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} — Luke at the Roost</title>
  <meta name="description" content="{meta_description}">
  <meta name="theme-color" content="#1a1209">
  <meta name="rating" content="adult">
  <link rel="canonical" href="{page_url}">

  <meta property="og:site_name" content="Luke at the Roost">
  <meta property="og:title" content="{title}">
  <meta property="og:description" content="{meta_description}">
  <meta property="og:image" content="{cover}">
  <meta property="og:url" content="{page_url}">
  <meta property="og:type" content="article">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="{title}">
  <meta name="twitter:description" content="{meta_description}">
  <meta name="twitter:image" content="{cover}">

  <link rel="icon" href="/favicon.ico" sizes="48x48">
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <link rel="icon" type="image/png" sizes="192x192" href="/favicon-192.png">
  <link rel="icon" type="image/png" sizes="48x48" href="/favicon-48.png">
  <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32.png">
  <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16.png">
  <link rel="apple-touch-icon" href="/apple-touch-icon.png">

  <link rel="alternate" type="application/rss+xml" title="Luke at the Roost RSS Feed" href="{feed}">
  <link rel="stylesheet" href="/css/style.css?v=7">

  <script type="application/ld+json">
{schema}
  </script>
</head>
<body>

  <a href="#main-content" class="skip-link">Skip to content</a>

  <nav class="site-nav">
    <a href="/" class="site-nav-brand">Luke at the Roost</a>
    <div class="site-nav-links">
      <a href="/how-it-works">How It Works</a>
      <a href="/clips">Clips</a>
      <a href="/stats">Stats</a>
    </div>
  </nav>

  <main id="main-content">

  <section class="page-header">
    <h1>{title}</h1>
    <p class="episode-meta">{meta_line}</p>
    <p class="page-subtitle">{description}</p>
  </section>

  <section class="episode-player">
    <audio controls preload="none" src="{audio_url}">
      Your browser does not support audio playback.
      <a href="{audio_url}">Download the episode</a>
    </audio>
  </section>

  <section class="episode-transcript">
    <h2>Transcript</h2>
{transcript}
  </section>

{episode_nav}
  </main>

  <footer class="footer"></footer>
  <script src="/js/footer.js"></script>
</body>
</html>
"""


def render_episode_page(episode, turns, prev_ep=None, next_ep=None) -> str:
    """Render a complete static episode page, transcript and schema included."""
    page_url = _page_url(episode.slug)
    return PAGE.format(
        title=_esc(episode.title),
        meta_description=_esc(_truncate(episode.description, 160)),
        description=_esc(episode.description),
        page_url=_esc(page_url),
        cover=_esc(COVER_IMAGE),
        feed=_esc(FEED_URL),
        audio_url=_esc(episode.audio_url),
        meta_line=_meta_line(episode),
        schema=_schema_block(episode, page_url),
        transcript=_transcript_html(turns),
        episode_nav=_episode_nav(prev_ep, next_ep),
    )


def _page_url(slug: str) -> str:
    return f"{SITE_URL}/episode/{slug}/"


def _esc(value) -> str:
    return html.escape(str(value or ""), quote=True)


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",.;:—- ") + "…"


def _schema_block(episode, page_url: str) -> str:
    ep_schema = {
        "@context": "https://schema.org",
        "@type": "PodcastEpisode",
        "url": page_url,
        "name": episode.title,
        "description": episode.description,
        "associatedMedia": {"@type": "MediaObject", "contentUrl": episode.audio_url},
        "partOfSeries": {
            "@type": "PodcastSeries",
            "name": "Luke at the Roost",
            "url": SITE_URL,
        },
        "contentLocation": {
            "@type": "Place",
            "name": "Big Bend, West Texas",
            "address": {
                "@type": "PostalAddress",
                "addressLocality": "Alpine",
                "addressRegion": "TX",
                "addressCountry": "US",
            },
        },
    }
    if episode.published_iso:
        ep_schema["datePublished"] = episode.published_iso
    if episode.duration_seconds:
        ep_schema["timeRequired"] = f"PT{episode.duration_seconds}S"
    if episode.number is not None:
        ep_schema["episodeNumber"] = episode.number

    breadcrumbs = {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": [
            {"@type": "ListItem", "position": 1, "name": "Home", "item": SITE_URL},
            {"@type": "ListItem", "position": 2, "name": episode.title, "item": page_url},
        ],
    }
    dumped = json.dumps([ep_schema, breadcrumbs], indent=2, ensure_ascii=False)
    return dumped.replace("<", "\\u003c")


def _meta_line(episode) -> str:
    parts = []
    published = _format_date(episode.published_iso)
    if published:
        parts.append(f'<time datetime="{_esc(episode.published_iso)}">{_esc(published)}</time>')
    duration = _format_duration(episode.duration_seconds)
    if duration:
        parts.append(_esc(duration))
    return " · ".join(parts)


def _format_date(published_iso: str) -> str:
    try:
        dt = datetime.fromisoformat(published_iso)
        return f"{dt.strftime('%B')} {dt.day}, {dt.year}"
    except (TypeError, ValueError):
        return ""


def _format_duration(seconds) -> str:
    if not seconds:
        return ""
    hours, minutes = divmod(int(seconds) // 60, 60)
    if hours:
        return f"{hours} hr {minutes} min"
    return f"{minutes} min"


def _transcript_html(turns) -> str:
    if not turns:
        return '    <p class="transcript-empty">Transcript not yet available for this episode.</p>'
    rows = [
        f'    <div class="transcript-turn">'
        f'<span class="transcript-speaker">{_esc(speaker)}</span>'
        f"<p>{_esc(text)}</p></div>"
        for speaker, text in turns
    ]
    return "\n".join(rows)


def _episode_nav(prev_ep, next_ep) -> str:
    links = []
    if prev_ep is not None:
        links.append(
            f'    <a class="episode-nav-prev" rel="prev" href="/episode/{_esc(prev_ep.slug)}/">'
            f"← {_esc(prev_ep.title)}</a>"
        )
    if next_ep is not None:
        links.append(
            f'    <a class="episode-nav-next" rel="next" href="/episode/{_esc(next_ep.slug)}/">'
            f"{_esc(next_ep.title)} →</a>"
        )
    if not links:
        return ""
    return '  <nav class="episode-nav">\n' + "\n".join(links) + "\n  </nav>\n"
