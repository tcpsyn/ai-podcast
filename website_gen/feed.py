import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

ITUNES_NS = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}

TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class Episode:
    number: int | None
    slug: str
    title: str
    description: str
    published_iso: str
    duration_seconds: int | None
    audio_url: str


def parse_feed(xml_text: str) -> list[Episode]:
    """Turn RSS feed XML into Episode records, skipping items with no usable link."""
    channel = ET.fromstring(xml_text).find("channel")
    if channel is None:
        return []

    episodes: list[Episode] = []
    for item in channel.findall("item"):
        slug = _slug_from_link(item.findtext("link"))
        if not slug:
            continue
        enclosure = item.find("enclosure")
        episodes.append(
            Episode(
                number=_as_int(item.findtext("itunes:episode", namespaces=ITUNES_NS)),
                slug=slug,
                title=(item.findtext("title") or "").strip(),
                description=_plain_text(item.findtext("description")),
                published_iso=_iso_date(item.findtext("pubDate")),
                duration_seconds=_as_int(item.findtext("itunes:duration", namespaces=ITUNES_NS)),
                audio_url=(enclosure.get("url") or "") if enclosure is not None else "",
            )
        )
    return episodes


def _slug_from_link(link: str | None) -> str:
    slug = (link or "").strip().rstrip("/")
    if "/episodes/" not in slug:
        return ""
    return slug.rsplit("/episodes/", 1)[-1]


def _plain_text(raw: str | None) -> str:
    text = TAG_RE.sub(" ", raw or "")
    return WHITESPACE_RE.sub(" ", html.unescape(text)).strip()


def _iso_date(raw: str | None) -> str:
    if not raw:
        return ""
    try:
        return parsedate_to_datetime(raw).isoformat()
    except (TypeError, ValueError):
        return ""


def _as_int(raw: str | None) -> int | None:
    try:
        return int((raw or "").strip())
    except ValueError:
        return None
