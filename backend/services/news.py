"""News service using local SearXNG for current events awareness in AI callers"""

import asyncio
import time
import re
from dataclasses import dataclass

import httpx

SEARXNG_URL = "http://localhost:8888"


@dataclass
class NewsItem:
    title: str
    source: str
    published: str
    content: str = ""


class NewsService:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._headlines_cache: list[NewsItem] = []
        self._headlines_ts: float = 0
        self._search_cache: dict[str, tuple[float, list[NewsItem]]] = {}

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=5.0)
        return self._client

    async def get_headlines(self) -> list[NewsItem]:
        # Cache for 30min
        if self._headlines_cache and time.time() - self._headlines_ts < 1800:
            return self._headlines_cache

        try:
            resp = await self.client.get(
                f"{SEARXNG_URL}/search",
                params={"q": "news", "format": "json", "categories": "news"},
            )
            resp.raise_for_status()
            items = self._parse_searxng(resp.json(), max_items=10)
            self._headlines_cache = items
            self._headlines_ts = time.time()
            return items
        except Exception as e:
            print(f"[News] Headlines fetch failed: {e}")
            self._headlines_ts = time.time()
            return self._headlines_cache

    async def search_topic(self, query: str) -> list[NewsItem]:
        cache_key = query.lower()

        if cache_key in self._search_cache:
            ts, items = self._search_cache[cache_key]
            if time.time() - ts < 600:
                return items

        # Evict oldest when cache too large
        if len(self._search_cache) > 50:
            oldest_key = min(self._search_cache, key=lambda k: self._search_cache[k][0])
            del self._search_cache[oldest_key]

        try:
            resp = await self.client.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json", "categories": "news"},
            )
            resp.raise_for_status()
            items = self._parse_searxng(resp.json(), max_items=5)
            self._search_cache[cache_key] = (time.time(), items)
            return items
        except Exception as e:
            print(f"[News] Search failed for '{query}': {e}")
            if cache_key in self._search_cache:
                return self._search_cache[cache_key][1]
            return []

    def _parse_searxng(self, data: dict, max_items: int = 10) -> list[NewsItem]:
        items = []
        for result in data.get("results", [])[:max_items]:
            title = result.get("title", "").strip()
            if not title:
                continue
            # Extract source from engines list or metadata
            engines = result.get("engines", [])
            source = engines[0] if engines else ""
            published = result.get("publishedDate", "")
            content = result.get("content", "").strip()
            items.append(NewsItem(title=title, source=source, published=published, content=content))
        return items

    def format_headlines_for_prompt(self, items: list[NewsItem]) -> str:
        lines = []
        for item in items:
            if item.source:
                lines.append(f"- {item.title} ({item.source})")
            else:
                lines.append(f"- {item.title}")
        return "\n".join(lines)

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "but", "and", "or", "if", "while", "because", "until", "about",
    "that", "this", "these", "those", "what", "which", "who", "whom",
    "it", "its", "he", "him", "his", "she", "her", "they", "them",
    "their", "we", "us", "our", "you", "your", "me", "my", "i",
    # Casual speech fillers
    "yeah", "well", "like", "man", "dude", "okay", "right", "know",
    "think", "mean", "really", "actually", "honestly", "basically",
    "literally", "stuff", "thing", "things", "something", "anything",
    "nothing", "everything", "someone", "anyone", "everyone", "nobody",
    "gonna", "wanna", "gotta", "kinda", "sorta", "dunno",
    "look", "see", "say", "said", "tell", "told", "talk", "talking",
    "feel", "felt", "guess", "sure", "maybe", "probably", "never",
    "always", "still", "even", "much", "many", "also", "got", "get",
    "getting", "going", "come", "came", "make", "made", "take", "took",
    "give", "gave", "want", "keep", "kept", "let", "put", "went",
    "been", "being", "doing", "having", "call", "called", "calling",
    "tonight", "today", "night", "time", "long", "good", "bad",
    "first", "last", "back", "down", "ever", "away", "cant", "dont",
    "didnt", "doesnt", "isnt", "wasnt", "wont", "wouldnt", "couldnt",
    "shouldnt", "aint", "stop", "start", "started", "help",
    # Radio show filler
    "welcome", "thanks", "thank", "show", "roost", "luke", "whats",
    "youre", "thats", "heres", "theyre", "ive", "youve", "weve",
    "sounds", "listen", "hear", "heard", "happen", "happened",
    "happening", "absolutely", "definitely", "exactly", "totally",
    "pretty", "little", "whole", "every", "point", "sense", "real",
    "great", "cool", "awesome", "amazing", "crazy", "weird", "funny",
    "tough", "hard", "wrong", "true", "trying", "tried", "works",
    "working", "anymore", "already", "enough", "though", "whatever",
    "theres", "making", "saying", "keeping", "possible", "instead",
    "front", "behind", "course", "talks", "happens", "watch",
    "everybodys", "pants", "husband", "client",
}


def extract_keywords(text: str, max_keywords: int = 3) -> list[str]:
    words = text.split()
    if len(words) < 8:
        return []  # Too short to extract meaningful topics

    keywords = []

    # Only look for proper nouns that are likely real topics (not caller names)
    proper_nouns = []
    for i, word in enumerate(words):
        clean = re.sub(r'[^\w]', '', word)
        if not clean or len(clean) < 3:
            continue
        is_sentence_start = i == 0 or (i > 0 and words[i - 1].rstrip()[-1:] in '.!?')
        if clean[0].isupper() and not is_sentence_start and clean.lower() not in STOP_WORDS:
            proper_nouns.append(clean)

    # Only use proper nouns if we found 2+ (single one is probably a name)
    if len(proper_nouns) >= 2:
        for noun in proper_nouns[:max_keywords]:
            if noun not in keywords:
                keywords.append(noun)
        if len(keywords) >= max_keywords:
            return keywords

    # Pass 2: uncommon words (>5 chars, not in stop words)
    for word in words:
        clean = re.sub(r'[^\w]', '', word).lower()
        if len(clean) > 5 and clean not in STOP_WORDS:
            if clean not in [k.lower() for k in keywords]:
                keywords.append(clean)
            if len(keywords) >= max_keywords:
                return keywords

    return keywords


news_service = NewsService()
