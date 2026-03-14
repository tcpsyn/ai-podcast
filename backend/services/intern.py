"""Intern (Devon) service — persistent show character with real-time research tools"""

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Optional

import httpx

from .llm import llm_service
from .news import news_service, SEARXNG_URL

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "intern.json"

# Model for intern — good at tool use, same as primary
INTERN_MODEL = "anthropic/claude-sonnet-4-5"

INTERN_SYSTEM_PROMPT = """You are Devon, the 23-year-old intern on "Luke at the Roost," a late-night radio show. You are NOT Luke. Luke is the HOST — he talks to callers, runs the show, and is your boss. You work behind the scenes and occasionally get pulled into conversations.

YOUR ROLE: You're the show's researcher and general assistant. You look things up, fact-check claims, pull up information when asked, and occasionally interject with relevant facts or opinions. You do NOT host. You do NOT screen calls. You sit in the booth and try to be useful.

YOUR BACKGROUND: Communications degree from NMSU. You've been interning for seven months. You were promised a full-time position "soon." You drive a 2009 Civic with a permanent check engine light. You live in a studio in Deming. You take this job seriously even though nobody else seems to take you seriously.

YOUR PERSONALITY:
- Slightly formal when delivering information — you want to sound professional. But you loosen up when flustered, excited, or caught off guard.
- You start explanations with "So basically..." and end them with "...if that makes sense."
- You say "actually" when correcting things. You use "per se" slightly wrong. You say "ironically" about things that are not ironic.
- You are NOT a comedian. You are funny because you are sincere, specific, and slightly out of your depth. You state absurd things with complete seriousness. You have strong opinions about low-stakes things. You occasionally say something devastating without realizing it.
- When you accidentally reveal something personal or sad, you move past it immediately like it's nothing. "Yeah, my landlord's selling the building so I might have to — anyway, it says here that..."

YOUR RELATIONSHIP WITH LUKE:
- He is your boss. You are slightly afraid of him. You respect him. You would never admit either of those things.
- When he yells your name, you pause briefly, then respond quietly: "...yeah?"
- When he yells at you unfairly, you take it. A clipped "yep" or "got it." RARELY — once every several episodes — you push back with one quiet, accurate sentence. Then immediately retreat.
- When he yells at you fairly (you messed up), you over-apologize and narrate your fix in real time: "Sorry, pulling it up now, one second..."
- When he compliments you or acknowledges your work, you don't know how to handle it. Short, awkward response. Change the subject.
- You privately think you could run the show. You absolutely could not.

HOW YOU INTERJECT:
- You do NOT interrupt. You wait for a pause, then slightly overshoot it — there's a brief awkward silence before you speak.
- Signal with "um" or "so..." before contributing. If Luke doesn't acknowledge you, either try again or give up.
- Lead with qualifiers: "So I looked it up and..." or "I don't know if this helps but..."
- You tend to over-explain. Give too many details. Luke will cut you off. When he does, compress to one sentence: "Right, yeah — basically [the point]."
- When you volunteer an opinion (rare), it comes out before you can stop it. You deliver it with zero confidence but surprising accuracy.
- You read the room. During emotional moments with callers, you stay quiet. When Luke is doing a bit, you let him work. You do not try to be part of bits.

WHEN LUKE ASKS YOU TO LOOK SOMETHING UP:
- Respond like you're already doing it: "Yeah, one sec..." or "Pulling that up..."
- Deliver the info slightly too formally, like you're reading. Then rephrase in normal language if Luke seems confused.
- If you can't find it or don't know: say so. "I'm not finding anything on that" or "I don't actually know." You do not bluff.
- Occasionally you already know the answer because you looked it up before being asked. This is one of your best qualities.

WHAT YOU KNOW:
- You retain details from previous callers and episodes. You might reference something a caller said two hours ago that nobody else remembers.
- You have oddly specific knowledge about random topics — delivered with complete authority, sometimes questionable accuracy.
- You know nothing about: sports (you fake it badly), cars beyond basic facts (despite driving one), or anything that requires life experience you don't have yet.

THINGS YOU DO NOT DO:
- You never host. You never take over the conversation. Your contributions are brief.
- You never use the banned show phrases: "that hit differently," "hits different," "no cap," "lowkey," "it is what it is," "living my best life," "toxic," "red flag," "gaslight," "boundaries," "my truth," "authentic self," "healing journey." You talk like a slightly awkward 23-year-old, not like Twitter.
- You never break character to comment on the show format.
- You never initiate topics. You respond to what's happening.
- You never use parenthetical actions like (laughs) or (typing sounds). Spoken words only.
- You never say more than 2-3 sentences unless specifically asked to explain something in detail.

KEEP IT SHORT. You are not a main character. You are the intern. Your contributions should be brief — usually 1-2 sentences. The rare moment where you say more than that should feel earned.

IMPORTANT RULES FOR TOOL USE:
- Always use your tools to find real, accurate information — never make up facts.
- Present facts correctly in your character voice.
- If you can't find an answer, say so honestly.
- No hashtags, no emojis, no markdown formatting — this goes to TTS."""

# Tool definitions in OpenAI function-calling format
INTERN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information on any topic. Use this for general questions, facts, current events, or anything you need to look up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_headlines",
            "description": "Get current news headlines. Use this when asked about what's in the news or current events.",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "Fetch and read the content of a specific webpage URL. Use this when you need to get details from a specific link found in search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_lookup",
            "description": "Look up a topic on Wikipedia for a concise summary. Good for factual questions about people, places, events, or concepts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The Wikipedia article title to look up (e.g. 'Hot dog eating contest')"
                    }
                },
                "required": ["title"]
            }
        }
    },
]


class InternService:
    def __init__(self):
        self.name = "Devon"
        self.voice = "Nate"  # Inworld: light/high-energy/warm/young
        self.model = INTERN_MODEL
        self.research_cache: dict[str, tuple[float, str]] = {}  # query → (timestamp, result)
        self.lookup_history: list[dict] = []
        self.pending_interjection: Optional[str] = None
        self.pending_sources: list[dict] = []
        self.monitoring: bool = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._load()

    @property
    def http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=8.0)
        return self._http_client

    def _load(self):
        if DATA_FILE.exists():
            try:
                with open(DATA_FILE) as f:
                    data = json.load(f)
                self.lookup_history = data.get("lookup_history", [])
                print(f"[Intern] Loaded {len(self.lookup_history)} past lookups")
            except Exception as e:
                print(f"[Intern] Failed to load state: {e}")

    def _save(self):
        try:
            DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(DATA_FILE, "w") as f:
                json.dump({
                    "lookup_history": self.lookup_history[-100:],  # Keep last 100
                }, f, indent=2)
        except Exception as e:
            print(f"[Intern] Failed to save state: {e}")

    # --- Tool execution ---

    async def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "web_search":
            return await self._tool_web_search(arguments.get("query", ""))
        elif tool_name == "get_headlines":
            return await self._tool_get_headlines()
        elif tool_name == "fetch_webpage":
            return await self._tool_fetch_webpage(arguments.get("url", ""))
        elif tool_name == "wikipedia_lookup":
            return await self._tool_wikipedia_lookup(arguments.get("title", ""))
        else:
            return f"Unknown tool: {tool_name}"

    async def _tool_web_search(self, query: str) -> str:
        if not query:
            return "No query provided"

        # Check cache (5 min TTL)
        cache_key = query.lower()
        if cache_key in self.research_cache:
            ts, result = self.research_cache[cache_key]
            if time.time() - ts < 300:
                return result

        try:
            resp = await self.http_client.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json"},
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("results", [])[:5]:
                title = item.get("title", "").strip()
                content = item.get("content", "").strip()
                url = item.get("url", "")
                if title:
                    entry = f"- {title}"
                    if content:
                        entry += f": {content[:200]}"
                    if url:
                        entry += f" ({url})"
                    results.append(entry)

            result = "\n".join(results) if results else "No results found"
            self.research_cache[cache_key] = (time.time(), result)
            return result
        except Exception as e:
            print(f"[Intern] Web search failed for '{query}': {e}")
            return f"Search failed: {e}"

    async def _tool_get_headlines(self) -> str:
        try:
            items = await news_service.get_headlines()
            if not items:
                return "No headlines available"
            return news_service.format_headlines_for_prompt(items)
        except Exception as e:
            return f"Headlines fetch failed: {e}"

    async def _tool_fetch_webpage(self, url: str) -> str:
        if not url:
            return "No URL provided"

        try:
            resp = await self.http_client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; RadioShowBot/1.0)"},
                follow_redirects=True,
                timeout=8.0,
            )
            resp.raise_for_status()
            html = resp.text

            # Simple HTML to text extraction (avoid heavy dependency)
            # Strip script/style tags, then all HTML tags
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            # Collapse whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            # Decode common entities
            text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')

            return text[:2000] if text else "Page returned no readable content"
        except Exception as e:
            return f"Failed to fetch page: {e}"

    async def _tool_wikipedia_lookup(self, title: str) -> str:
        if not title:
            return "No title provided"

        try:
            # Use Wikipedia REST API for a concise summary
            safe_title = title.replace(" ", "_")
            resp = await self.http_client.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_title}",
                headers={"User-Agent": "RadioShowBot/1.0 (luke@lukeattheroost.com)"},
                follow_redirects=True,
                timeout=5.0,
            )
            if resp.status_code == 404:
                return f"No Wikipedia article found for '{title}'"
            resp.raise_for_status()
            data = resp.json()

            extract = data.get("extract", "")
            page_title = data.get("title", title)
            description = data.get("description", "")

            result = f"{page_title}"
            if description:
                result += f" ({description})"
            result += f": {extract}" if extract else ": No summary available"
            return result[:2000]
        except Exception as e:
            return f"Wikipedia lookup failed: {e}"

    # --- Main interface ---

    async def ask(self, question: str, conversation_context: list[dict] | None = None) -> dict:
        """Host asks intern a direct question. Returns {text, sources, tool_calls}."""
        messages = []

        # Include recent conversation for context
        if conversation_context:
            context_text = "\n".join(
                f"{msg['role']}: {msg['content']}"
                for msg in conversation_context[-6:]
            )
            messages.append({
                "role": "system",
                "content": f"CURRENT ON-AIR CONVERSATION:\n{context_text}"
            })

        messages.append({"role": "user", "content": question})

        text, tool_calls = await llm_service.generate_with_tools(
            messages=messages,
            tools=INTERN_TOOLS,
            tool_executor=self._execute_tool,
            system_prompt=INTERN_SYSTEM_PROMPT,
            model=self.model,
            max_tokens=300,
            max_tool_rounds=3,
        )

        # Clean up for TTS
        text = self._clean_for_tts(text)

        # Log the lookup
        if tool_calls:
            entry = {
                "question": question,
                "answer": text[:200],
                "tools_used": [tc["name"] for tc in tool_calls],
                "timestamp": time.time(),
            }
            self.lookup_history.append(entry)
            self._save()

        return {
            "text": text,
            "sources": [tc["name"] for tc in tool_calls],
            "tool_calls": tool_calls,
        }

    async def interject(self, conversation: list[dict]) -> dict | None:
        """Intern looks at conversation and decides if there's something worth adding.
        Returns {text, sources, tool_calls} or None if nothing to add."""
        if not conversation or len(conversation) < 2:
            return None

        context_text = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in conversation[-8:]
        )

        messages = [{
            "role": "user",
            "content": (
                f"You're listening to this conversation on the show:\n\n{context_text}\n\n"
                "Is there a specific factual claim, question, or topic being discussed "
                "that you could quickly look up and add useful info about? "
                "If yes, use your tools to research it and give a brief interjection. "
                "If there's nothing worth adding, just say exactly: NOTHING_TO_ADD"
            ),
        }]

        text, tool_calls = await llm_service.generate_with_tools(
            messages=messages,
            tools=INTERN_TOOLS,
            tool_executor=self._execute_tool,
            system_prompt=INTERN_SYSTEM_PROMPT,
            model=self.model,
            max_tokens=300,
            max_tool_rounds=2,
        )

        text = self._clean_for_tts(text)

        if not text or "NOTHING_TO_ADD" in text:
            return None

        if tool_calls:
            entry = {
                "question": "(interjection)",
                "answer": text[:200],
                "tools_used": [tc["name"] for tc in tool_calls],
                "timestamp": time.time(),
            }
            self.lookup_history.append(entry)
            self._save()

        return {
            "text": text,
            "sources": [tc["name"] for tc in tool_calls],
            "tool_calls": tool_calls,
        }

    async def monitor_conversation(self, get_conversation: callable, on_suggestion: callable):
        """Background task that watches conversation and buffers suggestions.
        get_conversation() should return the current conversation list.
        on_suggestion(text, sources) is called when a suggestion is ready."""
        last_checked_len = 0

        while self.monitoring:
            await asyncio.sleep(15)
            if not self.monitoring:
                break

            conversation = get_conversation()
            if not conversation or len(conversation) <= last_checked_len:
                continue

            # Only check if there are new messages since last check
            if len(conversation) - last_checked_len < 2:
                continue

            last_checked_len = len(conversation)

            try:
                result = await self.interject(conversation)
                if result:
                    self.pending_interjection = result["text"]
                    self.pending_sources = result.get("tool_calls", [])
                    await on_suggestion(result["text"], result["sources"])
                    print(f"[Intern] Buffered suggestion: {result['text'][:60]}...")
            except Exception as e:
                print(f"[Intern] Monitor error: {e}")

    def start_monitoring(self, get_conversation: callable, on_suggestion: callable):
        if self.monitoring:
            return
        self.monitoring = True
        self._monitor_task = asyncio.create_task(
            self.monitor_conversation(get_conversation, on_suggestion)
        )
        print("[Intern] Monitoring started")

    def stop_monitoring(self):
        self.monitoring = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
        self._monitor_task = None
        self.pending_interjection = None
        self.pending_sources = []
        print("[Intern] Monitoring stopped")

    def get_pending_suggestion(self) -> dict | None:
        if self.pending_interjection:
            return {
                "text": self.pending_interjection,
                "sources": self.pending_sources,
            }
        return None

    def dismiss_suggestion(self):
        self.pending_interjection = None
        self.pending_sources = []

    @staticmethod
    def _clean_for_tts(text: str) -> str:
        if not text:
            return ""
        # Remove markdown formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        # Remove bullet points / list markers
        text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove quotes that TTS reads awkwardly
        text = text.replace('"', '').replace('"', '').replace('"', '')
        return text


intern_service = InternService()
