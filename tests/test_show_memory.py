import asyncio
import json
from datetime import datetime, timedelta, timezone

from backend.services.news import NewsItem
from backend.services.show_memory import ShowMemory, filter_items, window_days


def _mem(*texts, pinned=False):
    m = ShowMemory()
    for t in texts:
        m.add_fact(t, pinned=pinned)
    return m


def test_extractor_correction_replaces_fact():
    m = _mem("Luke is broadcasting from West Virginia tonight")
    fid = m.facts[0]["id"]
    m.apply_ops([{"op": "replace", "id": fid, "text": "Luke is back in Alpine tonight"}])
    assert [f["text"] for f in m.facts] == ["Luke is back in Alpine tonight"]


def test_pinned_fact_survives_extractor_ops():
    m = _mem("Luke's back is out", pinned=True)
    fid = m.facts[0]["id"]
    m.apply_ops([{"op": "remove", "id": fid}, {"op": "replace", "id": fid, "text": "x"}])
    assert m.facts[0]["text"] == "Luke's back is out"


def test_duplicate_add_is_skipped():
    m = _mem("Luke is in West Virginia")
    m.apply_ops([{"op": "add", "text": "  luke is in west virginia. "}])
    assert len(m.facts) == 1


def test_deleted_fact_is_not_re_added_by_extractor():
    m = _mem("Devon owes Luke $20")
    m.delete_fact(m.facts[0]["id"])
    m.apply_ops([{"op": "add", "text": "Devon owes Luke $20"}])
    assert m.facts == []


def test_malformed_ops_change_nothing():
    m = _mem("Luke is in West Virginia")
    before = [dict(f) for f in m.facts]
    m.apply_ops([{"op": "explode"}, {"nope": 1}, "garbage", {"op": "replace", "id": 999, "text": "x"}])
    assert m.facts == before


def test_fact_list_is_capped():
    m = ShowMemory()
    m.apply_ops([{"op": "add", "text": f"fact number {i}"} for i in range(30)])
    assert len(m.facts) == ShowMemory.MAX_FACTS


def test_round_trips_through_dict():
    m = _mem("Luke is in West Virginia", pinned=True)
    m.mark_call_started()
    m2 = ShowMemory()
    m2.load_dict(m.to_dict())
    assert m2.facts == m.facts and m2.calls_started is True


HEADLINES = [
    {"title": "Senate passes budget", "kind": "big", "summary": "Shutdown averted", "angle": "everyone's talking"},
    {"title": "Man marries his truck", "kind": "offbeat", "summary": "Ohio ceremony", "angle": "pure Devon bait"},
]


def test_devon_block_has_facts_and_opening_headlines_before_first_call():
    m = _mem("Luke is in West Virginia tonight")
    m.headlines = HEADLINES
    block = m.devon_block()
    assert "SHOW FACTS" in block and "West Virginia" in block
    assert "OPENING HEADLINES" in block and "Man marries his truck" in block


def test_devon_block_downgrades_headlines_after_first_call():
    m = ShowMemory()
    m.headlines = HEADLINES
    m.mark_call_started()
    block = m.devon_block()
    assert "OPENING HEADLINES" not in block
    assert "STORIES YOU KNOW ABOUT" in block and "Senate passes budget" in block


def test_devon_block_non_opening_uses_background_variant_before_first_call():
    m = ShowMemory()
    m.headlines = HEADLINES
    block = m.devon_block(opening=False)
    assert "OPENING HEADLINES" not in block
    assert "STORIES YOU KNOW ABOUT" in block and "Senate passes budget" in block


def test_devon_block_skips_malformed_headlines():
    m = ShowMemory()
    m.headlines = [{"kind": "big"}, {"title": ""}, {"title": "Man marries his truck"}]
    block = m.devon_block()
    assert "Man marries his truck" in block
    assert block.count("\n- ") == 1


def test_devon_block_with_only_malformed_headlines_is_empty():
    m = ShowMemory()
    m.headlines = [{"kind": "big"}]
    assert m.devon_block() == ""


def test_empty_memory_gives_empty_blocks():
    m = ShowMemory()
    assert m.devon_block() == "" and m.caller_block() == ""


def test_caller_block_lists_facts_only():
    m = _mem("Luke is in West Virginia tonight")
    m.headlines = HEADLINES
    block = m.caller_block()
    assert "West Virginia" in block and "truck" not in block


def test_extract_applies_llm_ops(monkeypatch):
    from backend.services import show_memory as sm
    m = ShowMemory()
    m.add_host_line("We drove out to my mom's place in West Virginia this week", schedule=False)

    async def fake_generate(**kw):
        assert "West Virginia" in kw["messages"][0]["content"]
        return json.dumps({"ops": [{"op": "add", "text": "Luke is broadcasting from his mom's house in West Virginia"}]})

    monkeypatch.setattr(sm.llm_service, "generate", fake_generate)
    asyncio.run(m.extract_facts())
    assert "West Virginia" in m.facts[0]["text"]
    assert m.processed_lines == 1


def test_extract_bad_json_leaves_facts(monkeypatch):
    from backend.services import show_memory as sm
    m = _mem("Luke is in West Virginia")
    m.add_host_line("blah", schedule=False)

    async def fake_generate(**kw):
        return "not json at all"

    monkeypatch.setattr(sm.llm_service, "generate", fake_generate)
    asyncio.run(m.extract_facts())
    assert [f["text"] for f in m.facts] == ["Luke is in West Virginia"]


def test_extract_noop_without_new_lines(monkeypatch):
    from backend.services import show_memory as sm
    called = []

    async def fake_generate(**kw):
        called.append(1)
        return "{}"

    monkeypatch.setattr(sm.llm_service, "generate", fake_generate)
    asyncio.run(ShowMemory().extract_facts())
    assert called == []


def test_window_days_clamped():
    now = datetime(2026, 11, 25, tzinfo=timezone.utc)
    assert window_days(None, now) == 1
    assert window_days(now - timedelta(hours=5), now) == 1
    assert window_days(now - timedelta(days=3, hours=2), now) == 4
    assert window_days(now - timedelta(days=40), now) == 7


def test_filter_drops_wikinews_old_and_duplicates():
    since = datetime(2026, 11, 20, tzinfo=timezone.utc)
    items = [
        NewsItem("Old blockade story", "wikinews", "2026-11-22T00:00:00"),
        NewsItem("Stale story", "brave.news", "2026-05-04T00:00:00"),
        NewsItem("Fresh story", "brave.news", "2026-11-24T10:00:00"),
        NewsItem("Fresh Story!", "bing news", ""),
        NewsItem("Undated story", "bing news", ""),
    ]
    assert [i.title for i in filter_items(items, since)] == ["Fresh story", "Undated story"]


def test_curation_failure_returns_empty(monkeypatch):
    from backend.services import show_memory as sm

    async def fake_recent(queries, time_range):
        return [NewsItem("Fresh story", "brave.news", "")]

    async def fake_generate(**kw):
        raise RuntimeError("openrouter down")

    async def fake_last():
        return None

    monkeypatch.setattr(sm.news_service, "search_recent", fake_recent)
    monkeypatch.setattr(sm.llm_service, "generate", fake_generate)
    monkeypatch.setattr(sm, "last_episode_date", fake_last)
    m = ShowMemory()
    asyncio.run(m.build_headlines())
    assert m.headlines == []


def test_curation_keeps_only_valid_picks(monkeypatch):
    from backend.services import show_memory as sm

    async def fake_recent(queries, time_range):
        return [NewsItem("A", "brave.news", ""), NewsItem("B", "bing news", "")]

    async def fake_generate(**kw):
        return json.dumps({"headlines": [
            {"title": "A", "kind": "big", "summary": "s", "angle": "a"},
            {"title": "", "kind": "offbeat"},
            {"kind": "offbeat", "summary": "no title"},
        ]})

    async def fake_last():
        return None

    monkeypatch.setattr(sm.news_service, "search_recent", fake_recent)
    monkeypatch.setattr(sm.llm_service, "generate", fake_generate)
    monkeypatch.setattr(sm, "last_episode_date", fake_last)
    m = ShowMemory()
    asyncio.run(m.build_headlines())
    assert [h["title"] for h in m.headlines] == ["A"]


def test_trimmed_checkpoint_keeps_unprocessed_lines_pending():
    m = ShowMemory()
    for i in range(100):
        m.add_host_line(f"line {i}", schedule=False)
    m.processed_lines = 90
    m2 = ShowMemory()
    m2.load_dict(m.to_dict())
    assert m2.host_lines[m2.processed_lines:] == [f"line {i}" for i in range(90, 100)]


async def _until(cond, steps=200):
    for _ in range(steps):
        if cond():
            return
        await asyncio.sleep(0)


def test_lines_added_during_extraction_get_picked_up(monkeypatch):
    from backend.services import show_memory as sm
    calls = []

    async def run():
        release = asyncio.Event()

        async def fake_generate(**kw):
            calls.append(kw["messages"][0]["content"])
            if len(calls) == 1:
                await release.wait()
            return json.dumps({"ops": []})

        monkeypatch.setattr(sm.llm_service, "generate", fake_generate)
        m = ShowMemory()
        for i in range(3):
            m.add_host_line(f"line {i}")
        await _until(lambda: calls)
        for i in range(3, 6):
            m.add_host_line(f"line {i}")
        release.set()
        await _until(lambda: m.processed_lines == 6)
        return m

    m = asyncio.run(run())
    assert m.processed_lines == 6
    assert len(calls) == 2 and "line 5" in calls[1]


def test_failed_extraction_retries_on_idle_timer(monkeypatch):
    from backend.services import show_memory as sm
    monkeypatch.setattr(sm, "EXTRACT_IDLE_SECONDS", 0)
    calls = []

    async def run():
        release = asyncio.Event()

        async def fake_generate(**kw):
            calls.append(kw["messages"][0]["content"])
            if len(calls) == 1:
                await release.wait()
                raise RuntimeError("openrouter down")
            return json.dumps({"ops": [{"op": "add", "text": "Luke is in West Virginia"}]})

        monkeypatch.setattr(sm.llm_service, "generate", fake_generate)
        m = ShowMemory()
        for i in range(3):
            m.add_host_line(f"line {i}")
        await _until(lambda: calls)
        m.add_host_line("line 3")
        release.set()
        await _until(lambda: m.processed_lines == 4)
        return m

    m = asyncio.run(run())
    assert m.processed_lines == 4
    assert [f["text"] for f in m.facts] == ["Luke is in West Virginia"]
    assert len(calls) == 2


def test_reset_mid_extraction_drops_stale_result(monkeypatch):
    from backend.services import show_memory as sm

    async def run():
        release, started = asyncio.Event(), asyncio.Event()

        async def fake_generate(**kw):
            started.set()
            await release.wait()
            return json.dumps({"ops": [{"op": "add", "text": "Luke is in West Virginia"}]})

        monkeypatch.setattr(sm.llm_service, "generate", fake_generate)
        m = ShowMemory()
        m.add_host_line("We're in West Virginia", schedule=False)
        task = asyncio.create_task(m.extract_facts())
        await started.wait()
        m.reset()
        release.set()
        await task
        return m

    m = asyncio.run(run())
    assert m.facts == [] and m.processed_lines == 0 and m._extracting is False


def test_reset_cancels_scheduled_extraction(monkeypatch):
    from backend.services import show_memory as sm

    async def run():
        started = asyncio.Event()

        async def fake_generate(**kw):
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(sm.llm_service, "generate", fake_generate)
        m = ShowMemory()
        m.add_host_line("only line")
        idle = m._idle_task
        m.reset()
        await asyncio.sleep(0)

        for i in range(3):
            m.add_host_line(f"line {i}")
        extract = m._extract_task
        await started.wait()
        m.reset()
        await asyncio.sleep(0)
        return idle.done(), extract.cancelled()

    assert asyncio.run(run()) == (True, True)


def test_host_line_during_idle_extraction_does_not_cancel_it(monkeypatch):
    from backend.services import show_memory as sm
    monkeypatch.setattr(sm, "EXTRACT_IDLE_SECONDS", 0)
    calls = []

    async def run():
        release = asyncio.Event()

        async def fake_generate(**kw):
            calls.append(kw["messages"][0]["content"])
            try:
                if len(calls) == 1:
                    await release.wait()
            finally:
                await asyncio.sleep(0)
            if len(calls) == 1:
                return json.dumps({"ops": [{"op": "add", "text": "Luke is in West Virginia"}]})
            return json.dumps({"ops": []})

        monkeypatch.setattr(sm.llm_service, "generate", fake_generate)
        m = ShowMemory()
        m.add_host_line("We're in West Virginia")
        await _until(lambda: calls)
        m.add_host_line("second line")
        release.set()
        await _until(lambda: m.processed_lines == 2)
        return m

    m = asyncio.run(run())
    assert [f["text"] for f in m.facts] == ["Luke is in West Virginia"]
    assert m.processed_lines == 2 and len(calls) == 2 and "second line" in calls[1]


def test_no_duplicate_extract_task_while_extracting(monkeypatch):
    from backend.services import show_memory as sm

    async def run():
        started = asyncio.Event()

        async def fake_generate(**kw):
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(sm.llm_service, "generate", fake_generate)
        m = ShowMemory()
        for i in range(3):
            m.add_host_line(f"line {i}")
        in_flight = m._extract_task
        await started.wait()
        for i in range(3, 6):
            m.add_host_line(f"line {i}")
        same = m._extract_task is in_flight
        m.reset()
        return same

    assert asyncio.run(run()) is True


def test_apply_ops_tolerates_string_ids_and_non_string_text():
    m = _mem("Luke is in West Virginia")
    fid = m.facts[0]["id"]
    m.apply_ops([
        {"op": "replace", "id": str(fid), "text": "Luke is back in Alpine"},
        {"op": "add", "text": 42},
        {"op": "remove", "id": "abc"},
        {"op": "remove", "id": None},
    ])
    assert [f["text"] for f in m.facts] == ["Luke is back in Alpine", "42"]


def test_last_episode_date_normalises_naive_to_utc(monkeypatch):
    from backend.services import show_memory as sm

    class FakeResp:
        text = "<rss><item><pubDate>Mon, 23 Nov 2026 03:00:00 -0000</pubDate></item></rss>"

        def raise_for_status(self):
            pass

    class FakeClient:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            return FakeResp()

    monkeypatch.setattr(sm.httpx, "AsyncClient", FakeClient)
    dt = asyncio.run(sm.last_episode_date())
    assert dt == datetime(2026, 11, 23, 3, tzinfo=timezone.utc)


def test_build_headlines_never_raises_on_bad_last_date(monkeypatch):
    from backend.services import show_memory as sm

    async def fake_last():
        return datetime(2026, 11, 23)

    monkeypatch.setattr(sm, "last_episode_date", fake_last)
    m = ShowMemory()
    asyncio.run(m.build_headlines())
    assert m.headlines == []


def test_reset_during_headline_build_drops_stale_result(monkeypatch):
    from backend.services import show_memory as sm
    m = ShowMemory()

    async def fake_recent(queries, time_range):
        m.reset()
        return [NewsItem("Stale", "brave.news", "")]

    async def fake_generate(**kw):
        return json.dumps({"headlines": [{"title": "Stale", "kind": "big", "summary": "s", "angle": "a"}]})

    async def fake_last():
        return None

    monkeypatch.setattr(sm.news_service, "search_recent", fake_recent)
    monkeypatch.setattr(sm.llm_service, "generate", fake_generate)
    monkeypatch.setattr(sm, "last_episode_date", fake_last)
    asyncio.run(m.build_headlines())
    assert m.headlines == []


def test_headlines_fresh_only_when_recent_and_present():
    m = ShowMemory()
    m.headlines_built_at = 1000.0
    assert not m.headlines_fresh(now=1010.0)
    m.headlines = list(HEADLINES)
    assert m.headlines_fresh(now=1000.0 + 29 * 60)
    assert not m.headlines_fresh(now=1000.0 + 31 * 60)
    m.reset()
    assert m.headlines == [] and not m.headlines_fresh()


def test_headlines_built_at_set_by_build_and_round_trips(monkeypatch):
    from backend.services import show_memory as sm

    async def fake_recent(queries, time_range):
        return [NewsItem("A", "brave.news", "")]

    async def fake_generate(**kw):
        return json.dumps({"headlines": [{"title": "A", "kind": "big", "summary": "s", "angle": "a"}]})

    async def fake_last():
        return None

    monkeypatch.setattr(sm.news_service, "search_recent", fake_recent)
    monkeypatch.setattr(sm.llm_service, "generate", fake_generate)
    monkeypatch.setattr(sm, "last_episode_date", fake_last)
    m = ShowMemory()
    asyncio.run(m.build_headlines())
    assert m.headlines_fresh()
    m2 = ShowMemory()
    m2.load_dict(m.to_dict())
    assert m2.headlines_built_at == m.headlines_built_at and m2.headlines_fresh()


def test_non_list_picks_do_not_raise(monkeypatch):
    from backend.services import show_memory as sm

    async def fake_recent(queries, time_range):
        return [NewsItem("A", "brave.news", "")]

    async def fake_generate(**kw):
        return json.dumps({"headlines": None})

    async def fake_last():
        return None

    monkeypatch.setattr(sm.news_service, "search_recent", fake_recent)
    monkeypatch.setattr(sm.llm_service, "generate", fake_generate)
    monkeypatch.setattr(sm, "last_episode_date", fake_last)
    m = ShowMemory()
    asyncio.run(m.build_headlines())
    assert m.headlines == []
