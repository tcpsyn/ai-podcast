"""Guards against the two model-config failures that have bitten us:
a routed model with no pricing entry (silently costs $0.00 on the dashboard),
and a stale model id that 404s at the OpenRouter API."""

from backend.config import settings
from backend.services.cost_tracker import OPENROUTER_PRICING
from backend.services.llm import OPENROUTER_MODELS, LLMService

_CALLER_DIALOG_MODEL_PARAMS = LLMService._CALLER_DIALOG_MODEL_PARAMS

# Retired (or expiring) on OpenRouter — kept here so a reintroduction fails loudly.
RETIRED_MODELS = {
    "anthropic/claude-3-haiku",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-sonnet-4-5",   # note the dash; the live id uses a dot
    "deepseek/deepseek-r1-distill-llama-70b",
    "google/gemini-2.5-flash",       # expires 2026-12-01
    "google/gemini-2.5-pro",         # expires 2026-12-01
    "google/gemini-flash-1.5",
    "minimax/minimax-m2-her",
    "mistralai/mistral-medium-3",
    "mistralai/mistral-small-creative",
    "moonshotai/kimi-k2",
    "qwen/qwen3-235b-a22b",
    "x-ai/grok-4",
    "x-ai/grok-4-fast",
    "x-ai/grok-4.1-fast",
}


def test_every_routed_model_has_pricing():
    """A routed model missing from OPENROUTER_PRICING records cost as $0.00."""
    unpriced = {
        cat: model
        for cat, model in settings.category_models.items()
        if model not in OPENROUTER_PRICING
    }
    assert not unpriced, f"routed models with no pricing entry: {unpriced}"


def test_no_retired_models_are_routed():
    routed = {
        cat: model
        for cat, model in settings.category_models.items()
        if model in RETIRED_MODELS
    }
    assert not routed, f"category routed to a retired model: {routed}"


def test_no_retired_models_in_the_pool():
    stale = sorted(set(OPENROUTER_MODELS) & RETIRED_MODELS)
    assert not stale, f"retired models still listed in OPENROUTER_MODELS: {stale}"


def test_no_retired_models_in_caller_dialog_params():
    stale = sorted(set(_CALLER_DIALOG_MODEL_PARAMS) & RETIRED_MODELS)
    assert not stale, f"per-model tuning keyed to retired models: {stale}"


def test_pool_models_are_priced():
    """Anything selectable should be costable."""
    unpriced = [m for m in OPENROUTER_MODELS if m not in OPENROUTER_PRICING]
    assert not unpriced, f"pool models with no pricing entry: {unpriced}"


def test_no_retired_model_ids_anywhere_in_the_codebase():
    """publish_episode.py, make_clips.py and relabel_transcripts.py each shipped
    a retired model id and only failed when someone ran them. Scan every source
    file so the next one fails here instead.

    cost_tracker.py is exempt: it intentionally keeps retired ids as pricing
    keys so historical cost records stay costable.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    exempt = {root / "backend" / "services" / "cost_tracker.py",
              pathlib.Path(__file__).resolve()}

    offenders = {}
    for path in root.rglob("*.py"):
        if path in exempt:
            continue
        # Skip venvs, vendored models, and git worktrees (separate checkouts
        # on other branches — not this tree's code).
        if any(part in {"venv", "mlx_models", ".git", ".claude", ".worktrees",
                        "node_modules", "remotion-demo"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hits = sorted(m for m in RETIRED_MODELS if f'"{m}"' in text or f"'{m}'" in text)
        if hits:
            offenders[str(path.relative_to(root))] = hits

    assert not offenders, f"retired model ids still referenced: {offenders}"


def test_mandatory_reasoning_models_pin_effort():
    """Sonnet 5.5 / Grok 4.7 can't disable reasoning, and reasoning tokens come
    out of max_tokens — unpinned, short caller budgets return empty text."""
    from backend.services.caller_gen import BATCH_MODEL
    from backend.services.llm import REASONING_EFFORT
    for model in (settings.category_models["caller_dialog"],
                  settings.category_models["devon_ask"], BATCH_MODEL):
        assert model in REASONING_EFFORT, f"{model} routed without a reasoning effort pin"


def test_no_retired_models_in_fallbacks():
    from backend.services.llm import FALLBACK_MODELS
    stale = sorted(set(FALLBACK_MODELS) & RETIRED_MODELS)
    assert not stale, f"retired models in FALLBACK_MODELS: {stale}"


def test_fallback_models_are_priced():
    from backend.services.llm import FALLBACK_MODELS
    unpriced = [m for m in FALLBACK_MODELS if m not in OPENROUTER_PRICING]
    assert not unpriced, f"fallback models with no pricing entry: {unpriced}"


def test_reasoning_headroom_skipped_when_reasoning_off():
    """effort "none" disables reasoning, so the extra headroom would just let
    cheap-category responses run long."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from backend.services.llm import REASONING_HEADROOM

    async def run(model):
        svc = LLMService()
        resp = MagicMock()
        resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        client = MagicMock(is_closed=False)
        client.post = AsyncMock(return_value=resp)
        svc._client = client
        await svc._call_openrouter_once([{"role": "user", "content": "hi"}], model, max_tokens=200)
        return client.post.call_args.kwargs["json"]

    off = asyncio.run(run("anthropic/claude-haiku-5.5"))
    assert off["reasoning"] == {"effort": "none"}
    assert off["max_tokens"] == 200

    on = asyncio.run(run("anthropic/claude-sonnet-5.5"))
    assert on["reasoning"] == {"effort": "low"}
    assert on["max_tokens"] == 200 + REASONING_HEADROOM


def test_show_facts_routes_to_cheap_model_with_reasoning_off():
    from backend.services.llm import REASONING_EFFORT
    model = settings.category_models.get("show_facts")
    assert model == "anthropic/claude-haiku-5.5"
    assert REASONING_EFFORT.get(model) == "none"
