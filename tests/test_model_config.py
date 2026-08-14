"""Guards against the two model-config failures that have bitten us:
a routed model with no pricing entry (silently costs $0.00 on the dashboard),
and a stale model id that 404s at the OpenRouter API."""

from backend.config import settings
from backend.services.cost_tracker import OPENROUTER_PRICING
from backend.services.llm import OPENROUTER_MODELS, LLMService

_CALLER_DIALOG_MODEL_PARAMS = LLMService._CALLER_DIALOG_MODEL_PARAMS

# Retired on OpenRouter — kept here so a reintroduction fails loudly.
RETIRED_MODELS = {
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-sonnet-4-5",   # note the dash; the live id uses a dot
    "google/gemini-flash-1.5",
    "mistralai/mistral-small-creative",
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
