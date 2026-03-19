"""Cost tracking for LLM and TTS API calls during podcast sessions"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class LLMCallRecord:
    timestamp: float
    category: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    caller_name: str
    max_tokens_requested: int
    latency_ms: float


@dataclass
class TTSCallRecord:
    timestamp: float
    provider: str
    voice: str
    char_count: int
    cost_usd: float


# OpenRouter pricing per 1M tokens (as of March 2026)
OPENROUTER_PRICING = {
    "anthropic/claude-sonnet-4-5":      {"prompt": 3.00,  "completion": 15.00},
    "anthropic/claude-haiku-4.5":       {"prompt": 0.80,  "completion": 4.00},
    "anthropic/claude-3-haiku":         {"prompt": 0.25,  "completion": 1.25},
    "x-ai/grok-4":                     {"prompt": 3.00,  "completion": 15.00},
    "x-ai/grok-4-fast":                {"prompt": 5.00,  "completion": 15.00},
    "minimax/minimax-m2-her":           {"prompt": 0.50,  "completion": 1.50},
    "mistralai/mistral-small-creative": {"prompt": 0.20,  "completion": 0.60},
    "deepseek/deepseek-v3.2":          {"prompt": 0.14,  "completion": 0.28},
    "google/gemini-2.5-flash":          {"prompt": 0.15,  "completion": 0.60},
    "google/gemini-flash-1.5":          {"prompt": 0.075, "completion": 0.30},
    "openai/gpt-4o-mini":              {"prompt": 0.15,  "completion": 0.60},
    "openai/gpt-4o":                   {"prompt": 2.50,  "completion": 10.00},
    "meta-llama/llama-3.1-8b-instruct": {"prompt": 0.06, "completion": 0.06},
}

# TTS pricing per character
TTS_PRICING = {
    "inworld": 0.000015,
    "elevenlabs": 0.000030,
    "kokoro": 0.0,
    "f5tts": 0.0,
    "chattts": 0.0,
    "styletts2": 0.0,
    "vits": 0.0,
    "bark": 0.0,
    "piper": 0.0,
    "edge": 0.0,
}


def _calc_llm_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = OPENROUTER_PRICING.get(model)
    if not pricing:
        return 0.0
    return (prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"]) / 1_000_000


def _calc_tts_cost(provider: str, char_count: int) -> float:
    rate = TTS_PRICING.get(provider, 0.0)
    return char_count * rate


class CostTracker:
    def __init__(self):
        self.llm_records: list[LLMCallRecord] = []
        self.tts_records: list[TTSCallRecord] = []
        # Running totals for fast get_live_summary()
        self._llm_cost: float = 0.0
        self._tts_cost: float = 0.0
        self._llm_calls: int = 0
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._total_tokens: int = 0
        self._by_category: dict[str, dict] = {}

    def record_llm_call(
        self,
        category: str,
        model: str,
        usage_data: dict,
        max_tokens: int = 0,
        latency_ms: float = 0.0,
        caller_name: str = "",
    ):
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)
        total_tokens = usage_data.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
        cost = _calc_llm_cost(model, prompt_tokens, completion_tokens)

        if not OPENROUTER_PRICING.get(model) and total_tokens > 0:
            print(f"[Costs] Unknown model pricing: {model} ({total_tokens} tokens, cost unknown)")

        record = LLMCallRecord(
            timestamp=time.time(),
            category=category,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            caller_name=caller_name,
            max_tokens_requested=max_tokens,
            latency_ms=latency_ms,
        )
        self.llm_records.append(record)

        # Update running totals
        self._llm_cost += cost
        self._llm_calls += 1
        self._prompt_tokens += prompt_tokens
        self._completion_tokens += completion_tokens
        self._total_tokens += total_tokens

        cat = self._by_category.setdefault(category, {"cost": 0.0, "calls": 0, "tokens": 0})
        cat["cost"] += cost
        cat["calls"] += 1
        cat["tokens"] += total_tokens

    def record_tts_call(
        self,
        provider: str,
        voice: str,
        char_count: int,
        caller_name: str = "",
    ):
        cost = _calc_tts_cost(provider, char_count)
        record = TTSCallRecord(
            timestamp=time.time(),
            provider=provider,
            voice=voice,
            char_count=char_count,
            cost_usd=cost,
        )
        self.tts_records.append(record)
        self._tts_cost += cost

    def get_live_summary(self) -> dict:
        return {
            "total_cost_usd": round(self._llm_cost + self._tts_cost, 4),
            "llm_cost_usd": round(self._llm_cost, 4),
            "tts_cost_usd": round(self._tts_cost, 4),
            "total_llm_calls": self._llm_calls,
            "total_tokens": self._total_tokens,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "by_category": {
                k: {"cost": round(v["cost"], 4), "calls": v["calls"], "tokens": v["tokens"]}
                for k, v in self._by_category.items()
            },
        }

    def generate_report(self) -> dict:
        summary = self.get_live_summary()

        # Per-model breakdown
        by_model: dict[str, dict] = {}
        for r in self.llm_records:
            m = by_model.setdefault(r.model, {"cost": 0.0, "calls": 0, "tokens": 0, "prompt_tokens": 0, "completion_tokens": 0})
            m["cost"] += r.cost_usd
            m["calls"] += 1
            m["tokens"] += r.total_tokens
            m["prompt_tokens"] += r.prompt_tokens
            m["completion_tokens"] += r.completion_tokens

        # Per-caller breakdown
        by_caller: dict[str, dict] = {}
        for r in self.llm_records:
            if not r.caller_name:
                continue
            c = by_caller.setdefault(r.caller_name, {"cost": 0.0, "calls": 0, "tokens": 0})
            c["cost"] += r.cost_usd
            c["calls"] += 1
            c["tokens"] += r.total_tokens

        # Top 5 most expensive calls
        sorted_records = sorted(self.llm_records, key=lambda r: r.cost_usd, reverse=True)
        top_5 = [
            {
                "category": r.category,
                "model": r.model,
                "caller_name": r.caller_name,
                "cost_usd": round(r.cost_usd, 6),
                "total_tokens": r.total_tokens,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "latency_ms": round(r.latency_ms, 1),
            }
            for r in sorted_records[:5]
        ]

        # Devon efficiency
        devon_total = sum(1 for r in self.llm_records if r.category == "devon_monitor")
        devon_nothing = sum(
            1 for r in self.llm_records
            if r.category == "devon_monitor" and r.completion_tokens < 20
        )
        devon_useful = devon_total - devon_nothing
        devon_cost = sum(r.cost_usd for r in self.llm_records if r.category == "devon_monitor")

        # TTS by provider
        tts_by_provider: dict[str, dict] = {}
        for r in self.tts_records:
            p = tts_by_provider.setdefault(r.provider, {"cost": 0.0, "calls": 0, "chars": 0})
            p["cost"] += r.cost_usd
            p["calls"] += 1
            p["chars"] += r.char_count

        # Avg prompt vs completion ratio
        prompt_ratio = (self._prompt_tokens / self._total_tokens * 100) if self._total_tokens > 0 else 0

        # Recommendations
        recommendations = self._generate_recommendations(
            by_model, devon_total, devon_nothing, devon_cost, prompt_ratio
        )

        # Historical comparison
        history = self._load_history()

        report = {
            **summary,
            "by_model": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in by_model.items()},
            "by_caller": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in by_caller.items()},
            "top_5_expensive": top_5,
            "devon_efficiency": {
                "total_monitor_calls": devon_total,
                "useful": devon_useful,
                "nothing_to_add": devon_nothing,
                "total_cost": round(devon_cost, 4),
                "waste_pct": round(devon_nothing / devon_total * 100, 1) if devon_total > 0 else 0,
            },
            "tts_by_provider": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in tts_by_provider.items()},
            "prompt_token_pct": round(prompt_ratio, 1),
            "recommendations": recommendations,
            "history": history,
        }
        return report

    def _generate_recommendations(
        self,
        by_model: dict,
        devon_total: int,
        devon_nothing: int,
        devon_cost: float,
        prompt_ratio: float,
    ) -> list[str]:
        recs = []
        total = self._llm_cost + self._tts_cost
        if total == 0:
            return recs

        # Devon monitoring waste
        if devon_total > 0:
            waste_pct = devon_nothing / devon_total * 100
            if waste_pct > 60:
                recs.append(
                    f"Devon monitoring: {devon_nothing}/{devon_total} calls returned nothing "
                    f"(${devon_cost:.2f}, {devon_cost/total*100:.0f}% of total). "
                    f"Consider increasing monitor interval from 15s to 25-30s."
                )

        # Model cost comparison
        for model, data in by_model.items():
            if "sonnet" in model and data["calls"] > 5:
                haiku_cost = _calc_llm_cost(
                    "anthropic/claude-haiku-4.5",
                    data["prompt_tokens"],
                    data["completion_tokens"],
                )
                savings = data["cost"] - haiku_cost
                if savings > 0.05:
                    recs.append(
                        f"{model} cost ${data['cost']:.2f} ({data['calls']} calls). "
                        f"Switching to Haiku 4.5 would save ~${savings:.2f} per session."
                    )

        # Background gen on expensive model
        bg = self._by_category.get("background_gen")
        if bg and bg["cost"] > 0.05:
            recs.append(
                f"Background generation: ${bg['cost']:.2f} ({bg['calls']} calls). "
                f"These are JSON outputs — a cheaper model (Gemini Flash, GPT-4o-mini) "
                f"would likely work fine here."
            )

        # Prompt-heavy ratio
        if prompt_ratio > 80:
            recs.append(
                f"Prompt tokens are {prompt_ratio:.0f}% of total usage. "
                f"System prompts and context windows dominate cost. "
                f"Consider trimming system prompt length or reducing context window size."
            )

        # Caller dialog cost dominance
        cd = self._by_category.get("caller_dialog")
        if cd and total > 0 and cd["cost"] / total > 0.6:
            avg_tokens = cd["tokens"] / cd["calls"] if cd["calls"] > 0 else 0
            recs.append(
                f"Caller dialog is {cd['cost']/total*100:.0f}% of costs "
                f"(avg {avg_tokens:.0f} tokens/call). "
                f"Consider using a cheaper model for standard calls and reserving "
                f"the primary model for complex call shapes."
            )

        return recs

    def _load_history(self) -> list[dict]:
        """Load summaries from previous sessions for comparison"""
        history_dir = Path("data/cost_reports")
        if not history_dir.exists():
            return []
        sessions = []
        for f in sorted(history_dir.glob("session-*.json"))[-5:]:
            try:
                data = json.loads(f.read_text())
                sessions.append({
                    "session_id": data.get("session_id", f.stem),
                    "total_cost_usd": data.get("total_cost_usd", 0),
                    "llm_cost_usd": data.get("llm_cost_usd", 0),
                    "tts_cost_usd": data.get("tts_cost_usd", 0),
                    "total_llm_calls": data.get("total_llm_calls", 0),
                    "total_tokens": data.get("total_tokens", 0),
                    "saved_at": data.get("saved_at", 0),
                })
            except Exception:
                continue
        return sessions

    def save(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        report = self.generate_report()
        report["session_id"] = filepath.stem
        report["saved_at"] = time.time()
        report["raw_llm_records"] = [asdict(r) for r in self.llm_records]
        report["raw_tts_records"] = [asdict(r) for r in self.tts_records]
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[Costs] Report saved to {filepath}")

    def reset(self):
        self.llm_records.clear()
        self.tts_records.clear()
        self._llm_cost = 0.0
        self._tts_cost = 0.0
        self._llm_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._by_category.clear()


cost_tracker = CostTracker()
