# Cost Dashboard Design

## Overview

A dedicated cost analytics dashboard at `/costs` that visualizes LLM and TTS spending across sessions with time-range filtering, model/category breakdowns, and drill-down into individual sessions and calls.

## Architecture

- **Route:** `/costs` served by FastAPI, standalone page matching the control panel's dark theme
- **Database:** SQLite (`data/costs.db`) for cross-session aggregation
- **Charts:** Chart.js (vanilla JS, no framework)
- **Data migration:** On first run, import existing `data/cost_reports/*.json` into SQLite
- **Dual write:** `cost_tracker.py` continues writing JSON reports (backward compat) and also writes to SQLite going forward

## Database Schema

### `sessions`
| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PK | Session ID |
| started_at | TIMESTAMP | Session start time |
| total_cost | REAL | Total cost USD |
| llm_cost | REAL | LLM cost USD |
| tts_cost | REAL | TTS cost USD |
| total_llm_calls | INTEGER | Number of LLM calls |
| total_tts_calls | INTEGER | Number of TTS calls |
| total_tokens | INTEGER | Total tokens used |
| prompt_tokens | INTEGER | Prompt tokens |
| completion_tokens | INTEGER | Completion tokens |

### `llm_calls`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| session_id | TEXT FK | References sessions.id |
| timestamp | TIMESTAMP | Call time |
| category | TEXT | background_gen, caller_dialog, devon_monitor, etc. |
| model | TEXT | Model identifier |
| prompt_tokens | INTEGER | Prompt tokens |
| completion_tokens | INTEGER | Completion tokens |
| cost | REAL | Cost USD |
| caller_name | TEXT | Caller name (nullable) |
| latency_ms | REAL | Response latency |

### `tts_calls`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| session_id | TEXT FK | References sessions.id |
| timestamp | TIMESTAMP | Call time |
| provider | TEXT | Inworld, ElevenLabs, etc. |
| voice | TEXT | Voice ID |
| char_count | INTEGER | Characters synthesized |
| cost | REAL | Cost USD |

## API Endpoints

All new endpoints under `/api/costs/`:

| Endpoint | Description |
|----------|-------------|
| `GET /api/costs/summary?period=today\|week\|month\|all` | Aggregated totals: spend, LLM/TTS split, call count, tokens, avg cost/session, % change vs previous period |
| `GET /api/costs/timeline?period=week\|month\|all&group_by=session\|day` | Time-series for line chart (cost over time) |
| `GET /api/costs/models?period=week\|month\|all` | Per-model breakdown for pie/bar charts |
| `GET /api/costs/categories?period=week\|month\|all` | Per-category breakdown (background_gen, caller_dialog, etc.) |
| `GET /api/costs/sessions?period=week\|month\|all` | Session list with totals, sortable |
| `GET /api/costs/session/{id}` | Single session detail: per-caller costs, expensive calls, recommendations |
| `GET /api/costs/expensive?period=week\|month\|all&limit=10` | Top N most expensive individual calls |

Existing `/api/costs` (live session) endpoint remains unchanged.

## Dashboard Layout

### Header
Time range selector tabs: Today / This Week / This Month / All Time. Clicking any tab refreshes all charts.

### Row 1 — Summary Cards (4 across)
- **Total Spend** with % change vs previous period
- **LLM / TTS Split** showing both values
- **Total Sessions** in period
- **Avg Cost Per Session**

### Row 2 — Two charts side by side
- **Left: Cost Over Time** — line chart, x-axis sessions or days, y-axis dollars. LLM and TTS as separate lines.
- **Right: Cost by Model** — doughnut chart with legend showing dollar amounts.

### Row 3 — Two charts side by side
- **Left: Cost by Category** — horizontal bar chart (background_gen, caller_dialog, devon_monitor, etc.)
- **Right: Cost Per Session Trend** — bar chart, each bar a session, colored above/below average.

### Row 4 — Tables
- **Most Expensive Calls** — top 10 LLM calls (model, category, caller, tokens, cost, timestamp)
- **Session List** — all sessions, sortable by date/cost, clickable for detail view.

### Session Detail View (click-through)
- Per-caller cost breakdown
- Call-by-call timeline
- Recommendations from existing `_generate_recommendations()` logic

## Visual Style
Matches the control panel's existing dark theme. Same fonts, colors, card styles.

## What's NOT in v1
- SignalWire cost tracking (future addition)
- Real-time WebSocket updates (polling on page load is sufficient)
- Cost alerts/budgets
- Export to CSV
