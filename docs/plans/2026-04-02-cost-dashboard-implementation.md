# Cost Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a cost analytics dashboard at `/costs` that visualizes LLM and TTS spending across sessions with time-range filtering, charts, and drill-down.

**Architecture:** SQLite database (`data/costs.db`) stores all cost records. Existing JSON reports are imported on first run. `cost_tracker.py` dual-writes to both JSON and SQLite. New API endpoints serve aggregated data. Standalone HTML page with Chart.js renders the dashboard.

**Tech Stack:** Python/FastAPI, SQLite, Chart.js (CDN), vanilla JS, CSS custom properties matching existing dark theme.

---

### Task 1: SQLite Database Module

**Files:**
- Create: `backend/services/cost_db.py`

**Step 1: Create the database module with schema**

Create `backend/services/cost_db.py` with:
- `init_db(db_path)` — creates tables if not exist, returns connection
- `import_json_reports(db_path, reports_dir)` — scans `data/cost_reports/*.json`, imports any sessions not already in the DB
- `get_db()` — returns a connection to `data/costs.db`, calls `init_db` on first use

```python
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "costs.db"
REPORTS_DIR = Path(__file__).parent.parent.parent / "data" / "cost_reports"

_connection = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at TIMESTAMP,
    total_cost REAL DEFAULT 0,
    llm_cost REAL DEFAULT 0,
    tts_cost REAL DEFAULT 0,
    total_llm_calls INTEGER DEFAULT 0,
    total_tts_calls INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp TIMESTAMP,
    category TEXT,
    model TEXT,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cost REAL DEFAULT 0,
    caller_name TEXT,
    latency_ms REAL DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS tts_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp TIMESTAMP,
    provider TEXT,
    voice TEXT,
    char_count INTEGER DEFAULT 0,
    cost REAL DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_llm_session ON llm_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_llm_timestamp ON llm_calls(timestamp);
CREATE INDEX IF NOT EXISTS idx_llm_category ON llm_calls(category);
CREATE INDEX IF NOT EXISTS idx_llm_model ON llm_calls(model);
CREATE INDEX IF NOT EXISTS idx_tts_session ON tts_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_tts_timestamp ON tts_calls(timestamp);
"""


def get_db():
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _connection.row_factory = sqlite3.Row
        _connection.executescript(SCHEMA)
        import_json_reports()
    return _connection


def import_json_reports():
    db = _connection
    if not REPORTS_DIR.exists():
        return
    existing = {row[0] for row in db.execute("SELECT id FROM sessions").fetchall()}
    for f in sorted(REPORTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        session_id = data.get("session_id", f.stem)
        if session_id in existing:
            continue
        saved_at = data.get("saved_at")
        started_at = datetime.fromtimestamp(saved_at).isoformat() if saved_at else None
        db.execute(
            "INSERT INTO sessions (id, started_at, total_cost, llm_cost, tts_cost, total_llm_calls, total_tts_calls, total_tokens, prompt_tokens, completion_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, started_at, data.get("total_cost_usd", 0), data.get("llm_cost_usd", 0), data.get("tts_cost_usd", 0), data.get("total_llm_calls", 0), len(data.get("raw_tts_records", [])), data.get("total_tokens", 0), data.get("prompt_tokens", 0), data.get("completion_tokens", 0)),
        )
        for rec in data.get("raw_llm_records", []):
            db.execute(
                "INSERT INTO llm_calls (session_id, timestamp, category, model, prompt_tokens, completion_tokens, cost, caller_name, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, rec.get("timestamp"), rec.get("category"), rec.get("model"), rec.get("prompt_tokens", 0), rec.get("completion_tokens", 0), rec.get("cost_usd", 0), rec.get("caller_name"), rec.get("latency_ms", 0)),
            )
        for rec in data.get("raw_tts_records", []):
            db.execute(
                "INSERT INTO tts_calls (session_id, timestamp, provider, voice, char_count, cost) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, rec.get("timestamp"), rec.get("provider"), rec.get("voice"), rec.get("char_count", 0), rec.get("cost_usd", 0)),
            )
        existing.add(session_id)
    db.commit()


def record_llm_call(session_id, timestamp, category, model, prompt_tokens, completion_tokens, cost, caller_name, latency_ms):
    db = get_db()
    db.execute(
        "INSERT INTO llm_calls (session_id, timestamp, category, model, prompt_tokens, completion_tokens, cost, caller_name, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, timestamp, category, model, prompt_tokens, completion_tokens, cost, caller_name, latency_ms),
    )
    db.commit()


def record_tts_call(session_id, timestamp, provider, voice, char_count, cost):
    db = get_db()
    db.execute(
        "INSERT INTO tts_calls (session_id, timestamp, provider, voice, char_count, cost) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, timestamp, provider, voice, char_count, cost),
    )
    db.commit()


def ensure_session(session_id, started_at=None):
    db = get_db()
    existing = db.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not existing:
        db.execute(
            "INSERT INTO sessions (id, started_at) VALUES (?, ?)",
            (session_id, started_at or datetime.now().isoformat()),
        )
        db.commit()


def update_session_totals(session_id):
    db = get_db()
    llm = db.execute(
        "SELECT COUNT(*) as calls, COALESCE(SUM(cost), 0) as cost, COALESCE(SUM(prompt_tokens), 0) as pt, COALESCE(SUM(completion_tokens), 0) as ct FROM llm_calls WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    tts = db.execute(
        "SELECT COUNT(*) as calls, COALESCE(SUM(cost), 0) as cost FROM tts_calls WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    total_tokens = llm["pt"] + llm["ct"]
    total_cost = llm["cost"] + tts["cost"]
    db.execute(
        "UPDATE sessions SET total_cost=?, llm_cost=?, tts_cost=?, total_llm_calls=?, total_tts_calls=?, total_tokens=?, prompt_tokens=?, completion_tokens=? WHERE id=?",
        (total_cost, llm["cost"], tts["cost"], llm["calls"], tts["calls"], total_tokens, llm["pt"], llm["ct"], session_id),
    )
    db.commit()


def _period_filter(period):
    now = datetime.now()
    if period == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        start = now - timedelta(days=now.weekday())
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        return None
    return start.isoformat()


def get_summary(period="all"):
    db = get_db()
    start = _period_filter(period)
    if start:
        where = "WHERE started_at >= ?"
        params = (start,)
        prev_start = _get_previous_period_start(period)
        prev_where = "WHERE started_at >= ? AND started_at < ?"
        prev_params = (prev_start, start)
    else:
        where = ""
        params = ()
        prev_where = None
        prev_params = ()

    row = db.execute(
        f"SELECT COUNT(*) as sessions, COALESCE(SUM(total_cost), 0) as total_cost, COALESCE(SUM(llm_cost), 0) as llm_cost, COALESCE(SUM(tts_cost), 0) as tts_cost, COALESCE(SUM(total_llm_calls), 0) as total_calls, COALESCE(SUM(total_tokens), 0) as total_tokens FROM sessions {where}",
        params,
    ).fetchone()

    result = {
        "total_cost": round(row["total_cost"], 4),
        "llm_cost": round(row["llm_cost"], 4),
        "tts_cost": round(row["tts_cost"], 4),
        "sessions": row["sessions"],
        "total_calls": row["total_calls"],
        "total_tokens": row["total_tokens"],
        "avg_cost_per_session": round(row["total_cost"] / max(row["sessions"], 1), 4),
    }

    if prev_where:
        prev = db.execute(
            f"SELECT COALESCE(SUM(total_cost), 0) as total_cost FROM sessions {prev_where}",
            prev_params,
        ).fetchone()
        prev_cost = prev["total_cost"]
        if prev_cost > 0:
            result["pct_change"] = round((row["total_cost"] - prev_cost) / prev_cost * 100, 1)
        else:
            result["pct_change"] = None
    else:
        result["pct_change"] = None

    return result


def _get_previous_period_start(period):
    now = datetime.now()
    if period == "today":
        return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    elif period == "week":
        start_of_week = now - timedelta(days=now.weekday())
        return (start_of_week - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    elif period == "month":
        first_of_month = now.replace(day=1)
        prev_month = first_of_month - timedelta(days=1)
        return prev_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    return None


def get_timeline(period="all", group_by="session"):
    db = get_db()
    start = _period_filter(period)
    if group_by == "day":
        if start:
            rows = db.execute(
                "SELECT DATE(started_at) as date, SUM(llm_cost) as llm_cost, SUM(tts_cost) as tts_cost, SUM(total_cost) as total_cost, COUNT(*) as sessions FROM sessions WHERE started_at >= ? GROUP BY DATE(started_at) ORDER BY date",
                (start,),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT DATE(started_at) as date, SUM(llm_cost) as llm_cost, SUM(tts_cost) as tts_cost, SUM(total_cost) as total_cost, COUNT(*) as sessions FROM sessions GROUP BY DATE(started_at) ORDER BY date"
            ).fetchall()
    else:
        if start:
            rows = db.execute(
                "SELECT id, started_at, llm_cost, tts_cost, total_cost FROM sessions WHERE started_at >= ? ORDER BY started_at",
                (start,),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT id, started_at, llm_cost, tts_cost, total_cost FROM sessions ORDER BY started_at"
            ).fetchall()
    return [dict(r) for r in rows]


def get_models(period="all"):
    db = get_db()
    start = _period_filter(period)
    if start:
        rows = db.execute(
            "SELECT l.model, COUNT(*) as calls, COALESCE(SUM(l.cost), 0) as cost, COALESCE(SUM(l.prompt_tokens), 0) as prompt_tokens, COALESCE(SUM(l.completion_tokens), 0) as completion_tokens FROM llm_calls l JOIN sessions s ON l.session_id = s.id WHERE s.started_at >= ? GROUP BY l.model ORDER BY cost DESC",
            (start,),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT model, COUNT(*) as calls, COALESCE(SUM(cost), 0) as cost, COALESCE(SUM(prompt_tokens), 0) as prompt_tokens, COALESCE(SUM(completion_tokens), 0) as completion_tokens FROM llm_calls GROUP BY model ORDER BY cost DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_categories(period="all"):
    db = get_db()
    start = _period_filter(period)
    if start:
        rows = db.execute(
            "SELECT l.category, COUNT(*) as calls, COALESCE(SUM(l.cost), 0) as cost, COALESCE(SUM(l.prompt_tokens + l.completion_tokens), 0) as tokens FROM llm_calls l JOIN sessions s ON l.session_id = s.id WHERE s.started_at >= ? GROUP BY l.category ORDER BY cost DESC",
            (start,),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT category, COUNT(*) as calls, COALESCE(SUM(cost), 0) as cost, COALESCE(SUM(prompt_tokens + completion_tokens), 0) as tokens FROM llm_calls GROUP BY category ORDER BY cost DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_sessions_list(period="all"):
    db = get_db()
    start = _period_filter(period)
    if start:
        rows = db.execute(
            "SELECT id, started_at, total_cost, llm_cost, tts_cost, total_llm_calls, total_tokens FROM sessions WHERE started_at >= ? ORDER BY started_at DESC",
            (start,),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT id, started_at, total_cost, llm_cost, tts_cost, total_llm_calls, total_tokens FROM sessions ORDER BY started_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_session_detail(session_id):
    db = get_db()
    session = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not session:
        return None
    by_caller = db.execute(
        "SELECT caller_name, COUNT(*) as calls, COALESCE(SUM(cost), 0) as cost, COALESCE(SUM(prompt_tokens + completion_tokens), 0) as tokens FROM llm_calls WHERE session_id = ? AND caller_name != '' GROUP BY caller_name ORDER BY cost DESC",
        (session_id,),
    ).fetchall()
    by_model = db.execute(
        "SELECT model, COUNT(*) as calls, COALESCE(SUM(cost), 0) as cost FROM llm_calls WHERE session_id = ? GROUP BY model ORDER BY cost DESC",
        (session_id,),
    ).fetchall()
    by_category = db.execute(
        "SELECT category, COUNT(*) as calls, COALESCE(SUM(cost), 0) as cost FROM llm_calls WHERE session_id = ? GROUP BY category ORDER BY cost DESC",
        (session_id,),
    ).fetchall()
    expensive = db.execute(
        "SELECT category, model, caller_name, cost, prompt_tokens, completion_tokens, latency_ms, timestamp FROM llm_calls WHERE session_id = ? ORDER BY cost DESC LIMIT 10",
        (session_id,),
    ).fetchall()
    tts = db.execute(
        "SELECT provider, COUNT(*) as calls, COALESCE(SUM(cost), 0) as cost, COALESCE(SUM(char_count), 0) as chars FROM tts_calls WHERE session_id = ? GROUP BY provider ORDER BY cost DESC",
        (session_id,),
    ).fetchall()
    return {
        "session": dict(session),
        "by_caller": [dict(r) for r in by_caller],
        "by_model": [dict(r) for r in by_model],
        "by_category": [dict(r) for r in by_category],
        "expensive_calls": [dict(r) for r in expensive],
        "tts_by_provider": [dict(r) for r in tts],
    }


def get_expensive_calls(period="all", limit=10):
    db = get_db()
    start = _period_filter(period)
    if start:
        rows = db.execute(
            "SELECT l.category, l.model, l.caller_name, l.cost, l.prompt_tokens, l.completion_tokens, l.latency_ms, l.timestamp, l.session_id FROM llm_calls l JOIN sessions s ON l.session_id = s.id WHERE s.started_at >= ? ORDER BY l.cost DESC LIMIT ?",
            (start, limit),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT category, model, caller_name, cost, prompt_tokens, completion_tokens, latency_ms, timestamp, session_id FROM llm_calls ORDER BY cost DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
```

**Step 2: Verify the module loads**

Run: `cd /Users/lukemacneil/code/ai-podcast && python -c "from backend.services.cost_db import get_db; db = get_db(); print('OK, sessions:', db.execute('SELECT COUNT(*) FROM sessions').fetchone()[0])"`

Expected: `OK, sessions: 18` (or however many JSON reports exist)

**Step 3: Commit**

```bash
git add backend/services/cost_db.py
git commit -m "Add SQLite cost database module with JSON import"
```

---

### Task 2: Integrate SQLite Writes into Cost Tracker

**Files:**
- Modify: `backend/services/cost_tracker.py`

**Step 1: Add dual-write to `record_llm_call`**

At the top of `cost_tracker.py`, add the import:
```python
from backend.services import cost_db
```

In `record_llm_call()` (around line 148, after appending to `self.llm_records`), add:
```python
try:
    cost_db.ensure_session(self._session_id)
    cost_db.record_llm_call(
        self._session_id, record.timestamp, record.category, record.model,
        record.prompt_tokens, record.completion_tokens, record.cost_usd,
        record.caller_name, record.latency_ms,
    )
except Exception:
    pass  # don't break show over analytics
```

**Step 2: Add dual-write to `record_tts_call`**

In `record_tts_call()` (around line 166, after appending to `self.tts_records`), add:
```python
try:
    cost_db.record_tts_call(
        self._session_id, record.timestamp, record.provider, record.voice,
        record.char_count, record.cost_usd,
    )
except Exception:
    pass
```

**Step 3: Add session_id tracking to `__init__`**

Add `self._session_id` to the constructor. Generate it from timestamp:
```python
self._session_id = f"session-{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
```

**Step 4: Update session totals on `save()`**

In the `save()` method, after writing the JSON file, add:
```python
try:
    cost_db.update_session_totals(self._session_id)
except Exception:
    pass
```

**Step 5: Commit**

```bash
git add backend/services/cost_tracker.py
git commit -m "Dual-write cost records to SQLite"
```

---

### Task 3: API Endpoints

**Files:**
- Modify: `backend/main.py` (add routes near existing `/api/costs` endpoints around line 10299)

**Step 1: Add new cost dashboard endpoints**

Add these routes near the existing cost endpoints (around line 10308):

```python
from backend.services import cost_db

@app.get("/api/costs/summary")
async def get_cost_summary(period: str = "all"):
    return cost_db.get_summary(period)

@app.get("/api/costs/timeline")
async def get_cost_timeline(period: str = "all", group_by: str = "session"):
    return cost_db.get_timeline(period, group_by)

@app.get("/api/costs/models")
async def get_cost_models(period: str = "all"):
    return cost_db.get_models(period)

@app.get("/api/costs/categories")
async def get_cost_categories(period: str = "all"):
    return cost_db.get_categories(period)

@app.get("/api/costs/sessions")
async def get_cost_sessions(period: str = "all"):
    return cost_db.get_sessions_list(period)

@app.get("/api/costs/session/{session_id}")
async def get_cost_session_detail(session_id: str):
    detail = cost_db.get_session_detail(session_id)
    if not detail:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    return detail

@app.get("/api/costs/expensive")
async def get_expensive_calls(period: str = "all", limit: int = 10):
    return cost_db.get_expensive_calls(period, limit)
```

**Step 2: Add route to serve the costs page**

Near the existing root route (around line 7654), add:

```python
@app.get("/costs")
async def costs_page():
    return FileResponse(frontend_dir / "costs.html")
```

**Step 3: Verify endpoints respond**

Run the server: `python -m uvicorn backend.main:app --reload --reload-dir backend --host 0.0.0.0 --port 8000`

Test: `curl -s http://localhost:8000/api/costs/summary?period=all | python -m json.tool`

Expected: JSON with total_cost, llm_cost, tts_cost, sessions, etc.

**Step 4: Commit**

```bash
git add backend/main.py
git commit -m "Add cost dashboard API endpoints"
```

---

### Task 4: Dashboard HTML

**Files:**
- Create: `frontend/costs.html`

**Step 1: Create the dashboard page**

Create `frontend/costs.html` — standalone HTML page with:
- Chart.js from CDN (`https://cdn.jsdelivr.net/npm/chart.js`)
- Link to `css/style.css` (shared theme) and `css/costs.css` (dashboard-specific)
- Script tag for `js/costs.js`
- Structure: header with time range tabs, 4 summary cards, 4 chart containers, 2 tables
- Use the same CSS variables as the control panel (`--bg`, `--bg-light`, `--accent`, `--text`, etc.)

Layout structure:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cost Dashboard - Luke at the Roost</title>
    <link rel="stylesheet" href="/css/style.css">
    <link rel="stylesheet" href="/css/costs.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <header class="costs-header">
        <h1>Cost Dashboard</h1>
        <a href="/" class="back-link">Back to Show</a>
        <nav class="period-tabs">
            <button class="period-tab active" data-period="all">All Time</button>
            <button class="period-tab" data-period="month">This Month</button>
            <button class="period-tab" data-period="week">This Week</button>
            <button class="period-tab" data-period="today">Today</button>
        </nav>
    </header>
    <main class="costs-main">
        <section class="summary-cards">
            <div class="card" id="card-total">
                <div class="card-label">Total Spend</div>
                <div class="card-value" id="total-spend">--</div>
                <div class="card-change" id="total-change"></div>
            </div>
            <div class="card" id="card-llm-tts">
                <div class="card-label">LLM / TTS</div>
                <div class="card-value" id="llm-tts-split">--</div>
            </div>
            <div class="card" id="card-sessions">
                <div class="card-label">Sessions</div>
                <div class="card-value" id="session-count">--</div>
            </div>
            <div class="card" id="card-avg">
                <div class="card-label">Avg / Session</div>
                <div class="card-value" id="avg-cost">--</div>
            </div>
        </section>
        <section class="chart-row">
            <div class="chart-container">
                <h3>Cost Over Time</h3>
                <canvas id="timeline-chart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Cost by Model</h3>
                <canvas id="model-chart"></canvas>
            </div>
        </section>
        <section class="chart-row">
            <div class="chart-container">
                <h3>Cost by Category</h3>
                <canvas id="category-chart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Cost Per Session</h3>
                <canvas id="session-chart"></canvas>
            </div>
        </section>
        <section class="tables-section">
            <div class="table-container">
                <h3>Most Expensive Calls</h3>
                <table id="expensive-table">
                    <thead>
                        <tr><th>Model</th><th>Category</th><th>Caller</th><th>Tokens</th><th>Cost</th><th>Latency</th></tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
            <div class="table-container">
                <h3>Sessions</h3>
                <table id="sessions-table">
                    <thead>
                        <tr><th>Date</th><th>LLM</th><th>TTS</th><th>Total</th><th>Calls</th><th></th></tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
        </section>
        <section class="session-detail hidden" id="session-detail">
            <h2>Session Detail: <span id="detail-session-id"></span></h2>
            <button class="close-detail" id="close-detail">Back</button>
            <div class="detail-grid">
                <div class="table-container">
                    <h3>By Caller</h3>
                    <table id="detail-caller-table">
                        <thead><tr><th>Caller</th><th>Calls</th><th>Cost</th></tr></thead>
                        <tbody></tbody>
                    </table>
                </div>
                <div class="table-container">
                    <h3>By Category</h3>
                    <table id="detail-category-table">
                        <thead><tr><th>Category</th><th>Calls</th><th>Cost</th></tr></thead>
                        <tbody></tbody>
                    </table>
                </div>
                <div class="table-container">
                    <h3>By Model</h3>
                    <table id="detail-model-table">
                        <thead><tr><th>Model</th><th>Calls</th><th>Cost</th></tr></thead>
                        <tbody></tbody>
                    </table>
                </div>
                <div class="table-container">
                    <h3>Most Expensive Calls</h3>
                    <table id="detail-expensive-table">
                        <thead><tr><th>Category</th><th>Model</th><th>Caller</th><th>Cost</th><th>Tokens</th><th>Latency</th></tr></thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </section>
    </main>
    <script src="/js/costs.js"></script>
</body>
</html>
```

**Step 2: Commit**

```bash
git add frontend/costs.html
git commit -m "Add cost dashboard HTML page"
```

---

### Task 5: Dashboard CSS

**Files:**
- Create: `frontend/css/costs.css`

**Step 1: Create dashboard-specific styles**

Create `frontend/css/costs.css` using the existing CSS variables from `style.css`. Key styles:
- `.costs-header` — flex row with title, back link, and period tabs
- `.period-tabs` / `.period-tab` — tab buttons, active state uses `--accent`
- `.summary-cards` — 4-column grid
- `.card` — `background: var(--bg-light)`, border, rounded corners matching `--radius`
- `.card-value` — large font, `color: var(--text)`
- `.card-change` — small text, green for negative (saving), red for positive (increase)
- `.chart-row` — 2-column grid
- `.chart-container` — padded card with canvas
- `.table-container` — styled tables matching dark theme
- `.session-detail` — full-width detail view
- Responsive: single column below 768px

Use `var(--bg)`, `var(--bg-light)`, `var(--accent)`, `var(--text)`, `var(--text-muted)`, `var(--radius)`, `var(--radius-sm)`, `var(--transition)` throughout.

**Step 2: Commit**

```bash
git add frontend/css/costs.css
git commit -m "Add cost dashboard CSS"
```

---

### Task 6: Dashboard JavaScript

**Files:**
- Create: `frontend/js/costs.js`

**Step 1: Create the dashboard JS**

Create `frontend/js/costs.js` with:

**State:**
```javascript
let currentPeriod = 'all';
let charts = {};  // store Chart.js instances for destroy/recreate
```

**Init:**
```javascript
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.period-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelector('.period-tab.active').classList.remove('active');
            tab.classList.add('active');
            currentPeriod = tab.dataset.period;
            loadDashboard();
        });
    });
    document.getElementById('close-detail').addEventListener('click', closeDetail);
    loadDashboard();
});
```

**Data loading — `loadDashboard()`:**
- Fetch all endpoints in parallel: summary, timeline, models, categories, sessions, expensive
- Call render functions for each section

**Render functions:**
- `renderSummary(data)` — populate the 4 summary cards, format as `$X.XX`, show % change with color
- `renderTimeline(data)` — Chart.js line chart with LLM and TTS as separate datasets, `--accent` and `--devon` colors
- `renderModels(data)` — Chart.js doughnut chart with model name labels
- `renderCategories(data)` — Chart.js horizontal bar chart
- `renderSessionBars(data)` — Chart.js bar chart, bars colored based on above/below average
- `renderExpensiveTable(data)` — populate table rows
- `renderSessionsTable(data)` — populate table rows with click handler to show detail
- `showSessionDetail(sessionId)` — fetch `/api/costs/session/{id}`, show detail section, populate tables
- `closeDetail()` — hide detail section

**Chart.js config notes:**
- Use dark theme: grid lines `rgba(245, 240, 229, 0.1)` (--text at 10%), tick color `var(--text-muted)`
- Chart colors palette: `#e8791d` (accent), `#c4944a` (devon), `#5a8a3c` (green), `#cc2222` (red), `#4a8ac4` (blue), `#8a5ac4` (purple), `#c4845a` (tan)
- Tooltips: dark background, light text
- Destroy existing chart instance before creating new one (prevents memory leaks on period switch)

**Utility:**
- `formatCost(n)` — returns `$X.XX` or `$X.XXXX` for small amounts
- `formatDate(iso)` — returns readable date
- `shortenModel(name)` — strip provider prefix from model names for chart labels

**Step 2: Commit**

```bash
git add frontend/js/costs.js
git commit -m "Add cost dashboard JavaScript with Chart.js"
```

---

### Task 7: Integration Test

**Step 1: Manual verification checklist**

Start the server and navigate to `http://localhost:8000/costs`:

1. Page loads with dark theme, no console errors
2. All Time tab is active by default, shows all 18 sessions
3. Summary cards show total spend, LLM/TTS split, session count, avg cost
4. Cost Over Time line chart renders with data points
5. Cost by Model doughnut shows model breakdown
6. Cost by Category bar chart shows category breakdown
7. Cost Per Session bars render with above/below-average coloring
8. Most Expensive Calls table populated
9. Sessions table populated, rows clickable
10. Click a session row → detail view shows with per-caller, per-model, per-category tables
11. Click "Back" → returns to main dashboard
12. Switch to "This Week" tab → all charts update (may show fewer/no data)
13. Switch to "This Month" → charts update with March data
14. Switch back to "All Time" → full data restored

**Step 2: Commit all remaining changes**

```bash
git add -A
git commit -m "Cost dashboard complete — SQLite backend, Chart.js frontend"
```

---

## File Summary

| Action | File |
|--------|------|
| Create | `backend/services/cost_db.py` |
| Modify | `backend/services/cost_tracker.py` |
| Modify | `backend/main.py` |
| Create | `frontend/costs.html` |
| Create | `frontend/css/costs.css` |
| Create | `frontend/js/costs.js` |

## Dependencies

- Chart.js loaded from CDN (no npm install needed)
- SQLite is stdlib (no pip install needed)
- No new Python packages required
