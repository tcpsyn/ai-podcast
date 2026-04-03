"""SQLite database for cross-session cost analytics"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "costs.db"
REPORTS_DIR = PROJECT_ROOT / "data" / "cost_reports"

_conn: sqlite3.Connection | None = None


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript("""
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
            session_id TEXT NOT NULL REFERENCES sessions(id),
            timestamp TIMESTAMP,
            category TEXT,
            model TEXT,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            cost REAL DEFAULT 0,
            caller_name TEXT,
            latency_ms REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS tts_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            timestamp TIMESTAMP,
            provider TEXT,
            voice TEXT,
            char_count INTEGER DEFAULT 0,
            cost REAL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_llm_session_ts_cat_model
            ON llm_calls(session_id, timestamp, category, model);
        CREATE INDEX IF NOT EXISTS idx_tts_session_ts
            ON tts_calls(session_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_llm_timestamp
            ON llm_calls(timestamp);
    """)
    conn.commit()
    return conn


def get_db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = init_db(DB_PATH)
        import_json_reports(_conn)
    return _conn


def import_json_reports(conn: sqlite3.Connection | None = None):
    if conn is None:
        conn = get_db()
    if not REPORTS_DIR.exists():
        return

    existing = {
        row[0]
        for row in conn.execute("SELECT id FROM sessions").fetchall()
    }

    for fp in sorted(REPORTS_DIR.glob("session-*.json")):
        try:
            data = json.loads(fp.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        session_id = data.get("session_id", fp.stem)
        if session_id in existing:
            continue

        saved_at = data.get("saved_at", 0)
        started_at = datetime.fromtimestamp(saved_at, tz=timezone.utc).isoformat() if saved_at else None

        conn.execute(
            "INSERT INTO sessions (id, started_at, total_cost, llm_cost, tts_cost, "
            "total_llm_calls, total_tts_calls, total_tokens, prompt_tokens, completion_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                started_at,
                data.get("total_cost_usd", 0),
                data.get("llm_cost_usd", 0),
                data.get("tts_cost_usd", 0),
                data.get("total_llm_calls", 0),
                len(data.get("raw_tts_records", [])),
                data.get("total_tokens", 0),
                data.get("prompt_tokens", 0),
                data.get("completion_tokens", 0),
            ),
        )

        for r in data.get("raw_llm_records", []):
            try:
                ts = datetime.fromtimestamp(r.get("timestamp", 0), tz=timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO llm_calls (session_id, timestamp, category, model, "
                    "prompt_tokens, completion_tokens, cost, caller_name, latency_ms) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_id, ts, r.get("category"), r.get("model"),
                        r.get("prompt_tokens", 0), r.get("completion_tokens", 0),
                        r.get("cost_usd", 0), r.get("caller_name", ""),
                        r.get("latency_ms", 0),
                    ),
                )
            except Exception:
                continue

        for r in data.get("raw_tts_records", []):
            try:
                ts = datetime.fromtimestamp(r.get("timestamp", 0), tz=timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO tts_calls (session_id, timestamp, provider, voice, "
                    "char_count, cost) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        session_id, ts, r.get("provider"), r.get("voice"),
                        r.get("char_count", 0), r.get("cost_usd", 0),
                    ),
                )
            except Exception:
                continue

        existing.add(session_id)

    conn.commit()


def record_llm_call(session_id, timestamp, category, model, prompt_tokens,
                     completion_tokens, cost, caller_name="", latency_ms=0.0):
    conn = get_db()
    ensure_session(session_id)
    ts = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat() if isinstance(timestamp, (int, float)) else timestamp
    conn.execute(
        "INSERT INTO llm_calls (session_id, timestamp, category, model, "
        "prompt_tokens, completion_tokens, cost, caller_name, latency_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, ts, category, model, prompt_tokens, completion_tokens,
         cost, caller_name, latency_ms),
    )
    conn.commit()


def record_tts_call(session_id, timestamp, provider, voice, char_count, cost):
    conn = get_db()
    ensure_session(session_id)
    ts = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat() if isinstance(timestamp, (int, float)) else timestamp
    conn.execute(
        "INSERT INTO tts_calls (session_id, timestamp, provider, voice, "
        "char_count, cost) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, ts, provider, voice, char_count, cost),
    )
    conn.commit()


def ensure_session(session_id, started_at=None):
    conn = get_db()
    existing = conn.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if existing:
        return
    ts = started_at
    if isinstance(started_at, (int, float)):
        ts = datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat()
    elif started_at is None:
        ts = datetime.now(timezone.utc).isoformat()
    conn.execute("INSERT INTO sessions (id, started_at) VALUES (?, ?)", (session_id, ts))
    conn.commit()


def update_session_totals(session_id):
    conn = get_db()
    llm = conn.execute(
        "SELECT COUNT(*) as cnt, COALESCE(SUM(cost),0) as cost, "
        "COALESCE(SUM(prompt_tokens),0) as pt, COALESCE(SUM(completion_tokens),0) as ct "
        "FROM llm_calls WHERE session_id = ?", (session_id,)
    ).fetchone()
    tts = conn.execute(
        "SELECT COUNT(*) as cnt, COALESCE(SUM(cost),0) as cost "
        "FROM tts_calls WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.execute(
        "UPDATE sessions SET total_cost=?, llm_cost=?, tts_cost=?, "
        "total_llm_calls=?, total_tts_calls=?, total_tokens=?, "
        "prompt_tokens=?, completion_tokens=? WHERE id=?",
        (
            llm["cost"] + tts["cost"], llm["cost"], tts["cost"],
            llm["cnt"], tts["cnt"],
            llm["pt"] + llm["ct"], llm["pt"], llm["ct"],
            session_id,
        ),
    )
    conn.commit()


def _period_filter(period: str) -> str | None:
    now = datetime.now(timezone.utc)
    if period == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    elif period == "week":
        return (now - timedelta(days=7)).isoformat()
    elif period == "month":
        return (now - timedelta(days=30)).isoformat()
    elif period == "all":
        return None
    return None


def _get_previous_period_start(period: str) -> str | None:
    now = datetime.now(timezone.utc)
    if period == "today":
        yesterday = now - timedelta(days=1)
        return yesterday.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    elif period == "week":
        return (now - timedelta(days=14)).isoformat()
    elif period == "month":
        return (now - timedelta(days=60)).isoformat()
    elif period == "all":
        return None
    return None


def _where_clause(period: str, ts_col: str = "started_at") -> tuple[str, list]:
    start = _period_filter(period)
    if start is None:
        return "", []
    return f"WHERE {ts_col} >= ?", [start]


def _pct_change(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return round((current - previous) / previous * 100, 1)


def get_summary(period: str = "all") -> dict:
    conn = get_db()
    where, params = _where_clause(period)
    row = conn.execute(
        f"SELECT COUNT(*) as sessions, COALESCE(SUM(total_cost),0) as total_cost, "
        f"COALESCE(SUM(llm_cost),0) as llm_cost, COALESCE(SUM(tts_cost),0) as tts_cost, "
        f"COALESCE(SUM(total_llm_calls),0) as llm_calls, "
        f"COALESCE(SUM(total_tts_calls),0) as tts_calls, "
        f"COALESCE(SUM(total_tokens),0) as tokens "
        f"FROM sessions {where}", params
    ).fetchone()

    result = {
        "sessions": row["sessions"],
        "total_cost": round(row["total_cost"], 4),
        "llm_cost": round(row["llm_cost"], 4),
        "tts_cost": round(row["tts_cost"], 4),
        "llm_calls": row["llm_calls"],
        "tts_calls": row["tts_calls"],
        "tokens": row["tokens"],
        "avg_cost_per_session": round(row["total_cost"] / max(row["sessions"], 1), 4),
    }

    # % change vs previous period
    prev_start = _get_previous_period_start(period)
    current_start = _period_filter(period)
    if prev_start and current_start:
        prev_row = conn.execute(
            "SELECT COALESCE(SUM(total_cost),0) as total_cost, "
            "COALESCE(SUM(total_llm_calls),0) as llm_calls "
            "FROM sessions WHERE started_at >= ? AND started_at < ?",
            [prev_start, current_start]
        ).fetchone()
        result["cost_change_pct"] = _pct_change(row["total_cost"], prev_row["total_cost"])
        result["calls_change_pct"] = _pct_change(row["llm_calls"], prev_row["llm_calls"])
    else:
        result["cost_change_pct"] = None
        result["calls_change_pct"] = None

    return result


def get_timeline(period: str = "all", group_by: str = "session") -> list[dict]:
    conn = get_db()
    where, params = _where_clause(period)

    if group_by == "day":
        rows = conn.execute(
            f"SELECT DATE(started_at) as date, COUNT(*) as sessions, "
            f"SUM(total_cost) as total_cost, SUM(llm_cost) as llm_cost, "
            f"SUM(tts_cost) as tts_cost, SUM(total_llm_calls) as llm_calls, "
            f"SUM(total_tokens) as tokens "
            f"FROM sessions {where} GROUP BY DATE(started_at) ORDER BY date", params
        ).fetchall()
        return [
            {
                "date": r["date"],
                "sessions": r["sessions"],
                "total_cost": round(r["total_cost"], 4),
                "llm_cost": round(r["llm_cost"], 4),
                "tts_cost": round(r["tts_cost"], 4),
                "llm_calls": r["llm_calls"],
                "tokens": r["tokens"],
            }
            for r in rows
        ]
    else:
        rows = conn.execute(
            f"SELECT id, started_at, total_cost, llm_cost, tts_cost, "
            f"total_llm_calls as llm_calls, total_tokens as tokens "
            f"FROM sessions {where} ORDER BY started_at", params
        ).fetchall()
        return [
            {
                "session_id": r["id"],
                "started_at": r["started_at"],
                "total_cost": round(r["total_cost"], 4),
                "llm_cost": round(r["llm_cost"], 4),
                "tts_cost": round(r["tts_cost"], 4),
                "llm_calls": r["llm_calls"],
                "tokens": r["tokens"],
            }
            for r in rows
        ]


def get_models(period: str = "all") -> list[dict]:
    conn = get_db()
    start = _period_filter(period)
    if start:
        rows = conn.execute(
            "SELECT l.model, COUNT(*) as calls, SUM(l.cost) as cost, "
            "SUM(l.prompt_tokens) as prompt_tokens, SUM(l.completion_tokens) as completion_tokens "
            "FROM llm_calls l JOIN sessions s ON l.session_id = s.id "
            "WHERE s.started_at >= ? GROUP BY l.model ORDER BY cost DESC", [start]
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT model, COUNT(*) as calls, SUM(cost) as cost, "
            "SUM(prompt_tokens) as prompt_tokens, SUM(completion_tokens) as completion_tokens "
            "FROM llm_calls GROUP BY model ORDER BY cost DESC"
        ).fetchall()
    return [
        {
            "model": r["model"],
            "calls": r["calls"],
            "cost": round(r["cost"], 4),
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
        }
        for r in rows
    ]


def get_categories(period: str = "all") -> list[dict]:
    conn = get_db()
    start = _period_filter(period)
    if start:
        rows = conn.execute(
            "SELECT l.category, COUNT(*) as calls, SUM(l.cost) as cost, "
            "SUM(l.prompt_tokens + l.completion_tokens) as tokens "
            "FROM llm_calls l JOIN sessions s ON l.session_id = s.id "
            "WHERE s.started_at >= ? GROUP BY l.category ORDER BY cost DESC", [start]
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT category, COUNT(*) as calls, SUM(cost) as cost, "
            "SUM(prompt_tokens + completion_tokens) as tokens "
            "FROM llm_calls GROUP BY category ORDER BY cost DESC"
        ).fetchall()
    return [
        {
            "category": r["category"],
            "calls": r["calls"],
            "cost": round(r["cost"], 4),
            "tokens": r["tokens"],
        }
        for r in rows
    ]


def get_sessions_list(period: str = "all") -> list[dict]:
    conn = get_db()
    where, params = _where_clause(period)
    rows = conn.execute(
        f"SELECT id, started_at, total_cost, llm_cost, tts_cost, "
        f"total_llm_calls, total_tts_calls, total_tokens "
        f"FROM sessions {where} ORDER BY started_at DESC", params
    ).fetchall()
    return [
        {
            "session_id": r["id"],
            "started_at": r["started_at"],
            "total_cost": round(r["total_cost"], 4),
            "llm_cost": round(r["llm_cost"], 4),
            "tts_cost": round(r["tts_cost"], 4),
            "total_llm_calls": r["total_llm_calls"],
            "total_tts_calls": r["total_tts_calls"],
            "total_tokens": r["total_tokens"],
        }
        for r in rows
    ]


def get_session_detail(session_id: str) -> dict | None:
    conn = get_db()
    session = conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if not session:
        return None

    by_caller = conn.execute(
        "SELECT caller_name, COUNT(*) as calls, SUM(cost) as cost, "
        "SUM(prompt_tokens + completion_tokens) as tokens "
        "FROM llm_calls WHERE session_id = ? AND caller_name != '' "
        "GROUP BY caller_name ORDER BY cost DESC", (session_id,)
    ).fetchall()

    by_model = conn.execute(
        "SELECT model, COUNT(*) as calls, SUM(cost) as cost, "
        "SUM(prompt_tokens) as prompt_tokens, SUM(completion_tokens) as completion_tokens "
        "FROM llm_calls WHERE session_id = ? GROUP BY model ORDER BY cost DESC", (session_id,)
    ).fetchall()

    by_category = conn.execute(
        "SELECT category, COUNT(*) as calls, SUM(cost) as cost, "
        "SUM(prompt_tokens + completion_tokens) as tokens "
        "FROM llm_calls WHERE session_id = ? GROUP BY category ORDER BY cost DESC", (session_id,)
    ).fetchall()

    expensive_calls = conn.execute(
        "SELECT timestamp, category, model, caller_name, cost, "
        "prompt_tokens, completion_tokens, latency_ms "
        "FROM llm_calls WHERE session_id = ? ORDER BY cost DESC LIMIT 10", (session_id,)
    ).fetchall()

    tts_by_provider = conn.execute(
        "SELECT provider, COUNT(*) as calls, SUM(cost) as cost, SUM(char_count) as chars "
        "FROM tts_calls WHERE session_id = ? GROUP BY provider ORDER BY cost DESC", (session_id,)
    ).fetchall()

    return {
        "session_id": session["id"],
        "started_at": session["started_at"],
        "total_cost": round(session["total_cost"], 4),
        "llm_cost": round(session["llm_cost"], 4),
        "tts_cost": round(session["tts_cost"], 4),
        "total_llm_calls": session["total_llm_calls"],
        "total_tts_calls": session["total_tts_calls"],
        "total_tokens": session["total_tokens"],
        "by_caller": [
            {"caller_name": r["caller_name"], "calls": r["calls"],
             "cost": round(r["cost"], 4), "tokens": r["tokens"]}
            for r in by_caller
        ],
        "by_model": [
            {"model": r["model"], "calls": r["calls"], "cost": round(r["cost"], 4),
             "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"]}
            for r in by_model
        ],
        "by_category": [
            {"category": r["category"], "calls": r["calls"],
             "cost": round(r["cost"], 4), "tokens": r["tokens"]}
            for r in by_category
        ],
        "expensive_calls": [
            {"timestamp": r["timestamp"], "category": r["category"], "model": r["model"],
             "caller_name": r["caller_name"], "cost": round(r["cost"], 6),
             "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
             "latency_ms": round(r["latency_ms"], 1)}
            for r in expensive_calls
        ],
        "tts_by_provider": [
            {"provider": r["provider"], "calls": r["calls"],
             "cost": round(r["cost"], 4), "chars": r["chars"]}
            for r in tts_by_provider
        ],
    }


def get_expensive_calls(period: str = "all", limit: int = 20) -> list[dict]:
    conn = get_db()
    start = _period_filter(period)
    if start:
        rows = conn.execute(
            "SELECT l.session_id, l.timestamp, l.category, l.model, l.caller_name, "
            "l.cost, l.prompt_tokens, l.completion_tokens, l.latency_ms "
            "FROM llm_calls l JOIN sessions s ON l.session_id = s.id "
            "WHERE s.started_at >= ? ORDER BY l.cost DESC LIMIT ?",
            [start, limit]
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT l.session_id, l.timestamp, l.category, l.model, l.caller_name, "
            "l.cost, l.prompt_tokens, l.completion_tokens, l.latency_ms "
            "FROM llm_calls l ORDER BY l.cost DESC LIMIT ?",
            [limit]
        ).fetchall()
    return [
        {
            "session_id": r["session_id"],
            "timestamp": r["timestamp"],
            "category": r["category"],
            "model": r["model"],
            "caller_name": r["caller_name"],
            "cost": round(r["cost"], 6),
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "latency_ms": round(r["latency_ms"], 1),
        }
        for r in rows
    ]
