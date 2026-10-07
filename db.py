"""SQLite persistence for meetings and their chat history."""
import json
import os
import sqlite3
from datetime import datetime, timezone

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "meetings.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT NOT NULL,
    transcript    TEXT NOT NULL,
    source_type   TEXT NOT NULL DEFAULT 'text',   -- text | file | audio
    source_name   TEXT,
    analysis_json TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id  INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_meeting ON chat_messages(meeting_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_conn(path: str | None = None) -> sqlite3.Connection:
    path = path or os.getenv("DATABASE_PATH") or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path: str | None = None) -> None:
    with get_conn(path) as conn:
        conn.executescript(SCHEMA)


def _row_to_meeting(row: sqlite3.Row, include_transcript: bool = True) -> dict:
    data = dict(row)
    data["analysis"] = json.loads(data.pop("analysis_json"))
    if not include_transcript:
        data.pop("transcript", None)
    return data


def create_meeting(title, transcript, analysis, source_type="text", source_name=None, path=None) -> int:
    now = _now()
    with get_conn(path) as conn:
        cur = conn.execute(
            "INSERT INTO meetings (title, transcript, source_type, source_name, analysis_json, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (title, transcript, source_type, source_name, json.dumps(analysis), now, now),
        )
        return cur.lastrowid


def list_meetings(path=None) -> list[dict]:
    with get_conn(path) as conn:
        rows = conn.execute(
            "SELECT id, title, source_type, analysis_json, created_at, updated_at,"
            " LENGTH(transcript) AS transcript_chars FROM meetings ORDER BY created_at DESC, id DESC"
        ).fetchall()
    out = []
    for r in rows:
        analysis = json.loads(r["analysis_json"])
        items = analysis.get("action_items", [])
        out.append({
            "id": r["id"],
            "title": r["title"],
            "source_type": r["source_type"],
            "created_at": r["created_at"],
            "sentiment": analysis.get("sentiment", {}).get("label"),
            "open_actions": sum(1 for i in items if not i.get("done")),
            "transcript_chars": r["transcript_chars"],
        })
    return out


def get_meeting(meeting_id: int, path=None) -> dict | None:
    with get_conn(path) as conn:
        row = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
    return _row_to_meeting(row) if row else None


def update_meeting(meeting_id: int, title=None, analysis=None, path=None) -> None:
    fields, values = [], []
    if title is not None:
        fields.append("title = ?")
        values.append(title)
    if analysis is not None:
        fields.append("analysis_json = ?")
        values.append(json.dumps(analysis))
    if not fields:
        return
    fields.append("updated_at = ?")
    values.extend([_now(), meeting_id])
    with get_conn(path) as conn:
        conn.execute(f"UPDATE meetings SET {', '.join(fields)} WHERE id = ?", values)


def delete_meeting(meeting_id: int, path=None) -> bool:
    with get_conn(path) as conn:
        cur = conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
        return cur.rowcount > 0


def add_message(meeting_id: int, role: str, content: str, path=None) -> dict:
    now = _now()
    with get_conn(path) as conn:
        cur = conn.execute(
            "INSERT INTO chat_messages (meeting_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (meeting_id, role, content, now),
        )
        return {"id": cur.lastrowid, "role": role, "content": content, "created_at": now}


def get_messages(meeting_id: int, path=None) -> list[dict]:
    with get_conn(path) as conn:
        rows = conn.execute(
            "SELECT id, role, content, created_at FROM chat_messages WHERE meeting_id = ? ORDER BY id",
            (meeting_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def clear_messages(meeting_id: int, path=None) -> None:
    with get_conn(path) as conn:
        conn.execute("DELETE FROM chat_messages WHERE meeting_id = ?", (meeting_id,))
