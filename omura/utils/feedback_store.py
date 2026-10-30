"""SQLite store for qualitative community feedback.

Feedback is deliberately kept in its own database so it can be exported, retained
and audited independently of the search catalog and analytics event stream.

Privacy notes
-------------
Only a salted SHA-256 of the client IP is ever persisted (``user_hash``); the raw
IP never reaches this module. Callers are responsible for hashing — see
``omura.routes.feedback``.

The database is created lazily on first use (import-safe) and every operation
opens its own short-lived connection, which keeps the store safe to call from
FastAPI's threadpool without sharing a connection across threads.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

DEFAULT_DB_PATH = Path(os.getenv("OMURA_FEEDBACK_DB_PATH", "data/feedback.sqlite"))

TARGET_TYPES = ("search", "result", "general", "report")
SENTIMENTS = ("up", "down")
STATUSES = ("new", "triaged", "resolved", "spam")

MAX_COMMENT_LEN = 4000
MAX_CONTACT_LEN = 320
MAX_TAGS = 8
MAX_TAG_LEN = 64

_CREATE_FEEDBACK = """
CREATE TABLE IF NOT EXISTS feedback (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    target_type TEXT NOT NULL DEFAULT 'general',
    target_id   TEXT,
    query       TEXT,
    rating      INTEGER,
    sentiment   TEXT,
    tags        TEXT NOT NULL DEFAULT '[]',
    comment     TEXT,
    contact     TEXT,
    page        TEXT,
    user_hash   TEXT,
    country     TEXT,
    user_agent  TEXT,
    status      TEXT NOT NULL DEFAULT 'new',
    meta        TEXT NOT NULL DEFAULT '{}'
)
"""

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status)",
)

_SELECT_COLUMNS = (
    "id, created_at, updated_at, target_type, target_id, query, rating, "
    "sentiment, tags, comment, contact, page, user_hash, country, user_agent, "
    "status, meta"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decode_row(row: sqlite3.Row) -> Dict[str, Any]:
    record = dict(row)
    try:
        record["tags"] = json.loads(record.get("tags") or "[]")
    except (TypeError, ValueError):
        record["tags"] = []
    try:
        record["meta"] = json.loads(record.get("meta") or "{}")
    except (TypeError, ValueError):
        record["meta"] = {}
    return record


def _validate_status(status: str) -> str:
    if status not in STATUSES:
        raise ValueError(f"invalid status {status!r}; expected one of {STATUSES}")
    return status


def _validate_target_type(target_type: str) -> str:
    if target_type not in TARGET_TYPES:
        raise ValueError(
            f"invalid target_type {target_type!r}; expected one of {TARGET_TYPES}"
        )
    return target_type


def _validate_sentiment(sentiment: Optional[str]) -> Optional[str]:
    if sentiment is not None and sentiment not in SENTIMENTS:
        raise ValueError(
            f"invalid sentiment {sentiment!r}; expected one of {SENTIMENTS}"
        )
    return sentiment


class FeedbackStore:
    """Thread-safe (connection-per-call) SQLite-backed feedback store."""

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        *,
        timeout: float = 30.0,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.timeout = timeout
        self._schema_lock = threading.Lock()
        self._schema_ready = False

    # ── connection / schema ────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self.db_path), timeout=self.timeout, check_same_thread=False
            )
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(_CREATE_FEEDBACK)
                for index_sql in _INDEXES:
                    conn.execute(index_sql)
                conn.commit()
            finally:
                conn.close()
            self._schema_ready = True

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._ensure_schema()
        conn = sqlite3.connect(
            str(self.db_path), timeout=self.timeout, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout = 30000")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── writes ─────────────────────────────────────────────────────────────

    def add(
        self,
        *,
        target_type: str = "general",
        target_id: Optional[str] = None,
        query: Optional[str] = None,
        rating: Optional[int] = None,
        sentiment: Optional[str] = None,
        tags: Optional[List[str]] = None,
        comment: Optional[str] = None,
        contact: Optional[str] = None,
        page: Optional[str] = None,
        user_hash: Optional[str] = None,
        country: Optional[str] = None,
        user_agent: Optional[str] = None,
        status: str = "new",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        _validate_target_type(target_type)
        _validate_sentiment(sentiment)
        _validate_status(status)

        if rating is not None and (not isinstance(rating, int) or not 1 <= rating <= 5):
            raise ValueError("rating must be an integer between 1 and 5")
        if comment is not None and len(comment) > MAX_COMMENT_LEN:
            raise ValueError(f"comment exceeds {MAX_COMMENT_LEN} characters")
        if contact is not None and len(contact) > MAX_CONTACT_LEN:
            raise ValueError(f"contact exceeds {MAX_CONTACT_LEN} characters")

        clean_tags = list(tags or [])
        if len(clean_tags) > MAX_TAGS:
            raise ValueError(f"at most {MAX_TAGS} tags are allowed")
        for tag in clean_tags:
            if not isinstance(tag, str):
                raise ValueError("tags must be strings")
            if len(tag) > MAX_TAG_LEN:
                raise ValueError(f"tag exceeds {MAX_TAG_LEN} characters")

        meta_obj = meta if meta is not None else {}
        if not isinstance(meta_obj, dict):
            raise ValueError("meta must be a JSON object")

        feedback_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback (
                    id, created_at, updated_at, target_type, target_id, query,
                    rating, sentiment, tags, comment, contact, page, user_hash,
                    country, user_agent, status, meta
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    now,
                    now,
                    target_type,
                    target_id,
                    query,
                    rating,
                    sentiment,
                    json.dumps(clean_tags, ensure_ascii=False),
                    comment,
                    contact,
                    page,
                    user_hash,
                    country,
                    user_agent,
                    status,
                    json.dumps(meta_obj, ensure_ascii=False, separators=(",", ":")),
                ),
            )
        record = self.get(feedback_id)
        assert record is not None
        return record

    def update_status(self, feedback_id: str, status: str) -> Dict[str, Any]:
        _validate_status(status)
        now = _now()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE feedback SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, feedback_id),
            )
            if cur.rowcount == 0:
                raise KeyError(feedback_id)
        record = self.get(feedback_id)
        assert record is not None
        return record

    # ── reads ──────────────────────────────────────────────────────────────

    def get(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_SELECT_COLUMNS} FROM feedback WHERE id = ?",
                (feedback_id,),
            ).fetchone()
        return _decode_row(row) if row is not None else None

    def list(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        if status is not None:
            _validate_status(status)
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))

        sql = f"SELECT {_SELECT_COLUMNS} FROM feedback"
        params: List[Any] = []
        if status is not None:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_decode_row(row) for row in rows]

    def count(self, status: Optional[str] = None) -> int:
        if status is not None:
            _validate_status(status)
        sql = "SELECT COUNT(*) FROM feedback"
        params: List[Any] = []
        if status is not None:
            sql += " WHERE status = ?"
            params.append(status)
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row[0]) if row is not None else 0

    def summary(self, days: int = 30) -> Dict[str, Any]:
        days = max(1, int(days))
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=days)
        since_iso = since.isoformat()

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, sentiment, target_type, rating, tags, created_at "
                "FROM feedback WHERE created_at >= ?",
                (since_iso,),
            ).fetchall()

        by_status = {name: 0 for name in STATUSES}
        by_sentiment = {name: 0 for name in SENTIMENTS}
        by_target_type = {name: 0 for name in TARGET_TYPES}
        tag_counts: Dict[str, int] = {}
        daily_counts: Dict[str, int] = {}
        ratings: List[int] = []

        for row in rows:
            status = row["status"]
            if status in by_status:
                by_status[status] += 1
            sentiment = row["sentiment"]
            if sentiment in by_sentiment:
                by_sentiment[sentiment] += 1
            target_type = row["target_type"]
            if target_type in by_target_type:
                by_target_type[target_type] += 1
            if row["rating"] is not None:
                ratings.append(int(row["rating"]))
            try:
                tags = json.loads(row["tags"] or "[]")
            except (TypeError, ValueError):
                tags = []
            for tag in tags:
                if isinstance(tag, str) and tag:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            created = str(row["created_at"])
            day = created[:10]
            daily_counts[day] = daily_counts.get(day, 0) + 1

        daily: List[Dict[str, Any]] = []
        cursor = since.date()
        end = now.date()
        while cursor <= end:
            key = cursor.isoformat()
            daily.append({"date": key, "count": daily_counts.get(key, 0)})
            cursor += timedelta(days=1)

        top_tags = [
            {"tag": tag, "count": count}
            for tag, count in sorted(
                tag_counts.items(), key=lambda item: (-item[1], item[0])
            )[:20]
        ]

        return {
            "days": days,
            "since": since_iso,
            "total": len(rows),
            "by_status": by_status,
            "by_sentiment": by_sentiment,
            "by_target_type": by_target_type,
            "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
            "rating_count": len(ratings),
            "top_tags": top_tags,
            "daily": daily,
        }

    # ── export ─────────────────────────────────────────────────────────────

    def export_jsonl(
        self,
        path: Union[str, Path],
        *,
        status: Optional[str] = None,
    ) -> int:
        """Write every matching record as JSON Lines. Returns the row count."""
        if status is not None:
            _validate_status(status)
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        sql = f"SELECT {_SELECT_COLUMNS} FROM feedback"
        params: List[Any] = []
        if status is not None:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at ASC, id ASC"

        written = 0
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(_decode_row(row), ensure_ascii=False, sort_keys=True)
                )
                handle.write("\n")
                written += 1
        return written


__all__ = [
    "DEFAULT_DB_PATH",
    "FeedbackStore",
    "MAX_COMMENT_LEN",
    "MAX_CONTACT_LEN",
    "MAX_TAGS",
    "SENTIMENTS",
    "STATUSES",
    "TARGET_TYPES",
]
