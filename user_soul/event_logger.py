"""EventLogger — real user event tracking for AARRR computation.

Provides the real-data counterpart to AI-simulated AARRR scores.
Supports InMemory (testing), SQLite (local prod), and HTTP (external analytics) backends.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import threading
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from user_soul.models import FeatureAAR


@dataclass
class Event:
    user_id: str
    event_name: str
    value: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
    feature_id: str = ""


@dataclass
class AAARREventMap:
    """Maps AARRR dimensions to event names in your event log."""
    acquisition: str = "first_session"
    activation: str = "key_action"
    retention: str = "day7_return"
    revenue: str = "purchase"
    referral: str = "shared"
    feature_used: str = "feature_used"


class EventLoggerBackend(Protocol):
    def log(self, event: Event) -> None: ...

    def query(
        self,
        event_name: str | None = None,
        user_id: str | None = None,
        feature_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Event]: ...

    def unique_users(
        self,
        event_name: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> set[str]: ...

    def count_events(
        self,
        event_name: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int: ...


class InMemoryEventLoggerBackend:
    def __init__(self) -> None:
        self._events: list[Event] = []
        self._lock = threading.Lock()

    def log(self, event: Event) -> None:
        with self._lock:
            self._events.append(event)

    def query(
        self,
        event_name: str | None = None,
        user_id: str | None = None,
        feature_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Event]:
        with self._lock:
            results = list(self._events)
        if event_name is not None:
            results = [e for e in results if e.event_name == event_name]
        if user_id is not None:
            results = [e for e in results if e.user_id == user_id]
        if feature_id is not None:
            results = [e for e in results if e.feature_id == feature_id]
        if start is not None:
            results = [e for e in results if e.timestamp >= start]
        if end is not None:
            results = [e for e in results if e.timestamp <= end]
        return results

    def unique_users(
        self,
        event_name: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> set[str]:
        events = self.query(event_name=event_name, start=start, end=end)
        return {e.user_id for e in events}

    def count_events(
        self,
        event_name: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        return len(self.query(event_name=event_name, start=start, end=end))


class SQLiteEventLoggerBackend:
    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            event_name TEXT NOT NULL,
            value REAL DEFAULT 1.0,
            timestamp TEXT NOT NULL,
            feature_id TEXT DEFAULT '',
            metadata TEXT DEFAULT '{}'
        )
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(self._CREATE_TABLE)

    def log(self, event: Event) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO events (user_id, event_name, value, timestamp, feature_id, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        event.user_id,
                        event.event_name,
                        event.value,
                        event.timestamp.isoformat(),
                        event.feature_id,
                        json.dumps(event.metadata),
                    ),
                )

    def _row_to_event(self, row: sqlite3.Row) -> Event:
        return Event(
            user_id=row["user_id"],
            event_name=row["event_name"],
            value=row["value"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            feature_id=row["feature_id"] or "",
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def query(
        self,
        event_name: str | None = None,
        user_id: str | None = None,
        feature_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Event]:
        clauses: list[str] = []
        params: list[object] = []
        if event_name is not None:
            clauses.append("event_name = ?")
            params.append(event_name)
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if feature_id is not None:
            clauses.append("feature_id = ?")
            params.append(feature_id)
        if start is not None:
            clauses.append("timestamp >= ?")
            params.append(start.isoformat())
        if end is not None:
            clauses.append("timestamp <= ?")
            params.append(end.isoformat())

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM events {where} ORDER BY timestamp ASC"

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    def unique_users(
        self,
        event_name: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> set[str]:
        events = self.query(event_name=event_name, start=start, end=end)
        return {e.user_id for e in events}

    def count_events(
        self,
        event_name: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        return len(self.query(event_name=event_name, start=start, end=end))


class HTTPEventLoggerBackend:
    def __init__(self, url: str, headers: dict | None = None) -> None:
        self._url = url
        self._headers = headers or {}

    def log(self, event: Event) -> None:
        payload = json.dumps(
            {
                "user_id": event.user_id,
                "event_name": event.event_name,
                "value": event.value,
                "timestamp": event.timestamp.isoformat(),
                "feature_id": event.feature_id,
                "metadata": event.metadata,
            }
        ).encode()
        try:
            req = urllib.request.Request(
                self._url,
                data=payload,
                method="POST",
                headers={"Content-Type": "application/json", **self._headers},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:
            print(f"[EventLogger] HTTP log failed: {exc}", file=sys.stderr)

    def query(
        self,
        event_name: str | None = None,
        user_id: str | None = None,
        feature_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Event]:
        return []

    def unique_users(
        self,
        event_name: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> set[str]:
        return set()

    def count_events(
        self,
        event_name: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        return 0


class EventLogger:
    def __init__(self, backend: EventLoggerBackend | None = None) -> None:
        self._backend = backend or InMemoryEventLoggerBackend()

    def log_event(
        self,
        user_id: str,
        event_name: str,
        value: float = 1.0,
        feature_id: str = "",
        **metadata: object,
    ) -> None:
        self._backend.log(
            Event(
                user_id=user_id,
                event_name=event_name,
                value=value,
                feature_id=feature_id,
                metadata=dict(metadata),
            )
        )

    def log_batch(self, events: list[Event]) -> None:
        for event in events:
            self._backend.log(event)

    def compute_aarrr(
        self,
        feature_id: str,
        event_map: AAARREventMap | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        min_sample: int = 10,
    ) -> FeatureAAR:
        em = event_map or AAARREventMap()

        all_events = self._backend.query(start=start, end=end)
        total_unique_users = len({e.user_id for e in all_events})

        feature_events = self._backend.query(feature_id=feature_id, start=start, end=end)
        sample_size = len(feature_events)

        if sample_size < min_sample:
            return FeatureAAR(
                feature_id=feature_id,
                acquisition=0.5,
                activation=0.5,
                retention=0.5,
                revenue=0.5,
                referral=0.5,
                confidence=0.0,
                archetype_votes={},
            )

        users_with_acquisition = self._backend.unique_users(
            event_name=em.acquisition, start=start, end=end
        )
        users_with_activation_event = self._backend.unique_users(
            event_name=em.activation, start=start, end=end
        )
        users_with_feature = {e.user_id for e in feature_events}
        users_with_activation = users_with_activation_event & users_with_feature
        users_with_retention = self._backend.unique_users(
            event_name=em.retention, start=start, end=end
        )
        users_with_referral = self._backend.unique_users(
            event_name=em.referral, start=start, end=end
        )
        revenue_events = self._backend.query(event_name=em.revenue, start=start, end=end)
        total_revenue_value = sum(e.value for e in revenue_events)

        acquisition = len(users_with_acquisition) / max(1, total_unique_users)
        activation = len(users_with_activation) / max(1, len(users_with_acquisition))
        retention = len(users_with_retention) / max(1, len(users_with_activation))
        revenue = total_revenue_value / max(1, total_unique_users)
        referral = len(users_with_referral) / max(1, total_unique_users)

        def clamp(v: float) -> float:
            return max(0.0, min(1.0, v))

        confidence = min(1.0, sample_size / 100.0)

        return FeatureAAR(
            feature_id=feature_id,
            acquisition=clamp(acquisition),
            activation=clamp(activation),
            retention=clamp(retention),
            revenue=clamp(revenue),
            referral=clamp(referral),
            confidence=confidence,
            archetype_votes={},
        )

    def query_metric(
        self,
        metric_name: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict:
        events = self._backend.query(event_name=metric_name, start=start, end=end)
        unique_users = {e.user_id for e in events}
        total_value = sum(e.value for e in events)
        mean_value = total_value / len(events) if events else 0.0
        return {
            "count": len(events),
            "unique_users": len(unique_users),
            "total_value": total_value,
            "mean_value": mean_value,
        }

    def export_events(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Event]:
        return self._backend.query(start=start, end=end)
