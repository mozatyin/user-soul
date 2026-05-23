from __future__ import annotations

import os
import tempfile
import threading
from datetime import datetime, timedelta

import pytest

from user_soul.event_logger import (
    AAARREventMap,
    Event,
    EventLogger,
    HTTPEventLoggerBackend,
    InMemoryEventLoggerBackend,
    SQLiteEventLoggerBackend,
)
from user_soul.models import FeatureAAR


def _make_logger() -> EventLogger:
    return EventLogger()


# ---------------------------------------------------------------------------
# InMemory basic
# ---------------------------------------------------------------------------

def test_log_single_event():
    logger = _make_logger()
    logger.log_event("u1", "session_started", feature_id="f1")
    events = logger.export_events()
    assert len(events) == 1
    assert events[0].user_id == "u1"
    assert events[0].event_name == "session_started"
    assert events[0].feature_id == "f1"


def test_log_batch():
    logger = _make_logger()
    batch = [
        Event(user_id="u1", event_name="click"),
        Event(user_id="u2", event_name="click"),
        Event(user_id="u3", event_name="purchase", value=9.99),
    ]
    logger.log_batch(batch)
    events = logger.export_events()
    assert len(events) == 3


def test_inmemory_thread_safety():
    backend = InMemoryEventLoggerBackend()
    logger = EventLogger(backend=backend)
    errors: list[Exception] = []

    def worker(uid: int) -> None:
        try:
            for _ in range(20):
                logger.log_event(f"u{uid}", "ping")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(logger.export_events()) == 200


# ---------------------------------------------------------------------------
# Query filters
# ---------------------------------------------------------------------------

def test_query_by_event_name():
    logger = _make_logger()
    logger.log_event("u1", "click")
    logger.log_event("u2", "purchase")
    logger.log_event("u3", "click")
    results = logger._backend.query(event_name="click")
    assert len(results) == 2
    assert all(e.event_name == "click" for e in results)


def test_query_by_user_id():
    logger = _make_logger()
    logger.log_event("alice", "click")
    logger.log_event("bob", "click")
    logger.log_event("alice", "purchase")
    results = logger._backend.query(user_id="alice")
    assert len(results) == 2
    assert all(e.user_id == "alice" for e in results)


def test_query_by_feature_id():
    logger = _make_logger()
    logger.log_event("u1", "feature_used", feature_id="feat_A")
    logger.log_event("u2", "feature_used", feature_id="feat_B")
    logger.log_event("u3", "feature_used", feature_id="feat_A")
    results = logger._backend.query(feature_id="feat_A")
    assert len(results) == 2
    assert all(e.feature_id == "feat_A" for e in results)


def test_query_by_time_range():
    logger = _make_logger()
    base = datetime(2025, 1, 1, 12, 0, 0)
    for i in range(5):
        logger._backend.log(Event("u1", "ping", timestamp=base + timedelta(hours=i)))
    start = base + timedelta(hours=1)
    end = base + timedelta(hours=3)
    results = logger._backend.query(start=start, end=end)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# Unique users / count
# ---------------------------------------------------------------------------

def test_unique_users():
    logger = _make_logger()
    for _ in range(3):
        logger.log_event("u1", "view")
    logger.log_event("u2", "view")
    logger.log_event("u3", "click")
    users = logger._backend.unique_users(event_name="view")
    assert users == {"u1", "u2"}


# ---------------------------------------------------------------------------
# compute_aarrr
# ---------------------------------------------------------------------------

def _populate_aarrr(logger: EventLogger, feature_id: str, n_users: int = 20) -> None:
    """Populate events so all AARRR dimensions have signal."""
    em = AAARREventMap()
    for i in range(n_users):
        uid = f"user_{i}"
        logger.log_event(uid, em.acquisition, feature_id=feature_id)
        if i < n_users * 0.8:
            logger.log_event(uid, em.activation, feature_id=feature_id)
        if i < n_users * 0.5:
            logger.log_event(uid, em.retention, feature_id=feature_id)
        if i < n_users * 0.3:
            logger.log_event(uid, em.revenue, value=1.99, feature_id=feature_id)
        if i < n_users * 0.2:
            logger.log_event(uid, em.referral, feature_id=feature_id)


def test_compute_aarrr_sufficient_data():
    logger = _make_logger()
    _populate_aarrr(logger, "feat_x", n_users=20)
    aar = logger.compute_aarrr("feat_x")
    assert isinstance(aar, FeatureAAR)
    assert aar.feature_id == "feat_x"
    assert aar.confidence > 0.0
    assert aar.acquisition > 0.0


def test_compute_aarrr_insufficient_data():
    logger = _make_logger()
    for i in range(5):
        logger.log_event(f"u{i}", "feature_used", feature_id="tiny")
    aar = logger.compute_aarrr("tiny")
    assert aar.confidence == 0.0
    assert aar.acquisition == 0.5
    assert aar.activation == 0.5


def test_compute_aarrr_acquisition_rate():
    logger = _make_logger()
    em = AAARREventMap()
    # 20 total users, 10 have acquisition event, all tagged to feature
    for i in range(20):
        logger.log_event(f"u{i}", "other_event", feature_id="f")
    for i in range(10):
        logger.log_event(f"u{i}", em.acquisition, feature_id="f")
    aar = logger.compute_aarrr("f", min_sample=1)
    assert pytest.approx(aar.acquisition, abs=0.1) == 10 / 20


def test_compute_aarrr_clamped_values():
    logger = _make_logger()
    em = AAARREventMap()
    # Force revenue > 1 per user to verify clamping
    for i in range(20):
        logger.log_event(f"u{i}", em.acquisition, feature_id="f2")
        logger.log_event(f"u{i}", em.revenue, value=100.0, feature_id="f2")
    aar = logger.compute_aarrr("f2", min_sample=1)
    assert 0.0 <= aar.acquisition <= 1.0
    assert 0.0 <= aar.activation <= 1.0
    assert 0.0 <= aar.retention <= 1.0
    assert 0.0 <= aar.revenue <= 1.0
    assert 0.0 <= aar.referral <= 1.0


def test_compute_aarrr_confidence_scales_with_sample():
    em = AAARREventMap()
    logger_small = _make_logger()
    logger_large = _make_logger()
    for i in range(15):
        logger_small.log_event(f"u{i}", em.acquisition, feature_id="f")
    for i in range(80):
        logger_large.log_event(f"u{i}", em.acquisition, feature_id="f")
    aar_small = logger_small.compute_aarrr("f", min_sample=1)
    aar_large = logger_large.compute_aarrr("f", min_sample=1)
    assert aar_large.confidence > aar_small.confidence


# ---------------------------------------------------------------------------
# query_metric
# ---------------------------------------------------------------------------

def test_query_metric_returns_correct_stats():
    logger = _make_logger()
    logger.log_event("u1", "purchase", value=10.0)
    logger.log_event("u2", "purchase", value=20.0)
    logger.log_event("u1", "purchase", value=5.0)
    stats = logger.query_metric("purchase")
    assert stats["count"] == 3
    assert stats["unique_users"] == 2
    assert pytest.approx(stats["total_value"]) == 35.0
    assert pytest.approx(stats["mean_value"]) == 35.0 / 3


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

def test_sqlite_backend_persists_events():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        backend1 = SQLiteEventLoggerBackend(db_path)
        logger1 = EventLogger(backend=backend1)
        logger1.log_event("u1", "login", feature_id="fx")
        logger1.log_event("u2", "login", feature_id="fx")

        # Reopen with fresh backend pointing at same file
        backend2 = SQLiteEventLoggerBackend(db_path)
        logger2 = EventLogger(backend=backend2)
        events = logger2.export_events()
        assert len(events) == 2
        assert {e.user_id for e in events} == {"u1", "u2"}
    finally:
        os.unlink(db_path)


def test_sqlite_backend_query_filter():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        backend = SQLiteEventLoggerBackend(db_path)
        logger = EventLogger(backend=backend)
        logger.log_event("alice", "click", feature_id="A")
        logger.log_event("bob", "purchase", feature_id="A")
        logger.log_event("alice", "click", feature_id="B")

        by_name = backend.query(event_name="click")
        assert len(by_name) == 2

        by_feature = backend.query(feature_id="A")
        assert len(by_feature) == 2

        by_user = backend.query(user_id="alice")
        assert len(by_user) == 2
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# HTTP backend
# ---------------------------------------------------------------------------

def test_http_backend_silent_on_failure():
    backend = HTTPEventLoggerBackend(url="http://localhost:19999/nonexistent")
    logger = EventLogger(backend=backend)
    # Must not raise
    logger.log_event("u1", "ping")


# ---------------------------------------------------------------------------
# Default backend
# ---------------------------------------------------------------------------

def test_event_logger_default_inmemory():
    logger = EventLogger()
    assert isinstance(logger._backend, InMemoryEventLoggerBackend)
    logger.log_event("u1", "test_event")
    assert len(logger.export_events()) == 1
