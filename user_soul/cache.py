"""Simulation result caching — three-tier hot/warm/cold architecture."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path

from user_soul.simulator import FeatureSignal

_CACHE_FILE = "simulation_cache.json"
_META_FILE = "simulation_meta.json"


def _feature_hash(features: list[dict]) -> str:
    """Stable hash of feature ID set — order independent."""
    ids = sorted(f["id"] for f in features)
    return hashlib.md5(json.dumps(ids).encode()).hexdigest()[:12]


def save_simulation_cache(
    state_dir: Path,
    features: list[dict],
    signals: list[FeatureSignal],
    status: str = "complete",
) -> None:
    """Write signals + metadata to state_dir."""
    signals_data = [dataclasses.asdict(s) for s in signals]
    (state_dir / _CACHE_FILE).write_text(
        json.dumps(signals_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "feature_hash": _feature_hash(features),
        "n_simulations": signals[0].n_simulations if signals else 0,
        "status": status,  # "partial" | "complete"
    }
    (state_dir / _META_FILE).write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def load_simulation_cache(
    state_dir: Path,
    features: list[dict],
) -> list[FeatureSignal] | None:
    """Load cached signals if feature hash matches. Returns None on any miss."""
    cache_path = state_dir / _CACHE_FILE
    meta_path = state_dir / _META_FILE
    if not cache_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("feature_hash") != _feature_hash(features):
            return None
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return [FeatureSignal(**item) for item in data]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
