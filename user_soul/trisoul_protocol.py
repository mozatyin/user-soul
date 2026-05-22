"""TriSoul Unified Message Protocol.

Shared by all four actors: PM-Soul, Code-Soul, ELTM, User-Soul.

The wire format is a plain dict with 3 mandatory envelope fields:
    {"id": uuid, "source": actor, "context_id": run_id, "type": ..., ...}

Actors return:
    {"acknowledged": bool, "ref_id": original_msg_id, ...}

TriSoulMessage is an OPTIONAL helper for constructing and parsing these dicts.
Actors never need to import it — they just see dicts with a few extra keys.
"""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


# ---------------------------------------------------------------------------
# Enums — shared vocabulary for all actors
# ---------------------------------------------------------------------------

class Actor(str, Enum):
    PM_SOUL = "pm-soul"
    CODE_SOUL = "code-soul"
    ELTM = "eltm"
    USER_SOUL = "user-soul"


class Severity(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


# ---------------------------------------------------------------------------
# make_dict — the primary API: produces a standard dict with envelope fields
# ---------------------------------------------------------------------------

def make_dict(
    type: str,
    source: str,
    context_id: str,
    *,
    id: str | None = None,
    severity: str | None = None,
    ref_id: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Build a protocol-compliant message dict.

    Every message in TriSoul carries 3 envelope fields:
      - id:         UUID for tracking and reply correlation
      - source:     who sent this ("pm-soul", "code-soul", "eltm", "user-soul")
      - context_id: pipeline run ID (e.g. "checkers_v3")

    Plus the existing "type" discriminator and all domain-specific fields.

    Usage:
        msg = make_dict("focus_area", "pm-soul", "run_1", area="behavior", current_score=0.3)
        response = codesoul.receive_from_pm(msg)
        assert response.get("ref_id") == msg["id"]
    """
    d: dict[str, Any] = {
        "id": id or str(uuid4()),
        "type": type,
        "source": source,
        "context_id": context_id,
    }
    if severity:
        d["severity"] = severity
    if ref_id:
        d["ref_id"] = ref_id
    d.update(fields)
    return d


def make_reply(
    original: dict[str, Any],
    *,
    acknowledged: bool = True,
    **fields: Any,
) -> dict[str, Any]:
    """Build a standard reply dict linked to the original message."""
    d: dict[str, Any] = {
        "acknowledged": acknowledged,
        "ref_id": original.get("id", ""),
    }
    d.update(fields)
    return d


def check_acknowledged(
    response: dict[str, Any],
    msg: dict[str, Any],
    log_fn=None,
) -> bool:
    """Check if a dispatch was acknowledged; log warning if not."""
    ack = response.get("acknowledged", False)
    if not ack and log_fn:
        target = msg.get("target", "?")
        msg_type = msg.get("type", "?")
        reason = response.get("response", response.get("reason", "unknown"))
        log_fn(f"  ⚠ {target} did not acknowledge '{msg_type}': {reason}")
    return ack


# ---------------------------------------------------------------------------
# TriSoulMessage — optional helper class for complex construction/parsing
# ---------------------------------------------------------------------------

@dataclass
class TriSoulMessage:
    """Optional wrapper for constructing protocol-compliant message dicts.

    Actors never receive this object — they receive the dict from to_dict().
    Use this when you need reply(), not_understood(), or structured construction.
    """
    type: str
    source: str
    target: str
    content: dict[str, Any]
    context_id: str

    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    ref_id: str | None = None
    severity: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Flatten to wire-format dict compatible with receive_from_pm()."""
        d: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "context_id": self.context_id,
        }
        if self.severity:
            d["severity"] = self.severity
        if self.ref_id:
            d["ref_id"] = self.ref_id
        d.update(self.content)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TriSoulMessage:
        envelope_keys = {"id", "type", "source", "target", "context_id",
                         "severity", "ref_id", "timestamp"}
        content = {k: v for k, v in d.items() if k not in envelope_keys}
        return cls(
            id=d.get("id", str(uuid4())),
            type=d.get("type", ""),
            source=d.get("source", ""),
            target=d.get("target", ""),
            content=content,
            context_id=d.get("context_id", ""),
            timestamp=d.get("timestamp", ""),
            ref_id=d.get("ref_id"),
            severity=d.get("severity"),
        )

    def reply(self, *, acknowledged: bool = True, **fields: Any) -> dict[str, Any]:
        return {"acknowledged": acknowledged, "ref_id": self.id, **fields}

    def not_understood(self, reason: str = "") -> dict[str, Any]:
        return {
            "acknowledged": False,
            "ref_id": self.id,
            "response": f"Unknown message type: {self.type}",
            "reason": reason,
        }
