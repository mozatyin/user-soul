"""Tests for the TriSoul unified message protocol (v2 — dict-based)."""
import sys
sys.path.insert(0, "/Users/mozat/user-soul")

from user_soul.trisoul_protocol import (
    Actor, Severity,
    TriSoulMessage,
    make_dict, make_reply, check_acknowledged,
)


def test_make_dict_envelope():
    d = make_dict("focus_area", "pm-soul", "run_1", area="behavior", current_score=0.3)
    assert d["type"] == "focus_area"
    assert d["source"] == "pm-soul"
    assert d["context_id"] == "run_1"
    assert d["area"] == "behavior"
    assert d["current_score"] == 0.3
    assert "id" in d
    assert len(d["id"]) > 10


def test_make_dict_with_severity():
    d = make_dict("user_verdict", Actor.USER_SOUL, "ctx", severity="P0", verdict="CRITICAL")
    assert d["severity"] == "P0"
    assert d["verdict"] == "CRITICAL"


def test_make_dict_no_severity():
    d = make_dict("user_verdict", Actor.USER_SOUL, "ctx", verdict="PASS")
    assert "severity" not in d


def test_make_dict_with_ref_id():
    d = make_dict("user_test_feedback", Actor.PM_SOUL, "ctx", ref_id="parent_123")
    assert d["ref_id"] == "parent_123"


def test_make_reply():
    original = make_dict("focus_area", "pm-soul", "ctx")
    reply = make_reply(original, status="applied", new_focus="behavior")
    assert reply["acknowledged"] is True
    assert reply["ref_id"] == original["id"]
    assert reply["status"] == "applied"


def test_make_reply_negative():
    original = make_dict("unknown_type", "pm-soul", "ctx")
    reply = make_reply(original, acknowledged=False, reason="unrecognized")
    assert reply["acknowledged"] is False
    assert reply["ref_id"] == original["id"]
    assert reply["reason"] == "unrecognized"


def test_check_acknowledged_true():
    msg = make_dict("test", "pm-soul", "ctx")
    resp = {"acknowledged": True, "ref_id": msg["id"]}
    assert check_acknowledged(resp, msg) is True


def test_check_acknowledged_false_logs():
    logged = []
    msg = make_dict("test", "pm-soul", "ctx", target="code-soul")
    resp = {"acknowledged": False, "reason": "not connected"}
    assert check_acknowledged(resp, msg, log_fn=logged.append) is False
    assert len(logged) == 1
    assert "not connected" in logged[0]


def test_trisoul_message_to_dict():
    msg = TriSoulMessage(
        type="focus_area",
        source="pm-soul",
        target="code-soul",
        content={"area": "behavior", "current_score": 0.3},
        context_id="ctx",
        severity="P0",
    )
    d = msg.to_dict()
    assert d["type"] == "focus_area"
    assert d["source"] == "pm-soul"
    assert d["context_id"] == "ctx"
    assert d["severity"] == "P0"
    assert d["area"] == "behavior"
    assert "id" in d


def test_trisoul_message_from_dict():
    d = {
        "id": "test-id-123",
        "type": "stagnation_alert",
        "source": "pm-soul",
        "target": "eltm",
        "context_id": "run_5",
        "severity": "P1",
        "recurring_bugs": ["bug1", "bug2"],
    }
    msg = TriSoulMessage.from_dict(d)
    assert msg.id == "test-id-123"
    assert msg.type == "stagnation_alert"
    assert msg.source == "pm-soul"
    assert msg.target == "eltm"
    assert msg.context_id == "run_5"
    assert msg.severity == "P1"
    assert msg.content["recurring_bugs"] == ["bug1", "bug2"]


def test_trisoul_message_roundtrip():
    msg = TriSoulMessage(
        type="user_test_feedback",
        source="user-soul",
        target="code-soul",
        content={"action_type": "fix", "description": "broken button"},
        context_id="ctx",
        ref_id="parent-verdict",
    )
    d = msg.to_dict()
    msg2 = TriSoulMessage.from_dict(d)
    assert msg2.type == msg.type
    assert msg2.source == msg.source
    assert msg2.ref_id == msg.ref_id
    assert msg2.content["action_type"] == "fix"


def test_trisoul_message_reply():
    msg = TriSoulMessage(
        type="fix_request",
        source="pm-soul",
        target="code-soul",
        content={"bugs": ["b1"]},
        context_id="ctx",
    )
    reply = msg.reply(status="fixed")
    assert reply["acknowledged"] is True
    assert reply["ref_id"] == msg.id
    assert reply["status"] == "fixed"


def test_trisoul_message_not_understood():
    msg = TriSoulMessage(
        type="unknown/v99",
        source="pm-soul",
        target="eltm",
        content={},
        context_id="ctx",
    )
    nu = msg.not_understood("unrecognized content type")
    assert nu["acknowledged"] is False
    assert nu["ref_id"] == msg.id
    assert "Unknown message type" in nu["response"]


def test_all_actors():
    assert set(a.value for a in Actor) == {"pm-soul", "code-soul", "eltm", "user-soul"}


def test_severity_enum():
    assert Severity.P0.value == "P0"
    assert Severity.P1.value == "P1"
    assert Severity.P2.value == "P2"
