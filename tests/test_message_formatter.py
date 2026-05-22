"""Tests for User-Soul message formatter (v2 — dict-based protocol)."""
import sys
sys.path.insert(0, "/Users/mozat/user-soul")

from user_soul.message_formatter import (
    format_action_as_dict,
    format_flat_verdict,
    format_graded_verdict,
    action_specs_to_dicts,
)
from user_soul.models import (
    ActionSpec,
    GradedPlaytestFeedback,
    PlaytestFeedback,
    PlaytestIssue,
)


def test_format_action_fix():
    spec = ActionSpec(
        owner="code-soul", action_type="fix", severity="P0",
        description="Buttons not responding",
        evidence=["click did nothing"],
        diagnosis_category="CODE_BUG",
    )
    d = format_action_as_dict(spec, "ctx_1")
    assert d["type"] == "user_test_feedback"
    assert d["source"] == "pm-soul"
    assert d["context_id"] == "ctx_1"
    assert d["severity"] == "P0"
    assert d["action_type"] == "fix"
    assert d["description"] == "Buttons not responding"
    assert "id" in d


def test_format_action_redesign():
    spec = ActionSpec(
        owner="eltm", action_type="redesign", severity="P1",
        description="Discoverability issue",
    )
    d = format_action_as_dict(spec, "ctx_2")
    assert d["type"] == "user_test_feedback"
    assert d["action_type"] == "redesign"
    assert d["severity"] == "P1"


def test_format_action_triage():
    spec = ActionSpec(
        owner="pm-soul", action_type="triage", severity="P2",
        description="Inconclusive",
    )
    d = format_action_as_dict(spec, "ctx_3")
    assert d["type"] == "user_test_feedback"
    assert d["severity"] == "P2"


def test_format_action_add_help():
    spec = ActionSpec(
        owner="eltm", action_type="add_help", severity="P1",
        description="Tutorial needed",
    )
    d = format_action_as_dict(spec, "ctx_4")
    assert d["type"] == "user_test_feedback"
    assert d["action_type"] == "add_help"


def test_format_action_with_ref_id():
    spec = ActionSpec(owner="code-soul", action_type="fix", severity="P0", description="bug")
    d = format_action_as_dict(spec, "ctx", ref_id="parent_msg_id")
    assert d["ref_id"] == "parent_msg_id"


def test_action_specs_to_dicts():
    specs = [
        ActionSpec(owner="code-soul", action_type="fix", severity="P0", description="bug1"),
        ActionSpec(owner="eltm", action_type="add_help", severity="P1", description="help1"),
        ActionSpec(owner="pm-soul", action_type="triage", severity="P2", description="unclear"),
    ]
    dicts = action_specs_to_dicts(specs, "ctx_5", verdict_ref_id="verdict_id")
    assert len(dicts) == 3
    assert all(d["ref_id"] == "verdict_id" for d in dicts)
    assert all(d["type"] == "user_test_feedback" for d in dicts)
    assert all(d["context_id"] == "ctx_5" for d in dicts)


def test_format_flat_verdict():
    fb = PlaytestFeedback(
        score=45, verdict="NEEDS_WORK",
        issues=[
            PlaytestIssue(severity="P0", description="crash", category="code_bug"),
            PlaytestIssue(severity="P1", description="confusing", category="ux_friction"),
        ],
        personas_completed=4, personas_total=5,
        suggestions=["fix crash first"],
    )
    d = format_flat_verdict(fb, "ctx_6")
    assert d["type"] == "user_verdict"
    assert d["source"] == "user-soul"
    assert d["context_id"] == "ctx_6"
    assert d["severity"] == "P0"
    assert d["verdict"] == "NEEDS_WORK"
    assert len(d["issues"]) == 2
    assert "id" in d


def test_format_graded_verdict_pass():
    fb = GradedPlaytestFeedback(
        score=90, verdict="PASS",
        personas_completed=6, personas_total=6,
    )
    d = format_graded_verdict(fb, "ctx_7")
    assert "severity" not in d
    assert d["verdict"] == "PASS"


def test_format_graded_verdict_with_diagnosis():
    from user_soul.game_knowledge import (
        DiagnosisCategory, DiagnosisItem, DifferentialDiagnosis,
    )
    diag = DifferentialDiagnosis(
        game_name="TestGame",
        tier_results={},
        diagnoses=[
            DiagnosisItem(
                category=DiagnosisCategory.CODE_BUG,
                severity="P0",
                description="All tiers fail on first click",
                evidence={"novice": "crash", "casual": "crash", "informed": "crash"},
                owner="code-soul",
            ),
        ],
        summary="Universal failure indicates code bug.",
    )
    fb = GradedPlaytestFeedback(
        score=15, verdict="CRITICAL",
        personas_completed=2, personas_total=6,
        diagnosis=diag,
    )
    fb.tier_feedbacks = {
        "novice": PlaytestFeedback(score=10, verdict="CRITICAL"),
        "casual": PlaytestFeedback(score=15, verdict="CRITICAL"),
        "informed": PlaytestFeedback(score=20, verdict="NEEDS_WORK"),
    }
    d = format_graded_verdict(fb, "ctx_8")
    assert d["severity"] == "P0"
    assert len(d["diagnosis"]) == 1
    assert d["diagnosis"][0]["category"] == "code_bug"
    assert d["tier_scores"]["novice"] == 10
