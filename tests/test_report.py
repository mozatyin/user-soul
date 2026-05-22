
from unittest.mock import patch as _patch

from user_soul.report import (
    _aggregate_bool, _aggregate_scale, _aggregate_text,
    aggregate, SimulationReport, MetricResult,
)
from user_soul.schema_extractor import EvaluationMetric
from user_soul.user_simulator import SessionResult
from user_soul.scenarios import ScenarioContext


CTX = ScenarioContext("evening", "calm", 1, "curiosity")


def test_aggregate_bool_true_rate():
    r = _aggregate_bool(["yes", "yes", "no", "yes"])
    assert abs(r.true_rate - 0.75) < 0.01


def test_aggregate_bool_handles_chinese_yes():
    r = _aggregate_bool(["是", "否", "是"])
    assert abs(r.true_rate - 0.667) < 0.01


def test_aggregate_bool_empty():
    r = _aggregate_bool([])
    assert r.true_rate == 0.0


def test_aggregate_scale_mean():
    r = _aggregate_scale(["4", "3", "5", "4"])
    assert abs(r.mean - 4.0) < 0.01


def test_aggregate_scale_distribution_keys():
    r = _aggregate_scale(["4", "4", "3"])
    assert 4 in r.distribution
    assert 3 in r.distribution
    assert 1 not in r.distribution


def test_aggregate_scale_empty():
    r = _aggregate_scale([])
    assert r.mean == 0.0


def test_aggregate_text_returns_samples():
    r = _aggregate_text(["教程太长", "UI 复杂", "第一局输了"])
    assert "教程太长" in r.samples
    assert len(r.samples) <= 10


def test_aggregate_full_report():
    metrics = [
        EvaluationMetric("ret", "bool", "回来吗？"),
        EvaluationMetric("eng", "scale_1_5", "投入度？"),
    ]
    sessions = [
        SessionResult(CTX, "叙述...", {"ret": "yes", "eng": "4"}),
        SessionResult(CTX, "叙述...", {"ret": "no",  "eng": "3"}),
        SessionResult(CTX, "叙述...", {"ret": "yes", "eng": "5"}),
    ]
    report = aggregate(sessions, metrics, "玩家", "游戏")
    assert isinstance(report, SimulationReport)
    assert report.n_simulations == 3
    assert report.user_type == "玩家"
    assert abs(report.metrics["ret"].true_rate - 0.667) < 0.01
    assert abs(report.metrics["eng"].mean - 4.0) < 0.01


def test_aggregate_missing_metric_values():
    """Sessions with no value for a metric → zero/empty result, no crash."""
    metrics = [EvaluationMetric("ret", "bool", "?")]
    sessions = [SessionResult(CTX, "叙述...", {})]  # no values parsed
    report = aggregate(sessions, metrics, "玩家", "游戏")
    assert report.metrics["ret"].true_rate == 0.0


def test_aggregate_text_skips_llm_with_fewer_than_3_values():
    """Text aggregation with api_key but < 3 values → no LLM call, empty themes."""
    r = _aggregate_text(["只有一条"], api_key="test")
    assert r.themes == []
    assert r.samples == ["只有一条"]


def test_aggregate_bool_populates_ci():
    r = _aggregate_bool(["yes"] * 18 + ["no"] * 12)  # 30 samples, true_rate=0.6
    assert r.n_samples == 30
    assert r.stdev is not None
    assert r.ci_95_low is not None and r.ci_95_high is not None
    assert r.ci_95_low < r.true_rate < r.ci_95_high


def test_aggregate_bool_ci_empty():
    r = _aggregate_bool([])
    assert r.n_samples == 0
    assert r.ci_95_low == 0.0
    assert r.ci_95_high == 0.0


def test_aggregate_scale_populates_ci():
    r = _aggregate_scale(["4", "3", "5", "4", "4", "3"])
    assert r.n_samples == 6
    assert r.stdev is not None and r.stdev > 0
    assert r.ci_95_low is not None and r.ci_95_high is not None
    assert r.ci_95_low < r.mean < r.ci_95_high


def test_aggregate_scale_single_value_ci():
    r = _aggregate_scale(["3"])
    assert r.n_samples == 1
    assert r.stdev == 0.0
    assert r.ci_95_low == 3.0
    assert r.ci_95_high == 3.0


def test_aggregate_auto_key_findings():
    metrics = [
        EvaluationMetric("ret", "bool", "回来吗？"),
        EvaluationMetric("eng", "scale_1_5", "投入度？"),
    ]
    sessions = [
        SessionResult(CTX, "叙述", {"ret": "yes", "eng": "4"}),
        SessionResult(CTX, "叙述", {"ret": "no",  "eng": "3"}),
        SessionResult(CTX, "叙述", {"ret": "yes", "eng": "5"}),
    ]
    with _patch("user_soul.core._llm_call") as mock_llm:
        mock_llm.return_value = ("Day-1 留存率为 67%，参与度均分 4.0。", 80)
        report = aggregate(sessions, metrics, "玩家", "游戏", api_key="test")
    mock_llm.assert_called()
    assert report.key_findings == "Day-1 留存率为 67%，参与度均分 4.0。"


def test_aggregate_no_key_findings_without_api_key():
    metrics = [EvaluationMetric("ret", "bool", "?"), EvaluationMetric("eng", "scale_1_5", "?")]
    sessions = [SessionResult(CTX, "叙述", {"ret": "yes", "eng": "4"})]
    report = aggregate(sessions, metrics, "玩家", "游戏", api_key=None)
    assert report.key_findings == ""


def test_aggregate_no_key_findings_with_single_metric():
    metrics = [EvaluationMetric("ret", "bool", "?")]
    sessions = [SessionResult(CTX, "叙述", {"ret": "yes"})]
    with _patch("user_soul.core._llm_call") as mock_llm:
        mock_llm.return_value = ("some findings", 50)
        report = aggregate(sessions, metrics, "玩家", "游戏", api_key="test")
    mock_llm.assert_not_called()
    # Single metric → no key_findings LLM call
    assert report.key_findings == ""


def test_simulation_report_day1_return_rate():
    metrics = [EvaluationMetric("ret", "bool", "?")]
    sessions = [SessionResult(CTX, "x", {"ret": "yes"}),
                SessionResult(CTX, "x", {"ret": "no"})]
    report = aggregate(sessions, metrics, "玩家", "游戏")
    assert abs(report.day1_return_rate - 0.5) < 0.01


def test_simulation_report_day1_return_rate_none_when_no_bool():
    metrics = [EvaluationMetric("eng", "scale_1_5", "?")]
    sessions = [SessionResult(CTX, "x", {"eng": "4"})]
    report = aggregate(sessions, metrics, "玩家", "游戏")
    assert report.day1_return_rate is None


def test_simulation_report_friction_themes():
    metrics = [EvaluationMetric("friction", "text", "?")]
    sessions = [SessionResult(CTX, "x", {"friction": "tutorial too long"})]
    report = aggregate(sessions, metrics, "玩家", "游戏")
    assert isinstance(report.friction_themes, list)


def test_simulation_report_locked_schema():
    metrics = [EvaluationMetric("ret", "bool", "回来吗？")]
    sessions = [SessionResult(CTX, "x", {"ret": "yes"})]
    report = aggregate(sessions, metrics, "玩家", "游戏")
    schema = report.locked_schema
    assert len(schema) == 1
    assert schema[0]["name"] == "ret"
    assert schema[0]["type"] == "bool"
    assert schema[0]["question"] == "回来吗？"


from user_soul.report import FeatureAAR, CoherenceReport

def test_feature_aar_fields():
    f = FeatureAAR(
        feature_id="invite_friends",
        acquisition=0.3, activation=0.5, retention=0.4,
        revenue=0.1, referral=0.9,
        confidence=0.8,
        archetype_votes={"Gamer": {"acquisition": 0.3, "referral": 0.9}},
    )
    assert f.feature_id == "invite_friends"
    assert f.referral == 0.9
    assert "Gamer" in f.archetype_votes

def test_coherence_report_is_coherent_true():
    r = CoherenceReport(
        selected_feature_ids=["ludo", "invite_friends"],
        missing_dependencies=[],
        blocked_journeys=[],
        reinstate_recommendations=[],
        is_coherent=True,
    )
    assert r.is_coherent

def test_coherence_report_is_coherent_false():
    r = CoherenceReport(
        selected_feature_ids=["ludo"],
        missing_dependencies=[{"feature_id": "invite_friends", "required_by": ["ludo"],
                               "reason": "ludo needs multiple players"}],
        blocked_journeys=["User tried to play Ludo but had no friends"],
        reinstate_recommendations=["invite_friends"],
        is_coherent=False,
    )
    assert not r.is_coherent
    assert "invite_friends" in r.reinstate_recommendations
