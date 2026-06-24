

from unittest.mock import patch, MagicMock
from user_soul.user_simulator import UserSimulator
from user_soul.domain_configs import GameDomainConfig
from user_soul.schema_extractor import EvaluationMetric


def test_prepare_extracts_schema():
    with patch("user_soul.schema_extractor.extract_evaluation_schema") as mock_extract:
        mock_extract.return_value = [EvaluationMetric("x", "bool", "?")]
        sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
        sim.prepare(product="游戏描述", goal="用户会留存吗？")
        mock_extract.assert_called_once()
        assert len(sim._metrics) == 1


def test_prepare_returns_self_for_chaining():
    with patch("user_soul.schema_extractor.extract_evaluation_schema") as mock_extract:
        mock_extract.return_value = []
        sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
        result = sim.prepare(product="游戏", goal="?")
        assert result is sim


def test_simulate_calls_llm_n_times():
    sim = UserSimulator("玩家", GameDomainConfig, api_key="test", use_behavioral_framework=False)
    sim._metrics = [EvaluationMetric("x", "bool", "?")]
    sim._product = "游戏"

    with patch("user_soul.core._llm_call") as mock_llm:
        mock_llm.return_value = ("叙述...\nx: yes", 200)
        sim.simulate(n_runs=5)

    assert mock_llm.call_count == 5
    assert len(sim._session_results) == 5


def test_simulate_uses_temperature_1():
    sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
    sim._metrics = [EvaluationMetric("x", "bool", "?")]
    sim._product = "游戏"

    with patch("user_soul.core._llm_call") as mock_llm:
        mock_llm.return_value = ("叙述...\nx: yes", 200)
        sim.simulate(n_runs=1)

    assert mock_llm.call_args[1]["temperature"] == 1.0


def test_simulate_uses_resolved_model():
    sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
    sim._metrics = [EvaluationMetric("x", "bool", "?")]
    sim._product = "游戏"

    with patch("user_soul.core._llm_call") as mock_llm:
        mock_llm.return_value = ("叙述...\nx: yes", 200)
        sim.simulate(n_runs=1)

    # _haiku_model resolves to the main model (Sonnet by default), never Opus.
    model = mock_llm.call_args[1].get("model", "")
    assert "claude" in model.lower()


def test_simulate_returns_self_for_chaining():
    sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
    sim._metrics = [EvaluationMetric("x", "bool", "?")]
    sim._product = "游戏"

    with patch("user_soul.core._llm_call") as mock_llm:
        mock_llm.return_value = ("叙述...\nx: yes", 200)
        result = sim.simulate(n_runs=2)

    assert result is sim


def test_simulate_stores_session_results():
    sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
    sim._metrics = [EvaluationMetric("ret", "bool", "回来吗？")]
    sim._product = "游戏"

    with patch("user_soul.core._llm_call") as mock_llm:
        mock_llm.return_value = ("他进入了游戏...\nret: yes", 200)
        sim.simulate(n_runs=3)

    from user_soul.user_simulator import SessionResult
    assert all(isinstance(r, SessionResult) for r in sim._session_results)
    assert sim._session_results[0].values.get("ret") == "yes"


def test_prepare_with_locked_metrics_skips_extraction():
    with patch("user_soul.schema_extractor.extract_evaluation_schema") as mock_extract:
        locked = [EvaluationMetric("ret", "bool", "回来吗？")]
        sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
        sim.prepare(product="游戏", locked_metrics=locked)
        mock_extract.assert_not_called()
        assert sim._metrics == locked


def test_prepare_with_locked_metrics_sets_product():
    locked = [EvaluationMetric("ret", "bool", "回来吗？")]
    sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
    sim.prepare(product="我的游戏PRD", locked_metrics=locked)
    assert sim._product == "我的游戏PRD"


def test_simulate_raises_if_prepare_not_called():
    sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
    import pytest
    with pytest.raises(RuntimeError, match="prepare"):
        sim.simulate(n_runs=1)
