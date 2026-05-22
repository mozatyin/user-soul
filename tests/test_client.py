"""Tests for MCVClient facade."""
from unittest.mock import patch, MagicMock
from user_soul.voter import MCVClient
from user_soul.report import SimulationReport, CompareReport
from user_soul.core import DecisionResult, Persona
from user_soul.schema_extractor import EvaluationMetric


def test_client_simulate_returns_simulation_report():
    client = MCVClient(api_key="test")
    mock_report = MagicMock(spec=SimulationReport)
    with patch("user_soul.voter.UserSimulator") as MockSim:
        instance = MockSim.return_value
        instance.prepare.return_value = instance
        instance.simulate.return_value = instance
        instance.report.return_value = mock_report
        with patch("user_soul.voter.build_domain_config") as mock_build:
            mock_build.return_value = MagicMock()
            result = client.simulate(
                product="游戏 PRD",
                user_type="18岁玩家",
                goal="用户会留存吗？",
                n_runs=5,
            )
    assert result is mock_report


def test_client_simulate_uses_provided_domain_config():
    from user_soul.domain_configs import GameDomainConfig
    client = MCVClient(api_key="test")
    with patch("user_soul.voter.UserSimulator") as MockSim:
        instance = MockSim.return_value
        instance.prepare.return_value = instance
        instance.simulate.return_value = instance
        instance.report.return_value = MagicMock()
        with patch("user_soul.voter.build_domain_config") as mock_build:
            client.simulate("prd", "玩家", "goal", domain_config=GameDomainConfig, n_runs=2)
    mock_build.assert_not_called()
    MockSim.assert_called_once_with("玩家", GameDomainConfig, api_key="test")


def test_client_simulate_auto_builds_domain_config_when_none():
    client = MCVClient(api_key="test")
    with patch("user_soul.voter.UserSimulator") as MockSim:
        instance = MockSim.return_value
        instance.prepare.return_value = instance
        instance.simulate.return_value = instance
        instance.report.return_value = MagicMock()
        with patch("user_soul.voter.build_domain_config") as mock_build:
            mock_build.return_value = MagicMock()
            client.simulate("prd", "用户", "goal", n_runs=2)
    mock_build.assert_called_once_with("prd", "test")


def test_client_compare_returns_compare_report():
    client = MCVClient(api_key="test")
    mock_compare = MagicMock(spec=CompareReport)
    with patch("user_soul.voter.UserSimulator") as MockSim:
        instance = MockSim.return_value
        instance.compare.return_value = mock_compare
        with patch("user_soul.voter.build_domain_config") as mock_build:
            mock_build.return_value = MagicMock()
            result = client.compare("prd_a", "prd_b", "玩家", "goal", n_runs=5)
    assert result is mock_compare
    instance.compare.assert_called_once_with(
        "prd_a", "prd_b",
        label_a="v_a", label_b="v_b",
        n_runs=5, locked_metrics=None, goal="goal",
    )


def test_client_decide_calls_persona_decider():
    p = Persona("p1", "Alice", "gamer", "18-24", ["fun"], ["ads"])
    client = MCVClient(api_key="test", personas=[p])
    mock_result = MagicMock(spec=DecisionResult)
    with patch("user_soul.voter.PersonaDecider") as MockPD:
        instance = MockPD.return_value
        instance.classify.return_value = mock_result
        result = client.decide(
            question="Which is better?",
            options=["A", "B"],
            context="context here",
        )
    assert result is mock_result
    instance.classify.assert_called_once_with(
        question="Which is better?",
        options=["A", "B"],
        context="context here",
    )


def test_client_decide_auto_generates_personas_when_none():
    client = MCVClient(api_key="test")
    with patch("user_soul.voter.PersonaDecider") as MockPD:
        instance = MockPD.return_value
        instance.classify.return_value = MagicMock()
        with patch("user_soul.voter.load_or_generate") as mock_gen:
            mock_gen.return_value = [
                Persona("p1", "Bob", "user", "25-35", [], [])
            ]
            client.decide("Q?", ["A", "B"], "ctx", product="app prd")
    mock_gen.assert_called_once()


def test_client_decide_raises_when_no_personas_and_no_product():
    import pytest
    client = MCVClient(api_key="test")
    with pytest.raises(ValueError, match="provide personas"):
        client.decide("Q?", ["A", "B"], "ctx")
