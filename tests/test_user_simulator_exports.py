


def test_user_simulator_importable_from_mcv():
    from user_soul import UserSimulator
    assert UserSimulator is not None


def test_domain_configs_importable_from_mcv():
    from user_soul import GameDomainConfig, AppDomainConfig, WebDomainConfig, DomainConfig
    assert GameDomainConfig.session_framing == "你开始了一局游戏"


def test_simulation_report_importable_from_mcv():
    from user_soul import SimulationReport, MetricResult
    assert SimulationReport is not None


def test_evaluation_metric_importable_from_mcv():
    from user_soul import EvaluationMetric
    assert EvaluationMetric is not None


def test_full_pipeline_smoke_test_no_llm():
    """Verify the pipeline wires together without calling LLM."""
    from unittest.mock import patch
    from user_soul import UserSimulator, GameDomainConfig
    from user_soul.schema_extractor import EvaluationMetric

    with patch("user_soul.schema_extractor.extract_evaluation_schema") as mock_schema, \
         patch("user_soul.core._llm_call") as mock_llm, \
         patch("user_soul.report._aggregate_text") as mock_text:

        mock_schema.return_value = [
            EvaluationMetric("ret", "bool", "回来吗？"),
            EvaluationMetric("eng", "scale_1_5", "投入度？"),
        ]
        mock_llm.return_value = ("他进入游戏...\nret: yes\neng: 4", 200)
        from user_soul.report import MetricResult
        mock_text.return_value = MetricResult(name="", type="text", themes=[], samples=[])

        sim = UserSimulator("游戏玩家", GameDomainConfig, api_key="test")
        sim.prepare(product="一款棋牌游戏", goal="用户会留存吗？")
        report = sim.simulate(n_runs=3).report()

    assert report.n_simulations == 3
    assert "ret" in report.metrics
    assert report.metrics["ret"].true_rate == 1.0   # all 3 returned "yes"
    assert abs(report.metrics["eng"].mean - 4.0) < 0.01
