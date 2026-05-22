

from unittest.mock import patch
from user_soul.schema_extractor import EvaluationMetric, extract_evaluation_schema


def test_extract_returns_evaluation_metrics():
    mock_resp = '[{"name": "retention", "type": "bool", "question": "会回来吗？"}, {"name": "engagement", "type": "scale_1_5", "question": "投入程度？"}]'
    with patch("user_soul.core._llm_call") as mock_llm:
        mock_llm.return_value = (mock_resp, 200)
        metrics = extract_evaluation_schema("用户会留存吗？", api_key="test")
    assert len(metrics) == 2
    assert isinstance(metrics[0], EvaluationMetric)
    assert metrics[0].name == "retention"
    assert metrics[0].type == "bool"
    assert metrics[1].type == "scale_1_5"


def test_extract_filters_invalid_types():
    mock_resp = '[{"name": "x", "type": "invalid_type", "question": "?"}, {"name": "y", "type": "bool", "question": "yes?"}]'
    with patch("user_soul.core._llm_call") as mock_llm:
        mock_llm.return_value = (mock_resp, 200)
        metrics = extract_evaluation_schema("test", api_key="test")
    assert len(metrics) == 1
    assert metrics[0].name == "y"


def test_extract_handles_malformed_json():
    with patch("user_soul.core._llm_call") as mock_llm:
        mock_llm.return_value = ("not json at all", 100)
        metrics = extract_evaluation_schema("test", api_key="test")
    assert metrics == []


def test_extract_uses_sonnet_model():
    mock_resp = '[{"name": "x", "type": "bool", "question": "?"}]'
    with patch("user_soul.core._llm_call") as mock_llm:
        mock_llm.return_value = (mock_resp, 200)
        extract_evaluation_schema("test", api_key="test")
    # schema extraction uses Sonnet (default model), not Haiku
    call_kwargs = mock_llm.call_args[1]
    model = call_kwargs.get("model")
    assert model is None or "haiku" not in str(model).lower()


def test_evaluation_metric_fields():
    m = EvaluationMetric(name="ret", type="bool", question="回来吗？")
    assert m.name == "ret"
    assert m.type == "bool"
    assert m.question == "回来吗？"
