

from user_soul.user_simulator import SessionResult, _build_session_prompt, _parse_session_output
from user_soul.schema_extractor import EvaluationMetric
from user_soul.domain_configs import GameDomainConfig
from user_soul.scenarios import ScenarioContext

METRICS = [
    EvaluationMetric(name="day1_return", type="bool", question="你想回来吗？"),
    EvaluationMetric(name="engagement", type="scale_1_5", question="投入程度？"),
    EvaluationMetric(name="drop_moment", type="text", question="哪里想退出？"),
]
CTX = ScenarioContext("evening", "competitive", 1, "want_to_rank_up")


def test_build_prompt_contains_user_type():
    prompt = _build_session_prompt("18岁手游玩家", CTX, "一款棋牌游戏", METRICS, GameDomainConfig)
    assert "18岁手游玩家" in prompt


def test_build_prompt_contains_all_metric_names():
    prompt = _build_session_prompt("玩家", CTX, "游戏", METRICS, GameDomainConfig)
    assert "day1_return" in prompt
    assert "engagement" in prompt
    assert "drop_moment" in prompt


def test_build_prompt_contains_session_framing():
    prompt = _build_session_prompt("玩家", CTX, "游戏", METRICS, GameDomainConfig)
    assert "你开始了一局游戏" in prompt


def test_build_prompt_with_screen_id():
    prompt = _build_session_prompt("玩家", CTX, "wireframe", METRICS, GameDomainConfig, screen_id="home_screen")
    assert "home_screen" in prompt


def test_build_prompt_no_opinion_words():
    prompt = _build_session_prompt("玩家", CTX, "游戏", METRICS, GameDomainConfig)
    lower = prompt.lower()
    assert "rate this" not in lower
    assert "score the" not in lower


def test_parse_session_output_extracts_values():
    raw = "用户进入了游戏大厅。他点击了快速匹配...\nday1_return: yes\nengagement: 4\ndrop_moment: 教程太长"
    values = _parse_session_output(raw, METRICS)
    assert values["day1_return"] == "yes"
    assert values["engagement"] == "4"
    assert values["drop_moment"] == "教程太长"


def test_parse_session_output_handles_missing():
    raw = "用户进入了游戏大厅，然后退出了。"
    values = _parse_session_output(raw, METRICS)
    assert "day1_return" not in values
    assert "engagement" not in values


def test_session_result_fields():
    sr = SessionResult(
        scenario=CTX,
        narrative="叙述...",
        values={"day1_return": "yes"},
    )
    assert sr.scenario is CTX
    assert sr.values["day1_return"] == "yes"
