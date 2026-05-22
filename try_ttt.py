import os
"""End-to-end User-Soul simulation for the REAL Tic-Tac-Toe game from our pipeline."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from user_soul.backends.anthropic import AnthropicBackend
from user_soul.engines.persona import PersonaEngine
from user_soul.engines.behavior import BehaviorEngine, _GAME_CONFIG
from user_soul.models import EvaluationMetric

API_KEY = os.environ["OPENROUTER_API_KEY"]

PRODUCT = """一款真实的 Tic-Tac-Toe 手机网页游戏（已实现，可运行）。

## 功能
- 主菜单：3个按钮 — vs AI / Local PvP / Online
- vs AI 模式：玩家执X先手，AI用minimax+alpha-beta剪枝（完美AI，不可能被击败）
- Local PvP：同一台设备轮流下
- Online：点击后显示"Waiting for opponent..."（未真正实现后端）
- 3x3棋盘，点击空格下子
- 胜利时高亮连线（黄色脉冲动画）
- 计分板：显示X和O的比分
- 结果画面：显示"You Win!" / "AI Wins!" / "It's a Draw!"
- Reset和Home按钮

## 视觉设计
- 紫蓝渐变背景 (linear-gradient #667eea → #764ba2)
- 白色圆角按钮+阴影
- X蓝色(#2563eb)，O红色(#dc2626)
- 简洁system-ui字体
- 最大宽度400px，移动端适配

## 已知限制
- 无广告、无内购、无段位系统
- Online模式是空壳（无服务器）
- AI是完美的minimax——玩家永远赢不了，最好结果是平局
- 无难度选择
- 无音效/动画（除胜利脉冲）
- 无教程/规则说明页面
- 返回主菜单会清零比分
"""

def main():
    print("=== User-Soul × 真实 Tic-Tac-Toe 游戏 ===\n")
    backend = AnthropicBackend(api_key=API_KEY)

    print("[1/3] 生成用户人设 (n=6)...")
    t0 = time.time()
    persona_engine = PersonaEngine(backend)
    pool = persona_engine.get_or_create(PRODUCT, n=6)
    t1 = time.time()
    print(f"  耗时: {t1-t0:.1f}s")
    for p in pool:
        print(f"    [{p.archetype_name}] {p.background_story[:80]}")
    print()

    print("[2/3] 行为模拟 (n_runs=10, adversarial=True)...")
    metrics = [
        EvaluationMetric("day_1_return_intent", "bool", "玩完第一局后，明天你还会打开这个游戏吗？"),
        EvaluationMetric("fun_rating", "scale_1_5", "这局游戏有多好玩？(1=很无聊, 5=非常好玩)"),
        EvaluationMetric("friction_points", "text", "有什么让你觉得不爽、困惑或想卸载的地方？"),
    ]
    behavior = BehaviorEngine(backend)
    t2 = time.time()
    report = behavior.simulate(
        PRODUCT, pool, metrics,
        n_runs=10,
        adversarial=True,
        domain_config=_GAME_CONFIG,
    )
    t3 = time.time()
    print(f"  耗时: {t3-t2:.1f}s")
    print(f"  总模拟数: {report.n_simulations}")
    print()

    print("[3/3] 模拟结果\n")
    print(f"  Day-1 留存率 (raw):      {report.day1_return_rate}")
    print(f"  Day-1 留存率 (adjusted): {report.day1_return_rate_adjusted}")
    print(f"  基准评价:                {report.benchmark_context}")
    print()

    for name, mr in report.metrics.items():
        if mr.type == "bool":
            ci_lo = f"{mr.ci_95_low:.0%}" if mr.ci_95_low is not None else "?"
            ci_hi = f"{mr.ci_95_high:.0%}" if mr.ci_95_high is not None else "?"
            tr = f"{mr.true_rate:.0%}" if mr.true_rate is not None else "?"
            print(f"  [{name}] true_rate={tr}, CI=[{ci_lo}, {ci_hi}], n={mr.n_samples}")
        elif mr.type == "scale_1_5":
            mean = f"{mr.mean:.2f}" if mr.mean is not None else "?"
            std = f"{mr.stdev:.2f}" if mr.stdev is not None else "?"
            print(f"  [{name}] mean={mean}/5, stdev={std}, n={mr.n_samples}")
        elif mr.type == "text":
            print(f"  [{name}] themes={mr.themes}")
    print()

    if report.key_findings:
        print(f"  关键发现:\n  {report.key_findings}")
    print()

    if report.adversarial_frictions:
        print(f"  对抗性摩擦点:")
        for f in report.adversarial_frictions:
            print(f"    - {f}")
    print()

    print(f"=== 总耗时: {t3-t0:.1f}s ===")

if __name__ == "__main__":
    main()
