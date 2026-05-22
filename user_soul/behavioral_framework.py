"""Behavioral simulation framework constants — Fogg BM + Hook Model + Peak-End Rule.

These constants are injected into UserSimulator session prompts to ground LLM
behavioral simulation in established psychological frameworks.

Research basis:
- Sycophancy deflation: AgentBench/AgentA/B show LLM agents over-rate cooperative
  scenarios by 30-40% vs real user data. SYCOPHANCY_DEFLATOR corrects this.
- Fogg Behavior Model: B = MAP (Motivation × Ability × Prompt). Friction > threshold → abandon.
- Hook Model (Nir Eyal): Trigger→Action→Variable Reward→Investment. Loop completion = retention signal.
- Peak-End Rule (Kahneman): Memory of experience = f(peak moment, end moment). Not the average.
- Average persona collapse: LLMs default to cooperative average users. Adversarial pass surfaces churners.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Sycophancy correction
# ---------------------------------------------------------------------------

SYCOPHANCY_DEFLATOR: float = 0.70
"""Multiply raw LLM-reported Day-1 return rate by this factor.

LLMs simulate overly cooperative users and inflate true_rate by ~30-40%.
Research: AgentBench (Liu et al. 2023) 6 actions vs human 16; AgentA/B (2025)
30% over-purchasing. 0.70 is conservative — use 0.65 for high-stakes decisions.
"""

# ---------------------------------------------------------------------------
# Cognitive budgets by user archetype
# ---------------------------------------------------------------------------

COGNITIVE_BUDGETS: dict[str, int] = {
    "casual":      8,   # distracted, multitasking — very low tolerance
    "core":       25,   # engaged, invested in product
    "social":     15,   # motivated by social proof, medium attention
    "goal":       20,   # task-focused, tolerates complexity for outcome
    "adversarial": 6,   # high churn risk — nearly zero patience
    "default":    15,   # fallback for unknown archetype
}
"""Cognitive budget units per archetype. Each UI element that requires reading/
understanding costs 1-3 units. Exceeding budget → session abandonment."""

# ---------------------------------------------------------------------------
# Behavioral framework prompt section (injected into session prompts)
# ---------------------------------------------------------------------------

BEHAVIORAL_FRAMEWORK_SECTION: str = """\
## 行为框架追踪（你必须遵循）
Fogg行为模型: 动机 × 能力。遇到任何能力障碍（注册墙/权限请求/说明文字超3行）必须记录为放弃点。
Hook模型四步: 触发→行动→可变奖励→投入。没有到达"可变奖励"步骤则 hook_completed=no。
峰端规则: 你的会话记忆 = 情感最高点 + 最终状态。高峰必须在前3分钟内出现才对留存有效。
认知负载: 你的认知预算={cognitive_budget}单位。每个需要阅读/理解的UI元素消耗1-3单位。超出预算→放弃。
"""

# ---------------------------------------------------------------------------
# Adversarial persona section (replaces normal user_type in adversarial pass)
# ---------------------------------------------------------------------------

ADVERSARIAL_PERSONA_SECTION: str = """\
## 高流失风险用户特征（你必须严格扮演此类用户）
- 碎片时间使用，5秒内无明确吸引点→立即划走
- 对注册/权限/邮箱验证极度抗拒，遇到直接退出
- 完全不读任何说明文字、教程或新手引导
- 每屏最多停留5秒，超时退出
- 对"明天再来""解锁更多""邀请好友"类承诺完全不信任
- 认知预算仅6单位，超出立即离开
"""

# ---------------------------------------------------------------------------
# Additional behavioral metrics injected alongside user-defined metrics
# ---------------------------------------------------------------------------

BEHAVIORAL_METRICS: list[tuple[str, str, str]] = [
    (
        "hook_completed",
        "bool",
        "本次会话完整经历了Hook循环（触发→行动→可变奖励→投入）吗？",
    ),
    (
        "end_state_sentiment",
        "bool",
        "会话结束时用户状态积极（想要更多）吗？",
    ),
]
"""List of (name, type, question) tuples. Injected by UserSimulator when
use_behavioral_framework=True. Measures Hook Model completion and Peak-End end state."""
