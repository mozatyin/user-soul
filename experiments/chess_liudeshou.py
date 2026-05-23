"""
实验：以刘德寿为核心 Persona，对国际象棋做 User Soul 功能过滤，输出 PRD。

刘德寿：52岁，上海，退休工程师，象棋六段，首次接触国际象棋。
目标：同样的 chess.com 研究数据，经 User Soul 过滤后产出刘德寿专属需求文档。
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from user_soul.backends.anthropic import AnthropicBackend
from user_soul.feature_filter import FeatureFilter, FeatureFilterReport, ScoredFeature
from user_soul.population import Archetype

# ─── API Key ─────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    env_path = Path("/Users/mozat/eltm/.env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                API_KEY = line.split("=", 1)[1].strip()
                break
if not API_KEY:
    raise RuntimeError("No API key found")

# ─── 棋类功能清单（来自 chess.com benchmark 扫描 + ELTM chess.json）──────────
CHESS_FEATURES = [
    # 新手流程 / Onboarding
    {"id": "onboard_welcome",        "name": "欢迎引导屏",               "description": "首次打开的品牌欢迎页和价值主张介绍",              "category": "onboarding", "source": "chess.com"},
    {"id": "onboard_skill_select",   "name": "棋力等级选择",              "description": "让用户选择新手/中级/高级，初始化对手难度",          "category": "onboarding", "source": "chess.com"},
    {"id": "onboard_tutorial",       "name": "基础规则互动教程",           "description": "逐步教棋子走法的交互式教程，有动画演示",            "category": "onboarding", "source": "chess.com"},
    {"id": "onboard_first_game",     "name": "引导首局对弈",              "description": "新手第一局由系统辅助提示合法走法",                 "category": "onboarding", "source": "chess.com"},
    {"id": "onboard_register",       "name": "账号注册/游客模式",          "description": "可注册保存进度，也可游客体验",                    "category": "onboarding", "source": "chess.com"},

    # 学习功能
    {"id": "learn_lessons",          "name": "结构化课程",                "description": "从开局到残局的系统课程，视频+练习",               "category": "learning",   "source": "chess.com"},
    {"id": "learn_puzzles",          "name": "每日棋题练习",              "description": "每天推送战术题，解题后有详解",                    "category": "learning",   "source": "chess.com"},
    {"id": "learn_opening",          "name": "开局库学习",                "description": "常见开局的步骤记忆和理解训练",                    "category": "learning",   "source": "chess.com"},
    {"id": "learn_endgame",          "name": "残局练习",                  "description": "经典残局局面的练习与解析",                        "category": "learning",   "source": "chess.com"},
    {"id": "learn_xiangqi_bridge",   "name": "象棋知识迁移提示",           "description": "针对会下象棋的用户，标注与象棋的异同",              "category": "learning",   "source": "localization"},
    {"id": "learn_analysis",         "name": "棋局分析引擎",              "description": "每盘对局后AI分析失误点，给出最佳走法",             "category": "learning",   "source": "chess.com"},
    {"id": "learn_coach",            "name": "AI教练模式",                "description": "对弈中实时提示策略建议（付费）",                   "category": "learning",   "source": "chess.com"},

    # 对弈模式
    {"id": "mode_vs_ai",             "name": "人机对战",                  "description": "与AI对弈，可调节10个难度等级",                    "category": "game_mode",  "source": "chess.com"},
    {"id": "mode_vs_human",          "name": "在线对人",                  "description": "随机匹配或邀请好友联网对弈",                      "category": "game_mode",  "source": "chess.com"},
    {"id": "mode_puzzle_rush",       "name": "解题冲关",                  "description": "限时内连续解题，计分竞速",                        "category": "game_mode",  "source": "chess.com"},
    {"id": "mode_daily_game",        "name": "通讯棋（慢棋）",             "description": "每天走一步的慢节奏对局，可多局并行",               "category": "game_mode",  "source": "chess.com"},
    {"id": "mode_bullet",            "name": "子弹棋（1分钟）",            "description": "极速对弈模式，每方仅1分钟",                       "category": "game_mode",  "source": "chess.com"},
    {"id": "mode_blitz",             "name": "快棋（3-5分钟）",            "description": "快节奏在线对弈",                                 "category": "game_mode",  "source": "chess.com"},
    {"id": "mode_rapid",             "name": "标准时制（10-30分钟）",      "description": "正式节奏对弈，适合认真学习",                      "category": "game_mode",  "source": "chess.com"},
    {"id": "mode_960",               "name": "国际象棋960",               "description": "随机开局变体玩法",                               "category": "game_mode",  "source": "chess.com"},

    # 游戏内控制
    {"id": "ctrl_legal_hints",       "name": "合法走法高亮",              "description": "点击棋子后显示所有可落子位置",                    "category": "gameplay",   "source": "chess.com"},
    {"id": "ctrl_undo",              "name": "悔棋",                      "description": "人机模式下可撤销上一步",                          "category": "gameplay",   "source": "chess.com"},
    {"id": "ctrl_flip_board",        "name": "翻转棋盘",                  "description": "从黑方视角查看棋盘",                             "category": "gameplay",   "source": "chess.com"},
    {"id": "ctrl_notation",          "name": "棋谱记录",                  "description": "实时显示代数记法棋谱",                            "category": "gameplay",   "source": "chess.com"},
    {"id": "ctrl_themes",            "name": "棋盘主题/棋子皮肤",          "description": "5种棋盘风格和棋子造型可选",                        "category": "gameplay",   "source": "chess.com"},
    {"id": "ctrl_sound",             "name": "音效开关",                  "description": "落子音效、将军提示音等",                          "category": "gameplay",   "source": "chess.com"},
    {"id": "ctrl_font_size",         "name": "界面字体大小调节",           "description": "支持大字体显示，适合视力不佳用户",                 "category": "gameplay",   "source": "accessibility"},

    # 社交 / 竞技
    {"id": "social_rating",          "name": "ELO等级分系统",             "description": "基于胜负的动态评分，显示玩家排名",                 "category": "social",     "source": "chess.com"},
    {"id": "social_leaderboard",     "name": "排行榜",                    "description": "全球/好友排名",                                  "category": "social",     "source": "chess.com"},
    {"id": "social_friends",         "name": "好友系统",                  "description": "添加好友、查看在线状态、邀请对弈",                 "category": "social",     "source": "chess.com"},
    {"id": "social_clubs",           "name": "棋友俱乐部",                "description": "加入兴趣群组，参与群赛",                          "category": "social",     "source": "chess.com"},
    {"id": "social_stream",          "name": "直播/观战",                 "description": "观看顶级棋手直播或录播",                          "category": "social",     "source": "chess.com"},
    {"id": "social_share",           "name": "分享棋局",                  "description": "将精彩局面分享到微信/微博",                        "category": "social",     "source": "localization"},

    # 留存 / 参与度
    {"id": "engage_streak",          "name": "每日打卡连击",              "description": "连续每日登录奖励，断签提醒",                      "category": "engagement", "source": "chess.com"},
    {"id": "engage_achievements",    "name": "成就徽章系统",              "description": "完成特定目标解锁徽章",                            "category": "engagement", "source": "chess.com"},
    {"id": "engage_progress_track",  "name": "学习进度追踪",              "description": "可视化展示课程完成度和棋力提升曲线",               "category": "engagement", "source": "chess.com"},
    {"id": "engage_notifications",   "name": "轮到你了提醒",              "description": "通讯棋模式下对手走棋后推送通知",                  "category": "engagement", "source": "chess.com"},

    # 本地化 / 无障碍
    {"id": "local_chinese",          "name": "完整中文界面",              "description": "所有规则说明、教程、界面文字中文化",               "category": "onboarding", "source": "localization"},
    {"id": "local_offline",          "name": "离线模式",                  "description": "无网络时仍可与AI对弈和做题",                      "category": "gameplay",   "source": "localization"},
]

# ─── 刘德寿核心 Persona 及围绕他的 4 个 Archetype ────────────────────────────
LIU_ARCHETYPES = [
    Archetype(
        name="刘德寿型 — 老年象棋高手转型者",
        description="50-60岁，退休，象棋功底深厚，初学国际象棋。视力一般，不爱竞技，喜欢慢节奏学习。",
        trait_constraints={},
        background_story=(
            "刘德寿，52岁，上海退休工程师，象棋六段，在棋友圈颇有声望。"
            "儿子在美国工作，建议他学国际象棋以便父子共同话题。"
            "他对棋很认真，愿意花时间系统学习，但不想被排名压力绑架。"
            "最希望的是：知道象棋和国际象棋的异同，能慢慢下完整局，看懂棋谱。"
            "最怕的是：界面字太小看不清，被初学者虐杀，浪费时间在无聊动画上。"
        ),
        frequency=0.35,
    ),
    Archetype(
        name="王淑珍型 — 中年陪读家长",
        description="45-55岁，家庭主妇或半退休，陪孙子/子女学棋，自己也想了解。",
        trait_constraints={},
        background_story=(
            "王淑珍，48岁，上海家庭主妇，孙子在学国际象棋，她想陪着一起学。"
            "完全零基础，需要最简单的入门引导。"
            "最关注：是否适合老年人操作，字够不够大，步骤是否清晰。"
            "对竞技排名毫无兴趣，只想能下一盘完整的棋和孙子对弈。"
        ),
        frequency=0.30,
    ),
    Archetype(
        name="张建国型 — 退休干部棋友社交型",
        description="55-65岁，退休，有棋友圈，学国际象棋是为了拓展社交圈或参加老年活动。",
        trait_constraints={},
        background_story=(
            "张建国，58岁，退休干部，原来下象棋，社区老年活动中心开始教国际象棋。"
            "他想跟上社区的棋友，不落后于人。"
            "最关注：能不能找到同龄人一起下慢棋，有没有俱乐部功能。"
            "对子弹棋这种快节奏完全不感兴趣，喜欢通讯棋这种慢节奏。"
        ),
        frequency=0.20,
    ),
    Archetype(
        name="陈晓明型 — 中年职场碎片学习者",
        description="40-50岁，在职，利用通勤/午休碎片时间学棋，有一定棋类基础。",
        trait_constraints={},
        background_story=(
            "陈晓明，44岁，上海中层管理，以前下过象棋，想在地铁上学国际象棋打发时间。"
            "时间碎片化，希望每次5-10分钟就能有收获。"
            "最关注：每日棋题、离线可用、进度保存。"
            "对竞技排名有一点兴趣，但不是核心需求。"
        ),
        frequency=0.15,
    ),
]

# ─── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("User Soul × ELTM 实验：刘德寿专属国际象棋需求文档")
    print("=" * 60)
    print(f"功能候选总数：{len(CHESS_FEATURES)}")
    print(f"Persona 核心：刘德寿（52岁，上海，退休工程师，象棋六段）")
    print(f"Archetype 数量：{len(LIU_ARCHETYPES)}")
    print()

    backend = AnthropicBackend(api_key=API_KEY)
    ff = FeatureFilter(backend)

    print("▶ 运行 FeatureFilter（调用 LLM 做 AARRR 评分）...")
    report: FeatureFilterReport = ff.filter(
        product_description=(
            "一款面向中国中老年用户的国际象棋学习 App，"
            "目标用户是有象棋基础、首次学国际象棋的50岁以上人群。"
            "核心价值：从象棋知识迁移、慢节奏学习、大字体无障碍设计。"
        ),
        raw_features=CHESS_FEATURES,
        target_segment="50-65岁有象棋基础的中国用户，首次学国际象棋，不爱竞技，重视学习和无障碍",
        archetypes=LIU_ARCHETYPES,
        top_n=20,
    )

    print(f"✓ 完成。must_have={len(report.must_have)}, "
          f"nice_to_have={len(report.nice_to_have)}, "
          f"skip={len(report.skip)}")
    print()

    prd = _render_prd(report)
    out_path = Path("/Users/mozat/a-docs/chess_liudeshou_prd.md")
    out_path.write_text(prd, encoding="utf-8")
    print(f"✓ PRD 已写入：{out_path}")
    print()
    print(prd[:3000])


def _render_prd(r: FeatureFilterReport) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# 国际象棋 App — 刘德寿专属需求文档",
        f"",
        f"> 生成时间：{now}  ",
        f"> 目标用户：{r.target_segment}  ",
        f"> 功能输入：{r.total_input} 个（来源：chess.com benchmark 扫描）  ",
        f"> User Soul 过滤后：{len(r.top_features)} 个核心功能",
        f"",
        f"---",
        f"",
        f"## Persona 核心：刘德寿",
        f"",
        f"| 属性 | 内容 |",
        f"|------|------|",
        f"| 年龄 | 52岁，上海，退休工程师 |",
        f"| 棋类背景 | 象棋六段，无国际象棋经验 |",
        f"| 学习动机 | 与在美国工作的儿子共同语言 |",
        f"| 核心诉求 | 系统学习、象棋知识迁移、慢节奏、无障碍 |",
        f"| 反感点 | 竞技压力、小字体、快节奏对弈、复杂社交 |",
        f"",
        f"**围绕刘德寿的用户群：**",
    ]
    for a in r.archetypes_used:
        lines.append(f"- {a}")
    lines += [
        f"",
        f"---",
        f"",
        f"## Phase 0.7 功能分层结果",
        f"",
        f"### ✅ Must Have（优先级分 ≥ 0.60）— {len(r.must_have)} 个",
        f"",
        f"| 功能 | 分类 | 优先级分 | AARRR 亮点 |",
        f"|------|------|----------|------------|",
    ]
    for f in sorted(r.must_have, key=lambda x: x.priority_score, reverse=True):
        aarrr_highlight = _aarrr_highlight(f)
        lines.append(
            f"| **{f.name}** | {f.category} | {f.priority_score:.2f} | {aarrr_highlight} |"
        )
    lines += [
        f"",
        f"### ⚠️ Nice to Have（0.35 ≤ 分 < 0.60）— {len(r.nice_to_have)} 个",
        f"",
        f"| 功能 | 分类 | 优先级分 |",
        f"|------|------|----------|",
    ]
    for f in sorted(r.nice_to_have, key=lambda x: x.priority_score, reverse=True):
        lines.append(f"| {f.name} | {f.category} | {f.priority_score:.2f} |")
    lines += [
        f"",
        f"### ❌ Skip（分 < 0.35）— {len(r.skip)} 个",
        f"",
        f"| 功能 | 分类 | 优先级分 | 理由 |",
        f"|------|------|----------|------|",
    ]
    for f in sorted(r.skip, key=lambda x: x.priority_score, reverse=True):
        reason = _skip_reason(f)
        lines.append(f"| {f.name} | {f.category} | {f.priority_score:.2f} | {reason} |")
    lines += [
        f"",
        f"---",
        f"",
        f"## 最终 PRD 功能范围（Top {len(r.top_features)}）",
        f"",
        f"按优先级排序，这是刘德寿专属版本的完整开发范围：",
        f"",
    ]
    for i, f in enumerate(r.top_features, 1):
        lines.append(f"### {i}. {f.name}")
        lines.append(f"")
        lines.append(f"- **分类**：{f.category} | **来源**：{f.source}")
        lines.append(f"- **描述**：{f.description}")
        tier_label = "Must Have" if f.classification == "must_have" else "Nice to Have"
        lines.append(f"- **优先级分**：{f.priority_score:.2f}（{tier_label}）")
        a = f.aarrr
        lines.append(
            f"- **AARRR**：留存 {a.retention:.2f} · 激活 {a.activation:.2f} · "
            f"获客 {a.acquisition:.2f} · 营收 {a.revenue:.2f} · 口碑 {a.referral:.2f}"
            f"  _(置信度 {a.confidence:.2f})_"
        )
        lines.append(f"")

    lines += [
        f"---",
        f"",
        f"## 与通用 Chess App PRD 的关键差异",
        f"",
        f"| 维度 | 通用 Chess App | 刘德寿版本 |",
        f"|------|--------------|-----------|",
        f"| 对弈模式 | 子弹棋/快棋为核心 | 慢棋/通讯棋/人机对战为核心 |",
        f"| 学习重点 | 开局库、战术题 | 象棋迁移提示、残局、规则教程 |",
        f"| 社交功能 | ELO排名、全球对战 | 本地棋友圈、无排名压力 |",
        f"| 无障碍 | 标准字体 | 大字体、简化操作 |",
        f"| 竞技功能 | 核心功能 | Skip |",
        f"| 中文化 | 可选 | Must Have |",
        f"",
        f"---",
        f"",
        f"_由 User Soul × ELTM 协同生成。功能评分基于 {len(LIU_ARCHETYPES)} 个 Persona Archetype 的 AARRR 仿真投票。_",
    ]
    return "\n".join(lines)


def _aarrr_highlight(f: ScoredFeature) -> str:
    a = f.aarrr
    dims = {
        "留存": a.retention, "激活": a.activation,
        "获客": a.acquisition, "营收": a.revenue, "口碑": a.referral,
    }
    top = max(dims, key=dims.get)
    return f"{top} {dims[top]:.2f}"


def _skip_reason(f: ScoredFeature) -> str:
    a = f.aarrr
    if f.category in ("social",) and a.retention < 0.35:
        return "竞技/社交对目标用户无吸引力"
    if "bullet" in f.id or "blitz" in f.id:
        return "快节奏与目标用户不匹配"
    if "stream" in f.id or "960" in f.id:
        return "高级玩法，超出初学者需求"
    if a.acquisition < 0.3 and a.retention < 0.3:
        return "AARRR 全维度偏低"
    return f"优先级分 {f.priority_score:.2f}"


if __name__ == "__main__":
    main()
