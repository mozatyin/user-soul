import os
"""End-to-end: Graded playtest on Checkers — verify full KGVT pipeline."""
import sys, time, json
sys.path.insert(0, "/Users/mozat/user-soul")

from user_soul.backends.anthropic import AnthropicBackend
from user_soul.engines.persona import PersonaEngine
from user_soul.playtest_bridge import run_graded_playtest
from user_soul.action_router import route_diagnosis, group_by_owner, format_action_summary
from user_soul.game_knowledge import GameKnowledge, KnowledgeTier, brief_for_tier

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
HTML_PATH = "/Users/mozat/pm-soul/projects/checkers_test/v1/index.html"

CHECKERS_GDD = {
    "game_name": "Checkers",
    "formal_rules": {
        "game_loop_type": "turn-based",
        "objective": "Capture all opponent pieces or block them from moving",
        "turn_structure": (
            "Players alternate turns. Each turn: "
            "1) Click one of your pieces to select it, "
            "2) Click an empty diagonal square to move there. "
            "Regular pieces move forward diagonally one square. "
            "To capture, jump over an opponent's piece to an empty square beyond it."
        ),
        "win_conditions": [
            "Capture all opponent pieces",
            "Block opponent so they cannot make any legal move",
        ],
        "special_rules": [
            "Pieces can only move diagonally on dark squares",
            "Regular pieces can only move forward (toward opponent's side)",
            "Capture is mandatory — if you can jump, you must jump",
            "Multi-jump: if after a jump another jump is available, you must continue jumping",
            "When a piece reaches the far end of the board, it becomes a King",
            "Kings can move diagonally both forward and backward",
            "You must select your own piece first, then select the destination",
            "Cannot move to an occupied square",
            "The board is 8x8 with pieces only on dark squares",
            "Red/black pieces start on opposite sides",
            "The center of the board is strategically advantageous for controlling the game",
            "Trading pieces when ahead is a strong strategy",
            "Keeping pieces on the back row prevents opponent kings",
            "A fork attack threatens two captures simultaneously",
        ],
        "setup": {
            "players": "2 (Red vs Black, or Player vs AI)",
            "initial_state": "Each player starts with 12 pieces on the dark squares of the first 3 rows",
        },
    },
    "render_hints": {"board_type": "grid_8x8"},
    "game_modes": [
        {"name": "vs AI"},
        {"name": "Local PvP"},
    ],
}


def on_progress(event):
    kind = event.get("kind", "")
    persona = event.get("persona", "")[:12]
    turn = event.get("turn", "")
    tier = event.get("tier", "?")
    if kind == "action":
        a = event.get("action", {})
        print(f"  [{tier:8s}] [{persona}] t{turn}: {a.get('action','?')} {a.get('selector','')[:30]} — {a.get('reason','')[:45]}")
    elif kind == "friction":
        print(f"  [{tier:8s}] [{persona}] t{turn}: FRICTION ({event.get('friction','')}) {event.get('detail','')[:60]}")


def main():
    backend = AnthropicBackend(api_key=API_KEY)

    # Step 1: Show knowledge extraction
    print("=" * 60)
    print("  Step 1: Knowledge Extraction from GDD")
    print("=" * 60)
    knowledge = GameKnowledge.from_gdd(CHECKERS_GDD)
    print(f"  Game: {knowledge.game_name}")
    print(f"  Category: {knowledge.game_category}")
    print(f"  Objective: {knowledge.objective}")
    print(f"  Interaction rules: {len(knowledge.interaction_rules)}")
    for r in knowledge.interaction_rules:
        print(f"    - {r}")
    print(f"  Strategy (FILTERED OUT): {len(knowledge.strategy_hints)}")
    for s in knowledge.strategy_hints:
        print(f"    [HIDDEN] {s}")

    print(f"\n  --- Tier Briefings ---")
    for tier in KnowledgeTier.all_tiers():
        brief = brief_for_tier(knowledge, tier)
        print(f"\n  [{tier.value.upper()}] ({len(brief)} chars):")
        for line in brief.strip().split("\n")[:5]:
            print(f"    {line[:80]}")
        if brief.count("\n") > 5:
            print(f"    ... ({brief.count(chr(10)) - 5} more lines)")

    # Step 2: Generate personas
    print(f"\n{'=' * 60}")
    print(f"  Step 2: Generate Personas")
    print(f"{'=' * 60}")
    t0 = time.time()
    engine = PersonaEngine(backend)
    pool = engine.get_or_create(
        "Checkers — classic board game with diagonal moves, jumping captures, "
        "king promotion. vs AI and local PvP.", n=6)
    print(f"  {len(pool)} personas in {time.time()-t0:.0f}s")
    for p in pool:
        print(f"    [{p.archetype_name}] {p.background_story[:60]}")

    # Step 3: Run graded playtest
    print(f"\n{'=' * 60}")
    print(f"  Step 3: Graded Playtest (3 tiers × 2 personas × 10 turns)")
    print(f"{'=' * 60}")
    t1 = time.time()
    feedback = run_graded_playtest(
        HTML_PATH, pool, backend, CHECKERS_GDD,
        k_turns=10, on_progress=on_progress,
    )
    t2 = time.time()

    # Step 4: Results
    print(f"\n{'=' * 60}")
    print(f"  Step 4: Results")
    print(f"{'=' * 60}")
    print(f"  Aggregate: {feedback.verdict} ({feedback.score:.0f}/100)")
    print(f"  Completed: {feedback.personas_completed}/{feedback.personas_total}")
    print(f"  Time: {t2-t1:.0f}s")

    print(f"\n  --- Per-Tier Scores ---")
    for tier, score in feedback.tier_scores.items():
        tier_fb = feedback.tier_feedbacks[tier]
        print(f"  {tier.upper():10s}  {score:.0f}/100  ({tier_fb.verdict})  "
              f"{tier_fb.personas_completed}/{tier_fb.personas_total} completed")

    print(f"\n  --- Differential Diagnosis ---")
    if feedback.diagnosis:
        print(feedback.diagnosis.summary)

    # Step 5: Action routing
    print(f"\n{'=' * 60}")
    print(f"  Step 5: PM-Soul Action Routing")
    print(f"{'=' * 60}")
    action_specs = route_diagnosis(feedback)
    print(format_action_summary(action_specs))

    by_owner = group_by_owner(action_specs)
    print(f"\n  --- Dispatch Summary ---")
    for owner in ["code-soul", "eltm", "pm-soul"]:
        specs = by_owner.get(owner, [])
        if specs:
            print(f"  {owner}: {len(specs)} action(s)")
            for s in specs:
                print(f"    [{s.severity}] {s.action_type}: {s.description[:70]}")
                if s.payload:
                    print(f"    payload keys: {list(s.payload.keys())}")

    print(f"\n  Primary owner: {feedback.primary_owner}")
    print(f"  Has blockers: {feedback.has_blockers}")
    print(f"  Total time: {t2-t0:.0f}s")


if __name__ == "__main__":
    main()
