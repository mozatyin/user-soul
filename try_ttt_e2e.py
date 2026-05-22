import os
"""End-to-end: UserSoulClient.playtest() on real tic-tac-toe game."""
import sys, time
sys.path.insert(0, "/Users/mozat/mcv")

from user_soul.backends.anthropic import AnthropicBackend
from user_soul.engines.persona import PersonaEngine
from user_soul.playtest_bridge import run_user_playtest
from user_soul.models import AgentProfile

API_KEY = os.environ["OPENROUTER_API_KEY"]
HTML_PATH = "/Users/mozat/pm-soul/projects/tictactoe_test/v1/index.html"

def on_progress(event):
    kind = event.get("kind", "")
    persona = event.get("persona", "")
    turn = event.get("turn", "")
    if kind == "action":
        action = event.get("action", {})
        print(f"  [{persona}] turn {turn}: {action.get('action','?')} {action.get('selector','')[:30]} — {action.get('reason','')[:50]}")
    elif kind == "friction":
        print(f"  [{persona}] turn {turn}: FRICTION ({event.get('friction','')}) {event.get('detail','')[:80]}")

def main():
    print("=== Playtest 端到端 (预生成人设) ===\n")
    backend = AnthropicBackend(api_key=API_KEY)

    # Pre-generate 3 personas to save time
    print("[1] 生成人设...")
    t0 = time.time()
    engine = PersonaEngine(backend)
    pool = engine.get_or_create("Tic-Tac-Toe mobile game", n=3)
    print(f"  {len(pool)} personas in {time.time()-t0:.1f}s")
    for p in pool:
        print(f"    [{p.archetype_name}] {p.background_story[:60]}")
    print()

    # Run playtest
    print("[2] Playwright playtest (k=10)...")
    t1 = time.time()
    feedback = run_user_playtest(
        HTML_PATH, pool, backend,
        k_turns=10, on_progress=on_progress,
    )
    t2 = time.time()
    print(f"\n  Playtest done in {t2-t1:.1f}s")

    print(f"\n=== PlaytestFeedback ===\n")
    print(f"  Verdict:   {feedback.verdict}")
    print(f"  Score:     {feedback.score:.1f}/100")
    print(f"  Completed: {feedback.personas_completed}/{feedback.personas_total}")
    print()

    if feedback.issues:
        print("  Issues:")
        for iss in feedback.issues:
            print(f"    [{iss.severity}] ({iss.category}) {iss.description}")
            for ev in iss.evidence[:2]:
                print(f"      evidence: {ev[:100]}")
        print()

    if feedback.suggestions:
        print("  Suggestions:")
        for s in feedback.suggestions[:3]:
            print(f"    - {s[:120]}")
        print()

    print(f"  Raw summary:\n  {feedback.raw_summary}")
    print(f"\n  Has blockers: {feedback.has_blockers}")
    print(f"  Total time: {t2-t0:.1f}s")

if __name__ == "__main__":
    main()
