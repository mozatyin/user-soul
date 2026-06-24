import os
"""Full loop: Playtest Checkers → PM-Soul routes → Fix → Re-test."""
import sys, time
sys.path.insert(0, "/Users/mozat/user-soul")

from user_soul.backends.anthropic import AnthropicBackend
from user_soul.engines.persona import PersonaEngine
from user_soul.playtest_bridge import run_user_playtest

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
HTML_PATH = "/Users/mozat/pm-soul/projects/checkers_test/v1/index.html"

def on_progress(event):
    kind = event.get("kind", "")
    persona = event.get("persona", "")[:12]
    turn = event.get("turn", "")
    if kind == "action":
        a = event.get("action", {})
        print(f"  [{persona}] t{turn}: {a.get('action','?')} {a.get('selector','')[:35]} — {a.get('reason','')[:50]}")
    elif kind == "friction":
        print(f"  [{persona}] t{turn}: FRICTION ({event.get('friction','')}) {event.get('detail','')[:70]}")

def main():
    backend = AnthropicBackend(api_key=API_KEY)

    print("=== Checkers Playtest ===\n")
    print("Generating personas...")
    t0 = time.time()
    engine = PersonaEngine(backend)
    pool = engine.get_or_create(
        "Checkers (跳棋/西洋跳棋) — classic board game with jumping captures, "
        "king promotion, mandatory captures. vs AI and local PvP modes.", n=3)
    print(f"  {len(pool)} personas in {time.time()-t0:.0f}s")
    for p in pool:
        print(f"    [{p.archetype_name}] {p.background_story[:70]}")

    print(f"\nPlaytest (k=15 turns per persona)...")
    t1 = time.time()
    feedback = run_user_playtest(
        HTML_PATH, pool, backend,
        k_turns=15, on_progress=on_progress,
    )
    t2 = time.time()

    print(f"\n{'='*60}")
    print(f"  Results: {feedback.verdict} ({feedback.score:.0f}/100)")
    print(f"  Completed: {feedback.personas_completed}/{feedback.personas_total}")
    print(f"  Time: {t2-t1:.0f}s")
    print(f"{'='*60}")

    if feedback.issues:
        print(f"\n  Issues:")
        for iss in feedback.issues:
            print(f"    [{iss.severity}] ({iss.category}) {iss.description}")
            for ev in iss.evidence[:2]:
                print(f"      {ev[:90]}")

    if feedback.suggestions:
        print(f"\n  Suggestions:")
        for s in feedback.suggestions[:5]:
            print(f"    - {s[:120]}")

    print(f"\n  Raw summary:\n  {feedback.raw_summary}")

    # PM-Soul routing
    print(f"\n{'='*60}")
    print(f"  PM-Soul Decision")
    print(f"{'='*60}")

    code_fixes = [i for i in feedback.issues if i.category != "design_issue"]
    design_issues = [i for i in feedback.issues if i.category == "design_issue"]

    for i in code_fixes:
        print(f"  → Code-Soul: [{i.severity}] {i.description[:70]}")
    for i in design_issues:
        print(f"  → ELTM:      [{i.severity}] {i.description[:70]}")

    if feedback.verdict == "PASS":
        print(f"\n  Decision: SHIP")
    elif feedback.has_blockers:
        print(f"\n  Decision: BLOCK — fix P0 issues before next round")
    else:
        print(f"\n  Decision: ITERATE — fix P1 issues, schedule re-test")

if __name__ == "__main__":
    main()
