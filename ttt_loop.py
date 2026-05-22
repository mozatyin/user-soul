"""Full loop: Playtest → PM-Soul routes → Fix → Re-test."""
import os
import sys, time, json, copy
sys.path.insert(0, "/Users/mozat/user-soul")
sys.path.insert(0, "/Users/mozat/pm-soul")

from user_soul.backends.anthropic import AnthropicBackend
from user_soul.engines.persona import PersonaEngine
from user_soul.playtest_bridge import run_user_playtest

API_KEY = os.environ["OPENROUTER_API_KEY"]
V1_PATH = "/Users/mozat/mcv/ttt_v1.html"
V2_PATH = "/Users/mozat/mcv/ttt_v2.html"

def on_progress(event):
    kind = event.get("kind", "")
    persona = event.get("persona", "")
    turn = event.get("turn", "")
    if kind == "action":
        action = event.get("action", {})
        print(f"    [{persona}] t{turn}: {action.get('action','?')} {action.get('selector','')[:30]} — {action.get('reason','')[:45]}")
    elif kind == "friction":
        print(f"    [{persona}] t{turn}: ⚠ {event.get('friction','')} — {event.get('detail','')[:60]}")

def run_playtest_round(label, html_path, pool, backend):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    t0 = time.time()
    feedback = run_user_playtest(
        html_path, pool, backend,
        k_turns=10, on_progress=on_progress,
    )
    elapsed = time.time() - t0
    print(f"\n  --- {label} Results ---")
    print(f"  Verdict:   {feedback.verdict}")
    print(f"  Score:     {feedback.score:.0f}/100")
    print(f"  Completed: {feedback.personas_completed}/{feedback.personas_total}")
    if feedback.issues:
        print(f"  Issues:")
        for iss in feedback.issues:
            print(f"    [{iss.severity}] ({iss.category}) {iss.description}")
    if feedback.suggestions:
        print(f"  Suggestions:")
        for s in feedback.suggestions[:3]:
            print(f"    - {s[:100]}")
    print(f"  Time: {elapsed:.0f}s")
    return feedback

def pm_soul_route(feedback):
    """PM-Soul reads feedback and decides actions."""
    print(f"\n{'='*60}")
    print(f"  PM-Soul: Routing feedback")
    print(f"{'='*60}\n")

    code_soul_fixes = []
    eltm_requests = []

    for iss in feedback.issues:
        action = {
            "description": iss.description,
            "category": iss.category,
            "severity": iss.severity,
            "evidence": iss.evidence[:2],
        }
        if iss.category == "design_issue":
            eltm_requests.append(action)
            print(f"  → ELTM:      [{iss.severity}] {iss.description[:70]}")
        else:
            code_soul_fixes.append(action)
            print(f"  → Code-Soul:  [{iss.severity}] {iss.description[:70]}")

    if not code_soul_fixes and not eltm_requests:
        print(f"  → No actions needed. Ship it!")

    return code_soul_fixes, eltm_requests

def code_soul_fix(fixes, backend):
    """Code-Soul fixes the game based on PM-Soul's instructions."""
    print(f"\n{'='*60}")
    print(f"  Code-Soul: Applying fixes")
    print(f"{'='*60}\n")

    with open(V1_PATH, "r") as f:
        html = f.read()

    fix_descriptions = "\n".join(
        f"- [{fix['severity']}] {fix['description']}"
        for fix in fixes
    )
    evidence_text = ""
    for fix in fixes:
        for ev in fix.get("evidence", []):
            evidence_text += f"  - {ev}\n"

    prompt = f"""你是一个前端开发者。用户测试发现了以下问题：

{fix_descriptions}

测试证据：
{evidence_text}

当前游戏 HTML 代码如下（完整代码）：
```html
{html}
```

请修复这些问题，输出完整的修复后 HTML。关键修复点：
1. 点击已占用格子时应该有视觉反馈（例如格子闪烁红色）
2. 确保所有按钮在各个屏幕状态下都可以正常点击
3. AI 思考后格子应该正确更新

只输出完整的 HTML 代码，不要任何解释。用 ```html 和 ``` 包裹。"""

    print("  Generating fix...")
    t0 = time.time()
    raw = backend.text(prompt, max_tokens=8000, model_tier="smart")
    elapsed = time.time() - t0
    print(f"  LLM response in {elapsed:.0f}s")

    # Extract HTML from response
    import re
    m = re.search(r'```html\s*(.*?)\s*```', raw, re.DOTALL)
    if m:
        fixed_html = m.group(1)
    elif "<html" in raw.lower():
        start = raw.lower().index("<html") if "<html" in raw.lower() else raw.lower().index("<!doctype")
        fixed_html = raw[start:]
    else:
        print("  ERROR: Could not extract HTML from LLM response")
        return False

    with open(V2_PATH, "w") as f:
        f.write(fixed_html)

    v1_size = len(html)
    v2_size = len(fixed_html)
    print(f"  v1: {v1_size} bytes → v2: {v2_size} bytes")
    print(f"  Saved to {V2_PATH}")
    return True

def main():
    backend = AnthropicBackend(api_key=API_KEY)

    # Generate personas once, reuse for both rounds
    print("Generating personas (shared across rounds)...")
    t0 = time.time()
    engine = PersonaEngine(backend)
    pool = engine.get_or_create("Tic-Tac-Toe mobile game with AI opponent", n=3)
    print(f"  {len(pool)} personas in {time.time()-t0:.0f}s")
    for p in pool:
        print(f"    [{p.archetype_name}] {p.background_story[:60]}")

    # Round 1: Playtest v1
    fb1 = run_playtest_round("Round 1: Playtest v1 (original)", V1_PATH, pool, backend)

    # PM-Soul routes
    code_fixes, eltm_reqs = pm_soul_route(fb1)

    if fb1.verdict == "PASS":
        print("\n  PM-Soul: v1 PASSED. No fixes needed.")
        return

    # Code-Soul fixes
    if code_fixes:
        ok = code_soul_fix(code_fixes, backend)
        if not ok:
            print("  Fix generation failed. Stopping.")
            return

        # Open v2 for human inspection
        import subprocess
        subprocess.run(["open", V2_PATH])
        print(f"\n  v2 已在浏览器打开。")

        # Round 2: Playtest v2
        fb2 = run_playtest_round("Round 2: Playtest v2 (fixed)", V2_PATH, pool, backend)

        # Compare
        print(f"\n{'='*60}")
        print(f"  对比")
        print(f"{'='*60}")
        print(f"  v1: {fb1.score:.0f}/100 ({fb1.verdict}) — {len(fb1.issues)} issues")
        print(f"  v2: {fb2.score:.0f}/100 ({fb2.verdict}) — {len(fb2.issues)} issues")
        delta = fb2.score - fb1.score
        print(f"  Delta: {'+' if delta >= 0 else ''}{delta:.0f} points")

        if fb2.verdict == "PASS":
            print(f"\n  PM-Soul: v2 PASSED. Ready to ship.")
        else:
            print(f"\n  PM-Soul: v2 still {fb2.verdict}. Another iteration needed.")
    else:
        print("\n  No code fixes needed. Issues are design-level → ELTM.")

if __name__ == "__main__":
    main()
