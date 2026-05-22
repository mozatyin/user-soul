import os
"""Real playtest: Playwright drives the actual TTT game, LLM picks actions by looking at screenshots."""
import sys, os, time, types

# Bypass code_soul.__init__ (has Python 3.9 union type issue)
sys.path.insert(0, "/Users/mozat/code-soul")
sys.modules['code_soul'] = types.ModuleType('code_soul')
sys.modules['code_soul'].__path__ = ['/Users/mozat/code-soul/code_soul']

from code_soul.playtest.runner import run_playtest
from code_soul.playtest.persona import PersonaProfile
from code_soul.playtest.report import FrictionReport
from code_soul.playtest.action_picker import pick_action, _coerce_action, _build_user_prompt, _SYSTEM_PROMPT

sys.path.insert(0, "/Users/mozat/mcv")
from user_soul.backends.anthropic import AnthropicBackend

API_KEY = os.environ["OPENROUTER_API_KEY"]
HTML_PATH = "/Users/mozat/pm-soul/projects/tictactoe_test/v1/index.html"
_backend = AnthropicBackend(api_key=API_KEY)

def _llm_caller(system, user, image_b64, api_key):
    """Use User-Soul's AnthropicBackend for vision calls."""
    if image_b64:
        import base64
        img_bytes = base64.b64decode(image_b64)
        prompt = f"{system}\n\n{user}"
        return _backend.vision(prompt, [img_bytes], max_tokens=400, model_tier="smart")
    else:
        prompt = f"{system}\n\n{user}"
        return _backend.text(prompt, max_tokens=400, model_tier="fast")

def _custom_action_picker(persona, screenshot, clickable, history, turn, k_turns, api_key=""):
    return pick_action(persona, screenshot, clickable, history, turn, k_turns,
                       api_key=api_key, llm_caller=_llm_caller)

PERSONAS = [
    PersonaProfile(
        name="casual_first_timer",
        profile=(
            "First-time user who doesn't know how to play. "
            "Clicks the most obvious button. Gives up if stuck for 3 turns."
        ),
        goals=["play one complete game against AI", "understand how to win"],
    ),
    PersonaProfile(
        name="competitive_gamer",
        profile=(
            "Experienced mobile gamer. Expects smooth UX. "
            "Will try to beat the AI. Gets bored if no challenge."
        ),
        goals=["try vs AI mode", "try to beat AI", "explore all game modes"],
    ),
    PersonaProfile(
        name="curious_explorer",
        profile=(
            "Pokes every button and menu. Wants to see all features. "
            "Frustrated by dead-ends or unfinished features."
        ),
        goals=["visit every screen", "try Online mode", "check all buttons"],
    ),
]

def on_progress(event):
    kind = event.get("kind", "")
    persona = event.get("persona", "")
    turn = event.get("turn", "")
    if kind == "action":
        action = event.get("action", {})
        act = action.get("action", "?")
        sel = action.get("selector", "")
        reason = action.get("reason", "")
        print(f"  [{persona}] turn {turn}: {act} {sel[:40]} — {reason[:60]}")
    elif kind == "friction":
        friction = event.get("friction", "")
        detail = event.get("detail", "")
        print(f"  [{persona}] turn {turn}: FRICTION ({friction}) {detail[:80]}")

def main():
    print(f"=== Real Playtest: Tic-Tac-Toe ===")
    print(f"HTML: {HTML_PATH}")
    print(f"Personas: {len(PERSONAS)}")
    print(f"Turns per persona: 12\n")

    t0 = time.time()
    report = run_playtest(
        html_path=HTML_PATH,
        personas=PERSONAS,
        k_turns=12,
        api_key=API_KEY,
        on_progress=on_progress,
        action_picker=_custom_action_picker,
    )
    elapsed = time.time() - t0

    print(f"\n=== Results ===\n")
    print(report.summary)
    print()

    if report.dead_ends:
        print(f"Dead ends:")
        for de in report.dead_ends:
            print(f"  - turn {de['turn']} ({de['persona']}): {de['detail'][:100]}")
        print()

    for pr in report.per_persona_results:
        print(f"--- {pr.persona} ---")
        print(f"  turns: {pr.turns_taken}, friction: {pr.friction_count}, completed: {pr.completed}, gave_up: {pr.gave_up}")
        for ev in pr.friction_events:
            print(f"  [{ev.kind}] turn {ev.turn}: {ev.detail[:100]}")
        print()

    print(f"Total time: {elapsed:.1f}s")

if __name__ == "__main__":
    main()
