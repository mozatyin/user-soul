"""Tests for user_soul.juice_scorer — the deterministic polish pre-gate.

juice_scorer is the ONLY unconditional User-Soul dependency in ELTM's renovate
loop (eltm/renovate.py:135 score_juice — no try/except, runs every round), yet it
shipped with zero tests. These tests lock in the scoring contract and add
regression guards for the §8.E false-positive fixes (commit 5f97e5b):
screen_shake / particles / flash_pulse must require call/keyframe forms, not bare
variable names.
"""
from __future__ import annotations

from user_soul.juice_scorer import score_juice, score_juice_dict, _SIGNAL_PATTERNS, JuiceReport


def _write(tmp_path, body: str):
    p = tmp_path / "game.html"
    p.write_text(f"<!doctype html><html><body>{body}</body></html>", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 1. Verdict tiers + normalization
# ---------------------------------------------------------------------------

def test_toy_game_scores_low(tmp_path):
    p = _write(tmp_path, "<h1>hi</h1><p>just text, no juice at all</p>")
    r = score_juice(p)
    assert isinstance(r, JuiceReport)
    assert r.score < 20
    assert r.verdict == "TOY"
    assert r.html_size > 0


def test_loaded_game_scores_high(tmp_path):
    body = """
    <style>
      @keyframes a{} @keyframes b{} @keyframes c{} @keyframes d{}
      @keyframes e{} @keyframes f{} @keyframes g{} @keyframes h{}
      .x{transition: all .3s; transform: scale(1.1); filter: blur(2px);
         box-shadow: 0 0 4px; background: linear-gradient(#fff,#000);
         animation-timing-function: cubic-bezier(.2,.8,.2,1);}
    </style>
    <script src="gsap.min.js"></script>
    <script>
      lottie.loadAnimation({});
      const ac = new AudioContext(); ac.createOscillator(); ac.createGain();
      new Audio('x.mp3').play(); playSound(); playSfx();
      const ctx = c.getContext('webgl');
      requestAnimationFrame(loop);
      function loop(){ screenShake(); createBurst(); spawnBurst(); confetti();
                       navigator.vibrate(20); }
      gsap.to(el,{x:1}); document.addEventListener('animationend', f);
      el.flash(); sparkle();
    </script>
    """
    r = score_juice(_write(tmp_path, body))
    assert r.score >= 45
    assert r.verdict in ("POLISHED", "ENTERPRISE")


def test_score_is_bounded_0_100(tmp_path):
    # 50 keyframes — far over the cap of 8 — score must still be <= 100.
    body = "<style>" + "".join(f"@keyframes k{i}{{}}" for i in range(50)) + "</style>"
    r = score_juice(_write(tmp_path, body))
    assert 0.0 <= r.score <= 100.0
    # raw hit count is reported uncapped in breakdown
    assert r.breakdown["keyframes"] == 50


def test_breakdown_covers_all_signals(tmp_path):
    r = score_juice(_write(tmp_path, "<p>x</p>"))
    assert set(r.breakdown) == {p[0] for p in _SIGNAL_PATTERNS}
    assert all(v == 0 for v in r.breakdown.values())


def test_dict_wrapper_matches(tmp_path):
    p = _write(tmp_path, "<script>screenShake();</script>")
    d = score_juice_dict(p)
    assert set(d) == {"score", "verdict", "breakdown", "html_size"}
    assert d["score"] == score_juice(p).score


# ---------------------------------------------------------------------------
# 2. §8.E false-positive regression guards (commit 5f97e5b)
# ---------------------------------------------------------------------------

def test_screen_shake_ignores_variable_name(tmp_path):
    # "var shake = false" must NOT count as screen shake juice.
    fp = score_juice(_write(tmp_path, "<script>var shake = false; let milkshake = 1;</script>"))
    assert fp.breakdown["screen_shake"] == 0
    # but a real call does count
    tp = score_juice(_write(tmp_path, "<script>screenShake();</script>"))
    assert tp.breakdown["screen_shake"] >= 1


def test_particles_ignores_bare_constant(tmp_path):
    # bare BURST_SIZE constant is not particle juice
    fp = score_juice(_write(tmp_path, "<script>const BURST_SIZE = 10;</script>"))
    assert fp.breakdown["particles"] == 0
    # a real burst function call does count
    tp = score_juice(_write(tmp_path, "<script>createBurst(x,y);</script>"))
    assert tp.breakdown["particles"] >= 1


def test_flash_effect_ignores_card_message_naming(tmp_path):
    # flashCard / flashMessage are UI naming, not flash juice (no blink/pulse/glow words)
    fp = score_juice(_write(tmp_path, "<script>flashCard(); flashMessage();</script>"))
    assert fp.breakdown["flash_effect"] == 0
    # a real flash effect call does count
    tp = score_juice(_write(tmp_path, "<script>flashEffect();</script>"))
    assert tp.breakdown["flash_effect"] >= 1
