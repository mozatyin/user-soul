"""Deterministic juice/game-feel scorer for HTML games.

Scans HTML for sensory feedback signals (audio, animation, particles,
haptic, screen shake) and produces a 0-100 quality score. No LLM calls.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class JuiceReport:
    score: float
    verdict: str
    breakdown: dict = field(default_factory=dict)


_SIGNAL_PATTERNS: list[tuple[str, str, int]] = [
    # (signal_name, regex_pattern, max_score_contribution)

    # Audio signals
    ("audio_context", r"new\s+(?:window\.)?AudioContext|new\s+(?:window\.)?webkitAudioContext", 8),
    ("audio_play", r"\.play\s*\(|\.playTone\s*\(|playSound\s*\(|playSfx\s*\(", 7),
    ("tone_freq", r"\.frequency\.(?:value|setValueAtTime)|createOscillator|oscillator", 5),

    # Animation signals
    ("keyframes", r"@keyframes\s+\w+", 8),
    ("css_transitions", r"transition\s*:\s*[^;]+(?:ease|linear|cubic-bezier|all\s+[\d.]+s)", 6),
    ("easing", r"cubic-bezier\s*\(|ease-(?:in|out|in-out)\b|easeOutBounce|easeInOut", 4),
    ("transform_anim", r"transform\s*:\s*(?:scale|rotate|translate)|\.style\.transform\s*=", 5),
    ("opacity_anim", r"opacity\s*:\s*[01]|\.style\.opacity\s*=", 3),

    # Particle/confetti signals
    ("particles", r"particle|confetti|burst|sparkle|emit(?:ter)?", 8),

    # Screen shake
    ("screen_shake", r"shake|screen.?shake|camera.?shake|offset.*random|translateX.*random", 6),

    # Haptic feedback
    ("haptic", r"navigator\.vibrate\s*\(|vibrateSafe\s*\(|vibrate\s*\(", 5),

    # Visual feedback
    ("flash_effect", r"flash|blink|pulse|glow|highlight.*anim", 4),
    ("color_gradient", r"linear-gradient|radial-gradient|createLinearGradient|createRadialGradient", 3),
    ("shadow_effect", r"box-shadow\s*:|text-shadow\s*:|shadowBlur|shadowColor", 3),

    # Game feel
    ("score_popup", r"popup|float.*text|damage.*number|score.*anim|\+\d+.*anim", 4),
    ("combo_system", r"combo|streak|chain|multiplier", 4),
    ("hit_effect", r"hit.?flash|hit.?effect|knockback|stun|impact|recoil", 5),
    ("smooth_movement", r"lerp|interpolat|smooth|tween|requestAnimationFrame", 4),

    # UI polish
    ("responsive_buttons", r"button.*hover|:hover\s*\{|btn.*active|:active\s*\{|cursor:\s*pointer", 3),
    ("loading_state", r"loading|spinner|progress.?bar|\.load", 2),
    ("game_over_anim", r"game.?over.*anim|victory.*anim|win.*screen|result.*screen", 3),
]


def score_juice(html_path: str) -> JuiceReport:
    """Score an HTML game file for juice/game-feel signals."""
    path = Path(html_path)
    if not path.exists():
        return JuiceReport(score=0, verdict="UNKNOWN", breakdown={})

    html = path.read_text(encoding="utf-8", errors="replace")
    html_lower = html.lower()

    breakdown: dict[str, int] = {}
    raw_score = 0.0
    max_possible = 0.0

    for signal_name, pattern, max_contrib in _SIGNAL_PATTERNS:
        matches = re.findall(pattern, html_lower if signal_name not in (
            "keyframes", "css_transitions", "audio_context",
        ) else html, re.IGNORECASE)
        count = len(matches)
        breakdown[signal_name] = count

        if count > 0:
            # Diminishing returns: first match = full value, extras add less
            signal_score = min(max_contrib, max_contrib * (0.6 + 0.4 * min(count, 5) / 5))
            raw_score += signal_score

        max_possible += max_contrib

    # Normalize to 0-100
    score = round(min(100, (raw_score / max_possible) * 100), 1) if max_possible > 0 else 0

    # Verdict thresholds
    if score >= 70:
        verdict = "ENTERPRISE"
    elif score >= 45:
        verdict = "POLISHED"
    elif score >= 20:
        verdict = "BASIC"
    else:
        verdict = "TOY"

    return JuiceReport(score=score, verdict=verdict, breakdown=breakdown)
