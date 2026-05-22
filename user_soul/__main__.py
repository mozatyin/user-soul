"""Offline simulation CLI — python -m mcv"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def trigger_background_simulation(
    state_dir: Path,
    n: int = 100,
) -> None:
    """Spawn background process to run full simulation. Does not block."""
    subprocess.Popen(
        [sys.executable, "-m", "mcv", "--state", str(state_dir), "--n", str(n)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_simulation(
    state_dir: Path,
    n: int,
    api_key: str,
) -> None:
    """Load features + personas from state_dir, run N simulations, write cache."""
    from user_soul.simulator import PersonaSimulator
    from user_soul.cache import save_simulation_cache

    features_path = state_dir / "features_for_simulation.json"
    personas_path = state_dir / "personas.json"

    if not features_path.exists():
        print(f"[mcv] No features_for_simulation.json in {state_dir} — skipping", flush=True)
        return
    if not personas_path.exists():
        print(f"[mcv] No personas.json in {state_dir} — skipping", flush=True)
        return

    features = json.loads(features_path.read_text(encoding="utf-8"))
    personas = json.loads(personas_path.read_text(encoding="utf-8"))

    print(f"[mcv] Running {n} simulations × {len(personas)} personas for {len(features)} features...", flush=True)
    sim = PersonaSimulator(personas, api_key)
    signals = sim.simulate(features, n_runs=n)
    save_simulation_cache(state_dir, features, signals, status="complete")
    print(f"[mcv] Done. Cache written to {state_dir}/simulation_cache.json", flush=True)


def main() -> None:
    import argparse, os
    parser = argparse.ArgumentParser(description="mcv offline simulation runner")
    parser.add_argument("--state", required=True, help="State directory path")
    parser.add_argument("--n", type=int, default=100, help="Number of simulation runs")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[mcv] ERROR: ANTHROPIC_API_KEY not set", flush=True)
        sys.exit(1)

    run_simulation(state_dir=Path(args.state), n=args.n, api_key=api_key)


if __name__ == "__main__":
    main()
