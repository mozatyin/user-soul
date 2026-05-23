"""ExperimentManager — Feature Gates, A/B Experiments, Layers, and Holdouts.

Real experiment assignment with deterministic hash-based user bucketing.
Equivalent to Statsig's core experiment infrastructure but framework-agnostic.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def _bucket(user_id: str, salt: str) -> float:
    """Returns float in [0.0, 1.0) deterministically from user_id + salt."""
    h = hashlib.md5(f"{user_id}:{salt}".encode()).hexdigest()
    return int(h, 16) / (16 ** len(h))


@dataclass
class ExperimentVariant:
    name: str
    weight: float
    config: dict = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    name: str
    variants: list[ExperimentVariant]
    holdout_pct: float = 0.0
    layer: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        total = sum(v.weight for v in self.variants)
        if total > 0 and abs(total - 1.0) > 0.001:
            self.variants = [
                ExperimentVariant(v.name, v.weight / total, v.config)
                for v in self.variants
            ]


@dataclass
class FeatureGateConfig:
    name: str
    rollout_pct: float
    description: str = ""


@dataclass
class HoldoutConfig:
    name: str
    holdout_pct: float
    description: str = ""


@dataclass
class LayerConfig:
    name: str
    description: str = ""


class ExperimentManager:
    def __init__(self) -> None:
        self._experiments: dict[str, ExperimentConfig] = {}
        self._gates: dict[str, FeatureGateConfig] = {}
        self._holdouts: dict[str, HoldoutConfig] = {}
        self._layers: dict[str, list[str]] = {}

    def add_experiment(self, config: ExperimentConfig) -> None:
        self._experiments[config.name] = config
        if config.layer is not None:
            self.add_to_layer(config.layer, config.name)

    def add_gate(self, config: FeatureGateConfig) -> None:
        self._gates[config.name] = config

    def add_holdout(self, config: HoldoutConfig) -> None:
        self._holdouts[config.name] = config

    def add_to_layer(self, layer_name: str, experiment_name: str) -> None:
        if layer_name not in self._layers:
            self._layers[layer_name] = []
        if experiment_name not in self._layers[layer_name]:
            self._layers[layer_name].append(experiment_name)

    def assign(self, user_id: str, experiment_name: str) -> str | None:
        """Assign user to a variant. Returns variant name, or None if in holdout.

        Algorithm:
        1. Check per-experiment holdout.
        2. Check layer mutual exclusivity — if user's layer slot doesn't map to
           this experiment, return "control" so callers always get a valid name.
        3. Assign to variant using cumulative weights.
        """
        exp = self._experiments.get(experiment_name)
        if exp is None:
            return None

        if exp.holdout_pct > 0.0:
            if _bucket(user_id, f"holdout:{experiment_name}") < exp.holdout_pct:
                return None

        if exp.layer is not None:
            layer_exps = sorted(self._layers.get(exp.layer, []))
            if len(layer_exps) > 1:
                b = _bucket(user_id, f"layer:{exp.layer}")
                slot = int(b * len(layer_exps))
                assigned_exp = layer_exps[slot]
                if assigned_exp != experiment_name:
                    control = next(
                        (v.name for v in exp.variants if v.name == "control"),
                        exp.variants[0].name,
                    )
                    return control

        b = _bucket(user_id, f"variant:{experiment_name}")
        cumulative = 0.0
        for v in exp.variants:
            cumulative += v.weight
            if b < cumulative:
                return v.name
        return exp.variants[-1].name

    def check_gate(self, user_id: str, gate_name: str) -> bool:
        """Evaluate a feature gate. Returns True if user is in the rollout."""
        gate = self._gates.get(gate_name)
        if gate is None:
            return False
        return _bucket(user_id, f"gate:{gate_name}") < gate.rollout_pct

    def is_in_holdout(self, user_id: str, holdout_name: str) -> bool:
        """Check if user is in a global holdout group."""
        holdout = self._holdouts.get(holdout_name)
        if holdout is None:
            return False
        return _bucket(user_id, f"global_holdout:{holdout_name}") < holdout.holdout_pct

    def get_variant_config(
        self,
        user_id: str,
        experiment_name: str,
        key: str,
        default=None,
    ):
        """Get config value for user's assigned variant."""
        variant_name = self.assign(user_id, experiment_name)
        if variant_name is None:
            return default
        exp = self._experiments.get(experiment_name)
        if exp is None:
            return default
        for v in exp.variants:
            if v.name == variant_name:
                return v.config.get(key, default)
        return default

    def get_experiment(self, user_id: str, experiment_name: str) -> "DynamicConfig":
        """Statsig-compatible: returns DynamicConfig populated with the user's variant config.

        Usage mirrors Statsig: exp = mgr.get_experiment(uid, "btn_test"); exp.get_value("color", "blue")
        """
        from user_soul.dynamic_config import DynamicConfig
        variant_name = self.assign(user_id, experiment_name)
        exp = self._experiments.get(experiment_name)
        if exp is None or variant_name is None:
            return DynamicConfig()
        for v in exp.variants:
            if v.name == variant_name:
                dc = DynamicConfig(dict(v.config))
                dc._name = experiment_name
                dc._variant = variant_name
                return dc
        return DynamicConfig()

    def get_layer(self, user_id: str, layer_name: str) -> "DynamicConfig":
        """Statsig Layer: returns merged config from the experiment this user is in for the layer."""
        from user_soul.dynamic_config import DynamicConfig
        experiments_in_layer = sorted(self._layers.get(layer_name, []))
        if not experiments_in_layer:
            return DynamicConfig()
        b = _bucket(user_id, f"layer:{layer_name}")
        slot = int(b * len(experiments_in_layer))
        assigned_exp = experiments_in_layer[slot]
        return self.get_experiment(user_id, assigned_exp)

    def get_all_assignments(self, user_id: str) -> dict[str, str | None]:
        """Returns {experiment_name: variant_name} for all registered experiments."""
        return {name: self.assign(user_id, name) for name in self._experiments}

    def export_config(self) -> dict:
        """Export all configs as dict for serialization."""
        return {
            "experiments": [
                {
                    "name": exp.name,
                    "variants": [
                        {"name": v.name, "weight": v.weight, "config": v.config}
                        for v in exp.variants
                    ],
                    "holdout_pct": exp.holdout_pct,
                    "layer": exp.layer,
                    "description": exp.description,
                }
                for exp in self._experiments.values()
            ],
            "gates": [
                {"name": g.name, "rollout_pct": g.rollout_pct, "description": g.description}
                for g in self._gates.values()
            ],
            "holdouts": [
                {"name": h.name, "holdout_pct": h.holdout_pct, "description": h.description}
                for h in self._holdouts.values()
            ],
        }

    @classmethod
    def from_config(cls, config: dict) -> ExperimentManager:
        """Reconstruct from exported config."""
        mgr = cls()
        for exp_data in config.get("experiments", []):
            variants = [
                ExperimentVariant(v["name"], v["weight"], v.get("config", {}))
                for v in exp_data["variants"]
            ]
            mgr.add_experiment(
                ExperimentConfig(
                    name=exp_data["name"],
                    variants=variants,
                    holdout_pct=exp_data.get("holdout_pct", 0.0),
                    layer=exp_data.get("layer"),
                    description=exp_data.get("description", ""),
                )
            )
        for g in config.get("gates", []):
            mgr.add_gate(
                FeatureGateConfig(g["name"], g["rollout_pct"], g.get("description", ""))
            )
        for h in config.get("holdouts", []):
            mgr.add_holdout(
                HoldoutConfig(h["name"], h["holdout_pct"], h.get("description", ""))
            )
        return mgr
