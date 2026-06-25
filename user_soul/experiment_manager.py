"""ExperimentManager — Feature Gates, A/B Experiments, Layers, and Holdouts.

Real experiment assignment with deterministic hash-based user bucketing.
Equivalent to Statsig's core experiment infrastructure but framework-agnostic.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from user_soul.targeting import (
    Condition, TargetingRule, evaluate_rules,
    REASON_DEFAULT, REASON_RULE, REASON_UNRECOGNIZED, REASON_HOLDOUT, REASON_DISABLED,
    REASON_STICKY,
)


class _UnitView:
    """Proxy that overrides user_id with the bucketing unit (for id_type !=
    userID) while delegating every other attribute to the real StatsigUser, so
    targeting conditions still read country/email/custom, etc."""
    __slots__ = ("_u", "user_id")

    def __init__(self, user, unit_id: str):
        self._u = user
        self.user_id = unit_id

    def __getattr__(self, name):
        return getattr(self._u, name)


def _rule_to_dict(r: TargetingRule) -> dict:
    return {
        "group_name": r.group_name, "pass_pct": r.pass_pct,
        "return_value": r.return_value, "id": r.id,
        "conditions": [{"operator": c.operator, "field": c.field, "target": c.target}
                       for c in r.conditions],
    }


def _rule_from_dict(d: dict) -> TargetingRule:
    return TargetingRule(
        group_name=d["group_name"],
        conditions=[Condition(c["operator"], c.get("field", ""), c.get("target"))
                    for c in d.get("conditions", [])],
        pass_pct=d.get("pass_pct", 1.0),
        return_value=d.get("return_value", True),
        id=d.get("id", ""),
    )

# 10,000 buckets — identical granularity to Statsig's evaluation engine.
_NUM_BUCKETS = 10_000


def _bucket_int(user_id: str, salt: str) -> int:
    """Deterministic 0..9999 bucket — matches Statsig exactly.

    Statsig hashes ``f"{salt}.{unit_id}"`` with SHA-256, takes the first 8 bytes
    as a big-endian uint64, then ``% 10000``.
    """
    digest = hashlib.sha256(f"{salt}.{user_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % _NUM_BUCKETS


def _bucket(user_id: str, salt: str) -> float:
    """Returns float in [0.0, 1.0) deterministically — Statsig SHA-256 bucketing.

    Thin wrapper over :func:`_bucket_int` so existing ``< pct`` (pct in 0..1)
    call-sites keep working while gaining wire-faithful hashing.
    """
    return _bucket_int(user_id, salt) / _NUM_BUCKETS


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
    # Targeting: rules a user must satisfy to be ELIGIBLE for the experiment.
    # Empty → everyone eligible (legacy behaviour). Mirrors Statsig targeting gate.
    targeting_rules: list[TargetingRule] = field(default_factory=list)
    salt: str = ""
    # Bucketing unit: "userID" (default) or a custom_ids key (e.g. "companyID").
    id_type: str = "userID"
    # Sticky bucketing: once assigned, keep the user's variant across weight edits.
    sticky: bool = False

    def __post_init__(self) -> None:
        if not self.salt:
            self.salt = self.name
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
    # Ordered targeting rules (Statsig-style). When present they are evaluated
    # first; rollout_pct is the implicit "everyone" default rule.
    rules: list[TargetingRule] = field(default_factory=list)
    salt: str = ""
    id_type: str = "userID"  # bucketing unit; custom_ids key for non-user units

    def __post_init__(self) -> None:
        if not self.salt:
            self.salt = self.name


@dataclass
class FeatureGate:
    """Result object for get_feature_gate() — mirrors Statsig's FeatureGate."""
    name: str
    value: bool
    rule_id: str
    group_name: str
    reason: str


@dataclass
class AssignmentResult:
    """Result object for assign_detailed() — variant + Statsig eval metadata."""
    experiment_name: str
    variant: str | None
    rule_id: str
    group_name: str
    reason: str


@dataclass
class HoldoutConfig:
    name: str
    holdout_pct: float
    description: str = ""


@dataclass
class LayerConfig:
    name: str
    description: str = ""


@dataclass
class DynamicConfigSpec:
    """A standalone Statsig DynamicConfig: a default value dict + targeting rules
    whose return_value is the dict served when the rule matches."""
    name: str
    defaults: dict = field(default_factory=dict)
    rules: list[TargetingRule] = field(default_factory=list)
    salt: str = ""
    id_type: str = "userID"

    def __post_init__(self) -> None:
        if not self.salt:
            self.salt = self.name


class ExperimentManager:
    def __init__(self) -> None:
        self._experiments: dict[str, ExperimentConfig] = {}
        self._gates: dict[str, FeatureGateConfig] = {}
        self._holdouts: dict[str, HoldoutConfig] = {}
        self._layers: dict[str, list[str]] = {}
        self._configs: dict[str, DynamicConfigSpec] = {}
        self._sticky: dict[tuple[str, str], str] = {}  # (unit_id, exp) → variant

    @staticmethod
    def _unit_id(user, id_type: str) -> str:
        """Resolve the bucketing unit. Statsig buckets on userID by default but can
        bucket on a custom unit (stableID, companyID, ...) via custom_ids."""
        if id_type and id_type != "userID":
            cid = (user.custom_ids or {}).get(id_type)
            if cid:
                return str(cid)
        return user.user_id

    def clear_sticky(self, user_or_id, experiment_name: str) -> None:
        """Forget a user's persisted sticky assignment for an experiment."""
        exp = self._experiments.get(experiment_name)
        id_type = exp.id_type if exp else "userID"
        unit = self._unit_id(self._as_user(user_or_id), id_type)
        self._sticky.pop((unit, experiment_name), None)

    def add_dynamic_config(self, spec: DynamicConfigSpec) -> None:
        self._configs[spec.name] = spec

    def evaluate_config(self, user_or_id, config_name: str) -> "DynamicConfig":
        """Statsig DynamicConfig evaluation: targeting rules → default dict."""
        from user_soul.dynamic_config import DynamicConfig
        spec = self._configs.get(config_name)
        if spec is None:
            return DynamicConfig(name=config_name, reason=REASON_UNRECOGNIZED)
        user = self._as_user(user_or_id)
        view = _UnitView(user, self._unit_id(user, spec.id_type))
        if spec.rules:
            ev = evaluate_rules(
                view, spec.rules, _bucket, spec.salt,
                default_value=None, pass_gate=self._pass_gate_cb(),
            )
            if ev.reason == REASON_RULE and isinstance(ev.value, dict):
                merged = dict(spec.defaults); merged.update(ev.value)
                return DynamicConfig(merged, name=config_name, rule_id=ev.rule_id,
                                     group_name=ev.group_name, reason=ev.reason)
        return DynamicConfig(dict(spec.defaults), name=config_name, reason=REASON_DEFAULT)

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

    def assign(self, user_or_id, experiment_name: str) -> str | None:
        """Assign user to a variant. Returns variant name, or None if excluded.

        Accepts a StatsigUser (targeting-aware) or a user_id string (legacy).
        """
        return self.assign_detailed(user_or_id, experiment_name).variant

    def assign_detailed(self, user_or_id, experiment_name: str) -> AssignmentResult:
        """Assign + Statsig eval metadata (rule_id / group_name / reason).

        Algorithm:
        0. Targeting — if the experiment has targeting_rules and the user matches
           none, they are NOT eligible (variant=None, reason Default).
        1. Per-experiment holdout (reason Holdout).
        2. Layer mutual exclusivity — if the user's layer slot maps to a different
           experiment, serve "control" (reason TargetingRule:layer).
        3. Variant assignment via cumulative weights (reason TargetingRule).
        """
        exp = self._experiments.get(experiment_name)
        if exp is None:
            return AssignmentResult(experiment_name, None, "", "", REASON_UNRECOGNIZED)
        user = self._as_user(user_or_id)
        uid = self._unit_id(user, exp.id_type)
        view = _UnitView(user, uid)

        # Sticky bucketing: a previously-persisted variant wins over re-evaluation.
        if exp.sticky and (uid, experiment_name) in self._sticky:
            v = self._sticky[(uid, experiment_name)]
            return AssignmentResult(experiment_name, v, f"variant:{v}", v, REASON_STICKY)

        if exp.targeting_rules:
            ev = evaluate_rules(
                view, exp.targeting_rules, _bucket, f"target:{exp.salt}",
                default_value=None, pass_gate=self._pass_gate_cb(),
            )
            if ev.reason != REASON_RULE:
                return AssignmentResult(experiment_name, None, "default", "Default", REASON_DEFAULT)

        if exp.holdout_pct > 0.0:
            if _bucket(uid, f"holdout:{exp.salt}") < exp.holdout_pct:
                return AssignmentResult(experiment_name, None, "holdout", "Holdout", REASON_HOLDOUT)

        if exp.layer is not None:
            layer_exps = sorted(self._layers.get(exp.layer, []))
            if len(layer_exps) > 1:
                b = _bucket(uid, f"layer:{exp.layer}")
                slot = int(b * len(layer_exps))
                assigned_exp = layer_exps[slot]
                if assigned_exp != experiment_name:
                    control = next(
                        (v.name for v in exp.variants if v.name == "control"),
                        exp.variants[0].name,
                    )
                    return AssignmentResult(
                        experiment_name, control, "layerExcluded", control, REASON_RULE)

        b = _bucket(uid, f"variant:{exp.salt}")
        cumulative = 0.0
        chosen = exp.variants[-1].name
        for v in exp.variants:
            cumulative += v.weight
            if b < cumulative:
                chosen = v.name
                break
        if exp.sticky:
            self._sticky[(uid, experiment_name)] = chosen
        return AssignmentResult(experiment_name, chosen, f"variant:{chosen}", chosen, REASON_RULE)

    def _as_user(self, user_or_id):
        """Accept a StatsigUser, a user-like proxy (_UnitView), or a bare
        user_id string (legacy). Only strings get wrapped."""
        if isinstance(user_or_id, str):
            from user_soul.statsig_user import StatsigUser
            return StatsigUser(user_id=user_or_id)
        return user_or_id

    def evaluate_gate(self, user_or_id, gate_name: str) -> FeatureGate:
        """Full Statsig-style gate evaluation: targeting rules → rollout default.

        Returns a FeatureGate carrying value + rule_id + group_name + reason.
        Unknown gate → value False, reason Unrecognized.
        """
        gate = self._gates.get(gate_name)
        if gate is None:
            return FeatureGate(gate_name, False, "", "", REASON_UNRECOGNIZED)
        user = self._as_user(user_or_id)

        unit = self._unit_id(user, gate.id_type)
        if gate.rules:
            ev = evaluate_rules(
                _UnitView(user, unit), gate.rules, _bucket, gate.salt,
                default_value=False, pass_gate=self._pass_gate_cb(),
            )
            # A rule that matched serves a (possibly False) value.
            if ev.reason in (REASON_RULE, REASON_DISABLED):
                return FeatureGate(gate_name, bool(ev.value), ev.rule_id, ev.group_name, ev.reason)
            # No rule matched → fall through to the implicit rollout default rule.

        passed = _bucket(unit, gate.salt) < gate.rollout_pct
        return FeatureGate(
            gate_name, passed,
            "default" if passed else "default",
            "Default", REASON_DEFAULT,
        )

    def check_gate(self, user_or_id, gate_name: str) -> bool:
        """Evaluate a feature gate. Returns True if user is in the rollout.

        Accepts a StatsigUser (targeting-aware) or a user_id string (legacy).
        """
        return self.evaluate_gate(user_or_id, gate_name).value

    def _pass_gate_cb(self):
        """Callback for pass_gate/fail_gate conditions referencing other gates."""
        return lambda user, gate_name: self.evaluate_gate(user, gate_name).value

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

    def get_experiment(self, user_or_id, experiment_name: str) -> "DynamicConfig":
        """Statsig-compatible: returns DynamicConfig populated with the user's variant config.

        Usage mirrors Statsig: exp = mgr.get_experiment(uid, "btn_test"); exp.get_value("color", "blue")
        """
        from user_soul.dynamic_config import DynamicConfig
        res = self.assign_detailed(user_or_id, experiment_name)
        exp = self._experiments.get(experiment_name)
        if exp is None or res.variant is None:
            return DynamicConfig(name=experiment_name, reason=res.reason)
        for v in exp.variants:
            if v.name == res.variant:
                return DynamicConfig(
                    dict(v.config), name=experiment_name, variant=res.variant,
                    rule_id=res.rule_id, group_name=res.group_name, reason=res.reason)
        return DynamicConfig(name=experiment_name, reason=res.reason)

    def get_layer_assignment(self, user_or_id, layer_name: str) -> str | None:
        """Which experiment in the layer is this user bucketed into (or None)."""
        experiments_in_layer = sorted(self._layers.get(layer_name, []))
        if not experiments_in_layer:
            return None
        uid = self._as_user(user_or_id).user_id
        b = _bucket(uid, f"layer:{layer_name}")
        slot = int(b * len(experiments_in_layer))
        return experiments_in_layer[slot]

    def get_layer(self, user_or_id, layer_name: str) -> "DynamicConfig":
        """Statsig Layer: returns merged config from the experiment this user is in for the layer."""
        from user_soul.dynamic_config import DynamicConfig
        assigned_exp = self.get_layer_assignment(user_or_id, layer_name)
        if assigned_exp is None:
            return DynamicConfig(name=layer_name)
        dc = self.get_experiment(user_or_id, assigned_exp)
        dc._name = layer_name
        dc._allocated_experiment = assigned_exp
        return dc

    def get_all_assignments(self, user_id: str) -> dict[str, str | None]:
        """Returns {experiment_name: variant_name} for all registered experiments."""
        return {name: self.assign(user_id, name) for name in self._experiments}

    def export_config(self) -> dict:
        """Export all configs as dict for serialization (targeting included)."""
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
                    "salt": exp.salt,
                    "targeting_rules": [_rule_to_dict(r) for r in exp.targeting_rules],
                }
                for exp in self._experiments.values()
            ],
            "gates": [
                {"name": g.name, "rollout_pct": g.rollout_pct, "description": g.description,
                 "salt": g.salt, "rules": [_rule_to_dict(r) for r in g.rules]}
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
                    salt=exp_data.get("salt", ""),
                    targeting_rules=[_rule_from_dict(r) for r in exp_data.get("targeting_rules", [])],
                )
            )
        for g in config.get("gates", []):
            mgr.add_gate(
                FeatureGateConfig(
                    g["name"], g["rollout_pct"], g.get("description", ""),
                    rules=[_rule_from_dict(r) for r in g.get("rules", [])],
                    salt=g.get("salt", ""),
                )
            )
        for h in config.get("holdouts", []):
            mgr.add_holdout(
                HoldoutConfig(h["name"], h["holdout_pct"], h.get("description", ""))
            )
        return mgr
