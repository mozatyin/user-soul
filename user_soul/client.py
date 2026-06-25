"""UserSoulClient — unified entry point for all User-Soul capabilities.

Statsig-compatible interface: check_gate / get_experiment / get_layer /
get_dynamic_config / log_event follow Statsig Python SDK naming exactly.
Developers can use Statsig's official docs as reference for these methods.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from user_soul.backend import LLMBackend
from user_soul.engines.persona import PersonaEngine
from user_soul.engines.behavior import BehaviorEngine
from user_soul.engines.vision import VisionEngine
from user_soul.engines.vote import VoteEngine
from user_soul.stages.research import ResearchPanel
from user_soul.stages.design_review import DesignReview
from user_soul.stages.module_uat import ModuleUAT
from user_soul.stages.launch import LaunchGate
from user_soul.models import (
    AgentProfile, EvaluationMetric,
    ResearchReport, DesignReviewReport, ModuleUATReport, LaunchReport,
    PlaytestFeedback, GradedPlaytestFeedback,
)
from user_soul.statsig_user import StatsigUser, StatsigOptions
from user_soul.experiment_manager import ExperimentManager, ExperimentConfig, FeatureGateConfig, HoldoutConfig
from user_soul.dynamic_config import DynamicConfig
from user_soul.event_logger import EventLogger, AAARREventMap
from user_soul.feature_filter import FeatureFilter, FeatureFilterReport
from user_soul.data_source import SimulatedDataSource, RealDataSource, HybridDataSource, create_data_source

if TYPE_CHECKING:
    from user_soul.pulse import PulseReport


class UserSoulClient:

    def __init__(
        self,
        backend: LLMBackend,
        mode: str = "ai",          # "ai" | "real" | "hybrid"
        event_logger: EventLogger | None = None,
        experiment_manager: ExperimentManager | None = None,
    ):
        self._backend = backend
        self._mode = mode
        self._persona = PersonaEngine(backend)
        self._behavior = BehaviorEngine(backend)
        self._vision = VisionEngine(backend)
        self._vote = VoteEngine(backend)

        # Real-data components (Statsig-equivalent)
        self._event_logger = event_logger or EventLogger()
        self._exp_manager = experiment_manager or ExperimentManager()

        # Override registry for testing (override_gate / override_config / override_experiment)
        self._overrides: dict = {}
        self._local_mode: bool = False

        # DataSource wired by mode
        self._data_source = create_data_source(
            mode=mode,
            backend=backend,
            event_logger=self._event_logger,
        )

    # ─── Statsig-compatible interface ────────────────────────────────────────

    # ─── Override registry (for testing / local_mode) ────────────────────────
    # Mirrors Statsig's override_gate / override_config / override_experiment

    def initialize(self, options: StatsigOptions | dict | None = None) -> None:
        """No-op for API compatibility. Configuration is passed in __init__."""
        if isinstance(options, StatsigOptions) and options.local_mode:
            self._local_mode = True

    # ─── Exposure logging (the behaviour that feeds Pulse) ───────────────────
    def _log_exposure(self, user: StatsigUser, event_name: str, name: str,
                      rule_id: str, group_name: str, reason: str, **extra) -> None:
        """Record a Statsig-style exposure event into the EventLogger.

        Exposures are what Pulse reads to attribute metric movements to variants;
        without them the experiment infra is blind. Disabled in local_mode.
        """
        if self._local_mode:
            return
        self._event_logger.log_event(
            user_id=user.user_id,
            event_name=event_name,
            value=1.0,
            name=name, rule_id=rule_id, group_name=group_name, reason=reason,
            **extra,
        )

    def check_gate(self, user: StatsigUser, gate_name: str,
                   log_exposure: bool = True) -> bool:
        """Evaluate a feature gate for a user (logs an exposure by default).

        Statsig docs: https://docs.statsig.com/server/pythonSDK#checking-a-gate
        """
        if f"gate:{gate_name}" in self._overrides:
            val = bool(self._overrides[f"gate:{gate_name}"])
            if log_exposure:
                self._log_exposure(user, "statsig::gate_exposure", gate_name,
                                   "override", "", "LocalOverride")
            return val
        gate = self._exp_manager.evaluate_gate(user, gate_name)
        if log_exposure:
            self._log_exposure(user, "statsig::gate_exposure", gate_name,
                               gate.rule_id, gate.group_name, gate.reason)
        return gate.value

    def get_feature_gate(self, user: StatsigUser, gate_name: str,
                         log_exposure: bool = True):
        """Statsig get_feature_gate → FeatureGate object (.value/.rule_id/.group_name)."""
        gate = self._exp_manager.evaluate_gate(user, gate_name)
        if log_exposure:
            self._log_exposure(user, "statsig::gate_exposure", gate_name,
                               gate.rule_id, gate.group_name, gate.reason)
        return gate

    def get_experiment(self, user: StatsigUser, experiment_name: str,
                       log_exposure: bool = True) -> DynamicConfig:
        """Get experiment config for a user (returns user's variant config).

        Statsig docs: https://docs.statsig.com/server/pythonSDK#getting-an-experiment
        Usage: exp = client.get_experiment(user, "btn_color_test")
               color = exp.get_value("color", "blue")
        """
        dc = self._exp_manager.get_experiment(user, experiment_name)
        if log_exposure:
            self._log_exposure(user, "statsig::config_exposure", experiment_name,
                               dc.rule_id, dc.group_name, dc.reason,
                               variant=dc._variant)
        return dc

    def get_layer(self, user: StatsigUser, layer_name: str):
        """Get a Layer for a user (mutually exclusive experiments).

        The returned Layer logs an exposure ONLY for the parameter actually read
        (Statsig layer parameter exposure), attributed to the allocated experiment.

        Statsig docs: https://docs.statsig.com/server/pythonSDK#layers
        """
        from user_soul.layer import Layer
        dc = self._exp_manager.get_layer(user, layer_name)
        allocated = getattr(dc, "_allocated_experiment", None)

        def _on_param(param: str, exp_name: str) -> None:
            self._log_exposure(user, "statsig::layer_exposure", layer_name,
                               dc.rule_id, dc.group_name, dc.reason,
                               parameter=param, allocated_experiment=exp_name)

        return Layer(layer_name, dc.value, allocated_experiment=allocated,
                     rule_id=dc.rule_id, group_name=dc.group_name, reason=dc.reason,
                     on_parameter_exposure=_on_param)

    def get_dynamic_config(self, user: StatsigUser, config_name: str,
                           log_exposure: bool = True) -> DynamicConfig:
        """Get a dynamic config object for a user.

        Priority: local override → registered DynamicConfigSpec (targeting) → empty.
        Statsig docs: https://docs.statsig.com/server/pythonSDK#dynamic-config
        """
        override = self._overrides.get(f"config:{config_name}")
        if override is not None:
            dc = DynamicConfig(override, name=config_name, reason="LocalOverride")
        else:
            dc = self._exp_manager.evaluate_config(user, config_name)
        if log_exposure:
            self._log_exposure(user, "statsig::config_exposure", config_name,
                               dc.rule_id, dc.group_name, dc.reason)
        return dc

    def log_event(
        self,
        user: StatsigUser,
        event_name: str,
        value: float = 0.0,
        metadata: dict | None = None,
    ) -> None:
        """Log a user event.

        Statsig docs: https://docs.statsig.com/server/pythonSDK#logging-an-event
        """
        self._event_logger.log_event(
            user_id=user.user_id,
            event_name=event_name,
            value=value,
            **(metadata or {}),
        )

    def override_gate(self, gate_name: str, value: bool) -> None:
        """Force a feature gate to a fixed value for all users (testing only).

        Statsig SDK: statsig.override_gate("new_feature", True)
        """
        self._exp_manager.add_gate(
            FeatureGateConfig(gate_name, rollout_pct=1.0 if value else 0.0)
        )
        self._overrides[f"gate:{gate_name}"] = value

    def override_config(self, config_name: str, value: dict) -> None:
        """Force a dynamic config to fixed values (testing only).

        Statsig SDK: statsig.override_config("prices", {"premium": 9.99})
        """
        self._overrides[f"config:{config_name}"] = value

    def override_experiment(self, experiment_name: str, value: dict) -> None:
        """Force an experiment to fixed variant values (testing only).

        Statsig SDK: statsig.override_experiment("btn_test", {"color": "red"})
        """
        self._overrides[f"experiment:{experiment_name}"] = value

    def remove_override(self, name: str) -> None:
        """Remove a previously set override.

        Statsig SDK: statsig.remove_override("new_feature")
        """
        for prefix in ("gate:", "config:", "experiment:"):
            self._overrides.pop(f"{prefix}{name}", None)

    # ─── Registration helpers (define gates / experiments / configs) ─────────
    def register_gate(self, config) -> None:
        self._exp_manager.add_gate(config)

    def register_experiment(self, config) -> None:
        self._exp_manager.add_experiment(config)

    def register_dynamic_config(self, spec) -> None:
        self._exp_manager.add_dynamic_config(spec)

    # ─── Manual exposure logging (Statsig manuallyLog*Exposure) ──────────────
    def manually_log_gate_exposure(self, user: StatsigUser, gate_name: str) -> None:
        gate = self._exp_manager.evaluate_gate(user, gate_name)
        self._log_exposure(user, "statsig::gate_exposure", gate_name,
                           gate.rule_id, gate.group_name, gate.reason)

    def manually_log_config_exposure(self, user: StatsigUser, config_name: str) -> None:
        dc = self._exp_manager.evaluate_config(user, config_name)
        self._log_exposure(user, "statsig::config_exposure", config_name,
                           dc.rule_id, dc.group_name, dc.reason)

    def manually_log_experiment_exposure(self, user: StatsigUser, experiment_name: str) -> None:
        dc = self._exp_manager.get_experiment(user, experiment_name)
        self._log_exposure(user, "statsig::config_exposure", experiment_name,
                           dc.rule_id, dc.group_name, dc.reason, variant=dc._variant)

    def manually_log_layer_parameter_exposure(
        self, user: StatsigUser, layer_name: str, parameter: str) -> None:
        dc = self._exp_manager.get_layer(user, layer_name)
        self._log_exposure(user, "statsig::layer_exposure", layer_name,
                           dc.rule_id, dc.group_name, dc.reason,
                           parameter=parameter,
                           allocated_experiment=getattr(dc, "_allocated_experiment", None))

    def get_exposures(self, event_name: str | None = None) -> list:
        """Return logged exposure events (for Pulse feeding / assertions)."""
        names = ([event_name] if event_name else
                 ["statsig::gate_exposure", "statsig::config_exposure", "statsig::layer_exposure"])
        out = []
        for n in names:
            out.extend(self._event_logger._backend.query(event_name=n))
        return out

    def get_config(self, user: StatsigUser, config_name: str) -> DynamicConfig:
        """Statsig primary name for get_dynamic_config.

        Statsig SDK: config = statsig.get_config(user, "prices")
        """
        return self.get_dynamic_config(user, config_name)

    def shutdown(self) -> None:
        """Flush events and clean up. No-op for in-process backends."""

    # ─── User Soul AI simulation (FeatureFilter, ABValidator) ────────────────

    def filter_features(
        self,
        product_description: str,
        raw_features: list[dict],
        target_segment: str,
        archetypes=None,
        top_n: int = 25,
    ) -> FeatureFilterReport:
        """Phase 0.7: Score and classify features using AI personas or real user data."""
        ff = FeatureFilter(self._backend, data_source=self._data_source)
        return ff.filter(
            product_description=product_description,
            raw_features=raw_features,
            target_segment=target_segment,
            archetypes=archetypes,
            top_n=top_n,
        )

    # ─── Persona-cohort decisions (VoteEngine) ───────────────────────────────

    def decide(self, question: str, options: list[str], context: str, *,
               personas: "list[AgentProfile] | None" = None,
               n: int = 5) -> "DecisionResult":
        """Classify a question across a persona cohort (e.g. Kano category).

        Returns a DecisionResult with the winning option, confidence (vote share)
        and full distribution. Personas are auto-generated from `context` if not given.
        """
        pool = personas or self._persona.get_or_create(context, n)
        return self._vote.classify(question, options, context, pool)

    def score(self, question: str, lo: float, hi: float, context: str, *,
              personas: "list[AgentProfile] | None" = None,
              n: int = 5) -> "DecisionResult":
        """Score a question on a numeric scale across a persona cohort.

        Returns a DecisionResult whose value is the cohort mean and whose
        confidence reflects agreement (1 − normalised stdev).
        """
        pool = personas or self._persona.get_or_create(context, n)
        return self._vote.score(question, lo, hi, context, pool)

    # ─── ELTM research-bridge capabilities (legacy MCVClient, now surfaced) ───

    def _mcv(self):
        """Construct the legacy MCVClient for capabilities not yet in the engine
        stack (coherence validation, friction attribution). Requires an api-keyed
        backend (AnthropicBackend)."""
        api_key = getattr(self._backend, "api_key", None)
        if not api_key:
            raise ValueError(
                "validate_coherence/attribute_frictions need a backend exposing "
                "`api_key` (e.g. AnthropicBackend); the given backend has none."
            )
        from user_soul.voter import MCVClient
        return MCVClient(api_key=api_key)

    def validate_coherence(self, product_description: str,
                           selected_features: list[dict],
                           dropped_features: list[dict] | None = None, *,
                           deep: bool = False) -> "CoherenceReport":
        """Catch dropped feature dependencies before build (ELTM Phase 0.8).

        Rule-based pass is free; pass deep=True to also run an LLM gap analysis
        for non-social dependency types. Returns a CoherenceReport.
        """
        return self._mcv().validate_coherence(
            product_description, selected_features, dropped_features, deep=deep)

    def attribute_frictions(self, product: str, frictions: list[str],
                            features: list[dict], *,
                            game_name: str = "",
                            original_slug: str = "") -> dict:
        """Map simulation friction themes to a Code-Soul reforge() defect manifest
        (ELTM Phase 8.8 → PDCA). Empty frictions → empty manifest, no LLM call."""
        return self._mcv().attribute_frictions(
            product, frictions, features,
            game_name=game_name, original_slug=original_slug)

    def get_pulse(
        self,
        experiment_name: str,
        metrics: dict,
    ) -> "PulseReport":
        """Statsig Pulse: compute significance analysis on A/B experiment metrics.

        metrics format for proportions: {name: (control_n, control_x, treatment_n, treatment_x)}
        metrics format for values: {name: ([control_vals], [treatment_vals])}
        """
        from user_soul.pulse import PulseComputer
        pc = PulseComputer()
        if metrics:
            first_val = next(iter(metrics.values()))
            if isinstance(first_val[0], list):
                return pc.compute_from_values(experiment_name, metrics)
            return pc.compute_from_rates(experiment_name, metrics)
        from user_soul.pulse import PulseReport
        return PulseReport(experiment_name, [], "underpowered", 0.0, 0, 0, "No metrics provided.")

    def create_persona_pool(self, product_description: str,
                            n: int = 12) -> list[AgentProfile]:
        return self._persona.get_or_create(product_description, n)

    def research(self, product_description: str, features: list[dict], *,
                 competitor_screenshots: list[tuple[str, bytes]] | None = None,
                 our_screenshot: bytes | None = None) -> ResearchReport:
        panel = ResearchPanel(self._backend)
        return panel.run(product_description, features,
                         competitor_screenshots=competitor_screenshots,
                         our_screenshot=our_screenshot)

    def review(self, product_description: str, screens: list[dict],
               target_flow: list[str], *,
               personas: list[AgentProfile] | None = None,
               wireframe_screenshots: list[bytes] | None = None,
               competitor_screenshots: list[tuple[str, bytes]] | None = None) -> DesignReviewReport:
        stage = DesignReview(self._backend)
        return stage.run(product_description, screens, target_flow,
                         personas=personas,
                         wireframe_screenshots=wireframe_screenshots,
                         competitor_screenshots=competitor_screenshots)

    def verify(self, product_description: str, *,
               personas: list[AgentProfile] | None = None,
               metrics: list[EvaluationMetric] | None = None,
               html_screenshots: list[bytes] | None = None,
               goal: str | None = None) -> ModuleUATReport:
        stage = ModuleUAT(self._backend)
        return stage.run(product_description,
                         personas=personas, metrics=metrics,
                         html_screenshots=html_screenshots, goal=goal)

    def launch(self, product_description: str,
               product_screenshots: list[bytes],
               competitor_screenshots: list[tuple[str, bytes]], *,
               personas: list[AgentProfile] | None = None,
               metrics: list[EvaluationMetric] | None = None,
               goal: str | None = None) -> LaunchReport:
        stage = LaunchGate(self._backend)
        return stage.run(product_description, product_screenshots,
                         competitor_screenshots,
                         personas=personas, metrics=metrics, goal=goal)

    def phase9_gate(self, product_description: str, *,
                    product_screenshots: list[bytes] | None = None,
                    competitor_screenshots: list[tuple[str, bytes]] | None = None,
                    reference_product: str | None = None,
                    user_type: str = "general user",
                    goal: str | None = None,
                    personas: list[AgentProfile] | None = None,
                    metrics: list[EvaluationMetric] | None = None) -> dict:
        """Unified ELTM-Flow Phase 9 launch gate — ONE entry, two modes:

        - `product_screenshots` given → full visual taste gate (`launch()` → VLM
          pairwise taste + behavioral sim → SHIP/IMPROVE/ABANDON).
        - otherwise → headless text A/B vs `reference_product` (ABValidator) →
          verdict mapped to SHIP/IMPROVE.

        Either way the verdict is ADVISORY: a human makes the final call
        (Constitution §4 — AI is never the final judge). Returns a normalized dict:
        {mode, recommendation, launch_ready, human_confirms, detail}.
        """
        if product_screenshots:
            report = self.launch(
                product_description, product_screenshots,
                competitor_screenshots or [],
                personas=personas, metrics=metrics, goal=goal)
            return {
                "mode": "visual",
                "recommendation": report.recommendation,
                "launch_ready": report.recommendation == "SHIP",
                "human_confirms": True,
                "detail": report,
            }

        from user_soul.ab_validator import ABValidator
        validator = ABValidator(self._backend,
                                api_key=getattr(self._backend, "api_key", None))
        rep = validator.validate(
            our_product=product_description,
            reference_product=reference_product or "industry benchmark",
            user_type=user_type,
            goal=goal or f"use {product_description[:40]} and decide whether to return",
        )
        return {
            "mode": "text",
            "recommendation": "SHIP" if rep.launch_ready else "IMPROVE",
            "launch_ready": rep.launch_ready,
            "human_confirms": True,
            "detail": rep,
        }

    def playtest(self, html_path: str, product_description: str, *,
                 personas: list[AgentProfile] | None = None,
                 k_turns: int = 12,
                 on_progress=None,
                 game_rules: str = "") -> PlaytestFeedback:
        from user_soul.playtest_bridge import run_user_playtest
        pool = personas or self._persona.get_or_create(product_description, n=5)
        return run_user_playtest(
            html_path, pool, self._backend,
            k_turns=k_turns, on_progress=on_progress,
            game_rules=game_rules,
        )

    def graded_playtest(self, html_path: str, product_description: str,
                        gdd: dict, *,
                        personas: list[AgentProfile] | None = None,
                        k_turns: int = 12,
                        on_progress=None) -> "GradedPlaytestFeedback":
        from user_soul.playtest_bridge import run_graded_playtest
        pool = personas or self._persona.get_or_create(product_description, n=6)
        return run_graded_playtest(
            html_path, pool, self._backend, gdd,
            k_turns=k_turns, on_progress=on_progress,
        )
