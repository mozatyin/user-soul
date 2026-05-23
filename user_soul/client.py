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

    def check_gate(self, user: StatsigUser, gate_name: str) -> bool:
        """Evaluate a feature gate for a user.

        Statsig docs: https://docs.statsig.com/server/pythonSDK#checking-a-gate
        """
        return self._exp_manager.check_gate(user.user_id, gate_name)

    def get_experiment(self, user: StatsigUser, experiment_name: str) -> DynamicConfig:
        """Get experiment config for a user (returns user's variant config).

        Statsig docs: https://docs.statsig.com/server/pythonSDK#getting-an-experiment
        Usage: exp = client.get_experiment(user, "btn_color_test")
               color = exp.get_value("color", "blue")
        """
        return self._exp_manager.get_experiment(user.user_id, experiment_name)

    def get_layer(self, user: StatsigUser, layer_name: str) -> DynamicConfig:
        """Get layer config for a user (mutually exclusive experiments).

        Statsig docs: https://docs.statsig.com/server/pythonSDK#layers
        """
        return self._exp_manager.get_layer(user.user_id, layer_name)

    def get_dynamic_config(self, user: StatsigUser, config_name: str) -> DynamicConfig:
        """Get a dynamic config object for a user.

        Statsig docs: https://docs.statsig.com/server/pythonSDK#dynamic-config
        """
        dc = DynamicConfig()
        dc._name = config_name
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

    def get_config(self, user: StatsigUser, config_name: str) -> DynamicConfig:
        """Statsig primary name for get_dynamic_config.

        Statsig SDK: config = statsig.get_config(user, "prices")
        """
        override = self._overrides.get(f"config:{config_name}")
        if override is not None:
            dc = DynamicConfig(override)
            dc._name = config_name
            return dc
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
