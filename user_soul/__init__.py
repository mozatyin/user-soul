"""User-Soul — user advocate across the entire product lifecycle."""
from user_soul.client import UserSoulClient
from user_soul.backend import LLMBackend
from user_soul.feature_filter import FeatureFilter, FeatureFilterReport, ScoredFeature
from user_soul.ab_validator import ABValidator, ABValidationReport, PDCAAction
from user_soul.eltm_adapter import extract_features, extract_benchmark_name, build_product_description
from user_soul.models import (
    AgentProfile, EvaluationMetric, PersonaStructure,
    SimulationReport, CompareReport, JourneyReport,
    ResearchReport, DesignReviewReport, ModuleUATReport, LaunchReport,
    PairwiseResult, ReviewResult, DecisionResult, FeatureAAR,
    PlaytestFeedback, PlaytestIssue,
    GradedPlaytestFeedback, ActionSpec,
)
from user_soul.playtest_bridge import extract_game_rules, run_graded_playtest
from user_soul.game_knowledge import (
    GameKnowledge, KnowledgeTier, brief_for_tier,
    DifferentialDiagnosis, TierResult, DiagnosisItem, DiagnosisCategory,
)
from user_soul.action_router import (
    route_diagnosis, route_flat_feedback,
    group_by_owner, format_action_summary,
)

# MCVClient — legacy voter interface
from user_soul.voter import MCVClient
from user_soul.core import Persona, PersonaDecider
from user_soul.personas import load_or_generate
from user_soul.simulator import PersonaSimulator, SimulationRun, FeatureSignal
from user_soul.cache import load_simulation_cache, save_simulation_cache
from user_soul.user_simulator import UserSimulator, SessionResult
from user_soul.domain_configs import DomainConfig, GameDomainConfig, AppDomainConfig, WebDomainConfig, build_domain_config
from user_soul.schema_extractor import extract_evaluation_schema
from user_soul.report import MetricResult, CompareReport as _CR, FeatureAAR as _FA, CoherenceReport
from user_soul.journey import simulate_journey
from user_soul.gate_ledger import GateLedger
from user_soul.population import (
    TraitDimension, Archetype, PersonaStructure as _PS2,
    PersonaPool, PopulationResearcher,
)
from user_soul.behavioral_framework import (
    SYCOPHANCY_DEFLATOR, BEHAVIORAL_FRAMEWORK_SECTION,
    ADVERSARIAL_PERSONA_SECTION, BEHAVIORAL_METRICS, COGNITIVE_BUDGETS,
)

__all__ = [
    # UserSoul
    "UserSoulClient", "LLMBackend",
    "AgentProfile", "EvaluationMetric", "PersonaStructure",
    "SimulationReport", "CompareReport", "JourneyReport",
    "ResearchReport", "DesignReviewReport", "ModuleUATReport", "LaunchReport",
    "PairwiseResult", "ReviewResult", "DecisionResult", "FeatureAAR",
    "PlaytestFeedback", "PlaytestIssue",
    "GradedPlaytestFeedback", "ActionSpec",
    "extract_game_rules", "run_graded_playtest",
    "GameKnowledge", "KnowledgeTier", "brief_for_tier",
    "DifferentialDiagnosis", "TierResult", "DiagnosisItem", "DiagnosisCategory",
    "route_diagnosis", "route_flat_feedback",
    "group_by_owner", "format_action_summary",
    # MCVClient / voter
    "MCVClient",
    "Persona", "PersonaDecider", "load_or_generate",
    "PersonaSimulator", "SimulationRun", "FeatureSignal",
    "load_simulation_cache", "save_simulation_cache",
    "UserSimulator", "SessionResult",
    "DomainConfig", "GameDomainConfig", "AppDomainConfig", "WebDomainConfig", "build_domain_config",
    "extract_evaluation_schema",
    "MetricResult", "CoherenceReport",
    "simulate_journey", "GateLedger",
    "TraitDimension", "Archetype",
    "PersonaPool", "PopulationResearcher",
    "SYCOPHANCY_DEFLATOR", "BEHAVIORAL_FRAMEWORK_SECTION",
    "ADVERSARIAL_PERSONA_SECTION", "BEHAVIORAL_METRICS", "COGNITIVE_BUDGETS",
    # Phase 0.7 — Feature Intelligence Filter
    "FeatureFilter", "FeatureFilterReport", "ScoredFeature",
    # Phase 8.8 — AB Validator vs Reference
    "ABValidator", "ABValidationReport", "PDCAAction",
    # ELTM Adapter — build_core() output → FeatureFilter input
    "extract_features", "extract_benchmark_name", "build_product_description",
]
