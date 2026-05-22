import json
from unittest.mock import patch
from user_soul.population import TraitDimension, Archetype, PersonaStructure, AgentProfile, PersonaPool, PopulationResearcher

def test_trait_dimension_fields():
    td = TraitDimension(
        name="ludo_familiarity",
        description="Real-world Ludo experience",
        low_label="never played Ludo before",
        high_label="expert Ludo player who teaches others",
        distribution="bimodal",
        mean=5.0,
        std=2.5,
        source="space2",
    )
    assert td.name == "ludo_familiarity"
    assert td.distribution == "bimodal"
    assert td.source == "space2"

def test_archetype_fields():
    arch = Archetype(
        name="Family Socializer",
        frequency=0.45,
        description="Plays Ludo with family, social motivation dominates",
        trait_constraints={"social_motivation": (6.0, 10.0), "ludo_familiarity": (3.0, 9.0)},
    )
    assert arch.frequency == 0.45
    assert arch.trait_constraints["social_motivation"] == (6.0, 10.0)

def test_persona_structure_fields():
    dims = [
        TraitDimension("social_motivation", "Social drive", "solo player", "social-first", "normal", 6.0, 2.0, "space2"),
    ]
    archs = [
        Archetype("Family Socializer", 0.60, "social player", {"social_motivation": (6.0, 10.0)}),
        Archetype("Solo Grinder",       0.40, "solo player",  {"social_motivation": (0.0, 4.0)}),
    ]
    ps = PersonaStructure(
        population_label="Arabic Ludo Players",
        product_context="Ludo mobile app for Arabic families",
        trait_dimensions=dims,
        archetypes=archs,
    )
    assert ps.population_label == "Arabic Ludo Players"
    assert len(ps.archetypes) == 2
    assert ps.research_notes == ""

def test_agent_profile_fields():
    ap = AgentProfile(
        agent_id="agent_001",
        archetype_name="Family Socializer",
        trait_vector={"social_motivation": 8.3, "ludo_familiarity": 5.1, "patience": 6.2},
    )
    assert ap.agent_id == "agent_001"
    assert ap.trait_vector["social_motivation"] == 8.3


def _make_structure() -> PersonaStructure:
    dims = [
        TraitDimension("social_motivation", "Social drive", "solo", "social", "normal", 6.0, 2.0, "space2"),
        TraitDimension("patience",           "Patience",    "quits fast", "very patient", "normal", 5.0, 2.0, "space2"),
    ]
    archs = [
        Archetype("Socializer", 0.60, "social", {"social_motivation": (6.0, 10.0)}),
        Archetype("Grinder",    0.40, "solo",   {"social_motivation": (0.0, 5.0)}),
    ]
    return PersonaStructure("Test Pop", "test product", dims, archs)

def test_pool_generates_n_agents():
    pool = PersonaPool(_make_structure())
    agents = pool.generate(30)
    assert len(agents) == 30

def test_pool_agents_have_unique_ids():
    pool = PersonaPool(_make_structure())
    agents = pool.generate(10)
    ids = [a.agent_id for a in agents]
    assert len(set(ids)) == 10

def test_pool_archetype_frequency_respected():
    """60% Socializer, 40% Grinder → at N=100, within ±15pp."""
    pool = PersonaPool(_make_structure())
    agents = pool.generate(100)
    socializers = sum(1 for a in agents if a.archetype_name == "Socializer")
    assert 45 <= socializers <= 75  # 60% ± 15pp

def test_pool_trait_constraints_respected():
    """Socializer agents must have social_motivation in [6.0, 10.0]."""
    pool = PersonaPool(_make_structure())
    agents = pool.generate(60)
    for a in agents:
        if a.archetype_name == "Socializer":
            assert 6.0 <= a.trait_vector["social_motivation"] <= 10.0
        elif a.archetype_name == "Grinder":
            assert 0.0 <= a.trait_vector["social_motivation"] <= 5.0

def test_pool_all_traits_in_0_10():
    pool = PersonaPool(_make_structure())
    agents = pool.generate(30)
    for a in agents:
        for val in a.trait_vector.values():
            assert 0.0 <= val <= 10.0

def test_pool_ids_unique_across_multiple_generate_calls():
    pool = PersonaPool(_make_structure())
    agents1 = pool.generate(5)
    agents2 = pool.generate(5)
    all_ids = [a.agent_id for a in agents1 + agents2]
    assert len(set(all_ids)) == 10

def test_human_story_returns_background_story():
    """to_human_story() returns the background_story when set."""
    story = "28岁，印度孟买，IT职员。通勤地铁时玩手机游戏打发时间。"
    agent = AgentProfile("a1", "Curious", {"patience": 6.0}, background_story=story)
    assert agent.to_human_story() == story

def test_human_story_fallback_when_no_story():
    """to_human_story() falls back to archetype name when no story is set."""
    agent = AgentProfile("a1", "Family Socializer", {"patience": 6.0})
    text = agent.to_human_story()
    assert "Family Socializer" in text

def test_archetype_background_story_field():
    """Archetype accepts background_story and defaults to empty string."""
    arch = Archetype("Casual Dabbler", 0.30, "low commitment user",
                     {"patience": (0.0, 5.0)},
                     background_story="35岁，泰国曼谷，家庭主妇。随手下载，不花钱在游戏上。")
    assert "泰国" in arch.background_story
    # Default is empty string
    arch2 = Archetype("Test", 0.70, "desc", {})
    assert arch2.background_story == ""

def test_persona_pool_propagates_background_story():
    """Agents from PersonaPool carry their archetype's background_story."""
    dims = [TraitDimension("patience", "Patience", "quits fast", "very patient", "normal", 5.0, 2.0, "space2")]
    story = "22岁，菲律宾马尼拉，快餐店员。下班后玩手机竞技游戏。"
    archs = [
        Archetype("Competitor", 1.0, "competitive user", {"patience": (4.0, 9.0)},
                  background_story=story),
    ]
    structure = PersonaStructure("Test", "test app", dims, archs)
    pool = PersonaPool(structure)
    agents = pool.generate(5)
    for agent in agents:
        assert agent.background_story == story
        assert agent.to_human_story() == story


def _mock_researcher_response() -> str:
    return json.dumps({
        "population_label": "Arabic Ludo App Users",
        "product_context": "Ludo mobile app for Arabic families",
        "trait_dimensions": [
            {
                "name": "ludo_familiarity",
                "description": "Prior Ludo experience",
                "low_label": "never played Ludo, confused by rules",
                "high_label": "expert player who knows all strategies",
                "distribution": "bimodal",
                "mean": 5.0,
                "std": 2.5,
                "source": "space2",
            },
            {
                "name": "social_motivation",
                "description": "Motivation to play with others vs solo",
                "low_label": "prefers solo play, ignores social features",
                "high_label": "only plays with friends and family",
                "distribution": "right_skewed",
                "mean": 7.0,
                "std": 2.0,
                "source": "space3",
            },
        ],
        "archetypes": [
            {
                "name": "Family Socializer",
                "frequency": 0.50,
                "description": "Plays with family, social-first motivation",
                "background_story": "32岁，利雅得，政府职员，三个孩子。和兄弟们周五晚上打纸牌是家庭传统。想在手机上继续这个传统。",
                "trait_constraints": {"social_motivation": [6.0, 10.0], "ludo_familiarity": [2.0, 9.0]},
            },
            {
                "name": "Competitive Grinder",
                "frequency": 0.30,
                "description": "Plays to win, high familiarity",
                "background_story": "22岁，开罗，大学生。手机上有FIFA Mobile和Clash Royale，习惯和人比赛，赢了会截图发群。",
                "trait_constraints": {"social_motivation": [2.0, 6.0], "ludo_familiarity": [6.0, 10.0]},
            },
            {
                "name": "Casual Dabbler",
                "frequency": 0.20,
                "description": "First-time user, low commitment",
                "background_story": "40岁，迪拜，家庭主妇。Facebook广告看到随手下载，主要玩消消乐，从不花钱在游戏上。",
                "trait_constraints": {"social_motivation": [3.0, 7.0], "ludo_familiarity": [0.0, 4.0]},
            },
        ],
        "research_notes": "Ludo is dominant in Arabic family gaming.",
    })

def test_researcher_returns_persona_structure():
    with patch("user_soul.core._llm_call", return_value=(_mock_researcher_response(), {})):
        researcher = PopulationResearcher(api_key="test")
        result = researcher.research("A Ludo mobile app for Arabic families")
    assert isinstance(result, PersonaStructure)
    assert result.population_label == "Arabic Ludo App Users"
    assert len(result.trait_dimensions) == 2
    assert len(result.archetypes) == 3

def test_researcher_trait_dimensions_parsed():
    with patch("user_soul.core._llm_call", return_value=(_mock_researcher_response(), {})):
        researcher = PopulationResearcher(api_key="test")
        result = researcher.research("A Ludo app")
    dim = result.trait_dimensions[0]
    assert dim.name == "ludo_familiarity"
    assert dim.distribution == "bimodal"
    assert dim.source == "space2"

def test_researcher_archetypes_parsed():
    with patch("user_soul.core._llm_call", return_value=(_mock_researcher_response(), {})):
        researcher = PopulationResearcher(api_key="test")
        result = researcher.research("A Ludo app")
    arch = result.archetypes[0]
    assert arch.name == "Family Socializer"
    assert arch.frequency == 0.50
    assert arch.trait_constraints["social_motivation"] == (6.0, 10.0)
    assert "利雅得" in arch.background_story  # human story preserved

def test_researcher_frequencies_sum_to_1():
    with patch("user_soul.core._llm_call", return_value=(_mock_researcher_response(), {})):
        researcher = PopulationResearcher(api_key="test")
        result = researcher.research("A Ludo app")
    total = sum(a.frequency for a in result.archetypes)
    assert abs(total - 1.0) < 0.01

def test_researcher_fallback_on_invalid_json():
    with patch("user_soul.core._llm_call", return_value=("not valid json at all", {})):
        researcher = PopulationResearcher(api_key="test")
        result = researcher.research("some product")
    assert isinstance(result, PersonaStructure)
    assert result.research_notes == "fallback — LLM parsing failed"
    assert len(result.archetypes) == 2


def test_prepare_with_pool_sets_pool():
    from user_soul.user_simulator import UserSimulator
    from user_soul.domain_configs import AppDomainConfig
    from user_soul.schema_extractor import EvaluationMetric
    sim = UserSimulator("test user", AppDomainConfig, api_key="test")
    agents = PersonaPool(_make_structure()).generate(6)
    metrics = [EvaluationMetric("day1_return", "bool", "会回来吗？")]
    sim.prepare_with_pool(product="test product", pool=agents, locked_metrics=metrics)
    assert sim._agent_pool == agents
    assert sim._metrics == metrics

def test_simulate_with_pool_uses_human_story():
    """Each session prompt should contain the agent's background story."""
    from user_soul.user_simulator import UserSimulator
    from user_soul.domain_configs import AppDomainConfig
    from user_soul.schema_extractor import EvaluationMetric
    from unittest.mock import patch, MagicMock

    # Build structure with real background stories
    dims = [TraitDimension("patience", "Patience", "quits fast", "very patient", "normal", 5.0, 2.0, "space2")]
    archs = [
        Archetype("Competitor", 1.0, "competitive", {"patience": (5.0, 10.0)},
                  background_story="22岁，马尼拉，快餐店员，赢了会截图发群。"),
    ]
    structure = PersonaStructure("Test", "test app", dims, archs)
    agents = PersonaPool(structure).generate(3)

    sim = UserSimulator("test user", AppDomainConfig, api_key="test", use_behavioral_framework=False)
    metrics = [EvaluationMetric("day1_return", "bool", "会回来吗？")]
    sim.prepare_with_pool(product="test product", pool=agents, locked_metrics=metrics)

    captured_prompts = []
    def fake_llm(prompt, api_key, **kwargs):
        captured_prompts.append(prompt)
        return "day1_return: yes", {}

    with patch("user_soul.core._llm_call", side_effect=fake_llm):
        sim.simulate(n_runs=3)

    assert len(captured_prompts) == 3
    # Every prompt must contain the human story, not trait labels
    for p in captured_prompts:
        assert "马尼拉" in p  # human story injected
        assert "social_motivation" not in p  # no raw trait names

def test_simulate_with_pool_cycles_when_n_exceeds_pool():
    """If n_runs > len(pool), cycles through pool."""
    from user_soul.user_simulator import UserSimulator
    from user_soul.domain_configs import AppDomainConfig
    from user_soul.schema_extractor import EvaluationMetric
    from unittest.mock import patch
    sim = UserSimulator("test user", AppDomainConfig, api_key="test")
    agents = PersonaPool(_make_structure()).generate(6)
    metrics = [EvaluationMetric("day1_return", "bool", "会回来吗？")]
    sim.prepare_with_pool(product="test product", pool=agents, locked_metrics=metrics)
    with patch("user_soul.core._llm_call", return_value=("day1_return: yes", {})):
        sim.simulate(n_runs=10)
    assert len(sim._session_results) == 10


def test_population_exports_available_at_mcv_root():
    import user_soul as mcv
    assert hasattr(mcv, "TraitDimension")
    assert hasattr(mcv, "Archetype")
    assert hasattr(mcv, "PersonaStructure")
    assert hasattr(mcv, "AgentProfile")
    assert hasattr(mcv, "PersonaPool")
    assert hasattr(mcv, "PopulationResearcher")
