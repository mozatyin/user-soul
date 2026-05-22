"""PopulationOS — research-driven heterogeneous user population generator.

Three spaces merged into PersonaStructure → PersonaPool → AgentProfile per session.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
import user_soul.core as _core


@dataclass
class TraitDimension:
    """One behavioral axis that distinguishes users of this product."""
    name: str           # snake_case, e.g. "ludo_familiarity"
    description: str    # human-readable, e.g. "Real-world Ludo experience"
    low_label: str      # behavior at value=0, e.g. "never played Ludo before"
    high_label: str     # behavior at value=10, e.g. "expert Ludo player who teaches others"
    distribution: str   # "normal" | "uniform" | "bimodal" | "right_skewed" | "left_skewed"
    mean: float         # 0-10 scale center
    std: float          # standard deviation on 0-10 scale
    source: str         # "space1" | "space2" | "space3"


@dataclass
class Archetype:
    """A named user cluster with frequency and trait constraints."""
    name: str
    frequency: float                                       # 0-1, must sum to 1.0 across all archetypes
    description: str
    trait_constraints: dict[str, tuple[float, float]]     # trait_name → (min_val, max_val)
    background_story: str = ""                            # human story injected into simulation prompt


@dataclass
class PersonaStructure:
    """Complete confirmed population model — output of research, input to PersonaPool."""
    population_label: str
    product_context: str
    trait_dimensions: list[TraitDimension]
    archetypes: list[Archetype]
    research_notes: str = ""


@dataclass
class AgentProfile:
    """Frozen per-session persona sampled from PersonaStructure."""
    agent_id: str
    archetype_name: str
    trait_vector: dict[str, float]    # {trait_name: value (0-10)}
    dims: list = field(default_factory=list, repr=False, compare=False)  # TraitDimension list from pool
    background_story: str = ""        # human story injected into simulation prompt

    def to_human_story(self) -> str:
        """Return the human background story for prompt injection.

        A real person description (age, city, job, apps they use, how they found
        this product) that the LLM can embody — not a ruleset to execute.

        The LLM infers behavioral tendencies from this context naturally,
        rather than being told what to do via trait labels or constraint rules.

        Falls back to archetype name when no story is available.
        """
        if self.background_story:
            return self.background_story
        return f"一个{self.archetype_name}类型的普通用户。"


_RESEARCH_PROMPT = """You are a behavioral researcher and product designer.

A product manager wants to simulate realistic users for their product:
"{product_description}"

Research and define the complete behavioral population for this product by combining:
- Space 1 (Product Intent): what the product does and who it targets
- Space 2 (Behavioral Research): what psychology and UX research says about this user type
- Space 3 (Market Context): cultural, competitive, and demographic context

Output ONLY valid JSON (no markdown):
{{
  "population_label": "short label for this user population",
  "product_context": "one sentence describing the product",
  "trait_dimensions": [
    {{
      "name": "snake_case_name",
      "description": "what this dimension measures",
      "low_label": "specific behavior when this trait is very low (0-3)",
      "high_label": "specific behavior when this trait is very high (7-10)",
      "distribution": "normal|uniform|bimodal|right_skewed|left_skewed",
      "mean": 5.0,
      "std": 2.0,
      "source": "space1|space2|space3"
    }}
  ],
  "archetypes": [
    {{
      "name": "Archetype Name",
      "frequency": 0.40,
      "description": "one-line summary of who they are",
      "background_story": "具体人物：年龄、城市/国家、职业、手机上已有哪些同类产品、每天什么时候用手机、怎么发现这个产品、一个自然的生活张力（想要什么 vs 有什么限制）。≤60字。只写这个人是谁，不写行为预测。",
      "trait_constraints": {{"trait_name": [min_val, max_val]}}
    }}
  ],
  "research_notes": "key insight about this user population"
}}

Rules:
- 4-8 trait_dimensions that actually predict different behaviors in this product
- 3-5 archetypes covering the full spectrum from most enthusiastic to least
- archetype frequencies must sum to 1.0
- trait_constraints only include dimensions that differentiate this archetype
- low_label and high_label must be specific behaviors, not just adjectives
- cover cultural/regional context in the dimensions if relevant
- background_story: write a REAL PERSON (age, location, job, existing apps, daily routine, discovery channel). Do NOT write behavioral predictions or rules. The story should make someone's natural patience/engagement level obvious from context, not stated explicitly."""


class PopulationResearcher:
    """Runs 3-space research to produce a PersonaStructure for human confirmation.

    One Sonnet call synthesizes product intent, behavioral research, and market context.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key

    def research(self, product_description: str) -> PersonaStructure:
        """Research user population for a product description.

        Returns a PersonaStructure ready for human confirmation.
        """
        prompt = _RESEARCH_PROMPT.format(product_description=product_description[:2000])
        raw, _ = _core._llm_call(prompt, self._api_key, max_tokens=2048)
        return self._parse(raw, product_description)

    def _parse(self, raw: str, product_description: str) -> PersonaStructure:
        """Parse LLM response into PersonaStructure. Returns minimal fallback on failure."""
        data = _core._safe_json(raw)
        if not data:
            return self._fallback(product_description)

        dims = []
        for d in data.get("trait_dimensions", []):
            if not isinstance(d, dict) or not d.get("name"):
                continue
            dims.append(TraitDimension(
                name=d["name"],
                description=d.get("description", ""),
                low_label=d.get("low_label", "low"),
                high_label=d.get("high_label", "high"),
                distribution=d.get("distribution", "normal"),
                mean=float(d.get("mean", 5.0)),
                std=float(d.get("std", 2.0)),
                source=d.get("source", "space2"),
            ))

        archs = []
        for a in data.get("archetypes", []):
            if not isinstance(a, dict) or not a.get("name"):
                continue
            raw_constraints = a.get("trait_constraints", {})
            constraints = {
                k: (float(v[0]), float(v[1]))
                for k, v in raw_constraints.items()
                if isinstance(v, (list, tuple)) and len(v) == 2
            }
            archs.append(Archetype(
                name=a["name"],
                frequency=float(a.get("frequency", 1.0 / max(len(data.get("archetypes", [1])), 1))),
                description=a.get("description", ""),
                trait_constraints=constraints,
                background_story=a.get("background_story", ""),
            ))

        # Normalise frequencies to sum to 1.0
        if archs:
            total = sum(a.frequency for a in archs)
            if total > 0:
                archs = [
                    Archetype(a.name, a.frequency / total, a.description, a.trait_constraints, a.background_story)
                    for a in archs
                ]

        if not dims or not archs:
            return self._fallback(product_description)

        return PersonaStructure(
            population_label=data.get("population_label", "App Users"),
            product_context=data.get("product_context", product_description[:100]),
            trait_dimensions=dims,
            archetypes=archs,
            research_notes=data.get("research_notes", ""),
        )

    def _fallback(self, product_description: str) -> PersonaStructure:
        """Minimal 2-archetype structure when LLM parsing fails."""
        dims = [
            TraitDimension(
                "engagement_drive", "Motivation to engage deeply",
                "passive, easily disengaged, skips most features",
                "highly motivated, explores all features eagerly",
                "normal", 5.0, 2.5, "space1",
            ),
        ]
        archs = [
            Archetype("Engaged User", 0.60, "actively uses the product", {"engagement_drive": (5.0, 10.0)}),
            Archetype("Passive User", 0.40, "minimal engagement",        {"engagement_drive": (0.0, 5.0)}),
        ]
        return PersonaStructure(
            population_label="App Users",
            product_context=product_description[:100],
            trait_dimensions=dims,
            archetypes=archs,
            research_notes="fallback — LLM parsing failed",
        )


class PersonaPool:
    """Samples N AgentProfiles from a confirmed PersonaStructure.

    Archetype assignment respects frequency weights.
    Trait values are sampled within archetype constraints using normal distribution,
    clipped to [0, 10] and to the archetype's [min, max] window.
    """

    def __init__(self, structure: PersonaStructure):
        self._structure = structure
        self._counter = 0

    def generate(self, n: int) -> list[AgentProfile]:
        """Generate N agents sampled from structure distributions."""
        agents = []
        archetypes = self._structure.archetypes
        dims = self._structure.trait_dimensions

        # Normalize frequencies in case they don't sum to exactly 1.0
        total = sum(a.frequency for a in archetypes)
        weights = [a.frequency / total for a in archetypes]

        for i in range(n):
            # Pick archetype by frequency weight
            arch = random.choices(archetypes, weights=weights, k=1)[0]

            # Sample trait vector
            trait_vector: dict[str, float] = {}
            for dim in dims:
                lo, hi = arch.trait_constraints.get(dim.name, (0.0, 10.0))
                # Center normal distribution within [lo, hi]
                center = (lo + hi) / 2
                span = (hi - lo) / 4  # ±2σ covers the range
                raw = random.gauss(center, max(span, 0.5))
                val = max(lo, min(hi, raw))    # clip to archetype window
                val = max(0.0, min(10.0, val)) # clip to absolute scale
                trait_vector[dim.name] = round(val, 2)

            self._counter += 1
            agent = AgentProfile(
                agent_id=f"agent_{self._counter:03d}",
                archetype_name=arch.name,
                trait_vector=trait_vector,
                dims=dims,
                background_story=arch.background_story,
            )
            agents.append(agent)

        return agents
