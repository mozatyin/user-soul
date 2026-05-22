"""Monte Carlo Voter — core types and PersonaDecider."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Persona:
    id: str
    name: str
    description: str
    cohort: str             # e.g. "College students 18-24, casual gamers"
    motivations: list[str] = field(default_factory=list)
    pain_points: list[str] = field(default_factory=list)


@dataclass
class DecisionResult:
    value: Any                          # str for classify, float for score, bool for validate
    confidence: float                   # 0.0–1.0, fraction of personas that agreed
    distribution: dict[str, float]      # option → fraction of votes
    mode: str                           # "fast" | "validated"
    tokens_used: int = 0
    raw_votes: list[Any] = field(default_factory=list)   # per-persona values


def _model_name(api_key: str) -> str:
    """Pick model. OpenRouter keys use openrouter base URL."""
    return "claude-sonnet-4-20250514"


def _haiku_model(api_key: str) -> str:
    """Haiku model ID — used for simulation (cheap, fast)."""
    return "claude-haiku-4-5-20251001"


# Module-level cache: probed once per process, reused across all _llm_call()s.
# None = not yet probed; "" = auto-bind works; "x.x.x.x" = use explicit address.
_LOCAL_ADDRESS_CACHE: str | None = None


def _resolve_local_address(target_host: str, port: int = 443) -> str | None:
    """Return the local IPv4 address that can reliably reach target_host, or None.

    Result is cached at module level — probed once per process so all subsequent
    LLM calls reuse the same interface (avoids intermittent utun VPN routing issues).
    """
    global _LOCAL_ADDRESS_CACHE
    if _LOCAL_ADDRESS_CACHE is not None:
        return _LOCAL_ADDRESS_CACHE or None  # "" → None (auto-bind OK)

    import socket as _socket

    # Enumerate non-loopback IPv4 addresses and find the first that can connect
    try:
        candidates = [
            info[4][0]
            for info in _socket.getaddrinfo(
                _socket.gethostname(), None, _socket.AF_INET, _socket.SOCK_STREAM
            )
            if not info[4][0].startswith("127.")
        ]
    except Exception:
        candidates = []

    for addr in candidates:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        s.settimeout(2)
        try:
            s.bind((addr, 0))
            s.connect((target_host, port))
            s.close()
            _LOCAL_ADDRESS_CACHE = addr
            return addr
        except Exception:
            try:
                s.close()
            except Exception:
                pass

    # Nothing found: let httpx handle it (will fail the same way, but cleanly)
    _LOCAL_ADDRESS_CACHE = ""
    return None


def _llm_call(
    prompt: str,
    api_key: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    model: str | None = None,
) -> tuple[str, int]:
    """Single LLM call → (response_text, tokens_used)."""
    import anthropic
    _model = model or _model_name(api_key)
    kwargs: dict = dict(
        model=_model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if temperature > 0.0:
        kwargs["temperature"] = temperature
    if api_key.startswith("sk-or-"):
        import httpx
        _local = _resolve_local_address("104.18.3.115")
        _transport = httpx.HTTPTransport(local_address=_local) if _local else None
        client = anthropic.Anthropic(
            api_key=api_key,
            base_url="https://openrouter.ai/api",
            http_client=httpx.Client(transport=_transport) if _transport else None,
        )
    else:
        client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(**kwargs)
    text = resp.content[0].text if resp.content else ""
    tokens = (resp.usage.input_tokens or 0) + (resp.usage.output_tokens or 0)
    return text, tokens


def _safe_json(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _to_float(val: Any, fallback: float) -> float:
    """Convert val to float, returning fallback on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return fallback


def _safe_json_arr(text: str) -> list:
    """Parse JSON array from LLM response."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
    return []


class PersonaDecider:
    def __init__(self, personas: list[Persona], api_key: str, mode: str | None = None):
        self.personas = personas
        self.api_key = api_key
        if mode is not None:
            self.mode = mode
        else:
            self.mode = os.environ.get("DECISION_MODE", "fast")

    def classify(
        self,
        question: str,
        options: list[str],
        context: str,
        batch: list[dict[str, Any]] | None = None,
    ) -> "DecisionResult | list[DecisionResult]":
        if self.mode == "validated":
            if batch:
                return self._validated_classify_batch(question, options, context, batch)
            return self._validated_classify_single(question, options, context)
        return self._fast_classify(question, options, context, batch)

    def _fast_classify(
        self,
        question: str,
        options: list[str],
        context: str,
        batch: list[dict[str, Any]] | None,
    ) -> "DecisionResult | list[DecisionResult]":
        persona = self.personas[0]
        options_str = " | ".join(options)

        if batch:
            items_json = json.dumps([
                {"id": item["id"], "name": item.get("name", item["id"])}
                for item in batch
            ])
            prompt = (
                f"You are: {persona.name} ({persona.cohort})\n"
                f"Motivations: {', '.join(persona.motivations)}\n"
                f"Pain points: {', '.join(persona.pain_points)}\n\n"
                f"Context: {context}\n"
                f"Question: {question}\n"
                f"Options: {options_str}\n\n"
                f"Items to classify:\n{items_json}\n\n"
                f'Reply with JSON array only: [{{"id": "...", "choice": "..."}}]'
            )
            raw, tokens = _llm_call(prompt, self.api_key, max_tokens=1024)
            arr = _safe_json_arr(raw)
            id_to_choice = {
                item.get("id", ""): item.get("choice", options[0])
                for item in arr
                if isinstance(item, dict)
            }
            per_item_tokens = tokens // max(len(batch), 1)
            results = []
            for item in batch:
                choice = id_to_choice.get(item["id"], options[0])
                results.append(DecisionResult(
                    value=choice,
                    confidence=1.0,
                    distribution={choice: 1.0},
                    mode=self.mode,
                    tokens_used=per_item_tokens,
                    raw_votes=[choice],
                ))
            return results

        # Single item
        prompt = (
            f"You are: {persona.name} ({persona.cohort})\n"
            f"Motivations: {', '.join(persona.motivations)}\n"
            f"Pain points: {', '.join(persona.pain_points)}\n\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Options: {options_str}\n\n"
            f'Reply with JSON only: {{"choice": "...", "reasoning": "one sentence"}}'
        )
        raw, tokens = _llm_call(prompt, self.api_key)
        data = _safe_json(raw)
        choice = data.get("choice", options[0])
        return DecisionResult(
            value=choice,
            confidence=1.0,
            distribution={choice: 1.0},
            mode=self.mode,
            tokens_used=tokens,
            raw_votes=[choice],
        )

    def score(
        self,
        question: str,
        lo: float,
        hi: float,
        context: str,
        batch: list[dict[str, Any]] | None = None,
    ) -> "DecisionResult | list[DecisionResult]":
        if self.mode == "validated":
            if batch:
                return self._validated_score_batch(question, lo, hi, context, batch)
            return self._validated_score_single(question, lo, hi, context)
        return self._fast_score(question, lo, hi, context, batch)

    def _fast_score(
        self,
        question: str,
        lo: float,
        hi: float,
        context: str,
        batch: list[dict[str, Any]] | None,
    ) -> "DecisionResult | list[DecisionResult]":
        persona = self.personas[0]

        def _clamp(v: float) -> float:
            return max(lo, min(hi, v))

        if batch:
            items_json = json.dumps([
                {"id": item["id"], "name": item.get("name", item["id"])}
                for item in batch
            ])
            prompt = (
                f"You are: {persona.name} ({persona.cohort})\n"
                f"Motivations: {', '.join(persona.motivations)}\n"
                f"Pain points: {', '.join(persona.pain_points)}\n\n"
                f"Context: {context}\n"
                f"Question: {question} (scale: {lo}–{hi})\n\n"
                f"Items to score:\n{items_json}\n\n"
                f'Reply with JSON array only: [{{"id": "...", "score": <number>}}]'
            )
            raw, tokens = _llm_call(prompt, self.api_key, max_tokens=1024)
            arr = _safe_json_arr(raw)
            id_to_score: dict[str, float] = {
                item.get("id", ""): _to_float(item.get("score"), (lo + hi) / 2)
                for item in arr
                if isinstance(item, dict)
            }
            per_item_tokens = tokens // max(len(batch), 1)
            results = []
            for item in batch:
                v = _clamp(id_to_score.get(item["id"], (lo + hi) / 2))
                results.append(DecisionResult(
                    value=v,
                    confidence=1.0,
                    distribution={"score": v},
                    mode=self.mode,
                    tokens_used=per_item_tokens,
                    raw_votes=[v],
                ))
            return results

        # Single item
        prompt = (
            f"You are: {persona.name} ({persona.cohort})\n"
            f"Motivations: {', '.join(persona.motivations)}\n"
            f"Pain points: {', '.join(persona.pain_points)}\n\n"
            f"Context: {context}\n"
            f"Question: {question} (scale: {lo}–{hi})\n\n"
            f'Reply with JSON only: {{"score": <number>, "reasoning": "one sentence"}}'
        )
        raw, tokens = _llm_call(prompt, self.api_key)
        data = _safe_json(raw)
        v = _clamp(_to_float(data.get("score"), (lo + hi) / 2))
        return DecisionResult(
            value=v,
            confidence=1.0,
            distribution={"score": v},
            mode=self.mode,
            tokens_used=tokens,
            raw_votes=[v],
        )

    def validate(self, assertion: str, context: str) -> DecisionResult:
        if self.mode == "validated":
            return self._validated_validate(assertion, context)
        return self._fast_validate(assertion, context)

    # ------------------------------------------------------------------ validated
    def _validated_validate(self, assertion: str, context: str) -> DecisionResult:
        votes: list[bool] = []
        total_tokens = 0
        for persona in self.personas:
            prompt = (
                f"You are: {persona.name} ({persona.cohort})\n"
                f"Motivations: {', '.join(persona.motivations)}\n"
                f"Pain points: {', '.join(persona.pain_points)}\n\n"
                f"Context: {context}\n"
                f"Assertion: {assertion}\n\n"
                f'Reply with JSON only: {{"result": true|false, "reasoning": "one sentence"}}'
            )
            raw, tokens = _llm_call(prompt, self.api_key)
            data = _safe_json(raw)
            votes.append(bool(data.get("result", False)))
            total_tokens += tokens
        true_frac = sum(votes) / len(votes)
        value = true_frac >= 0.5
        confidence = true_frac if value else (1 - true_frac)
        return DecisionResult(
            value=value,
            confidence=round(confidence, 4),
            distribution={"true": round(true_frac, 4), "false": round(1 - true_frac, 4)},
            mode=self.mode,
            tokens_used=total_tokens,
            raw_votes=votes,
        )

    def _validated_classify_single(
        self, question: str, options: list[str], context: str
    ) -> DecisionResult:
        from collections import Counter
        votes: list[str] = []
        total_tokens = 0
        options_str = " | ".join(options)
        for persona in self.personas:
            prompt = (
                f"You are: {persona.name} ({persona.cohort})\n"
                f"Motivations: {', '.join(persona.motivations)}\n"
                f"Pain points: {', '.join(persona.pain_points)}\n\n"
                f"Context: {context}\n"
                f"Question: {question}\n"
                f"Options: {options_str}\n\n"
                f'Reply with JSON only: {{"choice": "...", "reasoning": "one sentence"}}'
            )
            raw, tokens = _llm_call(prompt, self.api_key)
            data = _safe_json(raw)
            votes.append(data.get("choice", options[0]))
            total_tokens += tokens
        counts = Counter(votes)
        winner, winner_count = counts.most_common(1)[0]
        total = len(votes)
        dist = {opt: round(counts.get(opt, 0) / total, 4) for opt in options}
        return DecisionResult(
            value=winner,
            confidence=round(winner_count / total, 4),
            distribution=dist,
            mode=self.mode,
            tokens_used=total_tokens,
            raw_votes=votes,
        )

    def _validated_classify_batch(
        self, question: str, options: list[str], context: str,
        batch: list[dict[str, Any]]
    ) -> list[DecisionResult]:
        from collections import Counter
        id_votes: dict[str, list[str]] = {item["id"]: [] for item in batch}
        total_tokens = 0
        options_str = " | ".join(options)
        items_json = json.dumps([
            {"id": item["id"], "name": item.get("name", item["id"])} for item in batch
        ])
        for persona in self.personas:
            prompt = (
                f"You are: {persona.name} ({persona.cohort})\n"
                f"Motivations: {', '.join(persona.motivations)}\n"
                f"Pain points: {', '.join(persona.pain_points)}\n\n"
                f"Context: {context}\n"
                f"Question: {question}\n"
                f"Options: {options_str}\n\n"
                f"Items to classify:\n{items_json}\n\n"
                f'Reply with JSON array only: [{{"id": "...", "choice": "..."}}]'
            )
            raw, tokens = _llm_call(prompt, self.api_key, max_tokens=1024)
            arr = _safe_json_arr(raw)
            for item_vote in arr:
                if isinstance(item_vote, dict):
                    fid = item_vote.get("id", "")
                    if fid in id_votes:
                        id_votes[fid].append(item_vote.get("choice", options[0]))
            total_tokens += tokens
        per_item_tokens = total_tokens // max(len(batch), 1)
        results = []
        for item in batch:
            votes = id_votes[item["id"]] or [options[0]]
            counts = Counter(votes)
            winner, winner_count = counts.most_common(1)[0]
            total = len(votes)
            dist = {opt: round(counts.get(opt, 0) / total, 4) for opt in options}
            results.append(DecisionResult(
                value=winner,
                confidence=round(winner_count / total, 4),
                distribution=dist,
                mode=self.mode,
                tokens_used=per_item_tokens,
                raw_votes=votes,
            ))
        return results

    def _validated_score_single(
        self, question: str, lo: float, hi: float, context: str
    ) -> DecisionResult:
        import statistics
        votes: list[float] = []
        total_tokens = 0
        for persona in self.personas:
            prompt = (
                f"You are: {persona.name} ({persona.cohort})\n"
                f"Motivations: {', '.join(persona.motivations)}\n"
                f"Pain points: {', '.join(persona.pain_points)}\n\n"
                f"Context: {context}\n"
                f"Question: {question} (scale: {lo}–{hi})\n\n"
                f'Reply with JSON only: {{"score": <number>, "reasoning": "one sentence"}}'
            )
            raw, tokens = _llm_call(prompt, self.api_key)
            data = _safe_json(raw)
            v = max(lo, min(hi, _to_float(data.get("score"), (lo + hi) / 2)))
            votes.append(v)
            total_tokens += tokens
        avg = statistics.mean(votes)
        stdev = statistics.stdev(votes) if len(votes) > 1 else 0.0
        confidence = max(0.0, 1.0 - stdev / max(hi - lo, 1e-6))
        return DecisionResult(
            value=round(avg, 4),
            confidence=round(confidence, 4),
            distribution={"mean": round(avg, 4), "stdev": round(stdev, 4)},
            mode=self.mode,
            tokens_used=total_tokens,
            raw_votes=votes,
        )

    def _validated_score_batch(
        self, question: str, lo: float, hi: float, context: str,
        batch: list[dict[str, Any]]
    ) -> list[DecisionResult]:
        import statistics
        id_votes: dict[str, list[float]] = {item["id"]: [] for item in batch}
        total_tokens = 0
        items_json = json.dumps([
            {"id": item["id"], "name": item.get("name", item["id"])} for item in batch
        ])
        for persona in self.personas:
            prompt = (
                f"You are: {persona.name} ({persona.cohort})\n"
                f"Motivations: {', '.join(persona.motivations)}\n"
                f"Pain points: {', '.join(persona.pain_points)}\n\n"
                f"Context: {context}\n"
                f"Question: {question} (scale: {lo}–{hi})\n\n"
                f"Items to score:\n{items_json}\n\n"
                f'Reply with JSON array only: [{{"id": "...", "score": <number>}}]'
            )
            raw, tokens = _llm_call(prompt, self.api_key, max_tokens=1024)
            arr = _safe_json_arr(raw)
            for item_score in arr:
                if isinstance(item_score, dict):
                    fid = item_score.get("id", "")
                    if fid in id_votes:
                        v = max(lo, min(hi, _to_float(item_score.get("score"), (lo + hi) / 2)))
                        id_votes[fid].append(v)
            total_tokens += tokens
        per_item_tokens = total_tokens // max(len(batch), 1)
        results = []
        for item in batch:
            votes = id_votes[item["id"]] or [(lo + hi) / 2]
            avg = statistics.mean(votes)
            stdev = statistics.stdev(votes) if len(votes) > 1 else 0.0
            confidence = max(0.0, 1.0 - stdev / max(hi - lo, 1e-6))
            results.append(DecisionResult(
                value=round(avg, 4),
                confidence=round(confidence, 4),
                distribution={"mean": round(avg, 4), "stdev": round(stdev, 4)},
                mode=self.mode,
                tokens_used=per_item_tokens,
                raw_votes=votes,
            ))
        return results

    # ------------------------------------------------------------------ fast
    def _fast_validate(self, assertion: str, context: str) -> DecisionResult:
        persona = self.personas[0]
        prompt = (
            f"You are evaluating a product feature on behalf of this user persona:\n"
            f"Name: {persona.name}\n"
            f"Cohort: {persona.cohort}\n"
            f"Motivations: {', '.join(persona.motivations)}\n"
            f"Pain points: {', '.join(persona.pain_points)}\n\n"
            f"Context: {context}\n\n"
            f"Assertion: {assertion}\n\n"
            f'Reply with JSON only: {{"result": true|false, "reasoning": "one sentence"}}'
        )
        raw, tokens = _llm_call(prompt, self.api_key)
        data = _safe_json(raw)
        value = bool(data.get("result", False))
        return DecisionResult(
            value=value,
            confidence=1.0,
            distribution={"true": 1.0 if value else 0.0, "false": 0.0 if value else 1.0},
            mode=self.mode,
            tokens_used=tokens,
            raw_votes=[value],
        )
