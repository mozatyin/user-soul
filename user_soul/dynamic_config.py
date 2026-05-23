"""DynamicConfig — runtime key-value configuration with per-user and per-variant overrides.

Equivalent to Statsig DynamicConfig: returns different values to different users/variants.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ConfigRule:
    key: str
    value: Any
    user_id: str | None = None
    experiment: str | None = None
    variant: str | None = None


class DynamicConfig:
    """Statsig DynamicConfig-compatible: get_value() and value property match Statsig SDK exactly."""

    def __init__(self, defaults: dict | None = None):
        self._defaults: dict = defaults or {}
        self._rules: list[ConfigRule] = []
        self._name: str = ""
        self._variant: str = ""

    @property
    def value(self) -> dict:
        """Direct access to backing dict — matches Statsig's DynamicConfig.value."""
        return dict(self._defaults)

    def set(self, key: str, value: Any) -> None:
        self._defaults[key] = value

    def set_for_user(self, user_id: str, key: str, value: Any) -> None:
        self._rules.append(ConfigRule(key=key, value=value, user_id=user_id))

    def set_for_variant(self, experiment: str, variant: str, key: str, value: Any) -> None:
        self._rules.append(ConfigRule(key=key, value=value, experiment=experiment, variant=variant))

    def get(
        self,
        key: str,
        user_id: str | None = None,
        experiment: str | None = None,
        variant: str | None = None,
        default: Any = None,
    ) -> Any:
        # Priority: user_override > variant_override > global_default > default arg
        user_val = _MISSING = object()
        variant_val = _MISSING

        for rule in self._rules:
            if rule.key != key:
                continue
            if rule.user_id is not None and rule.user_id == user_id:
                user_val = rule.value
            elif (
                rule.user_id is None
                and rule.experiment is not None
                and rule.experiment == experiment
                and rule.variant is not None
                and rule.variant == variant
            ):
                variant_val = rule.value

        if user_val is not _MISSING:
            return user_val
        if variant_val is not _MISSING:
            return variant_val
        if key in self._defaults:
            return self._defaults[key]
        return default

    def get_value(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Statsig-identical alias for get(key, default).

        Statsig SDK: config.get_value("button_color", "blue")
        """
        return self.get(key, default=default)

    def get_typed(self, key: str, default: Any, expected_type: type) -> Any:
        """Type-safe getter — mirrors Statsig DynamicConfig.get_typed().

        Returns default if value is missing or wrong type.
        Statsig SDK: config.get_typed("price", 9.99, float)
        """
        val = self.get(key, default=default)
        if not isinstance(val, expected_type):
            return default
        return val

    def get_all(
        self,
        user_id: str | None = None,
        experiment: str | None = None,
        variant: str | None = None,
    ) -> dict:
        all_keys: set[str] = set(self._defaults.keys())
        for rule in self._rules:
            all_keys.add(rule.key)
        return {k: self.get(k, user_id=user_id, experiment=experiment, variant=variant) for k in all_keys}

    def as_json(self) -> str:
        payload = {
            "defaults": self._defaults,
            "rules": [
                {
                    "key": r.key,
                    "value": r.value,
                    "user_id": r.user_id,
                    "experiment": r.experiment,
                    "variant": r.variant,
                }
                for r in self._rules
            ],
        }
        return json.dumps(payload)

    @classmethod
    def from_json(cls, json_str: str) -> DynamicConfig:
        payload = json.loads(json_str)
        instance = cls(defaults=payload.get("defaults", {}))
        for r in payload.get("rules", []):
            instance._rules.append(ConfigRule(
                key=r["key"],
                value=r["value"],
                user_id=r.get("user_id"),
                experiment=r.get("experiment"),
                variant=r.get("variant"),
            ))
        return instance
