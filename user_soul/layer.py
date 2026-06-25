"""Layer — Statsig Layer object with PER-PARAMETER exposure logging.

The defining difference from a DynamicConfig: a Layer logs an exposure only for
the specific parameter the caller actually reads (and only once per parameter),
attributing it to the underlying allocated experiment. This prevents a layer that
hosts many experiments from over-counting exposures, and is exactly Statsig's
"layer parameter exposure" semantic.
"""
from __future__ import annotations

from typing import Any, Callable


class Layer:
    def __init__(
        self,
        name: str,
        values: dict | None = None,
        *,
        allocated_experiment: str | None = None,
        rule_id: str = "",
        group_name: str = "",
        reason: str = "Default",
        on_parameter_exposure: Callable[[str, str], None] | None = None,
    ):
        self._name = name
        self._values = values or {}
        self._allocated_experiment = allocated_experiment
        self._rule_id = rule_id
        self._group_name = group_name
        self._reason = reason
        self._on_exposure = on_parameter_exposure
        self._exposed: set[str] = set()

    def get(self, key: str, default: Any = None) -> Any:
        """Read a parameter — logs a one-time exposure for THIS key only."""
        present = key in self._values
        if present and self._on_exposure is not None and key not in self._exposed:
            self._exposed.add(key)
            self._on_exposure(key, self._allocated_experiment or "")
        return self._values.get(key, default)

    def get_value(self, key: str, default: Any = None) -> Any:
        """Statsig alias for get()."""
        return self.get(key, default)

    def get_typed(self, key: str, default: Any, expected_type: type) -> Any:
        val = self.get(key, default)
        return val if isinstance(val, expected_type) else default

    def get_name(self) -> str:
        return self._name

    @property
    def rule_id(self) -> str:
        return self._rule_id

    @property
    def group_name(self) -> str:
        return self._group_name

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def allocated_experiment_name(self) -> str | None:
        """Statsig: which experiment in the layer the user was allocated to."""
        return self._allocated_experiment
