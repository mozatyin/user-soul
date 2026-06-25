"""Targeting rule evaluation — mirrors Statsig's rule/condition engine.

Real Statsig feature gates and experiments do NOT just roll out by percentage:
each has an ordered list of *rules*, and each rule is an AND of *conditions*
evaluated against the StatsigUser (country, email, app_version, custom fields,
"passes another gate", etc.). The first rule whose conditions all pass then runs
a pass-percentage bucketing check; if the user buckets in, the rule's value /
group_name / rule_id is returned. This module implements that engine.

Hashing matches Statsig exactly: sha256(f"{salt}.{unit_id}") → first 8 bytes as
a big-endian uint64 → % 10000 buckets (see experiment_manager._bucket_int).
"""
from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable


# ─── Evaluation reasons (mirror Statsig's EvaluationReason strings) ──────────
REASON_DEFAULT = "Default"            # no rule matched → fell through to default value
REASON_RULE = "TargetingRule"         # a rule matched
REASON_OVERRIDE = "LocalOverride"     # forced via override_gate / override_config
REASON_UNRECOGNIZED = "Unrecognized"  # gate/config/experiment name not registered
REASON_HOLDOUT = "Holdout"            # excluded by holdout
REASON_DISABLED = "Disabled"          # rule pass-% bucketed the user out
REASON_STICKY = "Sticky"              # served the user's persisted variant


@dataclass
class Condition:
    """A single targeting predicate evaluated against a StatsigUser.

    field:    a StatsigUser attribute name ("country", "email", "app_version",
              "user_id"/"userID", "locale", "ip", "user_agent", "tier") or a
              custom field as "custom.<key>". Ignored for public/pass_gate types.
    operator: one of
              public | eq | neq | in | not_in |
              gt | gte | lt | lte |
              version_gt | version_gte | version_lt | version_lte |
              contains | starts_with | ends_with | regex |
              ip_in_cidr |
              before | after |
              in_segment | not_in_segment |
              pass_gate | fail_gate
    target:   the comparison value (scalar, list, CIDR(s), ISO date, segment
              name, or gate-name depending on operator).
    """
    operator: str
    field: str = ""
    target: Any = None


@dataclass
class TargetingRule:
    """An ordered rule: all conditions must pass, then pass_pct bucketing decides.

    group_name / id surface as Statsig's `group_name` / `rule_id`.
    return_value is what the gate/config serves when this rule wins (True for a
    boolean gate; a dict for a dynamic config; a variant name for assignment).
    """
    group_name: str
    conditions: list[Condition] = field(default_factory=list)
    pass_pct: float = 1.0
    return_value: Any = True
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = self.group_name


@dataclass
class Evaluation:
    """Result of evaluating a rule list: the value plus Statsig-style metadata."""
    value: Any
    rule_id: str
    group_name: str
    reason: str


def _parse_version(v: Any) -> tuple[int, ...]:
    """Parse '1.2.3' → (1, 2, 3); tolerant of build suffixes and missing parts."""
    if v is None:
        return (0,)
    s = str(v).strip()
    # keep only the leading dotted-numeric portion (drop "-beta", "+build", etc.)
    head = ""
    for ch in s:
        if ch.isdigit() or ch == ".":
            head += ch
        else:
            break
    parts = [p for p in head.split(".") if p != ""]
    if not parts:
        return (0,)
    return tuple(int(p) for p in parts)


def _cmp_versions(a: Any, b: Any) -> int:
    """Return -1/0/1 comparing two versions a and b semantically."""
    va, vb = _parse_version(a), _parse_version(b)
    n = max(len(va), len(vb))
    va = va + (0,) * (n - len(va))
    vb = vb + (0,) * (n - len(vb))
    return (va > vb) - (va < vb)


def _to_timestamp(v: Any) -> float | None:
    """Parse an ISO date/datetime string or epoch number → epoch seconds."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    try:
        return float(s)  # numeric epoch as string
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _resolve_field(user, field_name: str) -> Any:
    """Resolve a StatsigUser attribute or custom field by Statsig-ish field name."""
    if not field_name:
        return None
    fn = field_name.strip()
    if fn.startswith("custom."):
        return (user.custom or {}).get(fn[len("custom."):])
    alias = {
        "user_id": "user_id", "userid": "user_id",
        "email": "email", "country": "country", "locale": "locale",
        "ip": "ip", "user_agent": "user_agent", "useragent": "user_agent",
        "app_version": "app_version", "appversion": "app_version",
    }
    key = alias.get(fn.lower())
    if key:
        return getattr(user, key, None)
    if fn.lower() == "tier":
        return (user.statsig_environment or {}).get("tier")
    # fall back to custom dict by raw key
    return (user.custom or {}).get(fn)


def _eval_condition(
    cond: Condition,
    user,
    pass_gate: Callable[[object, str], bool] | None = None,
    segments: Callable[[object, str], bool] | None = None,
) -> bool:
    op = cond.operator
    if op in ("public", "everyone", "any"):
        return True

    if op in ("pass_gate", "fail_gate"):
        if pass_gate is None:
            return False
        passed = pass_gate(user, str(cond.target))
        return passed if op == "pass_gate" else (not passed)

    if op in ("in_segment", "not_in_segment"):
        if segments is None:
            return False
        member = segments(user, str(cond.target))
        return member if op == "in_segment" else (not member)

    actual = _resolve_field(user, cond.field)
    target = cond.target

    if op == "regex":
        if actual is None:
            return False
        try:
            return bool(re.search(str(target), str(actual)))
        except re.error:
            return False

    if op == "ip_in_cidr":
        if actual is None:
            return False
        cidrs = target if isinstance(target, (list, tuple)) else [target]
        try:
            ip = ipaddress.ip_address(str(actual))
        except ValueError:
            return False
        for c in cidrs:
            try:
                if ip in ipaddress.ip_network(str(c), strict=False):
                    return True
            except ValueError:
                continue
        return False

    if op in ("before", "after"):
        a, t = _to_timestamp(actual), _to_timestamp(target)
        if a is None or t is None:
            return False
        return a < t if op == "before" else a > t

    if op == "eq":
        return actual == target
    if op == "neq":
        return actual != target
    if op == "in":
        return actual in (target or [])
    if op == "not_in":
        return actual not in (target or [])

    if op in ("gt", "gte", "lt", "lte"):
        try:
            a, t = float(actual), float(target)
        except (TypeError, ValueError):
            return False
        return {"gt": a > t, "gte": a >= t, "lt": a < t, "lte": a <= t}[op]

    if op in ("version_gt", "version_gte", "version_lt", "version_lte"):
        c = _cmp_versions(actual, target)
        return {
            "version_gt": c > 0, "version_gte": c >= 0,
            "version_lt": c < 0, "version_lte": c <= 0,
        }[op]

    if op in ("contains", "starts_with", "ends_with"):
        if actual is None:
            return False
        a, t = str(actual), str(target)
        return {
            "contains": t in a,
            "starts_with": a.startswith(t),
            "ends_with": a.endswith(t),
        }[op]

    return False


def evaluate_rules(
    user,
    rules: list[TargetingRule],
    bucket_fn: Callable[[str, str], float],
    salt: str,
    default_value: Any,
    pass_gate: Callable[[object, str], bool] | None = None,
    segments: Callable[[object, str], bool] | None = None,
) -> Evaluation:
    """Evaluate an ordered rule list, Statsig-style.

    For the first rule whose conditions ALL pass, run pass-percentage bucketing
    (bucket_fn(user_id, "<salt>:<rule.id>") < rule.pass_pct). If bucketed in,
    return its value/group/rule_id with reason TargetingRule. If conditions pass
    but the user buckets out, that rule is "Disabled" and we fall through to the
    default. If no rule matches, return default with reason Default.
    """
    for rule in rules:
        if all(_eval_condition(c, user, pass_gate, segments) for c in rule.conditions):
            if rule.pass_pct >= 1.0 or bucket_fn(user.user_id, f"{salt}:{rule.id}") < rule.pass_pct:
                return Evaluation(rule.return_value, rule.id, rule.group_name, REASON_RULE)
            # conditions matched but bucketed out → this rule does not serve.
            return Evaluation(default_value, "default", "Default", REASON_DISABLED)
    return Evaluation(default_value, "default", "Default", REASON_DEFAULT)
