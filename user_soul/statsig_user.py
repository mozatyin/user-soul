"""StatsigUser — user identity object, identical to Statsig Python SDK's StatsigUser.

Developers can follow Statsig's official documentation to understand this object:
https://docs.statsig.com/server/pythonSDK
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StatsigOptions:
    """Initialization options — mirrors Statsig's StatsigOptions."""
    tier: str = "production"          # "production" | "staging" | "development"
    local_mode: bool = False           # skip network calls, use local overrides only
    disable_diagnostics: bool = False
    custom_logger: object = None


@dataclass
class StatsigUser:
    """User context passed to all gate/experiment/config evaluations.

    Field names and semantics are identical to Statsig's Python SDK StatsigUser.
    """
    user_id: str
    email: str = ""
    ip: str = ""
    user_agent: str = ""
    country: str = ""
    locale: str = ""
    app_version: str = ""
    custom: dict = field(default_factory=dict)
    private_attributes: dict = field(default_factory=dict)
    custom_ids: dict = field(default_factory=dict)
    statsig_environment: dict = field(default_factory=dict)  # e.g. {"tier": "staging"}

    def to_dict(self) -> dict:
        return {
            "userID": self.user_id,
            "email": self.email,
            "ip": self.ip,
            "userAgent": self.user_agent,
            "country": self.country,
            "locale": self.locale,
            "appVersion": self.app_version,
            "custom": self.custom,
            "customIDs": self.custom_ids,
        }
