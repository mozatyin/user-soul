"""SeriesDiagnostics — statistical anomaly detection over daily metric series.

A daily dashboard shows N metrics as N numbers (+ an AI prose summary). The
problem: with N≈180 metrics, several will look "red" on any given day by pure
chance, and a human/AI reader cannot tell a real move from noise, cannot rank by
severity, and cannot say "this dip is significant NOW". This turns the same
series into a statistically-grounded finding list:

  - per metric: z-score of the latest value vs its trailing baseline → p-value
  - across ALL metrics: Benjamini-Hochberg FDR so the multiple-testing false
    alarms a daily board inevitably throws are controlled
  - week-over-week change, trend slope, ratio/additive-aware, outlier-robust
  - light business-polarity heuristic so "significant + adverse" surfaces as a
    problem while "significant + favourable" surfaces as a positive mover

It reuses PulseComputer (benjamini_hochberg, _norm_cdf) so the stats are shared
with the Statsig/Pulse layer rather than reinvented.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from user_soul.pulse import PulseComputer, _norm_cdf


# Title/keyword hints that a metric's GOOD direction is DOWN (rising = problem).
_ADVERSE_UP_KEYWORDS = (
    "error", "错误", "fail", "失败", "crash", "崩溃", "latency", "延迟", "delay",
    "churn", "流失", "卸载", "uninstall", "退款", "refund", "cost", "成本",
    "complaint", "投诉", "drop", "掉线", "lag", "卡顿", "timeout", "超时",
    "bug", "异常", "exception", "block", "封禁", "bounce", "跳出",
)


@dataclass
class MetricFinding:
    metric_id: str
    title: str
    domain: str
    latest_date: str
    latest_value: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    p_value: float
    wow_change_pct: float          # week-over-week % change (vs 7 days ago)
    trend_slope: float             # per-day OLS slope over the window
    direction: str                 # "up" | "down" | "flat"
    metric_nature: str             # "additive" | "ratio" | ...
    significant_fdr: bool = False
    adverse: bool = False          # move is in the business-bad direction
    severity: str = "info"         # "P1" | "P2" | "watch" | "positive" | "info"

    def as_row(self) -> dict:
        return {
            "severity": self.severity, "metric": self.metric_id, "title": self.title,
            "domain": self.domain, "latest": self.latest_value,
            "z": round(self.z_score, 2), "p": round(self.p_value, 4),
            "wow%": round(self.wow_change_pct, 1), "dir": self.direction,
            "fdr_sig": self.significant_fdr, "adverse": self.adverse,
        }


def _polarity_adverse_up(title: str, domain: str) -> bool:
    blob = f"{title} {domain}".lower()
    return any(k in blob for k in _ADVERSE_UP_KEYWORDS)


class SeriesDiagnostics:
    def __init__(self, baseline_window: int = 14, alpha: float = 0.05,
                 min_baseline: int = 5, winsorize_pct: float = 0.0):
        self.baseline_window = baseline_window
        self.alpha = alpha
        self.min_baseline = min_baseline
        self.winsorize_pct = winsorize_pct
        self._pulse = PulseComputer(alpha=alpha)

    def _analyze_one(self, metric_id, title, domain, dates, values,
                     metric_nature="additive") -> MetricFinding | None:
        vals = [float(v) for v in values if v is not None]
        if len(vals) < self.min_baseline + 1:
            return None
        latest = vals[-1]
        baseline = vals[-(self.baseline_window + 1):-1]
        if len(baseline) < self.min_baseline:
            baseline = vals[:-1]
        if self.winsorize_pct > 0:
            baseline, _ = self._pulse.winsorize(baseline, self.winsorize_pct)

        mean = statistics.mean(baseline)
        std = statistics.pstdev(baseline) if len(baseline) > 1 else 0.0

        if std <= 1e-12:
            z, p = 0.0, 1.0
        else:
            z = (latest - mean) / std
            p = 2 * (1 - _norm_cdf(abs(z)))

        # week-over-week
        wow = 0.0
        if len(vals) >= 8 and abs(vals[-8]) > 1e-9:
            wow = (latest - vals[-8]) / abs(vals[-8]) * 100.0

        slope = self._ols_slope(vals[-(self.baseline_window + 1):])
        direction = "up" if z > 0.5 else "down" if z < -0.5 else "flat"

        return MetricFinding(
            metric_id=metric_id, title=title, domain=domain,
            latest_date=str(dates[-1]) if dates else "",
            latest_value=round(latest, 4),
            baseline_mean=round(mean, 4), baseline_std=round(std, 4),
            z_score=round(z, 4), p_value=round(p, 6),
            wow_change_pct=round(wow, 4), trend_slope=round(slope, 6),
            direction=direction, metric_nature=metric_nature,
            adverse=self._is_adverse(direction, title, domain),
        )

    @staticmethod
    def _is_adverse(direction: str, title: str, domain: str) -> bool:
        if direction == "flat":
            return False
        adverse_up = _polarity_adverse_up(title, domain)
        return (direction == "up") == adverse_up

    @staticmethod
    def _ols_slope(values: list[float]) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        xs = list(range(n))
        mx, my = (n - 1) / 2, statistics.mean(values)
        denom = sum((x - mx) ** 2 for x in xs)
        if denom <= 0:
            return 0.0
        return sum((xs[i] - mx) * (values[i] - my) for i in range(n)) / denom

    def diagnose(self, metrics: list[dict]) -> list[MetricFinding]:
        """metrics: list of {metric_id, title, domain, dates, values, metric_nature}.

        Returns findings ranked by severity then |z|, with FDR applied across the
        whole batch (so significance accounts for how many metrics were tested).
        """
        findings: list[MetricFinding] = []
        for m in metrics:
            f = self._analyze_one(
                m.get("metric_id", "?"), m.get("title", ""), m.get("domain", ""),
                m.get("dates", []), m.get("values", []),
                m.get("metric_nature", "additive"))
            if f is not None:
                findings.append(f)

        flags = self._pulse.benjamini_hochberg([f.p_value for f in findings], self.alpha)
        for f, ok in zip(findings, flags):
            f.significant_fdr = ok
            f.severity = self._severity(f)

        rank = {"P1": 0, "P2": 1, "positive": 2, "watch": 3, "info": 4}
        findings.sort(key=lambda f: (rank.get(f.severity, 9), -abs(f.z_score)))
        return findings

    def _severity(self, f: MetricFinding) -> str:
        if f.significant_fdr:
            if f.adverse:
                return "P1" if abs(f.z_score) >= 3 else "P2"
            return "positive"
        # not FDR-significant but a sizeable raw move → keep an eye on it
        if f.adverse and f.p_value < self.alpha and abs(f.z_score) >= 2:
            return "watch"
        return "info"

    @staticmethod
    def summary(findings: list[MetricFinding]) -> dict:
        from collections import Counter
        c = Counter(f.severity for f in findings)
        return {"total": len(findings), **dict(c)}
