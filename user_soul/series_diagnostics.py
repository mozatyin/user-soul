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
    low_sample: bool = False       # count metric below min_volume → thin/unstable
    trend_p_value: float = 1.0     # significance of the multi-day OLS slope
    trend_significant_fdr: bool = False  # trend survives FDR across the batch
    consecutive_extreme_days: int = 0   # trailing days on the same side of normal
    sustained: bool = False        # multi-day confirmed (not a one-day blip)
    slow_bleed: bool = False       # adverse trend (FDR) significant w/o a latest spike
    severity: str = "info"         # "P1" | "P2" | "watch" | "trend" | "positive" | "info"

    @property
    def confirmation(self) -> str:
        if self.slow_bleed:
            return f"slow-bleed {self.consecutive_extreme_days}d trend"
        if self.sustained:
            return f"{self.consecutive_extreme_days}d sustained"
        return "1d spike"

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
                 min_baseline: int = 5, winsorize_pct: float = 0.0,
                 min_volume: float = 0.0, min_persist_days: int = 3,
                 min_trend_drift: float = 0.15):
        self.baseline_window = baseline_window
        self.alpha = alpha
        self.min_baseline = min_baseline
        self.winsorize_pct = winsorize_pct
        # A move sustained this many trailing days counts as confirmed (not a blip).
        self.min_persist_days = min_persist_days
        # A trend must imply at least this relative drift over the window to count
        # as a slow bleed — kills statistically-nonzero-but-immaterial slopes.
        self.min_trend_drift = min_trend_drift
        # Count (additive) metrics whose magnitude is below this are treated as
        # thin/unstable: a "-100% WoW" on 2 users is noise, not a P1. Ratios are
        # exempt (their magnitude is 0..1, not a volume). 0 = guard off.
        self.min_volume = min_volume
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

        window = vals[-(self.baseline_window + 1):]
        slope, trend_p = self._trend_stats(window)
        direction = "up" if z > 0.5 else "down" if z < -0.5 else "flat"

        # Persistence: how many trailing days has the series sat on the latest
        # side of normal (|value-mean| > 0.5σ, same sign as the latest deviation).
        consec = 0
        if std > 1e-12:
            side = 1 if latest >= mean else -1
            for v in reversed(vals):
                if (v - mean) * side > 0 and abs(v - mean) > 0.5 * std:
                    consec += 1
                else:
                    break

        low_sample = (
            self.min_volume > 0
            and metric_nature == "additive"
            and max(abs(latest), abs(mean)) < self.min_volume
        )
        # materiality: the modelled change across the window vs the level
        trend_drift = abs(slope) * (len(window) - 1) / max(abs(mean), 1e-9)
        # the latest point must still sit on the adverse side (not already recovered)
        latest_on_adverse_side = self._is_adverse(
            "up" if latest >= mean else "down", title, domain)
        adverse_trend = (
            trend_p < self.alpha and trend_drift >= self.min_trend_drift
            and latest_on_adverse_side
            and self._is_adverse("up" if slope > 0 else "down", title, domain)
        )
        # Provisional (raw) — finalised under trend-FDR in diagnose().
        sustained = consec >= self.min_persist_days or adverse_trend
        # slow bleed candidate: a material adverse trend with no latest-day spike.
        slow_bleed = adverse_trend and p >= self.alpha

        return MetricFinding(
            metric_id=metric_id, title=title, domain=domain,
            latest_date=str(dates[-1]) if dates else "",
            latest_value=round(latest, 4),
            baseline_mean=round(mean, 4), baseline_std=round(std, 4),
            z_score=round(z, 4), p_value=round(p, 6),
            wow_change_pct=round(wow, 4), trend_slope=round(slope, 6),
            direction=direction, metric_nature=metric_nature,
            adverse=self._is_adverse(direction, title, domain),
            low_sample=low_sample,
            trend_p_value=round(trend_p, 6),
            consecutive_extreme_days=consec,
            sustained=sustained, slow_bleed=slow_bleed,
        )

    @staticmethod
    def _is_adverse(direction: str, title: str, domain: str) -> bool:
        if direction == "flat":
            return False
        adverse_up = _polarity_adverse_up(title, domain)
        return (direction == "up") == adverse_up

    @staticmethod
    def _ols_slope(values: list[float]) -> float:
        return SeriesDiagnostics._trend_stats(values)[0]

    @staticmethod
    def _trend_stats(values: list[float]) -> tuple[float, float]:
        """OLS slope over time + a two-sided p-value for slope != 0 (t-test,
        normal-approx tail). Answers 'is the multi-day trend real?'."""
        n = len(values)
        if n < 3:
            return 0.0, 1.0
        xs = list(range(n))
        mx, my = (n - 1) / 2, statistics.mean(values)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx <= 0:
            return 0.0, 1.0
        slope = sum((xs[i] - mx) * (values[i] - my) for i in range(n)) / sxx
        intercept = my - slope * mx
        sse = sum((values[i] - (intercept + slope * xs[i])) ** 2 for i in range(n))
        if sse <= 1e-18:
            return slope, 0.0  # perfect fit → unambiguous trend
        se_slope = math.sqrt((sse / (n - 2)) / sxx)
        if se_slope <= 0:
            return slope, 0.0
        t = slope / se_slope
        return slope, round(2 * (1 - _norm_cdf(abs(t))), 6)

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

        # FDR across BOTH the latest-day tests and the trend tests — otherwise the
        # slow-bleed section is ~5%·N chance trends (multiple testing on slopes too).
        flags = self._pulse.benjamini_hochberg([f.p_value for f in findings], self.alpha)
        trend_flags = self._pulse.benjamini_hochberg(
            [f.trend_p_value for f in findings], self.alpha)
        for f, ok, tok in zip(findings, flags, trend_flags):
            f.significant_fdr = ok
            f.trend_significant_fdr = tok
            f.slow_bleed = f.slow_bleed and tok          # require FDR-confirmed trend
            f.sustained = (f.consecutive_extreme_days >= self.min_persist_days) or f.slow_bleed
            f.severity = self._severity(f)

        rank = {"P1": 0, "P2": 1, "trend": 2, "positive": 3, "watch": 4, "info": 5}
        findings.sort(key=lambda f: (rank.get(f.severity, 9), -abs(f.z_score)))
        return findings

    def _severity(self, f: MetricFinding) -> str:
        if f.significant_fdr:
            if f.adverse:
                if f.low_sample:
                    return "watch"   # thin count data: not a real alarm
                base = "P1" if abs(f.z_score) >= 3 else "P2"
                # a single-day adverse spike that isn't sustained is not yet a
                # confirmed problem — hold at watch until it persists.
                return base if f.sustained else "watch"
            return "info" if f.low_sample else "positive"
        # slow bleed: an adverse multi-day trend the latest-day snapshot missed.
        if f.slow_bleed and not f.low_sample:
            return "trend"
        # not FDR-significant but a sizeable raw move → keep an eye on it
        if f.adverse and f.p_value < self.alpha and abs(f.z_score) >= 2 and not f.low_sample:
            return "watch"
        return "info"

    @staticmethod
    def summary(findings: list[MetricFinding]) -> dict:
        from collections import Counter
        c = Counter(f.severity for f in findings)
        return {"total": len(findings), **dict(c)}
