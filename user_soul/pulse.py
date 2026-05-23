"""PulseComputer — statistical significance analysis for A/B experiments.

Equivalent to Statsig Pulse: computes metric lifts with p-values and power estimates.
Uses z-test for proportions and Welch's t-test for continuous metrics.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass
class PulseMetric:
    name: str
    control_value: float
    treatment_value: float
    lift: float
    p_value: float
    significant: bool
    control_n: int
    treatment_n: int
    ci_lower: float
    ci_upper: float
    metric_type: str  # "proportion" or "continuous"


@dataclass
class PulseReport:
    experiment_name: str
    metrics: list[PulseMetric]
    overall_verdict: str       # "winner" | "loser" | "neutral" | "underpowered"
    power_estimate: float
    significant_wins: int
    significant_losses: int
    summary: str


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# Lookup table for two-tailed t critical values: df → t_0.025
_T_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
    40: 2.021, 60: 2.000, 120: 1.980,
}


def _t_pvalue_approx(t_stat: float, df: int) -> float:
    """Approximate two-tailed p-value from t-distribution using lookup table."""
    t_abs = abs(t_stat)
    sorted_dfs = sorted(_T_TABLE.keys())
    clamped_df = max(1, min(df, 120))
    closest_df = min(sorted_dfs, key=lambda x: abs(x - clamped_df))
    t_crit = _T_TABLE[closest_df]
    if t_abs >= t_crit:
        # Significant at 0.05 — estimate p more finely using normal approximation for large df
        if df >= 30:
            return 2 * (1 - _norm_cdf(t_abs))
        return 0.04  # conservative: just below 0.05
    else:
        # Not significant: estimate how far we are
        if df >= 30:
            return 2 * (1 - _norm_cdf(t_abs))
        ratio = t_abs / t_crit
        return max(0.05, min(1.0, 2 * (1 - _norm_cdf(t_abs * 0.9 if df >= 10 else t_abs * 0.7))))


class PulseComputer:
    def __init__(self, alpha: float = 0.05):
        self._alpha = alpha

    def z_test_proportion(
        self, n1: int, x1: int, n2: int, x2: int
    ) -> tuple[float, float, float, float]:
        if n1 < 2 or n2 < 2:
            return (0.0, 1.0, -1.0, 1.0)

        p1 = x1 / n1
        p2 = x2 / n2
        p_pool = (x1 + x2) / (n1 + n2)
        se = math.sqrt(max(1e-12, p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)))
        z = (p2 - p1) / se
        p_value = 2 * (1 - _norm_cdf(abs(z)))
        lift = (p2 - p1) / max(0.001, p1)

        # 95% CI on the lift via delta method
        se_lift = se / max(0.001, p1)
        ci_lower = lift - 1.96 * se_lift
        ci_upper = lift + 1.96 * se_lift

        return (round(lift, 6), round(p_value, 6), round(ci_lower, 6), round(ci_upper, 6))

    def welch_t_test(
        self, values1: list[float], values2: list[float]
    ) -> tuple[float, float, float, float]:
        if len(values1) < 2 or len(values2) < 2:
            return (0.0, 1.0, -1.0, 1.0)

        try:
            from scipy import stats as sp_stats
            t_stat, p_value = sp_stats.ttest_ind(values2, values1, equal_var=False)
            mean1 = statistics.mean(values1)
            mean2 = statistics.mean(values2)
            lift = (mean2 - mean1) / max(0.001, abs(mean1))
            n1, n2 = len(values1), len(values2)
            se1 = statistics.stdev(values1) / math.sqrt(n1)
            se2 = statistics.stdev(values2) / math.sqrt(n2)
            se_diff = math.sqrt(se1 ** 2 + se2 ** 2)
            se_lift = se_diff / max(0.001, abs(mean1))
            ci_lower = lift - 1.96 * se_lift
            ci_upper = lift + 1.96 * se_lift
            return (round(lift, 6), round(float(p_value), 6), round(ci_lower, 6), round(ci_upper, 6))
        except ImportError:
            pass

        n1, n2 = len(values1), len(values2)
        mean1 = statistics.mean(values1)
        mean2 = statistics.mean(values2)
        var1 = statistics.variance(values1)
        var2 = statistics.variance(values2)
        se = math.sqrt(max(1e-12, var1 / n1 + var2 / n2))
        t_stat = (mean2 - mean1) / se
        df = max(1, min(n1 - 1, n2 - 1))
        p_value = _t_pvalue_approx(t_stat, df)
        lift = (mean2 - mean1) / max(0.001, abs(mean1))
        se_lift = se / max(0.001, abs(mean1))
        ci_lower = lift - 1.96 * se_lift
        ci_upper = lift + 1.96 * se_lift
        return (round(lift, 6), round(p_value, 6), round(ci_lower, 6), round(ci_upper, 6))

    def power_estimate(
        self, n1: int, n2: int, effect_size: float = 0.1, alpha: float = 0.05
    ) -> float:
        if n1 < 1 or n2 < 1:
            return 0.0
        # Using Cohen's h / standard power formula for proportions:
        # power ≈ Phi(effect_size * sqrt(n_harmonic/2) - z_alpha/2)
        # With n_harmonic = harmonic mean of n1, n2
        n_harmonic = 2 * n1 * n2 / (n1 + n2)
        z_alpha = 1.96 if alpha == 0.05 else -math.log(alpha / 2)
        z_power = abs(effect_size) * math.sqrt(n_harmonic / 2) - z_alpha
        power = _norm_cdf(z_power)
        return round(min(0.99, max(0.0, power)), 4)

    def compute_from_rates(
        self,
        experiment_name: str,
        metrics: dict[str, tuple[int, int, int, int]],
    ) -> PulseReport:
        pulse_metrics: list[PulseMetric] = []
        for name, (cn, cc, tn, tc) in metrics.items():
            lift, p_value, ci_lower, ci_upper = self.z_test_proportion(cn, cc, tn, tc)
            p1 = cc / max(1, cn)
            p2 = tc / max(1, tn)
            pulse_metrics.append(PulseMetric(
                name=name,
                control_value=round(p1, 6),
                treatment_value=round(p2, 6),
                lift=lift,
                p_value=p_value,
                significant=p_value < self._alpha,
                control_n=cn,
                treatment_n=tn,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                metric_type="proportion",
            ))

        wins = sum(1 for m in pulse_metrics if m.significant and m.lift > 0)
        losses = sum(1 for m in pulse_metrics if m.significant and m.lift < 0)
        total_n1 = sum(m.control_n for m in pulse_metrics) // max(1, len(pulse_metrics))
        total_n2 = sum(m.treatment_n for m in pulse_metrics) // max(1, len(pulse_metrics))
        power = self.power_estimate(total_n1, total_n2)
        verdict = self._determine_verdict(wins, losses, power)

        report = PulseReport(
            experiment_name=experiment_name,
            metrics=pulse_metrics,
            overall_verdict=verdict,
            power_estimate=power,
            significant_wins=wins,
            significant_losses=losses,
            summary="",
        )
        report.summary = self._build_summary(report)
        return report

    def compute_from_values(
        self,
        experiment_name: str,
        metrics: dict[str, tuple[list[float], list[float]]],
    ) -> PulseReport:
        pulse_metrics: list[PulseMetric] = []
        for name, (ctrl_vals, trt_vals) in metrics.items():
            lift, p_value, ci_lower, ci_upper = self.welch_t_test(ctrl_vals, trt_vals)
            mean1 = statistics.mean(ctrl_vals) if ctrl_vals else 0.0
            mean2 = statistics.mean(trt_vals) if trt_vals else 0.0
            pulse_metrics.append(PulseMetric(
                name=name,
                control_value=round(mean1, 6),
                treatment_value=round(mean2, 6),
                lift=lift,
                p_value=p_value,
                significant=p_value < self._alpha,
                control_n=len(ctrl_vals),
                treatment_n=len(trt_vals),
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                metric_type="continuous",
            ))

        wins = sum(1 for m in pulse_metrics if m.significant and m.lift > 0)
        losses = sum(1 for m in pulse_metrics if m.significant and m.lift < 0)
        n1 = len(next(iter(metrics.values()))[0]) if metrics else 0
        n2 = len(next(iter(metrics.values()))[1]) if metrics else 0
        power = self.power_estimate(n1, n2)
        verdict = self._determine_verdict(wins, losses, power)

        report = PulseReport(
            experiment_name=experiment_name,
            metrics=pulse_metrics,
            overall_verdict=verdict,
            power_estimate=power,
            significant_wins=wins,
            significant_losses=losses,
            summary="",
        )
        report.summary = self._build_summary(report)
        return report

    def _determine_verdict(self, wins: int, losses: int, power: float) -> str:
        if power < 0.8 and wins == 0 and losses == 0:
            return "underpowered"
        if wins > losses and wins > 0:
            return "winner"
        if losses > wins and losses > 0:
            return "loser"
        return "neutral"

    def _build_summary(self, report: PulseReport) -> str:
        return (
            f"{report.significant_wins} wins, {report.significant_losses} losses, "
            f"power={report.power_estimate:.2f}. Verdict: {report.overall_verdict}."
        )
