from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


def _clip01(value: float) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _safe_div(numerator: float, denominator: float) -> float:
    denominator = float(denominator or 0.0)
    if denominator <= 0:
        return 0.0
    return float(numerator) / denominator


def _rate(value: Any) -> float:
    """Normalize a rate stored either as 0-1 ratio or 0-100 percentage."""
    try:
        out = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if out > 1.5:
        out = out / 100.0
    return _clip01(out)


def _first_existing(row: Mapping[str, Any], names: Iterable[str], default: float = 0.0) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            try:
                return float(row[name])
            except (TypeError, ValueError):
                return default
    return default


@dataclass(frozen=True)
class SCRIWeights:
    """Weights for leakage channels in the State Capacity Retention Index.

    The three leakage weights are normalized internally when they are used.
    They do not need to sum to one in configuration.
    """

    rent_capture: float = 0.50
    short_term_arbitrage: float = 0.25
    fiscal_misallocation: float = 0.25
    service_repair: float = 0.20
    information_repair: float = 0.20

    def normalized_leakage_weights(self) -> dict[str, float]:
        total = max(
            1e-12,
            float(self.rent_capture)
            + float(self.short_term_arbitrage)
            + float(self.fiscal_misallocation),
        )
        return {
            "rent_capture": float(self.rent_capture) / total,
            "short_term_arbitrage": float(self.short_term_arbitrage) / total,
            "fiscal_misallocation": float(self.fiscal_misallocation) / total,
        }


@dataclass(frozen=True)
class SCRIInput:
    """Single observation needed to compute SCRI."""

    fiscal_spending: float
    verified_food_security_outcome: float
    counterfactual_food_security_outcome: float = 0.0
    rent_capture_rate: float = 0.0
    short_term_arbitrage_rate: float = 0.0
    fiscal_misallocation_rate: float = 0.0
    service_coverage_rate: float = 0.0
    information_quality_rate: float = 0.0
    reference_retained_effect: float | None = None


@dataclass(frozen=True)
class SCRIResult:
    fiscal_spending: float
    verified_food_security_outcome: float
    counterfactual_food_security_outcome: float
    signed_food_security_gain: float
    adverse_food_security_effect: bool
    marginal_food_security_gain: float
    marginal_effect: float
    rent_capture_rate: float
    short_term_arbitrage_rate: float
    fiscal_misallocation_rate: float
    leakage_rate: float
    retained_marginal_effect: float
    reference_retained_effect: float
    scri: float
    repair_potential: float
    repair_adjusted_scri: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class SCRIEvaluator:
    """Compute the State Capacity Retention Index.

    SCRI is designed for the current climate-food-security storyline:
    climate risk -> policy support -> retained production capacity. It measures
    how much verified food-security effect is retained per fiscal yuan after
    deducting observable leakage channels such as rent capture and speculative
    land transfer.
    """

    def __init__(self, weights: SCRIWeights | None = None) -> None:
        self.weights = weights or SCRIWeights()

    def compute(self, item: SCRIInput) -> SCRIResult:
        fiscal_spending = max(0.0, float(item.fiscal_spending or 0.0))
        verified = max(0.0, float(item.verified_food_security_outcome or 0.0))
        counterfactual = max(0.0, float(item.counterfactual_food_security_outcome or 0.0))
        signed_gain = verified - counterfactual
        marginal_gain = max(0.0, signed_gain)
        marginal_effect = _safe_div(marginal_gain, fiscal_spending)

        rent_capture_rate = _rate(item.rent_capture_rate)
        short_term_arbitrage_rate = _rate(item.short_term_arbitrage_rate)
        fiscal_misallocation_rate = _rate(item.fiscal_misallocation_rate)
        leakage_weights = self.weights.normalized_leakage_weights()
        leakage_rate = _clip01(
            leakage_weights["rent_capture"] * rent_capture_rate
            + leakage_weights["short_term_arbitrage"] * short_term_arbitrage_rate
            + leakage_weights["fiscal_misallocation"] * fiscal_misallocation_rate
        )

        retained_marginal_effect = marginal_effect * (1.0 - leakage_rate)
        reference = float(item.reference_retained_effect or retained_marginal_effect or 0.0)
        scri = _safe_div(retained_marginal_effect, reference)

        service_coverage_rate = _rate(item.service_coverage_rate)
        information_quality_rate = _rate(item.information_quality_rate)
        repair_potential = _clip01(
            float(self.weights.service_repair) * service_coverage_rate
            + float(self.weights.information_repair) * information_quality_rate
        )
        repair_adjusted_scri = scri * (1.0 + repair_potential)

        return SCRIResult(
            fiscal_spending=fiscal_spending,
            verified_food_security_outcome=verified,
            counterfactual_food_security_outcome=counterfactual,
            signed_food_security_gain=signed_gain,
            adverse_food_security_effect=signed_gain < 0,
            marginal_food_security_gain=marginal_gain,
            marginal_effect=marginal_effect,
            rent_capture_rate=rent_capture_rate,
            short_term_arbitrage_rate=short_term_arbitrage_rate,
            fiscal_misallocation_rate=fiscal_misallocation_rate,
            leakage_rate=leakage_rate,
            retained_marginal_effect=retained_marginal_effect,
            reference_retained_effect=reference,
            scri=scri,
            repair_potential=repair_potential,
            repair_adjusted_scri=repair_adjusted_scri,
        )


def infer_reference_retained_effect(values: pd.Series, steps: pd.Series | None = None) -> float:
    series = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    series = series[series > 0]
    if series.empty:
        return 0.0
    if steps is None:
        return float(series.iloc[0])
    step_values = pd.to_numeric(steps, errors="coerce")
    min_step = step_values.min()
    first_period = pd.to_numeric(values[step_values == min_step], errors="coerce")
    first_period = first_period.replace([np.inf, -np.inf], np.nan).dropna()
    first_period = first_period[first_period > 0]
    if not first_period.empty:
        return float(first_period.median())
    return float(series.iloc[0])


def compute_scri_table(
    df: pd.DataFrame,
    *,
    scenario_col: str = "scenario",
    baseline_scenario: str = "baseline",
    control_scenario: str = "no_policy",
    id_cols: tuple[str, ...] = ("seed", "year", "step"),
    weights: SCRIWeights | None = None,
    reference_retained_effect: float | None = None,
) -> pd.DataFrame:
    """Compute SCRI from a unified-indicators or counterfactual dataframe.

    If both baseline and control scenarios are present, the verified outcome is
    the baseline outcome and the counterfactual outcome is the matched control
    outcome. Otherwise, the function treats the dataframe as already containing
    single-arm observations and uses zero as the counterfactual unless a
    `counterfactual_food_security_outcome` column is supplied.
    """

    if df.empty:
        return pd.DataFrame()

    if scenario_col in df.columns and {baseline_scenario, control_scenario}.issubset(set(df[scenario_col])):
        existing_id_cols = [col for col in id_cols if col in df.columns]
        if not existing_id_cols:
            raise ValueError(
                "SCRI paired computation needs at least one matching id column "
                f"from {id_cols!r}; otherwise baseline/control rows cannot be paired safely."
            )
        baseline = df[df[scenario_col] == baseline_scenario].copy()
        control = df[df[scenario_col] == control_scenario].copy()
        duplicate_baseline = baseline.duplicated(existing_id_cols).any()
        duplicate_control = control.duplicated(existing_id_cols).any()
        if duplicate_baseline or duplicate_control:
            raise ValueError(
                "SCRI paired computation found duplicate baseline/control keys; "
                f"matching columns={existing_id_cols!r}."
            )
        merged = baseline.merge(control, on=existing_id_cols, suffixes=("", "_control"))
        source = merged
        counterfactual_suffix = "_control"
    else:
        source = df.copy()
        counterfactual_suffix = None
        existing_id_cols = [col for col in id_cols if col in source.columns]

    evaluator = SCRIEvaluator(weights)
    rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        row_dict = row.to_dict()
        verified = _first_existing(
            row_dict,
            (
                "verified_food_security_outcome",
                "verified_grain_output",
                "total_grain_production",
                "grain_to_reserve",
            ),
        )
        if counterfactual_suffix:
            counterfactual = _first_existing(
                row_dict,
                (
                    f"verified_food_security_outcome{counterfactual_suffix}",
                    f"verified_grain_output{counterfactual_suffix}",
                    f"total_grain_production{counterfactual_suffix}",
                    f"grain_to_reserve{counterfactual_suffix}",
                ),
            )
        else:
            counterfactual = _first_existing(row_dict, ("counterfactual_food_security_outcome",), 0.0)

        fiscal_spending = _first_existing(
            row_dict,
            (
                "actual_total_fiscal",
                "total_grain_fiscal_spending",
                "total_subsidy_amount",
                "total_subsidy",
                "total_subsidy_cost",
            ),
        )
        item = SCRIInput(
            fiscal_spending=fiscal_spending,
            verified_food_security_outcome=verified,
            counterfactual_food_security_outcome=counterfactual,
            rent_capture_rate=_first_existing(
                row_dict,
                (
                    "rent_capture_rate",
                    "annual_rent_capitalization_rate",
                    "rent_capitalization_rate",
                ),
            ),
            short_term_arbitrage_rate=_first_existing(
                row_dict,
                ("short_term_arbitrage_rate", "arbitrage_transfer_rate"),
            ),
            fiscal_misallocation_rate=_first_existing(
                row_dict,
                ("fiscal_misallocation_rate", "fiscal_leakage_rate"),
            ),
            service_coverage_rate=_first_existing(row_dict, ("service_coverage_rate",)),
            information_quality_rate=_first_existing(
                row_dict,
                ("information_quality_rate", "monitoring_quality_rate", "ai_monitoring_quality"),
            ),
            reference_retained_effect=reference_retained_effect,
        )
        result = evaluator.compute(item).to_dict()
        for col in existing_id_cols:
            result[col] = row_dict[col]
        if scenario_col in row_dict:
            result[scenario_col] = baseline_scenario if counterfactual_suffix else row_dict[scenario_col]
        rows.append(result)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if reference_retained_effect is None:
        inferred = infer_reference_retained_effect(
            out["retained_marginal_effect"],
            out["step"] if "step" in out.columns else None,
        )
        if inferred > 0:
            out["reference_retained_effect"] = inferred
            out["scri"] = out["retained_marginal_effect"] / inferred
            out["repair_adjusted_scri"] = out["scri"] * (1.0 + out["repair_potential"])
    ordered = [col for col in ("seed", "year", "step", "scenario") if col in out.columns]
    ordered += [col for col in out.columns if col not in ordered]
    return out[ordered]
