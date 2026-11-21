from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from analysis.scri_evaluator import SCRIEvaluator, SCRIInput


CARBON_REFERENCE_KG_CO2E_PER_MU = 800.0
NITROGEN_REFERENCE_KG_N_PER_MU = 30.0


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    return max(0.0, min(1.0, float(value)))


def _rate(value: Any) -> float:
    numeric = _finite_float(value, 0.0)
    if numeric > 1.5:
        numeric /= 100.0
    return _clip01(numeric)


def _rate_or_nan(value: Any) -> float:
    numeric = _finite_float(value)
    if not math.isfinite(numeric):
        return float("nan")
    if numeric > 1.5:
        numeric /= 100.0
    return _clip01(numeric)


def _safe_ratio(numerator: Any, denominator: Any, *, default: float = float("nan")) -> float:
    num = _finite_float(numerator)
    den = _finite_float(denominator)
    if not math.isfinite(num) or not math.isfinite(den) or den <= 0:
        return default
    return num / den


def _livelihood_counts(indicators: Any) -> tuple[int, int, int]:
    return (
        int(_finite_float(getattr(indicators, "pure_farmer_count", 0), 0.0) or 0),
        int(_finite_float(getattr(indicators, "pluriactive_count", 0), 0.0) or 0),
        int(_finite_float(getattr(indicators, "non_farmer_count", 0), 0.0) or 0),
    )


def _smallholder_retention_component(indicators: Any, current_farmer_count: int) -> float:
    pure_count, pluriactive_count, non_farmer_count = _livelihood_counts(indicators)
    counted_farmers = pure_count + pluriactive_count + non_farmer_count
    if counted_farmers == current_farmer_count and current_farmer_count > 0:
        return _clip01(pure_count / current_farmer_count)

    reported_ratio = _finite_float(getattr(indicators, "pure_farmer_ratio", float("nan")))
    if math.isfinite(reported_ratio) and reported_ratio > 0:
        return _rate_or_nan(reported_ratio)

    return float("nan")


def _income_observed(indicators: Any, *, allow_zero_gini: bool = False) -> bool:
    income_fields = (
        "all_household_net_income",
        "all_household_gross_income",
        "farmer_avg_income",
        "pure_farmer_net_income",
    )
    if any(abs(_finite_float(getattr(indicators, field, float("nan")), 0.0)) > 0 for field in income_fields):
        return True
    if allow_zero_gini:
        return True
    gini = _finite_float(getattr(indicators, "farmer_income_gini", float("nan")))
    return math.isfinite(gini) and gini > 0.0


@dataclass(frozen=True)
class AxisDiagnostics:
    scri: float = float("nan")
    scri_status: str = "reference_required"
    scri_retained_marginal_effect: float = 0.0
    scri_leakage_rate: float = 0.0
    idi: float = float("nan")
    idi_status: str = "unavailable"
    idi_smallholder_retention_component: float = float("nan")
    idi_exit_penalty_component: float = float("nan")
    idi_income_component: float = float("nan")
    idi_distribution_component: float = float("nan")
    idi_exit_state_component: float = float("nan")
    aesi: float = float("nan")
    aesi_status: str = "unavailable"
    aesi_soil_component: float = float("nan")
    aesi_carbon_component: float = float("nan")
    aesi_nitrogen_component: float = float("nan")


def compute_scri_diagnostic(
    indicators: Any,
    *,
    reference_retained_effect: float | None = None,
) -> dict[str, Any]:
    fiscal_spending = max(0.0, _finite_float(getattr(indicators, "total_subsidy", 0.0), 0.0))
    if fiscal_spending <= 0.0:
        return {
            "scri": float("nan"),
            "scri_status": "no_fiscal_input",
            "scri_retained_marginal_effect": 0.0,
            "scri_leakage_rate": 0.0,
        }

    reference = reference_retained_effect
    if reference is None:
        reference = getattr(indicators, "scri_reference_retained_effect", None)
    reference = _finite_float(reference)

    result = SCRIEvaluator().compute(
        SCRIInput(
            fiscal_spending=fiscal_spending,
            verified_food_security_outcome=_finite_float(
                getattr(indicators, "total_grain_production", 0.0),
                0.0,
            ),
            counterfactual_food_security_outcome=_finite_float(
                getattr(indicators, "counterfactual_food_security_outcome", 0.0),
                0.0,
            ),
            rent_capture_rate=getattr(
                indicators,
                "annual_rent_capitalization_rate",
                getattr(indicators, "rent_capitalization_rate", 0.0),
            ),
            short_term_arbitrage_rate=getattr(indicators, "short_term_arbitrage_rate", 0.0),
            fiscal_misallocation_rate=getattr(indicators, "fiscal_misallocation_rate", 0.0),
            service_coverage_rate=getattr(indicators, "service_coverage_rate", 0.0),
            information_quality_rate=getattr(indicators, "information_quality_rate", 0.0),
            reference_retained_effect=reference if math.isfinite(reference) and reference > 0 else None,
        )
    )

    if not math.isfinite(reference) or reference <= 0:
        return {
            "scri": float("nan"),
            "scri_status": "reference_required",
            "scri_retained_marginal_effect": result.retained_marginal_effect,
            "scri_leakage_rate": result.leakage_rate,
        }

    return {
        "scri": result.scri,
        "scri_status": "computed",
        "scri_retained_marginal_effect": result.retained_marginal_effect,
        "scri_leakage_rate": result.leakage_rate,
    }


def compute_idi_diagnostic(indicators: Any) -> dict[str, Any]:
    current_farmer_count = int(_finite_float(getattr(indicators, "farmer_count", 0), 0.0) or 0)
    if current_farmer_count <= 0:
        return {
            "idi": float("nan"),
            "idi_status": "unavailable",
            "idi_smallholder_retention_component": float("nan"),
            "idi_exit_penalty_component": float("nan"),
            "idi_income_component": float("nan"),
            "idi_distribution_component": float("nan"),
            "idi_exit_state_component": float("nan"),
        }

    retention_component = _smallholder_retention_component(indicators, current_farmer_count)
    graveyard_status = str(getattr(indicators, "graveyard_observation_status", "missing") or "missing")
    if graveyard_status == "missing":
        exit_penalty_component = float("nan")
    elif graveyard_status == "observed_no_exits":
        exit_penalty_component = 1.0
    else:
        exit_rate = _rate_or_nan(getattr(indicators, "smallholder_exit_rate_graveyard", float("nan")))
        exit_penalty_component = 1.0 - exit_rate if math.isfinite(exit_rate) else float("nan")

    pure_count, _, _ = _livelihood_counts(indicators)
    pure_ratio_observed = math.isfinite(retention_component) and retention_component > 0.0
    pure_income_observed = (
        pure_count > 0
        or pure_ratio_observed
        or _finite_float(getattr(indicators, "pure_farmer_net_income", float("nan")), 0.0) > 0
    )
    if pure_income_observed:
        income_component = _clip01(
            _safe_ratio(
                getattr(indicators, "pure_farmer_net_income", float("nan")),
                getattr(indicators, "all_household_net_income", float("nan")),
            )
        )
    else:
        income_component = float("nan")

    if _income_observed(indicators, allow_zero_gini=current_farmer_count <= 1):
        distribution_component = _clip01(
            1.0 - _finite_float(getattr(indicators, "farmer_income_gini", float("nan")))
        )
    else:
        distribution_component = float("nan")

    exit_count = int(_finite_float(getattr(indicators, "farmer_exit_count", 0), 0.0) or 0)
    if graveyard_status == "missing":
        exit_state_component = float("nan")
    elif exit_count <= 0:
        exit_state_component = 1.0
    else:
        exit_income_ratio = _clip01(
            _safe_ratio(
                getattr(indicators, "graveyard_farmer_avg_exit_income", float("nan")),
                getattr(indicators, "all_household_net_income", float("nan")),
            )
        )
        exit_land_ratio = _clip01(
            _safe_ratio(
                getattr(indicators, "graveyard_farmer_avg_exit_land_area", float("nan")),
                getattr(indicators, "pure_farmer_mean_operated_area", float("nan")),
            )
        )
        components = [value for value in (exit_income_ratio, exit_land_ratio) if math.isfinite(value)]
        exit_state_component = float(np.mean(components)) if components else float("nan")

    weighted_components = [
        (0.30, retention_component),
        (0.20, exit_penalty_component),
        (0.25, income_component),
        (0.15, distribution_component),
        (0.10, exit_state_component),
    ]
    valid = [(weight, value) for weight, value in weighted_components if math.isfinite(value)]
    if not valid:
        idi = float("nan")
        status = "unavailable"
    else:
        weight_sum = sum(weight for weight, _ in valid)
        idi = sum(weight * value for weight, value in valid) / weight_sum
        status = "computed" if len(valid) == len(weighted_components) else "partial"

    return {
        "idi": _clip01(idi),
        "idi_status": status,
        "idi_smallholder_retention_component": retention_component,
        "idi_exit_penalty_component": exit_penalty_component,
        "idi_income_component": income_component,
        "idi_distribution_component": distribution_component,
        "idi_exit_state_component": exit_state_component,
    }


def compute_aesi_diagnostic(indicators: Any) -> dict[str, Any]:
    soil_component = _clip01(_finite_float(getattr(indicators, "soil_quality", float("nan"))))
    carbon_value = _finite_float(getattr(indicators, "carbon_footprint", float("nan")))
    nitrogen_value = _finite_float(getattr(indicators, "nitrogen_pollution", float("nan")))

    carbon_component = (
        _clip01(1.0 - carbon_value / CARBON_REFERENCE_KG_CO2E_PER_MU)
        if math.isfinite(carbon_value)
        else float("nan")
    )
    nitrogen_component = (
        _clip01(1.0 - nitrogen_value / NITROGEN_REFERENCE_KG_N_PER_MU)
        if math.isfinite(nitrogen_value)
        else float("nan")
    )

    if not math.isfinite(soil_component):
        return {
            "aesi": float("nan"),
            "aesi_status": "unavailable",
            "aesi_soil_component": soil_component,
            "aesi_carbon_component": carbon_component,
            "aesi_nitrogen_component": nitrogen_component,
        }

    components = [
        (0.40, soil_component),
        (0.30, carbon_component),
        (0.30, nitrogen_component),
    ]
    valid = [(weight, value) for weight, value in components if math.isfinite(value)]
    if len(valid) == len(components):
        status = "computed"
        weight_sum = 1.0
    elif len(valid) == 1:
        status = "soil_only"
        weight_sum = valid[0][0]
    else:
        status = "partial"
        weight_sum = sum(weight for weight, _ in valid)

    aesi = sum(weight * value for weight, value in valid) / weight_sum
    return {
        "aesi": _clip01(aesi),
        "aesi_status": status,
        "aesi_soil_component": soil_component,
        "aesi_carbon_component": carbon_component,
        "aesi_nitrogen_component": nitrogen_component,
    }


def compute_three_axis_diagnostics(
    indicators: Any,
    *,
    scri_reference_retained_effect: float | None = None,
) -> AxisDiagnostics:
    values: dict[str, Any] = {}
    values.update(
        compute_scri_diagnostic(
            indicators,
            reference_retained_effect=scri_reference_retained_effect,
        )
    )
    values.update(compute_idi_diagnostic(indicators))
    values.update(compute_aesi_diagnostic(indicators))
    return AxisDiagnostics(**values)


def apply_three_axis_diagnostics(
    indicators: Any,
    *,
    scri_reference_retained_effect: float | None = None,
) -> Any:
    diagnostics = compute_three_axis_diagnostics(
        indicators,
        scri_reference_retained_effect=scri_reference_retained_effect,
    )
    for key, value in diagnostics.__dict__.items():
        setattr(indicators, key, value)
    return indicators
