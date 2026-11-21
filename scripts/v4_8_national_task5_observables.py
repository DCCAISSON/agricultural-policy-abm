from __future__ import annotations

import math
import statistics
from typing import Final, TypedDict


MODEL_VALIDATION_YEARS: Final = {
    "NFPS": (2013, 2014, 2015),
    "CFPS": (2018, 2020, 2022),
}
ANNUAL_TRANSFER_OBSERVABLE: Final = (
    "annual_household_transfer_participation_across_clearing_and_village_channels"
)


class TargetMetadata(TypedDict):
    empirical_numerator: str
    empirical_denominator: str
    empirical_missing_rule: str
    empirical_outlier_rule: str
    empirical_aggregation_rule: str
    model_observable: str
    model_denominator: str
    model_aggregation_rule: str
    comparability_status: str


def model_years(
    evidence_channel: str,
    moment_id: str,
    *,
    validation: bool,
) -> tuple[int, ...]:
    if validation:
        try:
            return MODEL_VALIDATION_YEARS[evidence_channel]
        except KeyError as exc:
            raise ValueError(f"unsupported evidence channel: {evidence_channel}") from exc
    if evidence_channel == "CFPS":
        return 2010, 2012, 2014, 2016
    if evidence_channel != "NFPS":
        raise ValueError(f"unsupported evidence channel: {evidence_channel}")
    if moment_id.endswith("mu_share"):
        return 2010, 2011, 2012
    return 2011, 2012


def named_year_window(years: tuple[int, ...]) -> str:
    return "simulation_years_" + "|".join(str(year) for year in years)


def annual_transfer_participation(model: object) -> tuple[float, float]:
    farmers = list(getattr(model, "farmers", []) or [])
    farmer_ids = {
        str(getattr(farmer, "unique_id", id(farmer))): farmer
        for farmer in farmers
    }
    clearing_house = getattr(model, "land_clearing_house", None)
    transactions = list(getattr(clearing_house, "transactions", []) or [])
    rent_out_ids = {
        str(getattr(transaction.get("supplier"), "unique_id", id(transaction.get("supplier"))))
        for transaction in transactions
        if str(getattr(transaction.get("supplier"), "unique_id", id(transaction.get("supplier")))) in farmer_ids
    }
    rent_in_ids = {
        str(getattr(transaction.get("demander"), "unique_id", id(transaction.get("demander"))))
        for transaction in transactions
        if str(getattr(transaction.get("demander"), "unique_id", id(transaction.get("demander")))) in farmer_ids
    }
    current_step = int(getattr(model, "step_count", 0) or 0)
    for village in list(getattr(model, "village_collectives", []) or []):
        for record in list(getattr(village, "land_transfer_records", []) or []):
            if int(record.get("step", -1) or -1) != current_step:
                continue
            supplier_id = str(record.get("supplier_id", ""))
            demander_id = str(record.get("demander_id", ""))
            if supplier_id in farmer_ids:
                rent_out_ids.add(supplier_id)
            if demander_id in farmer_ids:
                rent_in_ids.add(demander_id)
    denominator = len(farmers)
    if denominator == 0:
        return 0.0, 0.0
    return len(rent_out_ids) / denominator, len(rent_in_ids) / denominator


def household_rent_burden_per_mu(model: object) -> float:
    farmers = list(getattr(model, "farmers", []) or [])
    farmer_ids = {id(farmer) for farmer in farmers}
    normalize = getattr(model, "_normalize_transfer_rent_total", None)
    total_rent = 0.0
    total_area = 0.0
    for patch in list(getattr(model, "land_patches", []) or []):
        lessee = getattr(patch, "leased_to", None)
        if lessee is None or id(lessee) not in farmer_ids:
            continue
        try:
            area = float(getattr(patch, "area", 0.0) or 0.0)
            raw_rent = float(
                getattr(patch, "rental_price", 0.0)
                or getattr(patch, "contracted_rental_price", 0.0)
                or 0.0
            )
        except (TypeError, ValueError):
            continue
        if area <= 0.0 or raw_rent <= 0.0:
            continue
        rent_total = float(normalize(raw_rent, area)) if callable(normalize) else raw_rent * area
        if math.isfinite(rent_total) and rent_total > 0.0:
            total_rent += rent_total
            total_area += area
    return total_rent / total_area if total_area > 0.0 else 0.0


def crop_gross_income(farmer: object) -> float:
    try:
        value = float(getattr(farmer, "agricultural_income", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, value) if math.isfinite(value) else 0.0


def household_area_shares(
    household_areas: list[float],
    enterprise_areas: list[float] | None = None,
) -> tuple[float, float, float, float]:
    del enterprise_areas
    positive = [area for area in household_areas if math.isfinite(area) and area > 0.0]
    denominator = len(positive)
    if denominator == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        sum(area < 5.0 for area in positive) / denominator,
        sum(5.0 <= area < 15.0 for area in positive) / denominator,
        sum(15.0 <= area < 50.0 for area in positive) / denominator,
        sum(area >= 50.0 for area in positive) / denominator,
    )


def winsorized_mean(values: list[float], upper_quantile: float = 0.99) -> float:
    finite_values = sorted(value for value in values if math.isfinite(value))
    if not finite_values:
        return 0.0
    position = (len(finite_values) - 1) * upper_quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    weight = position - lower_index
    upper = (
        finite_values[lower_index] * (1.0 - weight)
        + finite_values[upper_index] * weight
    )
    return statistics.mean(min(value, upper) for value in finite_values)


def target_metadata(moment_id: str) -> TargetMetadata:
    if moment_id in {
        "small_lt_5mu_share",
        "medium_5_15mu_share",
        "large_15_50mu_share",
        "household_ge_50mu_share",
    }:
        return {
            "empirical_numerator": "NFPS household-year records in the registered operated-area bin",
            "empirical_denominator": "NFPS household-year records with positive area_e",
            "empirical_missing_rule": "exclude missing or nonpositive area_e",
            "empirical_outlier_rule": "none; classify the reported operated area into fixed bins",
            "empirical_aggregation_rule": "pool household-year records across the empirical period",
            "model_observable": "current farmer-household operated-area bin share",
            "model_denominator": "current farmer agents with positive operated area",
            "model_aggregation_rule": "arithmetic mean over named simulation years and seeds",
            "comparability_status": "comparable_household_operated_area_stock",
        }
    if moment_id in {"rent_out_rate", "rent_in_rate"}:
        direction = "area_de2" if moment_id == "rent_out_rate" else "area_in1"
        return {
            "empirical_numerator": f"NFPS household-year records with positive {direction}",
            "empirical_denominator": "all NFPS household-year records in the empirical period",
            "empirical_missing_rule": "blank tenancy-flow field is treated as no recorded annual flow",
            "empirical_outlier_rule": "none; positive annual participation indicator",
            "empirical_aggregation_rule": "pooled annual household participation share",
            "model_observable": ANNUAL_TRANSFER_OBSERVABLE,
            "model_denominator": "all current farmer agents; enterprise transactions excluded from numerator",
            "model_aggregation_rule": "arithmetic mean of annual transaction participation over named years and seeds",
            "comparability_status": "annual_flow_comparable_all_implemented_transfer_channels",
        }
    if moment_id == "land_rent_per_mu":
        return {
            "empirical_numerator": "positive crop-specific rent expenditures rent_w through rent_o",
            "empirical_denominator": "corresponding positive crop sown area sow_w through sow_o",
            "empirical_missing_rule": "exclude crop components without both positive rent and positive sown area",
            "empirical_outlier_rule": "none",
            "empirical_aggregation_rule": "area-weighted rent per mu within year, then arithmetic mean across years",
            "model_observable": "annual household-operated leased-patch rent burden per transferred mu",
            "model_denominator": "positive-area patches leased to farmer households; enterprise leases excluded",
            "model_aggregation_rule": "arithmetic mean over named simulation years and seeds",
            "comparability_status": "partial_crop_rent_to_household_transfer_rent_burden",
        }
    if moment_id == "agricultural_income":
        return {
            "empirical_numerator": "nonnegative crop gross income inc_w through inc_o",
            "empirical_denominator": "all NFPS household-year records in each year",
            "empirical_missing_rule": "blank crop-income component is treated as zero",
            "empirical_outlier_rule": "winsorize household crop gross income at the within-year 99th percentile",
            "empirical_aggregation_rule": "household mean within year, then arithmetic mean across years",
            "model_observable": "within-year 99th-percentile-winsorized farmer crop gross revenue",
            "model_denominator": "all current farmer agents",
            "model_aggregation_rule": "arithmetic mean over named simulation years and seeds",
            "comparability_status": "comparable_crop_gross_income",
        }
    return {
        "empirical_numerator": "classified rural households in the registered CFPS livelihood category",
        "empirical_denominator": "rural households with a valid CFPS livelihood classification",
        "empirical_missing_rule": "exclude households without a valid livelihood classification",
        "empirical_outlier_rule": "not applicable",
        "empirical_aggregation_rule": "arithmetic mean of published year-specific rural-household shares",
        "model_observable": "current farmer-agent livelihood-category share",
        "model_denominator": "all current farmer agents",
        "model_aggregation_rule": "arithmetic mean over named simulation years and seeds",
        "comparability_status": "comparable_rural_household_livelihood_share",
    }
