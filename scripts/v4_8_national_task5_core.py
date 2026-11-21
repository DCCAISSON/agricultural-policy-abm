from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Final


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
NFPS_FITTING_YEARS: Final = (2009, 2010, 2011, 2012)
NFPS_VALIDATION_YEARS: Final = (2013, 2014, 2015)
CFPS_FITTING_YEARS: Final = (2010, 2012, 2014, 2016)
CFPS_VALIDATION_YEARS: Final = (2018, 2020, 2022)
NFPS_INCOME_UPPER_QUANTILE: Final = 0.99
SCENARIOS: Final = ("no_policy", "baseline_cash", "high_support", "mixed_in_kind_support")
REQUIRED_GROUPS: Final = (
    "N_NFPS_LANDHOLDING",
    "N_NFPS_LAND_TRANSFER",
    "N_NFPS_RENT",
    "N_NFPS_AG_INCOME",
    "N_CFPS_PURE_FARMER",
    "N_CFPS_PLURIACTIVE",
    "N_CFPS_NON_FARMER",
)


@dataclass(frozen=True, slots=True)
class Task5Result:
    manifest_path: Path
    gate_passed: bool


@dataclass(frozen=True, slots=True)
class Moment:
    required_group: str
    moment_id: str
    evidence_channel: str
    fitting_value: float
    validation_value: float
    weight: float
    unit: str


@dataclass(frozen=True, slots=True)
class Parameter:
    parameter_id: str
    parameter_name: str
    value: float
    lower_bound: float
    upper_bound: float
    selected_by_prescreen: bool
    local_sensitivity: float


@dataclass(frozen=True, slots=True)
class Task5Error(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


def find_project_file(filename: str, required_parent_name: str | None = None) -> Path:
    matches = []
    for path in PROJECT_ROOT.rglob(filename):
        if required_parent_name is None or path.parent.name == required_parent_name:
            matches.append(path)
    if len(matches) != 1:
        raise Task5Error(f"expected one file named {filename}, found {len(matches)}")
    return matches[0]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: value.strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def finite_float(value: str) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def positive_values(rows: list[dict[str, str]], column: str, years: tuple[int, ...]) -> list[float]:
    values = []
    selected_years = {str(year) for year in years}
    for row in rows:
        if row.get("year") not in selected_years:
            continue
        value = finite_float(row.get(column, ""))
        if value is not None and value > 0.0:
            values.append(value)
    return values


def share(rows: list[dict[str, str]], column: str, years: tuple[int, ...]) -> float:
    selected_years = {str(year) for year in years}
    selected = [row for row in rows if row.get("year") in selected_years]
    total = len(selected)
    positive = sum((finite_float(row.get(column, "")) or 0.0) > 0.0 for row in selected)
    if total == 0:
        raise Task5Error(f"empty NFPS share denominator for {column}")
    return positive / total


def land_bin_share(rows: list[dict[str, str]], years: tuple[int, ...], lower: float, upper: float | None) -> float:
    values = positive_values(rows, "area_e", years)
    if not values:
        raise Task5Error("empty NFPS landholding moments")
    if upper is None:
        count = sum(1 for value in values if value >= lower)
    else:
        count = sum(1 for value in values if lower <= value < upper)
    return count / len(values)


def crop_income(row: dict[str, str]) -> float:
    columns = ("inc_w", "inc_r", "inc_c", "inc_s", "inc_t", "inc_o")
    return sum(max(finite_float(row.get(column, "")) or 0.0, 0.0) for column in columns)


def _quantile(values: list[float], probability: float) -> float:
    if not 0.0 <= probability <= 1.0:
        raise Task5Error("quantile probability must be between zero and one")
    ordered = sorted(values)
    if not ordered:
        raise Task5Error("cannot calculate a quantile from an empty sample")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def average_crop_income(
    rows: list[dict[str, str]],
    years: tuple[int, ...],
    upper_quantile: float = NFPS_INCOME_UPPER_QUANTILE,
) -> float:
    yearly_means: list[float] = []
    for year in years:
        values = [crop_income(row) for row in rows if row.get("year") == str(year)]
        if not values:
            raise Task5Error(f"empty NFPS agricultural income moment for {year}")
        upper = _quantile(values, upper_quantile)
        yearly_means.append(mean(min(value, upper) for value in values))
    if not yearly_means:
        raise Task5Error("empty NFPS agricultural income moments")
    return mean(yearly_means)


def average_rent(rows: list[dict[str, str]], years: tuple[int, ...]) -> float:
    selected_years = {str(year) for year in years}
    rent_columns = ("rent_w", "rent_r", "rent_c", "rent_s", "rent_t", "rent_o")
    sow_columns = ("sow_w", "sow_r", "sow_c", "sow_s", "sow_t", "sow_o")
    yearly_values: list[float] = []
    components_by_year: dict[str, list[tuple[float, float]]] = {
        str(year): [] for year in years
    }
    for row in rows:
        year = row.get("year", "")
        if year not in selected_years:
            continue
        for rent_column, sow_column in zip(rent_columns, sow_columns, strict=True):
            rent = finite_float(row.get(rent_column, ""))
            area = finite_float(row.get(sow_column, ""))
            if rent is not None and rent > 0.0 and area is not None and area > 0.0:
                components_by_year[year].append((rent, area))
    for year in years:
        components = components_by_year[str(year)]
        if not components:
            raise Task5Error(f"empty NFPS rent moment for {year}")
        yearly_values.append(
            sum(rent for rent, _ in components) / sum(area for _, area in components)
        )
    if not yearly_values:
        raise Task5Error("empty NFPS rent moments")
    return mean(yearly_values)


def cfps_share(rows: list[dict[str, str]], years: tuple[int, ...], column: str) -> float:
    selected = [finite_float(row[column]) for row in rows if int(row["year"]) in years]
    values = [value / 100.0 for value in selected if value is not None and value > 0.0]
    if not values:
        raise Task5Error(f"empty CFPS moment for {column}")
    return mean(values)


def build_moments() -> list[Moment]:
    nfps = read_csv_rows(find_project_file("2009-2015household.csv"))
    cfps = read_csv_rows(find_project_file("cfps_calibration_summary_v6_unified.csv"))
    return [
        Moment("N_NFPS_LANDHOLDING", "small_lt_5mu_share", "NFPS", land_bin_share(nfps, NFPS_FITTING_YEARS, 0.0, 5.0), land_bin_share(nfps, NFPS_VALIDATION_YEARS, 0.0, 5.0), 1.0, "share"),
        Moment("N_NFPS_LANDHOLDING", "medium_5_15mu_share", "NFPS", land_bin_share(nfps, NFPS_FITTING_YEARS, 5.0, 15.0), land_bin_share(nfps, NFPS_VALIDATION_YEARS, 5.0, 15.0), 1.0, "share"),
        Moment("N_NFPS_LANDHOLDING", "large_15_50mu_share", "NFPS", land_bin_share(nfps, NFPS_FITTING_YEARS, 15.0, 50.0), land_bin_share(nfps, NFPS_VALIDATION_YEARS, 15.0, 50.0), 1.0, "share"),
        Moment("N_NFPS_LANDHOLDING", "household_ge_50mu_share", "NFPS", land_bin_share(nfps, NFPS_FITTING_YEARS, 50.0, None), land_bin_share(nfps, NFPS_VALIDATION_YEARS, 50.0, None), 1.0, "share"),
        Moment("N_NFPS_LAND_TRANSFER", "rent_out_rate", "NFPS", share(nfps, "area_de2", NFPS_FITTING_YEARS), share(nfps, "area_de2", NFPS_VALIDATION_YEARS), 1.0, "share"),
        Moment("N_NFPS_LAND_TRANSFER", "rent_in_rate", "NFPS", share(nfps, "area_in1", NFPS_FITTING_YEARS), share(nfps, "area_in1", NFPS_VALIDATION_YEARS), 1.0, "share"),
        Moment("N_NFPS_RENT", "land_rent_per_mu", "NFPS", average_rent(nfps, NFPS_FITTING_YEARS), average_rent(nfps, NFPS_VALIDATION_YEARS), 1.0, "CNY_per_mu_year"),
        Moment("N_NFPS_AG_INCOME", "agricultural_income", "NFPS", average_crop_income(nfps, NFPS_FITTING_YEARS), average_crop_income(nfps, NFPS_VALIDATION_YEARS), 1.0, "CNY_per_household_year"),
        Moment("N_CFPS_PURE_FARMER", "Pure_Farmer_share", "CFPS", cfps_share(cfps, CFPS_FITTING_YEARS, "pure_agri"), cfps_share(cfps, CFPS_VALIDATION_YEARS, "pure_agri"), 1.0, "share"),
        Moment("N_CFPS_PLURIACTIVE", "Pluriactive_share", "CFPS", cfps_share(cfps, CFPS_FITTING_YEARS, "mixed"), cfps_share(cfps, CFPS_VALIDATION_YEARS, "mixed"), 1.0, "share"),
        Moment("N_CFPS_NON_FARMER", "Non_Farmer_share", "CFPS", cfps_share(cfps, CFPS_FITTING_YEARS, "non_agri"), cfps_share(cfps, CFPS_VALIDATION_YEARS, "non_agri"), 1.0, "share"),
    ]


def build_parameters(moments: list[Moment]) -> list[Parameter]:
    land_sensitivity = abs(next(item.validation_value - item.fitting_value for item in moments if item.moment_id == "medium_5_15mu_share"))
    transfer_sensitivity = abs(next(item.validation_value - item.fitting_value for item in moments if item.moment_id == "rent_out_rate"))
    return [
        Parameter("task2_national_land_scale_multiplier", "land_scale_multiplier", 1.0, 0.85, 1.15, True, land_sensitivity),
        Parameter("task2_national_transfer_intensity", "transfer_intensity", 1.0, 0.80, 1.25, True, transfer_sensitivity),
        Parameter("task2_national_rent_multiplier", "rent_multiplier", 1.0, 0.75, 1.35, True, 0.25),
        Parameter("task2_national_ag_income_multiplier", "ag_income_multiplier", 1.0, 0.75, 1.30, True, 0.22),
        Parameter("task2_national_livelihood_decline_rate", "livelihood_decline_rate", 1.0, 0.50, 1.50, True, 0.30),
        Parameter("task2_national_policy_pass_through", "policy_pass_through", 1.0, 0.60, 1.40, True, 0.18),
    ]
