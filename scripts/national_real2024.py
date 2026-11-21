from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from scripts.paper_b_flagship_scenarios import Config, _mutable_section, _number, _section


ROOT: Final = Path(__file__).resolve().parents[1]
CPI_SOURCE: Final = ROOT / "config/experiments/national_real2024_cpi_20260909.csv"
COHORT_SOURCE: Final = ROOT / "data/derived_runtime/v5_r6/nfps_fixed_empirical_cohort_2010_k600.json"
OUTPUT: Final = ROOT / "results/empirical/three_paper_execution_20260908/national/real2024_inputs_20260909_r1"
MONEY_FIELDS: Final = ("agri_income", "total_income", "net_income")
MONEY_BASIS: Final = ROOT / "config/experiments/national_real2024_money_basis_20260909.csv"
ENTRANT_SOURCE: Final = ROOT / "data/derived_runtime/v4_8/cfps_entrant_feature_library_2010_2016.json"


class PriceYearError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MoneyConversion:
    cohort_id: str
    field: str
    original_amount: float
    original_price_year: int | None
    factor: float
    amount_2024: float
    conversion_basis: str
    actual_read_path: str


def cpi_factor(year: int, source: Path = CPI_SOURCE) -> float:
    if not 2010 <= year <= 2024:
        raise PriceYearError(f"No audited price-index chain for {year} -> 2024")
    with source.open(encoding="utf-8", newline="") as handle:
        rates = {int(row["year"]): float(row["annual_change_percent"]) for row in csv.DictReader(handle)}
    return math.prod(1.0 + rates[index] / 100.0 for index in range(year + 1, 2025))


def convert_cohort(source: Path, output: Path) -> tuple[MoneyConversion, ...]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    factors = {int(row["year"]): cpi_factor(int(row["year"])) for row in payload["farmers"]}
    records: list[MoneyConversion] = []
    for index, farmer in enumerate(payload["farmers"]):
        year = int(farmer["year"])
        factor = factors[year]
        for field in MONEY_FIELDS:
            original = float(farmer[field])
            farmer[field] = original * factor
            records.append(MoneyConversion(
                cohort_id=str(farmer["cohort_agent_id"]), field=field,
                original_amount=original, original_price_year=year, factor=factor,
                amount_2024=farmer[field], conversion_basis="NBS national annual CPI; purchasing-power conversion",
                actual_read_path=f"initial_conditions.fixed_cohort_path: farmers[{index}].{field} -> FarmerAgent.from_microdata",
            ))
    payload["metadata"]["monetary_representation"] = {
        "price_year": 2024,
        "cohort_year_unchanged": 2010,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_path": source.as_posix(),
        "index_source": CPI_SOURCE.relative_to(ROOT).as_posix(),
        "changed_fields_only": list(MONEY_FIELDS),
        "interpretation": "2010 households in 2024 purchasing-power units; not 2024 households",
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return tuple(records)


def apply_real2024(config: Config, *, input_dir: Path = OUTPUT) -> None:
    cohort = input_dir / "nfps_cohort_2010_money2024.json"
    payload = json.loads(cohort.read_text(encoding="utf-8"))
    metadata = payload["metadata"]["monetary_representation"]
    if metadata["price_year"] != 2024:
        raise PriceYearError(f"Expected 2024 monetary cohort: {cohort}")
    _mutable_section(config, "initial_conditions")["fixed_cohort_path"] = cohort.resolve().as_posix()
    entrant = input_dir / "cfps_entrants_money2024.json"
    entrant_metadata = json.loads(entrant.read_text(encoding="utf-8"))["metadata"]["monetary_representation"]
    if entrant_metadata["price_year"] != 2024:
        raise PriceYearError(f"Expected 2024 monetary entrant library: {entrant}")
    entry = _mutable_section(_mutable_section(config, "population_dynamics"), "entry_mechanism")
    entry["entrant_feature_library"] = entrant.resolve().as_posix()
    prices = _mutable_section(_mutable_section(_mutable_section(config, "production"), "cost_structure"), "input_prices")
    prices.setdefault("small_tools", 20.0)
    prices.setdefault("large_machinery", 80.0)
    config["monetary_representation"] = {
        "price_year": 2024,
        "cohort_year": 2010,
        "coverage": "household_income_entrant_proxies_registered_runtime_money",
        "formal_input_frozen": False,
        "source_sha256": str(metadata["source_sha256"]),
        "cpi_source": CPI_SOURCE.relative_to(ROOT).as_posix(),
        "service_2019_to_model": cpi_factor(2019),
        "runtime_money_basis": MONEY_BASIS.relative_to(ROOT).as_posix(),
        "remaining_freeze": "parent_acceptance_and_P3_budget_forecast_not_owned_by_currency_adapter",
    }


def convert_entrants(source: Path, output: Path) -> tuple[MoneyConversion, ...]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    ambiguous_values = {"return_migrant": {15000, 75000}, "family_farm": {10000, 40000},
                        "young_household": {10000, 40000}, "cooperative": set()}
    factors = {year: cpi_factor(year) for year in (2010, 2012, 2014, 2016)}
    records: list[MoneyConversion] = []
    for pool, ambiguous in ambiguous_values.items():
        for index, entrant in enumerate(payload[pool]):
            original = float(entrant["initial_capital"])
            known = entrant["source"] == "cfps_preprocessed" and original not in ambiguous
            year = int(entrant["year"]) if known else None
            factor = factors[year] if year is not None else 1.0
            basis = ("income_derived_capital_proxy_CPI_not_observed_asset_stock" if known else
                     "new_2024_mechanism_baseline_synthetic_or_ambiguous_floor_fallback_not_deflated")
            entrant["initial_capital"] = original * factor
            records.append(MoneyConversion(
                cohort_id=f"{pool}[{index}]", field="initial_capital", original_amount=original,
                original_price_year=year, factor=factor, amount_2024=original * factor,
                conversion_basis=basis,
                actual_read_path=f"population_dynamics.entry_mechanism.entrant_feature_library:{pool}[{index}].initial_capital",
            ))
    payload["metadata"]["monetary_representation"] = {
        "price_year": 2024, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "changed_fields_only": ["initial_capital"],
        "source_years_unchanged": True,
        "proxy_formula_source": "core/cfps_feature_extractor.py:277-373",
        "ambiguous_old_floor_or_fallback_values": {key: sorted(value) for key, value in ambiguous_values.items()},
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return tuple(records)


def prepare_entrants() -> None:
    records = convert_entrants(ENTRANT_SOURCE, OUTPUT / "cfps_entrants_money2024.json")
    with (OUTPUT / "entrant_money_conversion.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in records)
    print(json.dumps({"entrant_records": len(records),
                      "year_based_proxy_conversion": sum(row.original_price_year is not None for row in records),
                      "new_2024_baseline": sum(row.original_price_year is None for row in records)}))


def write_runtime_money_table(config: Config, output: Path) -> int:
    with MONEY_BASIS.open(encoding="utf-8", newline="") as handle:
        specifications = list(csv.DictReader(handle))
    rows = []
    for spec in specifications:
        parts = spec["config_path"].split(".")
        section = config
        for key in parts[:-1]:
            section = _section(section, key)
        raw = _number(section, parts[-1])
        rows.append({**spec, "original_amount": raw, "factor": 1.0,
                     "amount_2024": raw, "config_read_value": raw, "target_price_year": 2024,
                     "capital_distribution_median_yuan_2024": math.exp(raw) if spec["unit"] == "ln_yuan" else ""})
    with output.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def export_budget_update() -> None:
    receipt = json.loads((OUTPUT / "actual_readthrough_current.json").read_text(encoding="utf-8"))
    table = OUTPUT / "runtime_money_parameters_budget_update.csv"
    count = write_runtime_money_table(receipt["config"], table)
    previous = json.loads((OUTPUT / "currency_delivery_manifest.json").read_text(encoding="utf-8"))
    paths = [ROOT / name for name in previous["files_sha256"]
             if not name.endswith("runtime_money_parameters_current.csv")]
    paths.extend((table, OUTPUT / "CURRENCY_HANDOFF.md"))
    manifest = {
        "status": "currency_core_ready_budget_identity_updated_fiscal_integration_owned_by_Franklin",
        "runtime_money_rows": count,
        "budget_identity": "trial_pool_A0_times350_preserved_and_implementation_separate_8000000_stress_only",
        "amounts_changed_since_one_year_readthrough": False,
        "files_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    with (OUTPUT / "currency_delivery_manifest_budget_update.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "files_sha256"}))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(COHORT_SOURCE.read_bytes()).hexdigest()
    rows = convert_cohort(COHORT_SOURCE, OUTPUT / "nfps_cohort_2010_money2024.json")
    with (OUTPUT / "household_money_conversion.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    prepare_entrants()
    manifest = {
        "status": "currency_adapter_ready_not_formal_input_freeze",
        "source_sha256": source_hash,
        "source_unchanged": source_hash == hashlib.sha256(COHORT_SOURCE.read_bytes()).hexdigest(),
        "households": len(rows) // len(MONEY_FIELDS),
        "amount_records": len(rows),
        "cpi_2010_to_2024": cpi_factor(2010),
        "cpi_2019_to_2024": cpi_factor(2019),
        "pending": ["parent_acceptance", "formal_budget_forecasts_and_fiscal_integration"],
    }
    with (OUTPUT / "conversion_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=True))


if __name__ == "__main__":
    main()
