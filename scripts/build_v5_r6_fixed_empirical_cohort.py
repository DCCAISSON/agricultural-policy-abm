#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run:
# python scripts/build_v5_r6_fixed_empirical_cohort.py
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v5_r6_calibration_core import CohortMember, CohortSource, build_fixed_cohort


DEFAULT_WEIGHTS: Final = (
    PROJECT_ROOT
    / "project_management"
    / "evidence"
    / "v5_r"
    / "r2"
    / "nfps_calibration_weights_2010.csv"
)
DEFAULT_NFPS: Final = (
    PROJECT_ROOT / "校准用数据源" / "全国农村固定观察点" / "2009-2015household.csv"
)
DEFAULT_OUTPUT: Final = (
    PROJECT_ROOT
    / "data"
    / "derived_runtime"
    / "v5_r6"
    / "nfps_fixed_empirical_cohort_2010_k600.json"
)
DEFAULT_DIAGNOSTICS: Final = (
    PROJECT_ROOT
    / "project_management"
    / "evidence"
    / "v5_r"
    / "r6"
    / "cohort_marginal_diagnostics.csv"
)
REGION_LABELS: Final = {
    "east": "东部",
    "central": "中部",
    "west": "西部",
    "northeast": "东北",
}


class CohortBuildError(RuntimeError):
    pass


def _number(value: str, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def _crop_income(row: dict[str, str]) -> float:
    return math.fsum(
        max(0.0, _number(row.get(column, "")))
        for column in ("inc_w", "inc_r", "inc_c", "inc_s", "inc_t", "inc_o")
    )


def _livelihood(agri_ratio: float) -> str:
    if agri_ratio > 0.5:
        return "Pure_Farmer"
    if agri_ratio > 0.2:
        return "Pluriactive"
    return "Non_Farmer"


def _farmer_scale(farm_size: str) -> str:
    return {
        "lt_5": "small",
        "ge_5_lt_15": "medium",
        "ge_15_lt_50": "family",
        "ge_50": "family",
    }[farm_size]


def _read_sources(path: Path) -> tuple[CohortSource, ...]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    return tuple(
        CohortSource(
            observation_id=row["observation_id"],
            source_row=int(row["source_row"]),
            representation_weight_q=float(row["representation_weight_q"]),
            common_region=row["common_region"],
            farm_size=row["farm_size"],
            land_role=row["land_role"],
            operated_area_mu=float(row["operated_area_mu"]),
        )
        for row in rows
        if row["national_representation_eligible"] == "True"
    )


def _read_nfps(path: Path, source_rows: set[int]) -> dict[int, dict[str, str]]:
    selected: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for source_row, row in enumerate(csv.DictReader(handle), 2):
            if source_row in source_rows:
                selected[source_row] = row
    if set(selected) != source_rows:
        raise CohortBuildError("NFPS source-row join is incomplete")
    return selected


def _marginal_rows(
    sources: tuple[CohortSource, ...],
    cohort: tuple[CohortMember, ...],
) -> list[dict[str, str | int | float]]:
    source_mass: dict[tuple[str, str], float] = defaultdict(float)
    cohort_count: dict[tuple[str, str], int] = defaultdict(int)
    for source in sources:
        for dimension, category in (
            ("common_region", source.common_region),
            ("farm_size", source.farm_size),
            ("land_role", source.land_role),
        ):
            source_mass[(dimension, category)] += source.representation_weight_q
    for member in cohort:
        for dimension, category in (
            ("common_region", member.common_region),
            ("farm_size", member.farm_size),
            ("land_role", member.land_role),
        ):
            cohort_count[(dimension, category)] += 1
    return [
        {
            "dimension": dimension,
            "category": category,
            "source_weighted_share": source_share,
            "cohort_share": cohort_count[(dimension, category)] / len(cohort),
            "absolute_difference": abs(
                cohort_count[(dimension, category)] / len(cohort) - source_share
            ),
            "cohort_count": cohort_count[(dimension, category)],
        }
        for (dimension, category), source_share in sorted(source_mass.items())
    ]


def build(weights: Path, nfps: Path, size: int, output: Path, diagnostics: Path) -> None:
    sources = _read_sources(weights)
    cohort = build_fixed_cohort(sources, size)
    raw = _read_nfps(nfps, {member.source_row for member in cohort})
    farmers = []
    for member in cohort:
        row = raw[member.source_row]
        total_income = max(0.0, _number(row.get("inc_total", "")))
        agri_income = _crop_income(row)
        agri_ratio = agri_income / total_income if total_income > 0.0 else 0.0
        farmers.append(
            {
                "cohort_agent_id": member.cohort_agent_id,
                "source_observation_id": member.source_observation_id,
                "source_row": member.source_row,
                "source_representation_weight_q": member.source_representation_weight_q,
                "cohort_probability_q": member.cohort_probability_q,
                "representation_weight_q": member.runtime_representation_weight,
                "year": 2010,
                "hhid": row.get("hhid", ""),
                "region": REGION_LABELS[member.common_region],
                "common_region": member.common_region,
                "land_area_mu": member.operated_area_mu,
                "land_rent_in_area": max(0.0, _number(row.get("area_in1", ""))),
                "land_rent_out_area": max(0.0, _number(row.get("area_de2", ""))),
                "land_role": member.land_role,
                "farmer_scale": _farmer_scale(member.farm_size),
                "agri_income": agri_income,
                "total_income": total_income,
                "net_income": _number(row.get("inc_net", ""), total_income),
                "agri_ratio": agri_ratio,
                "livelihood": _livelihood(agri_ratio),
                "family_size": max(1, int(_number(row.get("hhnumbers", ""), 3.0))),
            }
        )
    payload = {
        "metadata": {
            "source_weight_file": str(weights.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "source_nfps_file": str(nfps.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "source_population_size": len(sources),
            "cohort_size": size,
            "selection_method": "deterministic_joint_cell_largest_remainder_and_systematic_pps",
            "source_weight_role": "selection_probability",
            "cohort_probability_q": 1.0 / size,
            "runtime_representation_weight": 1.0,
            "runtime_weight_reason": "equal_mass_post_resampling_avoids_double_weighting_annual_mu_accounts",
        },
        "farmers": farmers,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    diagnostic_rows = _marginal_rows(sources, cohort)
    diagnostics.parent.mkdir(parents=True, exist_ok=True)
    with diagnostics.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(diagnostic_rows[0]))
        writer.writeheader()
        writer.writerows(diagnostic_rows)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"cohort={len(farmers)} max_margin_error={max(float(row['absolute_difference']) for row in diagnostic_rows):.8f} sha256={digest}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--nfps", type=Path, default=DEFAULT_NFPS)
    parser.add_argument("--size", type=int, default=600)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    args = parser.parse_args()
    build(args.weights.resolve(), args.nfps.resolve(), args.size, args.output.resolve(), args.diagnostics.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
