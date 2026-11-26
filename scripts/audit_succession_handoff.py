"""Independently key old paired results; leave the frozen pulse files untouched."""

import argparse
import csv
from collections import defaultdict
from hashlib import sha256
import json
from math import fsum, isclose
from pathlib import Path

from scripts.run_succession_annual_allocation import csv_write, design_inputs, json_write
from scripts.run_succession_shared_budget import INSTRUMENTS, METRICS, interval, paired, setup


def keyed(rows: list[dict], columns: tuple[str, ...]) -> dict:
    result = {tuple(row[k] for k in columns): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate values in key {columns}")
    return result


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def audit_pulse(source: Path, output: Path) -> None:
    design, _, _ = design_inputs()
    options = design["old_pulse_check"]
    abs_tol, rel_tol = options["absolute_tolerance"], options["relative_tolerance"]
    digest = (source / "design.json").read_bytes()
    pulse = json.loads(digest)
    receipt = json.loads((source / "receipt.json").read_bytes())
    if sha256(digest).hexdigest() != receipt["design_sha256"]:
        raise ValueError("Old design differs from receipt")
    lo, hi = pulse["seeds_inclusive"]
    seeds, years = range(lo, hi + 1), range(1, pulse["horizon"] + 1)
    arms = {"none": (("none", 0), ("none", 0))}
    low, high = setup(pulse)[0].caps
    for policy in INSTRUMENTS:
        arms.update({f"{policy}_A": ((policy, high), ("none", 0)),
                     f"{policy}_B": (("none", 0), (policy, high)),
                     f"{policy}_split": ((policy, low), (policy, low))})
    rows = [
        dict(raw, seed=int(raw["seed"]), year=int(raw["year"]),
             **{m: float(raw[m]) for m in (*METRICS, "guarantee_mu", "reservation", "common_budget")})
        for raw in load_csv(source / "region_year.csv")
    ]
    regions = ("A", "B")
    by_key = keyed(rows, ("seed", "arm", "year", "region"))
    if set(by_key) != {(s, a, y, r) for s in seeds for a in arms for y in years for r in regions}:
        raise ValueError("Old regional grid incomplete")
    for (s, a, y, r), row in by_key.items():
        action = arms[a][regions.index(r)] if y == 1 else ("none", 0)
        if (row["policy"], row["reservation"]) != action or not (
            0 <= row["spending"] <= row["committed"] + 1e-7 <= row["reservation"] + 2e-7
        ):
            raise ValueError("Old fiscal ledger differs from frozen menu")
        if y > 1 and any(not isclose(by_key[s, a, y-1, r][f"{who}_closing_cash"],
                                     row[f"{who}_opening_cash"], abs_tol=1e-8)
                         for who in ("owner", "tenant")):
            raise ValueError("Old cash rollover differs")
    estimates, decisions = paired(rows, pulse)
    columns = ("instrument", "focal", "reference", "region", "window", "metric")
    expected = keyed(load_csv(source / "paired_effects.csv"), columns)
    actual = keyed(estimates, columns)
    if set(expected) != set(actual):
        raise ValueError("Old paired estimate keys differ")
    max_abs = defaultdict(float)
    for key, row in actual.items():
        for k, v in row.items():
            previous = expected[key][k]
            if k in ("mean", "low", "high", "confidence"):
                max_abs[k] = max(max_abs[k], abs(v-float(previous)))
                if not isclose(v, float(previous), abs_tol=abs_tol, rel_tol=rel_tol):
                    raise ValueError(f"Old contrast tolerance exceeded: {key} {k}")
            elif str(v) != previous:
                raise ValueError(f"Old contrast category differs: {key} {k}")
    old_decisions = {r["instrument"]: r for r in receipt["decisions"]}
    for row in decisions:
        ref = old_decisions[row["instrument"]]
        if row["supported_reversal"] != ref["supported_reversal"] or any(
            not isclose(a, b, abs_tol=abs_tol, rel_tol=rel_tol)
            for field in ("first_year_interval", "cumulative_interval")
            for a, b in zip(row[field], ref[field], strict=True)
        ):
            raise ValueError("Old training gate changed")
    if any(r["supported_reversal"] for r in decisions) != receipt["stage_2_eligible"]:
        raise ValueError("Old overall training gate changed")
    supplemental = []
    for instrument in INSTRUMENTS:
        for focal, ref in (("A", "B"), ("split", "A"), ("split", "B")):
            for window, span in (("first_year", [1]), ("cumulative", years)):
                for metric in ("grain_kg", "spending"):
                    for region in ("total", *regions):
                        rs = regions if region == "total" else (region,)
                        values = [fsum(by_key[s, f"{instrument}_{focal}", y, r][metric] -
                                       by_key[s, f"{instrument}_{ref}", y, r][metric]
                                       for y in span for r in rs) for s in seeds]
                        center, low95, high95 = interval(values, 0.05)
                        _, lower, upper = interval(values, 0.05/options["supplement_simultaneous_family"])
                        supplemental.append(dict(instrument=instrument, focal=focal, reference=ref,
                                                 window=window, metric=metric, region=region, n=len(values),
                                                 mean=center, low95=low95, high95=high95,
                                                 joint_low=lower if region=="total" else "",
                                                 joint_high=upper if region=="total" else ""))
    output.mkdir(parents=True, exist_ok=False)
    csv_write(output / "pulse_supplement.csv", supplemental)
    json_write(output / "pulse_readback.json", dict(
        region_years=len(rows), estimates=len(estimates), maxima=dict(max_abs),
        absolute_tolerance=abs_tol, relative_tolerance=rel_tol,
        stage_2_eligible=receipt["stage_2_eligible"], decisions=decisions,
        supplemental_rows=len(supplemental),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pulse", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit_pulse(args.pulse, args.output)
