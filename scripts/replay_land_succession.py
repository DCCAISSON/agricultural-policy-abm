"""Independently rerun prespecified worlds and reconcile the published account rows."""
import argparse
import ast
import csv
import gzip
import json
from hashlib import sha256
from pathlib import Path

from scripts.run_land_succession import (
    Experiment, _init_evaluation, designs, run_experiment,
)
from scripts.succession_q import LearnedPolicy


def replay(directory: Path, receipt: Path) -> None:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    for filename in ("core/land_succession.py", "core/succession_allocation.py",
                     "agents/government_agent.py", "scripts/run_land_succession.py",
                     "scripts/succession_q.py"):
        actual = sha256((root / filename).read_bytes()).hexdigest()
        if actual != manifest["files"][filename]:
            raise ValueError(f"Runtime source fingerprint mismatch: {filename}")
    models = json.loads((directory / "q_table.json").read_text(encoding="utf-8"))
    selected = sorted([name for name in models if name.startswith("q_base_")])[:1]
    q = []
    for name in selected:
        data = models[name]
        keys = {ast.literal_eval(key): value for key, value in data["states"].items()}
        q.append(LearnedPolicy(name, data["profile"], data["aggregated"],
                               {key: tuple(x["q"]) for key, x in keys.items()},
                               {key: tuple(x["visits"]) for key, x in keys.items()},
                               data["coverage_passed"]))
    _init_evaluation(tuple(q))
    cases = designs()
    jobs = []
    for group in ("factorial", "fixed_contract", "diagnostic", "duration", "plot_order", "evaluation"):
        for name in cases[group]:
            if group == "diagnostic" and name not in (
                "cash_reset", "exit_reset", "friction_off", "recovery", "reentry",
            ):
                continue
            seed = manifest["seeds"][group][0]
            jobs.append(Experiment(group, name, seed, cases[group][name]))
    counts = {}
    selected_worlds = {(j.group, j.condition, str(j.seed)) for j in jobs}
    replayed = [run_experiment(job) for job in jobs]
    for name in ("annual", "plots", "operators", "transactions", "fiscal", "withdrawal_cohort"):
        suffix = ".csv" if name == "annual" else ".csv.gz"
        opener = open if name == "annual" else gzip.open
        expected = (row for block in replayed for row in block.get(name, ()))
        with opener(directory / (name + suffix), "rt", encoding="utf-8", newline="") as ledger:
            actual = (row for row in csv.DictReader(ledger)
                      if (row["group"], row["condition"], row["seed"]) in selected_worlds
                      and (not row["strategy"].startswith("q_") or row["strategy"] in selected))
            count = 0
            for lhs, rhs in zip(expected, actual, strict=True):
                rendered = {key: "" if value is None else str(value) for key, value in lhs.items()}
                if rendered != rhs:
                    raise ValueError(f"Replay mismatch: {name} record {count}, {rhs['group']}/{rhs['condition']}")
                count += 1
            counts[name] = count
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump({"status": "passed", "world_anchors": sorted(selected_worlds),
                   "learned_strategies": selected, "exact_csv_rows": counts}, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    replay(args.run, args.receipt)
