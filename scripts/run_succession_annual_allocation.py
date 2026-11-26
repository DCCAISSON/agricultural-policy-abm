"""Freeze policies on development paths, then evaluate on independent synthetic paths."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import gzip
from hashlib import sha256
from io import TextIOWrapper
import json
from pathlib import Path
import platform
from statistics import mean

import numpy as np
import scipy

from scripts.run_succession_shared_budget import ROOT, setup, verify_base
from scripts.succession_annual_learning import explore, fit, p0_tail, run_path, state_key

DESIGN = ROOT / "docs/全国经营接续/年度分配/design.json"


def seeds_of(bounds: list[int]) -> range:
    return range(bounds[0], bounds[1] + 1)


def design_inputs() -> tuple[dict, str, dict[str, str]]:
    source = DESIGN.read_bytes().replace(b"\r\n", b"\n")
    design = json.loads(source)
    verify_base(design)
    budget, _ = setup(design)
    blocks = [design["fixed_development_seeds"], *design["training_repeats"], design["evaluation_seeds"]]
    seeds = [s for block in blocks for s in seeds_of(block)]
    if len(seeds) != len(set(seeds)) or set(seeds).intersection(range(330001, 330129)):
        raise ValueError("Seed ranges overlap")
    if len(budget.feasible_moves()) != 13:
        raise ValueError("Unexpected joint menu")
    hashes = {f"scripts/{name}.py": sha256((ROOT / f"scripts/{name}.py").read_bytes()
                                          .replace(b"\r\n", b"\n")).hexdigest()
              for name in ("succession_annual_learning", "run_succession_annual_allocation",
                           "run_succession_shared_budget")}
    return design, sha256(source).hexdigest(), hashes


def json_write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def csv_write(path: Path, rows: list[dict]) -> None:
    if path.suffix == ".gz":
        with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                _csv(stream, rows)
    else:
        with path.open("w", encoding="utf-8", newline="") as stream:
            _csv(stream, rows)


def _csv(stream: TextIOWrapper, rows: list[dict]) -> None:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def fixed_job(task: tuple[int, dict]) -> list[list[dict]]:
    seed, design = task
    budget, _ = setup(design)
    return [run_path(seed, design, f"fixed_{i:02}", fixed=i) for i, _ in enumerate(budget.feasible_moves())]


def train(out: Path, workers: int) -> None:
    design, digest, hashes = design_inputs()
    out.mkdir(parents=True, exist_ok=False)
    fixed_seeds = seeds_of(design["fixed_development_seeds"])
    print("Development: all 13 fixed policies", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        dev = list(pool.map(fixed_job, [(seed, design) for seed in fixed_seeds], chunksize=2))
        fixed_rows = [dict(seed=seed, action=i, grain_kg=sum(r["grain_kg"] for r in path),
                           spending=sum(r["spending"] for r in path))
                      for seed, paths in zip(fixed_seeds, dev, strict=True) for i, path in enumerate(paths)]
        menu = setup(design)[0].feasible_moves()
        scores = [mean(r["grain_kg"] for r in fixed_rows if r["action"] == i) for i, _ in enumerate(menu)]
        tails = p0_tail([paths[0] for paths in dev], design["horizon"])
        frozen = dict(design_sha256=digest, code_sha256=hashes, actions=menu,
                      fixed_scores=scores, strong_fixed=max(range(len(scores)), key=scores.__getitem__),
                      pooled_p0_continuation=tails, repeats=[])
        replay = []
        for repeat, bounds in enumerate(design["training_repeats"], 1):
            print(f"Development: shared openings, repeat {repeat}", flush=True)
            keys = seeds_of(bounds)
            paths = pool.map(explore, keys, [design] * len(keys), chunksize=4)
            openings = [row for path in paths for row in path]
            table, coverage = fit(openings, design, tails)
            frozen["repeats"].append(dict(repeat=repeat, coverage=coverage, table=table))
            replay.extend(dict(repeat=repeat, seed=seed, observation=state_key(state), action=i,
                               grain_kg=grain, successor=state_key(successor), withdrawal_grain_kg=tail)
                          for seed, state, branches in openings
                          for i, (grain, successor, tail) in enumerate(branches))
    csv_write(out / "development_fixed.csv", fixed_rows)
    csv_write(out / "development_openings.csv.gz", replay)
    json_write(out / "policy_frozen.json", frozen)
    json_write(out / "training_receipt.json", dict(
        design_sha256=digest, fixed_dev_paths=len(fixed_rows), training_action_rows=len(replay),
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        freeze_sha256=sha256((out/"policy_frozen.json").read_bytes()).hexdigest(),
    ))
    print(f"Frozen: strong fixed action {frozen['strong_fixed']}; coverage "
          f"{[r['coverage']['coverage'] for r in frozen['repeats']]}", flush=True)


def eval_job(task: tuple[int, dict, dict]) -> list[dict]:
    seed, design, frozen = task
    rows = [row for path in fixed_job((seed, design)) for row in path]
    for repeat in frozen["repeats"]:
        n = repeat["repeat"]
        for strategy in (f"q_{n}", f"myopic_{n}"):
            rows.extend(run_path(seed, design, strategy, table=repeat["table"]))
            if n == design["primary_repeat"]:
                rows.extend(run_path(seed, design, strategy, table=repeat["table"], cash_reset=True))
    return rows


def evaluate(frozen_dir: Path, out: Path, workers: int) -> None:
    design, digest, hashes = design_inputs()
    raw = (frozen_dir / "policy_frozen.json").read_bytes()
    frozen = json.loads(raw)
    receipt = json.loads((frozen_dir / "training_receipt.json").read_text(encoding="utf-8"))
    if (digest != frozen["design_sha256"] or hashes != frozen["code_sha256"]
            or sha256(raw).hexdigest() != receipt["freeze_sha256"]):
        raise ValueError("Frozen policy provenance mismatch")
    out.mkdir(parents=True, exist_ok=False)
    seeds = seeds_of(design["evaluation_seeds"])
    print(f"Evaluation: {len(seeds)} independent paired seeds", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for path in pool.map(eval_job, [(seed, design, frozen) for seed in seeds], chunksize=2)
                for row in path]
    csv_write(out / "region_year.csv.gz", rows)
    json_write(out / "evaluation_receipt.json", dict(
        design_sha256=digest, paired_seeds=len(seeds), region_year_rows=len(rows),
        policy_frozen_sha256=receipt["freeze_sha256"],
    ))
    print(f"Evaluation complete: {len(rows)} region-years", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "train":
        train(args.output, args.workers)
    elif args.frozen:
        evaluate(args.frozen, args.output, args.workers)
    else:
        parser.error("evaluate requires --frozen")
