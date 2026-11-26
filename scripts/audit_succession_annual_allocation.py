"""Audit frozen strategies and estimate differences on complete paired paths."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
from hashlib import sha256
import json
from math import fsum, isclose
from pathlib import Path
from statistics import mean

from scripts.audit_succession_handoff import keyed, load_csv
from scripts.run_succession_annual_allocation import csv_write, design_inputs, json_write, seeds_of
from scripts.run_succession_shared_budget import METRICS, interval, setup
from scripts.succession_annual_learning import fit, grain_bin, observation, roots, state_key


def gz_csv(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def state(value: str) -> tuple[int, ...]:
    return tuple(map(int, value.split(":")))


def audit_training(path: Path, frozen: dict, design: dict) -> None:
    menu = setup(design)[0].feasible_moves()
    fixed = keyed(load_csv(path / "development_fixed.csv"), ("seed", "action"))
    seeds = seeds_of(design["fixed_development_seeds"])
    require(set(fixed)=={(str(s), str(i)) for s in seeds for i in range(len(menu))},
            "Development fixed grids differ")
    means = [mean(float(fixed[str(s), str(i)]["grain_kg"]) for s in seeds) for i in range(len(menu))]
    require(frozen["fixed_scores"] == means and frozen["strong_fixed"] == max(range(len(means)), key=means.__getitem__),
            "Fixed comparator selection differs")
    data = defaultdict(dict)
    sequence = defaultdict(dict)
    for row in gz_csv(path / "development_openings.csv.gz"):
        repeat, seed = int(row["repeat"]), int(row["seed"])
        key, next_key, action = state(row["observation"]), state(row["successor"]), int(row["action"])
        require(next_key[0] == key[0]-1 and (key[-1] == next_key[-1] == design["budget_yuan"]),
                "Development observation clock or budget differs")
        require(action not in data[repeat, seed, key], "Repeated action at development opening")
        require(sequence[repeat, seed].setdefault(key[0], key) == key, "More than one observed state per opening")
        data[repeat, seed, key][action] = (float(row["grain_kg"]), next_key, float(row["withdrawal_grain_kg"]))
    expected = {(i, seed) for i, bounds in enumerate(design["training_repeats"], 1) for seed in seeds_of(bounds)}
    require(set(sequence)==expected, "Development seed grid differs")
    years = set(range(1, design["active_years"]+1))
    for keys in sequence.values():
        require(set(keys)==years, "Development year grid differs")
    pool = {int(k): v for k, v in frozen["pooled_p0_continuation"].items()}
    for repetition in frozen["repeats"]:
        samples = []
        i = repetition["repeat"]
        for (rep, seed, key), branches in data.items():
            if rep == i:
                require(set(branches)==set(range(len(menu))), "Action support differs within opening")
                samples.append((seed, key, [branches[j] for j in range(len(menu))]))
        table, coverage = fit(samples, design, pool)
        require(table == repetition["table"] and coverage == repetition["coverage"], "Frozen fitted table differs")


def audit_evaluation(train: Path, evaluation: Path, output: Path) -> None:
    design, digest, code = design_inputs()
    raw = (train / "policy_frozen.json").read_bytes()
    frozen = json.loads(raw)
    freeze = sha256(raw).hexdigest()
    rec = json.loads((evaluation / "evaluation_receipt.json").read_bytes())
    trec = json.loads((train / "training_receipt.json").read_bytes())
    require(rec["policy_frozen_sha256"]==trec["freeze_sha256"]==freeze and digest==frozen["design_sha256"]
            ==rec["design_sha256"] and code==frozen["code_sha256"], "Frozen provenance differs")
    audit_training(train, frozen, design)
    budget, _ = setup(design)
    menu = budget.feasible_moves()
    require(json.loads(json.dumps(menu)) == frozen["actions"], "Frozen menu differs")
    seeds, years = seeds_of(design["evaluation_seeds"]), range(1, design["horizon"]+1)
    regions = ("A", "B")
    repeats = {r["repeat"]: r for r in frozen["repeats"]}
    primary = design["primary_repeat"]
    accounts, paths = {}, {}
    modes = (
        [(f"fixed_{i:02}", 0) for i in range(len(menu))] +
        [(f"{k}_{i}", 0) for i in repeats for k in ("q", "myopic")] +
        [(f"{k}_{primary}", 1) for k in ("q", "myopic")]
    )
    for row in gz_csv(evaluation / "region_year.csv.gz"):
        key = (int(row["seed"]), row["strategy"], int(row["cash_reset"]), int(row["year"]), row["region"])
        require(key not in accounts, "Duplicate evaluation region-year")
        accounts[key] = dict(row, action=int(row["action"]), **{
            k: float(row[k]) for k in (*METRICS, "reservation", "guarantee_mu")})
    require(set(accounts)=={(s, k, c, y, r) for s in seeds for k, c in modes for y in years for r in regions}
            and len(accounts)==rec["region_year_rows"] and len(seeds)==rec["paired_seeds"], "Evaluation grids differ")
    counts, shares = Counter(), Counter()
    for s in seeds:
        start = roots(s, design)
        for k, c in modes:
            observed = observation(start, design)
            for y in years:
                rows = [accounts[s, k, c, y, r] for r in regions]
                remaining, reason = observed[0], "fixed"
                kind, num = k.split("_")
                if remaining <= 0:
                    action, reason = 0, "withdrawal"
                elif kind == "fixed":
                    action = int(num)
                else:
                    row = repeats[int(num)]["table"].get(state_key(observed))
                    reason = "unseen" if row is None else "supported" if row["supported"] else "sparse"
                    action = row[kind] if reason == "supported" else 0
                require(sum(r["reservation"] for r in rows)<=budget.common and
                        fsum(r["committed"] for r in rows)<=budget.common+1e-7 and
                        fsum(r["spending"] for r in rows)<=budget.common+1e-7, "Common cap exceeded")
                if remaining > 0:
                    counts[k, c, reason] += 1
                    shares[k, c, y, action] += 1
                for i, r in enumerate(rows):
                    policy, cap = menu[action][i]
                    guarantee = cap/start[i].land
                    require((r["action"], r["reason"], r["observation"], r["policy"],
                             r["reservation"], r["guarantee_mu"]) ==
                            (action, reason, state_key(observed), policy, cap, guarantee),
                            "Action, lagged public state, or prior reservation differs")
                    require(0 <= r["spending"] <= r["committed"]+1e-7 <= cap+2e-7, "Fiscal order differs")
                    if cap == 0:
                        require(all(r[x]==0 for x in ("spending", "committed", "owner_support", "tenant_support")),
                                "Payment at P0")
                    else:
                        for value in (r["spending"], r["committed"]):
                            plots = value / (guarantee * 4)
                            require(isclose(plots, round(plots), abs_tol=1e-8), "Paid or committed amount per plot differs")
                    require(isclose(r["spending"], r["owner_support"]+r["tenant_support"], abs_tol=1e-8)
                            and 0<=r["grain_area"]<=start[i].land and 0<=r["leased_idle"]<=r["leased_area"]
                            and 0<=r["retained_grain"]+r["continuity_area"]<=r["grain_area"]
                            and 0<=r["continuity_area"]<=r["successor_grain"]<=r["grain_area"],
                            "Land/production/support subaccounts differ")
                    if y>1 and not c:
                        previous = accounts[s, k, c, y-1, r["region"]]
                        require(all(isclose(previous[f"{w}_closing_cash"], r[f"{w}_opening_cash"], abs_tol=1e-7)
                                    for w in ("owner", "tenant")), "Cash rollover differs")
                observed = (remaining-1, *(grain_bin(r["grain_kg"], w.land, design)
                                           for r, w in zip(rows, start)), budget.common)
            if c:
                for region in regions:
                    require(all(accounts[s, k, 0, 1, region][m] == accounts[s, k, 1, 1, region][m]
                                for m in METRICS), "Cash sensitivity perturbs initial-year comparison")
            for label, window in (("first", [1]), ("active", range(1, design["active_years"]+1)),
                                  ("withdrawal", range(design["active_years"]+1, design["horizon"]+1)),
                                  ("total", years)):
                for region in (*regions, "joint"):
                    rows = [accounts[s,k,c,y,r] for y in window for r in (regions if region=="joint" else (region,))]
                    measures = {m: fsum(row[m] for row in rows) for m in (*METRICS, "reservation")}
                    if label == "total":
                        for m in ("grain_kg", "retained_grain", "continuity_area", "leased_idle"):
                            measures[f"stop_{m}"] = paths[s,k,c,"withdrawal",region][f"{m}"]
                    paths[s,k,c,label,region] = measures
    output.mkdir(parents=True, exist_ok=False)
    csv_write(output/"path_metrics.csv.gz", [dict(seed=s,strategy=k,cash_reset=c,window=w,region=r,**d)
                                            for (s,k,c,w,r),d in paths.items() if w=="total"])
    csv_write(output/"action_counts.csv", [dict(strategy=k,cash_reset=c,year=y,action=a,count=n)
                                         for (k,c,y,a),n in sorted(shares.items())])
    csv_write(output/"fallback.csv", [dict(strategy=k,cash_reset=c,reason=r,count=n,active_openings=len(seeds)*design["active_years"])
                                    for (k,c,r),n in sorted(counts.items())])
    decisions = summarize(paths, seeds, modes, frozen, design, output)
    cash_stress = [r for (_, _, c, _, _), r in accounts.items() if not c and r["reservation"] > 0]
    tariff = []
    for (policy, cap) in sorted({(r["policy"], r["reservation"]) for r in cash_stress}):
        rows = [r for r in cash_stress if (r["policy"], r["reservation"]) == (policy, cap)]
        tariff.append(dict(policy=policy, reservation=cap, guarantee_yuan_per_mu=rows[0]["guarantee_mu"],
                           funded_region_years=len(rows), positive_payment=sum(r["spending"]>0 for r in rows),
                           mean_commitment=mean(r["committed"] for r in rows),
                           mean_payment=mean(r["spending"] for r in rows)))
    csv_write(output/"effective_tiers.csv", tariff)
    json_write(output/"audit_receipt.json", dict(
        status="passed", region_years=len(accounts), paths=len(seeds)*len(modes), holdout_seeds=len(seeds),
        training_tables_refit=len(repeats), frozen_sha256=freeze, budget_max=budget.common,
        checks=["paired grid", "developer-only fixed selection and policy refit", "lagged observation",
                "support and fallback", "local and common fiscal accounting", "last two years P0",
                "grain/land/support subaccounts", "cash rollover and initial cash sensitivity identity"],
    ))
    json_write(output/"research_decision.json", decisions)


def summarize(paths: dict, seeds: range, modes: list[tuple[str, int]], frozen: dict,
              design: dict, output: Path) -> dict:
    def vector(strategy: str, reset: int, window: str, region: str, metric: str) -> list[float]:
        return [paths[s,strategy,reset,window,region][metric] for s in seeds]
    measures = (*METRICS, "reservation", "stop_grain_kg", "stop_retained_grain", "stop_continuity_area", "stop_leased_idle")
    primary = design["primary_repeat"]
    controls = (f"fixed_{frozen['strong_fixed']:02}", f"myopic_{primary}")
    summary, pairs, mechanism = [], [], []
    for k,c in modes:
        for r in ("A", "B", "joint"):
            row = {m: mean(vector(k,c,"total",r,m)) for m in measures}
            summary.append(dict(strategy=k,cash_reset=c,region=r,n=len(seeds),**row))
    for rep in frozen["repeats"]:
        n = rep["repeat"]
        for ref in (controls[0], f"myopic_{n}"):
            for r in ("joint", "A", "B"):
                for m in measures:
                    primary_test = n==primary and r=="joint" and m in ("grain_kg", "stop_grain_kg", "spending")
                    alpha = design["primary_alpha"]/design["primary_family_size"] if primary_test else 0.05
                    a, b = vector(f"q_{n}",0,"total",r,m), vector(ref,0,"total",r,m)
                    mid, lo, hi = interval([x-y for x,y in zip(a,b)], alpha)
                    pairs.append(dict(focal=f"q_{n}",reference=ref,region=r,metric=m,n=len(seeds),
                                      mean=mid,lower=lo,upper=hi,confidence=1-alpha,primary=primary_test))
    for r in ("A", "B", "joint"):
        for w in ("first", "active", "withdrawal", "total"):
            for m in ("grain_kg", "retained_grain", "continuity_area", "matched_area", "renewed_area", "leased_idle",
                      "tenant_opening_cash", "tenant_closing_cash", "owner_closing_cash", "spending"):
                q,m0 = f"q_{primary}", f"myopic_{primary}"
                vals = {}
                for c in (0,1):
                    vals[c] = [a-b for a,b in zip(vector(q,c,w,r,m),vector(m0,c,w,r,m))]
                for contrast, series in (("carry", vals[0]), ("reset", vals[1]),
                                         ("carry_minus_reset", [a-b for a,b in zip(vals[0],vals[1])])):
                    mid, lo, hi = interval(series,0.05)
                    mechanism.append(dict(contrast=contrast,region=r,window=w,metric=m,
                                          n=len(seeds),mean=mid,lower=lo,upper=hi))
    csv_write(output/"strategy_means.csv", summary)
    csv_write(output/"paired_effects.csv", pairs)
    csv_write(output/"cash_sensitivity.csv", mechanism)
    candidates = [r for r in summary if r["region"]=="joint" and r["cash_reset"]==0]
    csv_write(output/"tested_frontier.csv", [
        dict(strategy=r["strategy"],grain_kg=r["grain_kg"],spending=r["spending"],
             stop_grain_kg=r["stop_grain_kg"],not_dominated=not any(
                 o["grain_kg"]>=r["grain_kg"] and o["spending"]<=r["spending"] and
                 (o["grain_kg"]>r["grain_kg"] or o["spending"]<r["spending"]) for o in candidates))
        for r in candidates
    ])
    main = {}
    for ref in controls:
        tests = {r["metric"]:r for r in pairs if r["primary"] and r["reference"]==ref}
        grain, stop, payment = (tests[m] for m in ("grain_kg", "stop_grain_kg", "spending"))
        tolerance = design["cost_equivalence_yuan"]
        margin = design["grain_and_stop_noninferiority_margin_kg"]
        equivalent = -tolerance<=payment["lower"] and payment["upper"]<=tolerance
        saving = payment["upper"]<0
        no_stop_loss = stop["lower"]>=-margin
        main[ref] = dict(tests=tests, payment_equivalence_established=equivalent,
                         lower_payment_established=saving,
                         improvement_established=no_stop_loss and (
                             grain["lower"]>0 and (equivalent or saving)
                             or saving and grain["lower"]>=-margin))
    coverage = frozen["repeats"][primary-1]["coverage"]["coverage"]
    return dict(primary_repeat=primary, primary_comparisons=main,
                coverage=coverage, coverage_pass=coverage>=design["common_state_action_coverage_required"],
                fiscal_gain=coverage>=design["common_state_action_coverage_required"] and
                all(v["improvement_established"] for v in main.values()),
                evaluation_unit="complete paired two-region synthetic path",
                empirical_validation="not run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("training", "evaluation", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    audit_evaluation(args.training, args.evaluation, args.output)
