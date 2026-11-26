"""Reaggregate frozen synthetic 7a ledgers for fallback C; never run the model."""

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from hashlib import file_digest
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import t

WORKTREE = Path(__file__).resolve().parents[1]
FROZEN = WORKTREE.parents[1] / "land_succession_transmission_final_20261126"
FORMAL = FROZEN / "formal"
OUT = WORKTREE / "docs" / "全国经营接续" / "7a回读"
SNAPSHOT = "7a304a8ab33af8b2ca9ee3225428c900c8f5d02d"
REF = FROZEN / "worktree/results/experiments/land_succession_transmission_final_20261126"
KEY = ("group", "condition", "strategy", "seed", "year")
POLICIES = ("none", "grain_pre", "grain_post", "takeover_pre", "takeover_post")
REPEATS = tuple(f"q_base_{r:02d}" for r in range(1, 6))
YEARS = range(1, 9)
GAMMA = .92
METRICS = ("grain_area", "grain_kg", "retained_grain", "continuity_area", "reactivated_area",
           "unfilled_area", "matched_area", "leased_area", "held_area", "leased_idle", "idle_area",
           "cash_crop_area", "cash_unplanted", "labor_unplanted", "owner_area", "tenant_area",
           "budget", "announced_rate", "committed", "spending", "owner_support", "tenant_support",
           "rent_total", "owner_rent_income", "owner_income", "tenant_income")
Cell = str | float | int | bool | None
Key = tuple[str, str, str, str, str]


def require(ok: bool, label: str) -> None:
    if not ok:
        raise RuntimeError(label)


def read(path: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as file:
        yield from csv.DictReader(file)


def key(r: dict[str, str]) -> Key:
    return r["group"], r["condition"], r["strategy"], r["seed"], r["year"]


def write(name: str, rows: list[dict[str, Cell]]) -> None:
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: None if isinstance(v, float) and np.isnan(v) else v for k, v in row.items()})


def stats(x: np.ndarray) -> dict[str, Cell]:
    vals = x[np.isfinite(x)]
    n = len(vals)
    mean = float(vals.mean()) if n else None
    half = float(t.ppf(.975, n - 1) * vals.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    return dict(n=n, mean=mean, ci95_low=mean-half if half is not None else None,
                ci95_high=mean+half if half is not None else None)


def digest(path: Path, unpack: bool = False) -> str:
    with (gzip.open(path, "rb") if unpack else path.open("rb")) as file:
        return file_digest(file, "sha256").hexdigest()


def main() -> None:
    provenance = json.loads((REF / "audit_provenance.json").read_text(encoding="utf-8"))
    manifest = json.loads((FORMAL / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((FORMAL / "ledger_verification.json").read_text(encoding="utf-8"))
    sums = dict(line.split()[::-1] for line in (REF / "SHA256SUMS.txt").read_text().splitlines())
    names = ("manifest.json", "annual.csv", "operators.csv.gz", "fiscal.csv.gz", "transactions.csv.gz",
             "training_sar.csv.gz", "training_checkpoints.csv", "q_table.json", "ledger_verification.json",
             "cessation_cohorts.csv")
    hashes = {name: digest(FORMAL / name) for name in names}
    for name, sha in hashes.items():
        expected = (digest(REF / "cessation_cohorts.csv.gz", unpack=True) if name == "cessation_cohorts.csv"
                    else provenance["inputs"].get(name, sums.get(name)))
        require(sha == expected, f"Frozen source mismatch: {name}")
    require(audit["checks"] == "passed" and not audit["failure_counts"]
            and audit["analysis_sha256"] == provenance["audit_source_sha256"], "Frozen audit not passed")
    require(manifest["synthetic"] and tuple(manifest["policies"]) == POLICIES and manifest["years"] == 8
            and manifest["q"]["discount"] == GAMMA and manifest["q"]["weights"]["base"] == [1., .18, .14],
            "Frozen design mismatch")
    groups = ("factorial", "fixed_contract", "evaluation", "duration")
    seeds = {g: tuple(map(str, manifest["seeds"][g])) for g in groups}
    conditions = {g: tuple(manifest["scenario"][g]) for g in groups}
    require(all(len(seeds[g]) == len(set(seeds[g])) == n for g, n in zip(groups, (100, 100, 100, 50))),
            "Unexpected seed coverage")
    training: set[str] = set()
    for r in read(FORMAL / "training_sar.csv.gz"):
        training.add(r["seed"])
    require(not any(a & b for a, b in ((training, set(seeds["factorial"])),
                (training, set(seeds["evaluation"])), (set(seeds["factorial"]), set(seeds["evaluation"])))),
            "Training, fixed selection, or evaluation seeds overlap")

    annual = {key(r): r for r in read(FORMAL / "annual.csv") if r["group"] in groups}
    tables = json.loads((FORMAL / "q_table.json").read_text(encoding="utf-8"))
    strategies = {"factorial": POLICIES, "fixed_contract": POLICIES,
                  "evaluation": (*POLICIES, "rule", *tables),
                  "duration": ("none", *(f"{p}_y{y}" for p, y in product(POLICIES[1:], (1, 3, 8))))}
    expected = {k for g in groups for k in product((g,), conditions[g], strategies[g], seeds[g], map(str, YEARS))}
    require(expected == annual.keys(), "Frozen annual panel incomplete")

    def panel(g: str, c: str, s: str, field: str) -> np.ndarray:
        return np.array([[float(annual[g, c, s, seed, str(y)][field]) for y in YEARS] for seed in seeds[g]])

    def check_fixed() -> dict[str, int]:
        fixed = {k: r for k, r in annual.items() if k[0] == "fixed_contract"}
        fiscal: defaultdict[Key, np.ndarray] = defaultdict(lambda: np.zeros(2))
        post_stop_rows = 0
        for r in read(FORMAL / "fiscal.csv.gz"):
            if r["group"] == "fixed_contract":
                fiscal[key(r)] += [float(r["commitment"]), float(r["support"])]
            if r["group"] == "duration" and r["strategy"] == "takeover_pre_y3" and int(r["year"]) > 3:
                post_stop_rows += 1
        require(not post_stop_rows and all(np.allclose(fiscal[k], [float(r["committed"]), float(r["spending"])],
                rtol=0, atol=1e-8) for k, r in fixed.items()), "Fixed fiscal or post-stop ledger mismatch")
        # Each annual P0 ledger is written first. Compare realized resources, not post-decision outcomes.
        fields = ("opening", "post_rent_cash", "liquidity", "labor_capacity", "off_farm_wage", "weather")
        base: dict[tuple[str, ...], tuple[str, ...]] = {}
        counts: Counter[Key] = Counter()
        for r in read(FORMAL / "operators.csv.gz"):
            if r["group"] != "fixed_contract":
                continue
            k = key(r)
            draw_key = (r["condition"], r["seed"], r["year"], r["id"])
            values = tuple(r[f] for f in fields)
            if r["strategy"] == "none":
                require(draw_key not in base, "Duplicate fixed P0 operator")
                base[draw_key] = values
            else:
                require(base.get(draw_key) == values, "Fixed contract resource or draw mismatch")
            counts[k] += 1
        require(fixed.keys() == counts.keys() and set(counts.values()) == {32}, "Fixed operator coverage mismatch")
        return dict(fixed_arm_years=len(fixed), fixed_operator_years_compared=sum(counts.values())-len(base),
                    fixed_fiscal_arm_years=len(fixed), duration_p3_post_stop_fiscal_rows=post_stop_rows)

    checks = check_fixed()
    trades: defaultdict[Key, dict[str, tuple[float, float, bool]]] = defaultdict(dict)
    for r in read(FORMAL / "transactions.csv.gz"):
        if r["group"] == "factorial" and r["buyer"]:
            trades[key(r)][r["plot"]] = (float(r["area"]), float(r["rent"]), r["matched"] == "True")

    def rent(g: str, c: str, left: str, right: str, new: bool) -> dict[str, np.ndarray]:
        result = defaultdict(lambda: np.full((len(seeds[g]), 8), np.nan))
        for i, seed in enumerate(seeds[g]):
            for year in YEARS:
                a, b = ({p: (mu, yuan) for p, (mu, yuan, matched) in trades[g, c, s, seed, str(year)].items()
                         if not new or matched} for s in (left, right))
                am, bm = (sum(mu for mu, _ in arm.values()) for arm in (a, b))
                ar, br = (sum(yuan for _, yuan in arm.values()) for arm in (a, b))
                field = "matched_area" if new else "leased_area"
                require(am == float(annual[g, c, left, seed, str(year)][field])
                        and bm == float(annual[g, c, right, seed, str(year)][field]), "Trade area mismatch")
                price = ar/am-br/bm if am and bm else np.nan
                shared = a.keys() & b.keys()
                shared_mu = sum(a[p][0] for p in shared)
                for f, v in dict(rent_mu=price, rent_total=ar-br, both_markets_observed=bool(am and bm),
                                 rent_price_component=price*(am+bm)/2 if am and bm else 0.,
                                 rent_area_component=(ar/am+br/bm)*(am-bm)/2 if am and bm else ar-br,
                                 shared_rent_mu=sum(a[p][1]-b[p][1] for p in shared)/shared_mu if shared else np.nan,
                                 shared_rent_area=shared_mu).items():
                    result[("new_" if new else "active_") + f][i, year-1] = v
        return dict(result)

    core: list[dict[str, Cell]] = []
    timing = (("grain_pre", "grain_post"), ("takeover_pre", "takeover_post"))
    for g in ("factorial", "fixed_contract"):
        pairs = (*((p, "none") for p in POLICIES[1:]), *timing) if g == "factorial" else timing
        for c, (a, b) in product(conditions[g], pairs):
            delta = {f: panel(g, c, a, f)-panel(g, c, b, f) for f in METRICS}
            if g == "factorial" and b == "none":
                delta.update(rent(g, c, a, b, False) | rent(g, c, a, b, True))
                require(np.allclose(delta["rent_total"], delta["active_rent_total"], atol=1e-8), "Rent cash mismatch")
            if g == "fixed_contract":
                require(all(not delta[f].any() for f in ("budget", "committed", "matched_area", "rent_total")),
                        "Fixed timing comparison not contract- and commitment-equivalent")
            for f, values in delta.items():
                n = np.isfinite(values).sum(axis=1)
                means = np.divide(np.nansum(values, axis=1), n, out=np.full(len(n), np.nan), where=n>0)
                core.append(dict(group=g, condition=c, strategy=a, reference=b, metric=f,
                                 paired_seed_years=int(n.sum()), **stats(means)))

    stop: list[dict[str, Cell]] = []
    for f in ("grain_kg", "grain_area", "continuity_area", "retained_grain", "spending", "committed",
              "held_area", "matched_area", "leased_idle", "owner_closing_cash", "tenant_closing_cash"):
        d = {c: (panel("duration", c, "takeover_pre_y3", f)-panel("duration", c, "none", f))[:, 3:6]
             for c in conditions["duration"]}
        for label, x in (d | {"3yr_minus_1yr": d["contract_3yr"]-d["middle"]}).items():
            for window, v in zip(("post1", "post2", "post3", "post1to3_mean"), (*x.T, x.mean(axis=1))):
                stop.append(dict(term=label, window=window, metric=f, **stats(v)))

    cohort = {(r["condition"], r["seed"], r["event_year"], r["arm"]): r
              for r in read(FORMAL / "cessation_cohorts.csv") if r["stop_after"] == "3" and r["event_year"] in "1 2 3".split()}
    paid = [dict(term=c, event_year=y, arm=a, reference=b, metric=f,
                 **stats(np.array([float(cohort[c, i, y, a][f])-float(cohort[c, i, y, b][f]) if b
                                   else float(cohort[c, i, y, a][f]) for i in seeds["duration"]])))
            for c, y, (a, b), f in product(conditions["duration"], ("1", "2", "3"),
                (("stopped", ""), ("stopped", "never"), ("stopped", "continuous")),
                ("cohort_plots", "cohort_mu", "grain_kg", "original_recipient_mu", "idle_area", "support"))]

    weights = np.power(GAMMA, np.arange(8))
    reducers = {"discounted_sum": weights, "annual_mean": np.full(8, 1/8), "undiscounted_sum": np.ones(8)}
    selection = {p: np.stack([panel("factorial", c, p, "reward_base") for c in conditions["factorial"]])
                 for p in POLICIES}
    selected = max(POLICIES, key=lambda p: (selection[p] @ weights).mean())
    ranking = [dict(strategy=p, policy=f"P{POLICIES.index(p)}", selected=p==selected,
                    conditions=len(conditions["factorial"]), seeds=len(seeds["factorial"]),
                    **{f: float((selection[p] @ w).mean()) for f, w in reducers.items()}) for p in POLICIES]
    evaluation: list[dict[str, Cell]] = []
    for c in conditions["evaluation"]:
        for score, w in reducers.items():
            outcome = {s: panel("evaluation", c, s, "reward_base") @ w for s in (*POLICIES, "rule", *REPEATS)}
            outcome["q_base_mean"] = np.mean([outcome[s] for s in REPEATS], axis=0)
            hindsight = max(POLICIES, key=lambda p: outcome[p].mean())
            for s, values in outcome.items():
                for ref in ("", *((*POLICIES, "rule") if s.startswith("q_") else ())):
                    evaluation.append(dict(condition=c, score=score, strategy=s, reference=ref,
                                           selection_fixed=selected, hindsight_fixed=hindsight,
                                           **stats(values-outcome[ref] if ref else values)))
    components: list[dict[str, Cell]] = []
    for c, (mode, w) in product(conditions["evaluation"], reducers.items()):
        for s in (*POLICIES, "rule", "q_base_mean"):
            x = {f: np.mean([panel("evaluation", c, arm, f) for arm in (REPEATS if s=="q_base_mean" else (s,))], axis=0)
                 for f in ("reward_base", "grain_kg", "continuity_area", "spending", "budget", "total_area",
                           "owner_income", "tenant_income")}
            a, b, d = x["grain_kg"]/(450*x["total_area"]), .18*x["continuity_area"]/x["total_area"], -.14*x["spending"]/x["budget"]
            require(np.allclose(a+b+d, x["reward_base"], atol=1e-12, rtol=0), "Reward components disagree")
            components.append(dict(condition=c, score=mode, strategy=s, **{f: float((v @ w).mean()) for f, v in
                (x | dict(grain_score=a, continuity_score=b, fiscal_score=d)).items() if f not in ("budget", "total_area")}))

    coverage = {r["strategy"]: r for r in read(FORMAL / "training_checkpoints.csv")}
    fallbacks = Counter((r["condition"], r["strategy"], r["policy_fallback"]) for k, r in annual.items()
                        if k[0] == "evaluation" and k[2] in tables)
    q_audit = [dict(condition=c, strategy=s, final_stage=coverage[s]["stage"],
                    covered_actions=int(coverage[s]["covered_common_actions"]), common_actions=int(coverage[s]["common_actions"]),
                    coverage=float(coverage[s]["common_action_coverage"]), passed=tables[s]["coverage_passed"],
                    evaluation_actions=sum(n for (cond, arm, _), n in fallbacks.items() if (cond, arm)==(c, s)),
                    unseen_state=fallbacks[c, s, "unseen_state"], no_supported_action=fallbacks[c, s, "no_supported_action"])
               for c, s in product(conditions["evaluation"], tables)]
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("core_paired.csv", core), ("stop_paired.csv", stop), ("stop_paid_cohort.csv", paid),
                           ("fixed_selection.csv", ranking), ("q_evaluation.csv", evaluation),
                           ("q_components.csv", components), ("q_coverage.csv", q_audit)):
        write(filename, rows)
    (OUT / "sources_and_checks.json").write_text(json.dumps(dict(snapshot=SNAPSHOT, source="formal",
        audit_provenance=(REF / "audit_provenance.json").relative_to(FROZEN).as_posix(), sha256=hashes, ledger_audit=audit, checks=checks,
        gamma=GAMMA, training_unique_seeds=len(training), selected_fixed=selected), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, default=FROZEN)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    FROZEN, OUT = args.frozen, args.output
    FORMAL = FROZEN / "formal"
    REF = FROZEN / "worktree/results/experiments/land_succession_transmission_final_20261126"
    main()
