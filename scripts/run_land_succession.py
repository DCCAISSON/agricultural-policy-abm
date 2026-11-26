"""Run the synthetic land-succession experiment. Use --smoke before --formal."""

import argparse
import csv
import gzip
import io
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from itertools import islice, product
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from core.land_succession import CROPS, POLICIES, Scenario, Succession
from scripts.succession_q import (
    COMMON_STATE_SHARE, DISCOUNT, LEARNING_RATE, MIN_ACTION_VISITS, MIN_COVERAGE,
    SAR_FIELDS, STATE_NAMES, THRESHOLDS, TRANSPARENT_RULE, WEIGHTS, YEARS, LearnedPolicy,
    TrainingJob, observation, reward, train, transparent,
)

EXPECTED_POLICIES = ("none", "grain_pre", "grain_post", "takeover_pre", "takeover_post")
SEEDS = tuple(range(301, 401))
SENS_SEEDS = SEEDS[:50]
EVAL_SEEDS = tuple(range(200001, 200101))
TRAIN_STARTS = tuple(range(10001, 60001, 10000))
GATE = "tests/test_land_succession_transmission.py"
INDEPENDENT_AUDIT = ('python -B -c "from pathlib import Path; from scripts.succession_analysis '
                     'import analyze; analyze(Path(\'../formal\'))"')
CONTRACT_FIELDS = ("assigned_operator", "rent_per_mu", "contract_start", "contract_end",
                   "released", "posted", "matched")
SOLID_ORDER = ("original", "reverse", "shuffle1", "shuffle2", "shuffle3")
Row = dict[str, str | int | float | bool | None]
Ledgers = dict[str, list[Row]]


def designs() -> dict[str, dict[str, Scenario]]:
    base = Scenario(search_failure=.08, cash=1., labor=1.)
    factorial = {}
    for search, cash, labor in product((False, True), repeat=3):
        name = f'{"high" if search else "low"}_search__{"tight" if cash else "ample"}_cash__{"tight" if labor else "ample"}_labor'
        factorial[name] = replace(base, search_failure=.48 if search else .08,
                                  cash=.56 if cash else 1., labor=.60 if labor else 1.)
    middle = replace(base, search_failure=.25, cash=.73, labor=.77)
    changes = {
        "middle": middle,
        "friction_off": replace(middle, search_failure=0., fee=0.),
        "search_zero": replace(middle, search_failure=0.),
        "search_blocked": replace(middle, search_failure=1.),
        "fee_off": replace(middle, fee=0.),
        "fee_high": replace(middle, fee=560.),
        "no_tenants": replace(middle, entrants=0),
        "few_tenants": replace(middle, entrants=4),
        "many_tenants": replace(middle, entrants=16),
        "cash_relief": replace(middle, cash=4.),
        "cash_tight": replace(middle, cash=.15),
        "labor_relief": replace(middle, labor=3.),
        "labor_tight": replace(middle, labor=.15),
        "rent_low": replace(middle, ask_per_mu=65.),
        "rent_high": replace(middle, ask_per_mu=195.),
        "start_free": replace(middle, start_cost=0.),
        "start_high": replace(middle, start_cost=2000.),
        "cash_reset": replace(middle, carry_cash=False),
        "exit_reset": replace(middle, carry_exit=False),
        "no_feedback": replace(middle, carry_cash=False, carry_exit=False),
        "contract_3yr": replace(middle, contract_years=3),
        "recovery": replace(middle, recovery=True),
        "reentry": replace(middle, reentry=True),
        "wage_low": replace(middle, outside_wage=.7),
        "wage_high": replace(middle, outside_wage=1.3),
        "rate_low": replace(middle, rate=60.),
        "rate_high": replace(middle, rate=120.),
        "budget_tight": replace(middle, budget=360.),
        "budget_slack": replace(middle, budget=20_000.),
    }
    return {
        "factorial": factorial, "fixed_contract": factorial, "diagnostic": changes,
        "duration": {k: changes[k] for k in ("middle", "contract_3yr")},
        "plot_order": {"middle": middle, "contract_3yr": changes["contract_3yr"]},
        "evaluation": {
            "unseen_middle": middle,
            "unseen_cash_labor": replace(middle, search_failure=.18, cash=.66, labor=.49),
            "unseen_search": replace(middle, search_failure=.66, cash=.88, labor=.92),
        },
    }


def write_gzip(stack: ExitStack, path: Path) -> io.TextIOWrapper:
    raw = stack.enter_context(path.open("wb"))
    stream = stack.enter_context(gzip.GzipFile(filename="", fileobj=raw, mode="wb",
                                              compresslevel=1, mtime=0))
    return stack.enter_context(io.TextIOWrapper(stream, encoding="utf-8", newline=""))


def _record(log: Callable, world: Succession, key: Row, out: Row,
            plots: list[Row], actors: list[Row], trades: list[Row],
            state: tuple[int, ...], fallback: str = "", available_years: int = YEARS) -> Row:
    for label, rows in (("plots", plots), ("operators", actors), ("transactions", trades)):
        log(label, (entry | key for entry in rows))
    log("fiscal", (key | {k: p[k] for k in ("plot", "operator", "crop", "eligible",
                                            "commitment", "expected_support", "support")}
                   for p in plots if p["eligible"] or p["commitment"] or p["support"]))
    row = out | key | dict(
        previous_state=str(state), unseen_state=fallback == "unseen_state",
        policy_fallback=fallback, planned_support_years=available_years,
        total_area=world.land, contract_years=world.scenario.contract_years,
        reward=reward(world, out), **{f"reward_{name}": reward(world, out, name) for name in WEIGHTS},
    )
    log("annual", (row,))
    return row


def simulate(seed: int, scenario: Scenario, strategy: str, learned: LearnedPolicy | None,
             log: Callable, group: str, condition: str, *, policy: str | None = None,
             active: int = YEARS, order: str = "original") -> list[Row]:
    world = Succession(seed, scenario)
    if order == "reverse":
        world.plots.reverse()
    elif order.startswith("shuffle"):
        np.random.default_rng([seed, 731, int(order[-1])]).shuffle(world.plots)
    follow: dict[str, tuple[Row, int]] = {}
    rows = []
    for year in range(1, YEARS + 1):
        state = observation(world)
        fallback = ""
        if learned is not None:
            action, fallback = learned.choose(state)
            actual = POLICIES[action]
        elif strategy == "rule":
            actual = POLICIES[transparent(state)]
        else:
            actual = (policy or strategy) if year <= active else "none"
        out, plots, actors, trades = world.step(actual, ledger=True)
        key = dict(group=group, condition=condition, seed=seed, year=year, strategy=strategy, policy=actual)
        rows.append(_record(log, world, key, out, plots, actors, trades, state, fallback,
                            0 if strategy == "none" else active))
        if group != "duration" or active == YEARS:
            continue
        if year <= active:
            follow.update({p["plot"]: (p, year) for p in plots if p["support"] > 0})
        elif year <= active + 3:
            by_id = {a["id"]: a for a in actors}
            for plot in plots:
                pid = plot["plot"]
                if pid not in follow:
                    continue
                prior, receipt_year = follow[pid]
                actor = by_id.get(plot["operator"], {})
                log("withdrawal_cohort", (key | dict(
                    plot=pid, last_receipt_year=receipt_year, last_paid_operator=prior["operator"],
                    last_payment=prior["support"], years_after_stop=year-active,
                    operator=plot["operator"], same_operator=plot["operator"] == prior["operator"],
                    crop=plot["crop"], physical_area=plot["physical_area"],
                    grain_kg=plot["harvest"] if plot["crop"] == "grain" else 0.,
                    profit=actor.get("profit"), closing_cash=actor.get("closing"),
                ),))
    return rows


def fixed_contract(seed: int, scenario: Scenario, condition: str, log: Callable) -> None:
    world = Succession(seed, scenario)
    for year in range(1, YEARS + 1):
        state = observation(world)
        origin = world.fork()
        out, plots, actors, trades = world.step("none", ledger=True)
        key = dict(group="fixed_contract", condition=condition, seed=seed, year=year)
        _record(log, world, key | dict(strategy="none", policy="none"),
                out, plots, actors, trades, state, available_years=0)
        contracts = {p["plot"]: tuple(p[field] for field in CONTRACT_FIELDS) for p in plots}
        opening = {a["id"]: a["opening"] for a in actors}
        commitments = {}
        for rule in POLICIES[1:]:
            copy = origin.fork()
            result, rows, accounts, transactions = copy.step(rule, ledger=True, fixed=plots)
            replay = {p["plot"]: tuple(p[field] for field in CONTRACT_FIELDS) for p in rows}
            if replay != contracts or {a["id"]: a["opening"] for a in accounts} != opening:
                raise RuntimeError(f"Fixed contract/opening balance changed: {condition}/{seed}/{year}/{rule}")
            commitments[rule] = {p["plot"]: (p["eligible"], p["commitment"], p["expected_support"]) for p in rows}
            if rule.endswith("_post") and commitments[rule] != commitments[rule.replace("_post", "_pre")]:
                raise RuntimeError(f"Fixed eligibility/commitment changed: {condition}/{seed}/{year}/{rule}")
            _record(log, copy, key | dict(strategy=rule, policy=rule), result, rows, accounts, transactions, state)


@dataclass(frozen=True, slots=True)
class Experiment:
    group: str
    condition: str
    seed: int
    scenario: Scenario


def run_experiment(job: Experiment) -> Ledgers:
    entries: Ledgers = {}

    def log(label: str, rows: Iterable[Row]) -> None:
        entries.setdefault(label, []).extend(rows)

    group, condition, seed, scenario = job.group, job.condition, job.seed, job.scenario
    if group == "fixed_contract":
        fixed_contract(seed, scenario, condition, log)
    elif group == "duration":
        simulate(seed, scenario, "none", None, log, group, condition)
        for policy, active in product(POLICIES[1:], (1, 3, YEARS)):
            simulate(seed, scenario, f"{policy}_y{active}", None, log, group, condition,
                     policy=policy, active=active)
    elif group == "plot_order":
        for order in SOLID_ORDER:
            simulate(seed, scenario, f"takeover_pre_{order}", None, log, group, condition,
                     policy="takeover_pre", order=order)
    else:
        for strategy in POLICIES + (("rule",) if group == "evaluation" else ()):
            simulate(seed, scenario, strategy, None, log, group, condition)
        if group == "evaluation":
            for learned in EVAL_POLICIES:
                simulate(seed, scenario, learned.strategy, learned, log, group, condition)
    return entries


EVAL_POLICIES: tuple[LearnedPolicy, ...] = ()


def _init_evaluation(policies: tuple[LearnedPolicy, ...]) -> None:
    global EVAL_POLICIES
    EVAL_POLICIES = policies


def run(out: Path, *, formal: bool, workers: int) -> None:
    if tuple(POLICIES) != EXPECTED_POLICIES:
        raise RuntimeError("The final core must expose the ordered P0-P4 policy menu.")
    cases = designs()
    seeds = {g: EVAL_SEEDS if g == "evaluation" else
             SEEDS if g in ("factorial", "fixed_contract") else SENS_SEEDS for g in cases}
    if not formal:
        seeds = {g: s[:1] for g, s in seeds.items()}
    starts = TRAIN_STARTS if formal else TRAIN_STARTS[:1]
    paths = 2048 if formal else 8
    train_seeds = {seed for start in starts for offset in (0, 5000)
                   for seed in range(start + offset, start + offset + 2*paths)}
    if train_seeds & set(EVAL_SEEDS + SEEDS) or set(cases["factorial"].values()) & set(cases["evaluation"].values()):
        raise RuntimeError("Learning and evaluation overlap.")

    out.mkdir(parents=True, exist_ok=False)
    if formal:
        subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        GATE, f"--junitxml={out / 'gate.xml'}"], cwd=ROOT, check=True)
    files = ("core/land_succession.py", "core/succession_allocation.py", "modules/land_clearing_house.py",
             "agents/government_agent.py", "scripts/run_land_succession.py", "scripts/succession_q.py",
             "scripts/succession_analysis.py", GATE)
    manifest = dict(
        mode="formal" if formal else "smoke", years=YEARS, workers=workers, policies=POLICIES,
        seeds=seeds, synthetic=True, train_seed_starts=starts, train_initial_paths=paths,
        q=dict(weights=WEIGHTS, state=STATE_NAMES, thresholds=THRESHOLDS, rate=LEARNING_RATE,
               discount=DISCOUNT, common_state_share=COMMON_STATE_SHARE,
               min_action_visits=MIN_ACTION_VISITS, min_coverage=MIN_COVERAGE,
               extension_paths=2*paths, aggregation="drop high_spending only",
               aggregate_seed_offset=5000, reference="physical area times nominal grain yield",
               area_denominator="fixed physical area", unseen="transparent rule"),
        withdrawal=dict(horizons=[1, 3, YEARS], followup_years=3,
                        cohort="plots paid at least once during the support window"),
        transparent_rule=TRANSPARENT_RULE,
        command=["python", "-B", "-m", "scripts.run_land_succession",
                 "../formal" if formal else "../smoke", "--formal" if formal else "--smoke",
                 "--workers", str(workers)],
        independent_audit=INDEPENDENT_AUDIT.replace("../formal", "../smoke") if not formal else INDEPENDENT_AUDIT,
        versions=dict(python=sys.version, numpy=np.__version__),
        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        worktree_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
        scenario={group: {name: asdict(s) for name, s in design.items()} for group, design in cases.items()},
        files={path: sha256((ROOT / path).read_bytes()).hexdigest() for path in files if (ROOT / path).is_file()},
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    train_jobs = [TrainingJob(name, i, start, tuple(cases["factorial"].items()), paths)
                  for name in WEIGHTS for i, start in enumerate(starts, 1)]
    models = {}
    q_records = []
    written: Counter[str] = Counter()
    with ExitStack() as stack:
        writers = {}

        def write(label: str, entries: Iterable[Row]) -> None:
            for row in entries:
                if label not in writers:
                    target = (stack.enter_context((out / f"{label}.csv").open("w", encoding="utf-8", newline=""))
                              if label in ("annual", "training_checkpoints", "q_visits")
                              else write_gzip(stack, out / f"{label}.csv.gz"))
                    writers[label] = csv.DictWriter(target, fieldnames=list(row))
                    writers[label].writeheader()
                writers[label].writerow(row)
                written[label] += 1

        sar = csv.writer(write_gzip(stack, out / "training_sar.csv.gz"))
        sar.writerow(SAR_FIELDS)
        ctx = get_context("spawn")
        with ctx.Pool(workers) as pool:
            for fitted in pool.imap(train, train_jobs):
                learned = fitted.policy
                q_records.append(learned)
                models[learned.strategy] = dict(
                    profile=learned.profile, aggregated=learned.aggregated, coverage_passed=learned.coverage_passed,
                    states={str(s): dict(q=learned.table[s], visits=learned.visits[s]) for s in sorted(learned.visits)},
                )
                sar.writerows(fitted.sar)
                written["training_sar"] += len(fitted.sar)
                write("training_checkpoints", fitted.checkpoints)
                write("q_visits", fitted.visits)
        (out / "q_table.json").write_text(json.dumps(models, indent=2), encoding="utf-8")
        with ctx.Pool(workers, initializer=_init_evaluation, initargs=(tuple(q_records),)) as pool:
            for group, design in cases.items():
                jobs = (Experiment(group, name, seed, case) for name, case in design.items() for seed in seeds[group])
                before = written["annual"]
                while batch := tuple(islice(jobs, 2*workers)):
                    for accounts in pool.imap(run_experiment, batch):
                        for label, rows in accounts.items():
                            write(label, rows)
                print(f"{group}: {written['annual'] - before} year rows", flush=True)
    (out / "execution.json").write_text(json.dumps(dict(status="completed", rows=written), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog=f"After execution, run the independent audit: {INDEPENDENT_AUDIT}")
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="One paired seed per condition, small learning check.")
    mode.add_argument("--formal", action="store_true", help="Run the gated full experiment.")
    parser.add_argument("--workers", default=6, type=int, choices=range(1, 9))
    args = parser.parse_args()
    run(args.output.resolve(), formal=args.formal, workers=args.workers)
