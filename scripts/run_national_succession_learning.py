"""Paired policy diagnostic with a coverage-aware government and a fitted threshold rule."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import math
from pathlib import Path

from scripts import run_national_government_q_20261119 as legacy

YEARS = 8
TRAIN_SEED = 23
EVAL_SEEDS = (31, 47)
FIXED = {"base": 0, "area": 1, "scale80": 2}
RULES = {"rule_area": 1, "rule_scale80": 2}
GAP = .60
SWEEPS = 20


def signal(year, grain, reference):
    return "initial" if year == 1 else "gap" if grain < GAP * reference else "other"


def threshold(state, target):
    return target if state == "gap" else 0


def fit(rows):
    samples = defaultdict(list)
    for row in rows:
        ref = row["grain_reference_kg"]
        state = signal(row["step"], row["state_observations"]["grain_output_kg"], ref)
        future = None if row["done"] else signal(
            row["step"] + 1, row["outcome_observations"]["grain_output_kg"], ref)
        samples[state, row["action"]].append((row["reward"], future))
    q = dict.fromkeys(samples, 0.0)
    for _ in range(SWEEPS):
        maxima = {s: max(v for (k, _), v in q.items() if k == s) for s, _ in q}
        q = {key: math.fsum(r + .95 * maxima.get(n, 0.0) for r, n in values) / len(values)
             for key, values in samples.items()}
    return [{"state": s, "action": a, "visits": len(samples[s, a]), "q": q[s, a]}
            for s, a in sorted(samples)]


def frozen_choice(state, covered, rule_target):
    visited = [r for r in covered if r["state"] == state and r["visits"] > 0]
    if not visited:
        return threshold(state, rule_target), "rule_fallback", 0
    best = max(visited, key=lambda r: (r["q"], -r["action"]))
    return best["action"], "covered_greedy", best["visits"]


def episode(root, name, seed, policy, expected, inputs, artifact, full):
    base, economics, budget, current = legacy.load_base()
    if current != inputs:
        raise RuntimeError("Frozen inputs changed")
    model = legacy.build_model(base, economics, seed)
    from agents.government_agent import get_farmer_net_income
    q = model.config["policy"]["government_q_project"]
    q.update(episode_end_step=YEARS, uncapped_income_reward=True,
             h0_income_reference_yuan=math.fsum(get_farmer_net_income(f)
                 for f in model.initial_household_cohort) / 600,
             budget_area_menu=math.fsum(p.area for p in model.land_patches))
    # Use the same grain reference as the frozen prior experiment, not a seed-varying target.
    q["grain_reference_kg"] = artifact["grain_reference_kg"]
    gov = model.government
    legacy.configure_government(gov, {}, seed, False)
    choices = []

    def choose(_state, action_count=3):
        assert action_count == 3
        year = model.step_count
        prior = gov.government_q_sar[-1]["outcome_observations"]["grain_output_kg"] if year > 1 else 0.0
        state = signal(year, prior, q["grain_reference_kg"])
        if policy in FIXED:
            action, source, count = FIXED[policy], "fixed", None
        elif policy in RULES or policy == "rule_base":
            action, source, count = threshold(state, RULES.get(policy, 0)), "threshold_rule", None
        else:
            action, source, count = frozen_choice(state, artifact["coverage"], artifact["rule_target"])
        choices.append(dict(year=year, state=state, prior_grain_kg=prior, action=action,
                            source=source, visits=count, feasible_actions=[0, 1, 2]))
        return action

    gov._choose_action_rl = choose
    path = root / name
    if full:
        from scripts.national_plot_use_observer import snapshot_plot_use

        def observe(current, directory):
            legacy.old.write(directory / "actual_plot_use.json", snapshot_plot_use(current))
    else:
        observe = None
    result = legacy.run_episode(model, budget, path, seed, expected, False, observe_year=observe)
    assert len(choices) == YEARS
    assert all(c["action"] == r["action"] for c, r in zip(choices, gov.government_q_sar))
    legacy.old.write(path / "choices.json", choices)
    if full:
        account = model.production_succession_account
        legacy.old.write(path / "succession_events.json", dict(
            releases=[asdict(row) for row in account.releases],
            allocations=[asdict(row) for row in account.allocations]))
    print(f"{name}: complete", flush=True)
    return dict(result, name=name, policy=policy)


def run(root, phase):
    write, read = legacy.old.write, legacy.old.read
    _, _, _, inputs = legacy.load_base()
    expected = legacy.source_hashes()
    driver = str(Path(__file__).resolve())
    reference = 1379348.1549567
    if phase == "train":
        root.mkdir(parents=True)
        write(root / "design.json", dict(sources=expected, inputs=inputs, grain_reference_kg=reference,
             driver_sha256=legacy.old.digest(Path(driver)), years=YEARS, grain_gap_fraction=GAP,
             fit_sweeps=SWEEPS, train_seed=TRAIN_SEED, evaluation_seeds=EVAL_SEEDS))
        artifact = {"grain_reference_kg": reference}
        plan = [(f"train_{p}", TRAIN_SEED, p) for p in (*FIXED, *RULES)]
    else:
        design, artifact = read(root / "design.json"), read(root / "training.json")
        if (design["sources"], design["inputs"], design["driver_sha256"]) != (
                expected, inputs, legacy.old.digest(Path(driver))):
            raise RuntimeError("Sources or frozen inputs changed between phases")
        observer = Path(__file__).with_name("national_plot_use_observer.py")
        write(root / "evaluation_source.json", {str(observer.resolve()): legacy.old.digest(observer)})
        rule_policy = artifact["rule_policy"]
        plan = [(f"eval_{seed}_{name}", seed, policy) for seed in EVAL_SEEDS
                for name, policy in [*((p, p) for p in FIXED), ("rule", rule_policy), ("covered", "covered")]]
    with ProcessPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(episode, root, name, seed, policy, expected, inputs, artifact, phase == "evaluate")
                 for name, seed, policy in plan]
        results = [task.result() for task in tasks]
    for seed in {r["seed"] for r in results}:
        paths = [r for r in results if r["seed"] == seed]
        for key in ("initial_sha256", "initial_exogenous_sha256", "annual_exogenous_sha256"):
            assert len({r[key] for r in paths}) == 1, (seed, key)
    if phase == "train":
        fixed = [r for r in results if r["policy"] in FIXED]
        rows = [read(f) for r in fixed for f in sorted((root / r["name"]).glob("year_*/government_sar.json"))]
        best = max(fixed, key=lambda r: r["discounted_reward"])["policy"]
        rule = max([results[0], *results[3:]], key=lambda r: r["discounted_reward"])["policy"]
        write(root / "training.json", dict(episodes=results, best_fixed=best,
            rule_policy="rule_base" if rule == "base" else rule, rule_target=RULES.get(rule, 0),
            actual_transitions=len(rows), coverage=fit(rows), grain_reference_kg=reference))
    else:
        write(root / "COMPLETED.json", dict(episodes=results, pairs_checked=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("phase", choices=["train", "evaluate"])
    args = parser.parse_args()
    run(args.root.resolve(), args.phase)
