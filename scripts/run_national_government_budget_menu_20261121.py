"""Eight-year diagnostic of the revised annual policy menu, using the frozen 600-household cohort."""
import argparse
from dataclasses import asdict
import math
from pathlib import Path

from scripts import run_national_government_q_20261119 as legacy

YEARS = 8
TRAIN_SEEDS = (17, 23, 17)
EVAL_SEEDS = (31, 47)
FIXED = (("learned", None), ("base_only", 0), ("grain_area", 1), ("scale80", 2))


def run(output, finish_local=False):
    base, economics, budget, inputs = legacy.load_base()
    expected = legacy.source_hashes()
    driver = Path(__file__).resolve()
    write = legacy.old.write
    table, count, refs, results = {}, 0, None, []
    plan = [(f"train_{i+1:02d}", seed, True, None) for i, seed in enumerate(TRAIN_SEEDS)]
    plan += [(f"eval_{seed}_{name}", seed, False, action)
             for seed in EVAL_SEEDS for name, action in FIXED]
    plan.append(("eval_31_grain_area_local", 31, False, 1))
    if finish_local:
        design = legacy.old.read(output / "design.json")
        prior_driver = design["sources"].pop(str(driver))
        if (design["sources"] != expected or design["inputs"] != inputs
                or (output / "COMPLETED.json").exists() or (output / plan[-1][0]).exists()):
            raise ValueError("Finish-local requires unchanged model and 11 completed pre-local paths")
        for name, seed, train, action in plan[:-1]:
            directory = output / name
            result = legacy.old.read(directory / "COMPLETED.json")
            cfg = legacy.old.read(directory / "effective_config.json")["policy"]["government_q_project"]
            current_refs = tuple(cfg[k] for k in ("grain_reference_kg", "h0_income_reference_yuan", "budget_area_menu"))
            refs = refs or current_refs
            before, after = (legacy.decode_q(legacy.old.read(directory / f"q_{k}.json")) for k in ("before", "after"))
            if (current_refs != refs or result["seed"] != seed or result["training"] != train
                    or result["years"] != YEARS or before != table
                    or result["training_step_count_before"] != count
                    or result["training_step_count_after"] != count + (YEARS if train else 0)
                    or (not train and after != table)):
                raise ValueError(f"Invalid carried-forward episode: {name}")
            rows = [legacy.old.read(directory / f"year_{year:02d}/government_sar.json")
                    for year in range(1, YEARS + 1)]
            if action is not None and any(r["action"] != action for r in rows):
                raise ValueError(f"Fixed policy differs: {name}")
            if train:
                table, count = after, result["training_step_count_after"]
            results.append(dict(result, episode=name,
                action_counts=[sum(r["action"] == a for r in rows) for a in range(3)]))
        write(output / "local_continuation.json", dict(from_driver=prior_driver,
            to_driver=legacy.old.digest(driver), reused_paths=[r["episode"] for r in results],
            reason="prepared configuration had no land_market block; local path never started"))
        plan = plan[-1:]
    else:
        output.mkdir(parents=True)
        write(output / "design.json", dict(years=YEARS, training_seeds=TRAIN_SEEDS,
            evaluation_seeds=EVAL_SEEDS, policies=FIXED, weights=legacy.WEIGHTS,
            income_rule="asinh(I/I_initial-1)",
            support_rate="annual trial pool / fixed initial total land; same rate for both active rules",
            boundary="diagnostic; not nationally calibrated or a same-horizon replication of run_02",
            geography="one paired fixed grain_area sensitivity uses 25-cell-side locality for new ordinary leases only",
            inputs=inputs, sources={**expected, str(driver): legacy.old.digest(driver)}))
    for name, seed, train, action in plan:
        if any(legacy.old.digest(Path(p)) != sha for p, sha in inputs.items()):
            raise RuntimeError("Frozen inputs changed")
        model = legacy.build_model(base, economics, seed)
        if name.endswith("_local"):
            model.config.setdefault("land_market", {})["synthetic_locality_cell_size"] = 25
        q = model.config["policy"]["government_q_project"]
        from agents.government_agent import get_farmer_net_income
        observed_refs = (q["grain_reference_kg"],
            math.fsum(get_farmer_net_income(f) for f in model.initial_household_cohort) / 600,
            math.fsum(p.area for p in model.land_patches))
        if refs is None:
            refs = observed_refs
        if not math.isclose(refs[2], observed_refs[2], abs_tol=1e-8):
            raise RuntimeError("Reference land differs")
        q.update(episode_end_step=YEARS, grain_reference_kg=refs[0],
            h0_income_reference_yuan=refs[1], budget_area_menu=refs[2], uncapped_income_reward=True)
        legacy.configure_government(model.government, table, seed, train, action, training_steps=count)
        result = legacy.run_episode(model, budget, output / name, seed, expected, train,
                                    full_audit=not train and (action is None or (seed == 31 and action == 1)))
        rows = model.government.government_q_sar
        for row in rows:
            actual = row["action"]
            assert row["program_rule"] == ("scale_eligibility" if actual == 2 else "area_universal")
            assert actual == action or action is None
            assert row["income_reward_rule"] == "asinh_centered_on_fixed_initial_income"
            assert row["nominal_rate_yuan_per_mu"] == (
                row["trial_budget_yuan"] / (refs[2] * row["trial_to_model"]) if actual else 0)
        if train:
            table = legacy.decode_q(legacy.q_rows(model.government.q_table))
            count = model.government.training_step_count
        else:
            account = model.production_succession_account
            write(output / name / "land_succession.json", dict(
                releases=[asdict(event) for event in account.releases],
                allocations=[asdict(event) for event in account.allocations],
                annual_destination=[asdict(row) for row in account.annual_land_destination(
                    tuple(model.simulation_base_year + y for y in range(1, YEARS + 1)))]))
        result.update(episode=name, action_counts=[sum(r["action"] == a for r in rows) for a in range(3)])
        results.append(result)
        print(f"{name} complete: {result['elapsed_seconds']:.1f}s", flush=True)
        del model
    for seed in EVAL_SEEDS:
        group = [r for r in results if r["seed"] == seed]
        for key in ("initial_sha256", "initial_exogenous_sha256", "annual_exogenous_sha256"):
            if len({r[key] for r in group}) != 1:
                raise RuntimeError(f"Paired evaluation differs: {seed}/{key}")
    write(output / "COMPLETED.json", dict(episodes=results, training_updates=count, references=refs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--finish-local", action="store_true")
    args = parser.parse_args()
    run(args.output.resolve(), finish_local=args.finish_local)
