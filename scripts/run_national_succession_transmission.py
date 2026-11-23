"""Study-only paired branches of the existing annual government policy model."""
import argparse
import csv
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import yaml

from core.p3_project_contracts import P3Program
from modules.land_clearing_house import LandClearingHouse
from scripts import run_national_government_q_20261119 as legacy
from scripts.national_plot_use_observer import snapshot_plot_use

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/diagnostics/national_succession_learning_20261121/run_01"
TARGET = "enterprise_20"
old = legacy.old


def parcels(model):
    def identity(actor):
        return None if actor is None else str(actor.unique_id)

    return {str(p.get_patch_id()): (identity(p.current_operator), identity(p.leased_to),
                                  identity(p.owner), p.abandoned)
            for p in model.land_patches}


def rows_to_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def episode(job, output, inputs, sources, reference):
    name, seed, actions, controls, anchor = job
    base, economics, budget, current = legacy.load_base()
    assert current == inputs
    model = legacy.build_model(base, economics, seed)
    gov = model.government
    from agents.government_agent import get_farmer_net_income
    q = model.config["policy"]["government_q_project"]
    q.update(episode_end_step=len(actions), uncapped_income_reward=True,
             h0_income_reference_yuan=math.fsum(get_farmer_net_income(f)
                 for f in model.initial_household_cohort) / len(model.initial_household_cohort),
             budget_area_menu=math.fsum(p.area for p in model.land_patches),
             grain_reference_kg=reference)
    legacy.configure_government(gov, {}, seed, False)
    prior_trial, offsets = defaultdict(float), {}
    signatures, income = [], []
    default_ratio = P3Program.expected_payment_ratio
    default_clear = LandClearingHouse.clear_market
    default_begin = gov._begin_government_q_project

    def ratio(program, actor_id, pool, nominal):
        if model.step_count == 2 and "expectation" in controls and pool == "area":
            return 0.0
        return default_ratio(program, actor_id, pool, nominal)

    def clear(house, *args, **kwargs):
        if model.step_count != 2 or "new_leases" not in controls:
            return default_clear(house, *args, **kwargs)
        assert house.model is model
        return None

    def begin(program):
        chosen = default_begin(program)
        if model.step_count != 2 or anchor is None:
            return chosen
        chosen = chosen.model_copy(update={"nominal_rate": anchor["rate"],
                   "scenario_rate_source": anchor["id"] + "_rate_only"})
        model.config["fiscal_system"]["p3_support"] = chosen.model_dump()
        gov._government_q_pending["nominal_rate_yuan_per_mu"] = chosen.nominal_rate
        return chosen

    def choose(_state, action_count=3):
        assert action_count == 3
        if model.step_count == 2:
            actors = old.annual_actors(model)
            signatures.append(legacy.fingerprint({
                "land": parcels(model), "cash": {k: a.capital for k, a in actors.items()},
                "prior_outcome": gov.government_q_sar[-1]["outcome_observations"],
                "random": model.random.getstate(),
                "exogenous": old.exogenous_snapshot(model),
            }))
            for identity, amount in prior_trial.items():
                if "cash" in controls and amount > 0 and identity in actors:
                    actor = actors[identity]
                    withheld = min(max(0.0, actor.capital), amount)
                    actor.capital -= withheld
                    offsets[identity] = {"year1_payment": amount, "held_aside": withheld}
        return actions[model.step_count - 1]

    def observe(current, directory):
        year = current.step_count
        if year == 1:
            for row in gov.p3_project_payments:
                if row["layer"] == "trial":
                    prior_trial[row["recipient_id"]] += row["paid_yuan"]
            if "cash" in controls:
                # Record the offset in the opening account at the first stage next year.
                assert prior_trial
        if year == 2:
            old.write(directory / "actual_plot_use.json", snapshot_plot_use(current))
            actor = next(a for a in current.agricultural_enterprises if str(a.unique_id) == TARGET)
            old.write(directory / "target.json", {
                "year1_trial_paid": prior_trial.get(TARGET, 0.0),
                "cash_offsets": offsets,
                "cash_account": getattr(actor, "last_preproduction_cash_account", None),
                "grain_area": actor.grain_area, "economic_area": actor.cash_crop_area,
                "grain_kg": current.revision_harvest_records.get(TARGET, {}).get("grain_physical_kg", 0.0),
                "decision": [r for r in current.v4_8_fixed_rule_trace
                             if r["year"] == current.simulation_base_year + year
                             and TARGET in r.values()],
                "land_after_production": parcels(current),
            })
        for actor in current.initial_household_cohort:
            settlement = getattr(actor, "last_income_settlement", None) or {}
            if settlement.get("step") != year:
                settlement = {}
            income.append({"step": year, "actor": str(actor.unique_id),
                           "cash_net": get_farmer_net_income(actor), **settlement})

    gov._choose_action_rl = choose
    gov._begin_government_q_project = begin
    path = output / name

    default_check = old.check_annual_accounts

    def check(current, opening, land, snapshot):
        # The diagnostic hold-aside is an opening balance adjustment, not an expense.
        if current.step_count == 2:
            opening = {k: v - offsets.get(k, {}).get("held_aside", 0.0)
                       for k, v in opening.items()}
        return default_check(current, opening, land, snapshot)

    with patch.object(P3Program, "expected_payment_ratio", ratio), \
         patch.object(LandClearingHouse, "clear_market", clear), \
         patch.object(old, "check_annual_accounts", check):
        result = legacy.run_episode(model, budget, path, seed, sources, False, observe_year=observe)
    assert [row["action"] for row in gov.government_q_sar] == list(actions)
    assert len(signatures) == 1
    old.write(path / "household_income.json", income)
    old.write(path / "prefix.json", signatures[0])
    print(f"{name}: complete", flush=True)
    return dict(result, name=name, controls=controls, anchor=anchor, prefix=signatures[0])


def run(output):
    _, _, _, inputs = legacy.load_base()
    sources = legacy.source_hashes()
    reference = old.read(OLD / "training.json")["grain_reference_kg"]
    catalogue = yaml.safe_load((ROOT / "config/policies/regional_grain_support.yaml").read_text(encoding="utf-8"))
    entries = {r["id"]: r for r in catalogue["entries"]}
    anchors = []
    for tag, key in (("tongdao", "hunan_tongdao_2024_rice_target_price"),
                     ("leshan", "sichuan_leshan_shizhong_2024_rice_target_price")):
        entry = entries[key]
        rate = entry["rates"][0]
        assert entry["year"] == 2024 and rate["unit"] == "CNY/mu"
        anchors.append((tag, {"id": key, "rate": rate["amount"], "status": entry["rate_status"],
                              "url": entry["source"]["url"]}))
    jobs = [(f"branch_{s}_{a}", s, (2, a, 0, 0), (), None)
            for s in (31, 47) for a in range(3)]
    jobs += [(f"original_47_{a}", 47, (a,) * 8, (), None) for a in (0, 2)]
    jobs += [(f"mechanism_{name}", 47, (2, action), controls, None)
             for name, action, controls in (
                 ("no_cash", 2, ("cash",)), ("no_expectation", 2, ("expectation",)),
                 ("neither", 2, ("cash", "expectation")),
                 ("no_new_leases_base", 0, ("new_leases",)),
                 ("no_new_leases_scale", 2, ("new_leases",)))]
    jobs += [(f"anchor_{tag}", 47, (2, 1), (), anchor) for tag, anchor in anchors]
    output.mkdir(parents=True)
    old.write(output / "design.json", {"jobs": jobs, "sources": sources, "frozen_inputs": inputs})
    with ProcessPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(episode, job, output, inputs, sources, reference) for job in jobs]
        results = [task.result() for task in pending]
    for seed in (31, 47):
        paired = [r for r in results if r["seed"] == seed and r["name"] != "original_47_0"]
        assert len({r["prefix"] for r in paired}) == 1
        for year in range(1, 5):
            exogenous = [old.read(output / r["name"] / f"year_{year:02d}/exogenous_shocks.json")
                         for r in paired if r["years"] >= year]
            assert all(row == exogenous[0] for row in exogenous)
    old.write(output / "COMPLETED.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output.resolve())
