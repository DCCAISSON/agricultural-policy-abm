"""Three-year paired tests on the frozen national cohort; no policy learning."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import math
import shutil
from contextlib import nullcontext
from unittest.mock import patch

from agents.government_agent import get_farmer_net_income
from scripts.run_national_succession_transmission import legacy, OLD, old, parcels, LandClearingHouse
from scripts.national_plot_use_observer import snapshot_plot_use


def jobs():
    for seed in (103, 109):
        for margin in (0, 50, 75):
            for policy in ("base", "scale"):
                yield seed, margin, policy, "open"
        for policy in ("base", "scale"):
            yield seed, 50, policy, "closed"
        yield seed, 50, "late", "open"


def episode(job, output, inputs, sources, reference):
    from scripts.national_actor_trace import write_actor_trace
    seed, margin, policy, market = job
    directory = output / f"s{seed}_m{margin}_{policy}_{market}"
    base, economics, budget, observed = legacy.load_base()
    assert inputs == observed
    rule = base["runtime_fixed_rule"]
    assert rule["inertia_basis"] == "yuan_per_mu" and rule["minimum_inertia_margin_yuan_per_mu"] == 50
    rule["minimum_inertia_margin_yuan_per_mu"] = margin
    model = legacy.build_model(base, economics, seed)
    gov = model.government
    model.config["policy"]["government_q_project"].update(
        episode_end_step=3, uncapped_income_reward=True, grain_reference_kg=reference,
        budget_area_menu=math.fsum(p.area for p in model.land_patches),
        h0_income_reference_yuan=math.fsum(get_farmer_net_income(f) for f in
                                          model.initial_household_cohort) / 600)
    legacy.configure_government(gov, {}, seed, False)
    actions = {"base": (0, 0, 0), "scale": (2, 2, 2), "late": (0, 0, 2)}[policy]
    gov._choose_action_rl = lambda state, action_count=3: actions[model.step_count - 1]
    skipped = []
    def skip_new_leases(house, model_config=None):
        assert house.model is model
        skipped.append(model.step_count)
        house.transactions = []
        house.last_supply_count, house.last_demand_count = len(house.supply_offers), len(house.demand_offers)
        house._capture_flow_audit()
        house.supply_offers, house.demand_offers = [], []
    opening_land = legacy.fingerprint(parcels(model))

    def observe(current, year):
        if market == "closed":
            assert skipped == list(range(1, current.step_count + 1))
            assert not current.land_clearing_house.transactions
        old.write(year / "actual_plot_use.json", snapshot_plot_use(current))
        write_actor_trace(current, year)

    with (patch.object(LandClearingHouse, "clear_market", skip_new_leases)
          if market == "closed" else nullcontext()):
        result = legacy.run_episode(model, budget, directory, seed, sources, False,
                                    full_audit=True, observe_year=observe)
    assert tuple(r["action"] for r in gov.government_q_sar) == actions
    assert result["frozen_q_unchanged"] and not result["training"]
    print(directory.name, "complete", flush=True)
    return dict(result, path=directory.name, initial_land_sha256=opening_land)


def run(output, previous=None):
    from scripts import national_actor_trace
    _, _, _, inputs = legacy.load_base()
    sources = legacy.source_hashes()
    files = [Path(__file__), Path(national_actor_trace.__file__), Path(legacy.__file__).with_name(
        "national_plot_use_observer.py"), Path(legacy.__file__).with_name("run_national_succession_transmission.py")]
    ref = old.read(OLD / "training.json")["grain_reference_kg"]
    output.mkdir(parents=True)
    old.write(output / "design.json", dict(jobs=list(jobs()), sources=sources, frozen_inputs=inputs,
        observation_sources={str(p): old.digest(p) for p in files}, grain_reference_kg=ref,
        contrast="additional scale-qualified support vs base rights retained; no government training",
        shocks="common initial cohort and bound external streams, not synchronized endogenous draws"))
    pending, rows, reused = [], [], []
    if previous:
        before = old.read(previous / "design.json")
        assert before["sources"] == sources and before["frozen_inputs"] == inputs
        assert before["observation_sources"][str(files[0])] == old.digest(previous / "original_runner.py")
        assert all(before["observation_sources"][str(p)] == old.digest(p) for p in files[1:])
    for j in jobs():
        s, m, policy, market = j
        name = f"s{s}_m{m}_{policy}_{market}"
        src = previous / name if previous else None
        if src and market == "open" and (src / "COMPLETED.json").exists():
            summary = old.read(src / "COMPLETED.json")
            assert summary["seed"] == s and summary["years"] == 3 and summary["frozen_q_unchanged"]
            acts = [old.read(src / f"year_{t:02d}/government_sar.json")["action"] for t in (1, 2, 3)]
            assert acts == {"scale": [2]*3, "base": [0]*3, "late": [0,0,2]}[policy]
            assert old.read(src / "effective_config.json")["runtime_fixed_rule"]["minimum_inertia_margin_yuan_per_mu"] == m
            shutil.copytree(src, output / name)
            rows.append(dict(summary, path=name, initial_land_sha256=None))
            reused.append(name)
        else:
            pending.append(j)
    old.write(output / "reuse.json", {"source": str(previous) if previous else None, "completed_open_only": reused,
              "initial_parcel_hash": "not persisted by original workers; retained null, compare initial operator snapshots for all"})
    with ProcessPoolExecutor(max_workers=3) as pool:
        tasks = [pool.submit(episode, j, output, inputs, sources, ref) for j in pending]
        rows.extend(t.result() for t in tasks)
    for seed in (103, 109):
        group = [r for r in rows if r["seed"] == seed]
        for key in ("initial_sha256", "initial_exogenous_sha256", "annual_exogenous_sha256"):
            assert len({r[key] for r in group}) == 1, (seed, key)
        assert len({r["initial_land_sha256"] for r in group if r["initial_land_sha256"]}) == 1
        assert len({old.digest(output/r["path"]/"initial_operators.json") for r in group}) == 1
    assert all(old.digest(p) == old.read(output / "design.json")["observation_sources"][str(p)] for p in files)
    old.write(output / "COMPLETED.json", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reuse-open", type=Path)
    args = parser.parse_args()
    run(args.output.resolve(), args.reuse_open)
