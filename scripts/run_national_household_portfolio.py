"""Run one behavior-only job, without overwriting any experiment or archive."""
import argparse
from datetime import datetime
import math
from pathlib import Path

from agents.government_agent import get_farmer_net_income
from scripts.national_actor_trace import write_actor_trace
from scripts.run_national_inertia_pairs import OLD, legacy, old, parcels, snapshot_plot_use


ROOT = old.ROOT / "results/diagnostics/national_household_portfolio"
ARCHIVE = old.ROOT / "results/diagnostics/national_inertia_pairs_20261121/run_02"
POLICY = {"base": 0, "ordinary_post": 1, "scale80_post": 2}
FILES = ("decision_trace.json", "actors.jsonl", "cash_flow.json", "fiscal_account.json",
         "actual_harvests.json", "actual_plot_use.json", "plot_production_status.json",
         "land.json", "lease_transactions.json", "government_sar.json")


def run(seed, rule, policy="base", paired_old=None):
    if seed not in (103, 109) or rule not in ("old", "new") or policy != "base":
        raise ValueError("behavior phase: seeds 103/109 and base only; policy phase locked")
    output = ROOT / "runs" / (f"s{seed}_{rule}_{policy}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    archive = ARCHIVE / f"s{seed}_m50_base_open"
    cfg, economics, budget, inputs = legacy.load_base()
    sources = legacy.source_hashes()
    assert inputs == old.read(ARCHIVE / "design.json")["frozen_inputs"]
    portfolio = rule == "new"
    cfg["runtime_fixed_rule"]["household_crop_portfolio"] = portfolio
    share_path = ROOT / "reference/initial_grain_shares.json"
    if portfolio:
        if paired_old is None:
            raise ValueError("supply an audited old run of the same seed")
        manifest = old.read(paired_old / "audit.json")
        assert manifest["ok"] and (manifest["seed"], manifest["rule"]) == (seed, "old")
        shares = old.read(share_path)
    model = legacy.build_model(cfg, economics, seed)
    if portfolio:
        ids = {str(a.unique_id) for a in model.initial_household_cohort}
        assert shares and shares.keys() <= ids
        assert all(type(x) in (float, int) and 0 <= x <= 1 for x in shares.values())
        model.household_initial_grain_shares = shares
    model.config["policy"]["government_q_project"].update(
        episode_end_step=3, uncapped_income_reward=True,
        grain_reference_kg=old.read(OLD / "training.json")["grain_reference_kg"],
        budget_area_menu=math.fsum(p.area for p in model.land_patches),
        h0_income_reference_yuan=math.fsum(get_farmer_net_income(f) for f in model.initial_household_cohort)/600)
    expected_cfg = old.read(archive / "effective_config.json")
    expected_cfg["runtime_fixed_rule"]["household_crop_portfolio"] = portfolio
    assert model.config == expected_cfg
    assert model.config["runtime_fixed_rule"]["minimum_inertia_margin_yuan_per_mu"] == 50
    legacy.configure_government(model.government, {}, seed, False)
    # The inertia experiment fixed the choice directly, without counting unknown Q states.
    model.government._choose_action_rl = lambda state, action_count=3: POLICY[policy]
    output.mkdir(parents=True)
    path = output / "episode"
    old.write(output / "design.json", dict(seed=seed, rule=rule, policy=policy, policy_locked=True,
              source_hashes=sources, runner_sha256=old.digest(Path(__file__)), inputs=inputs,
              initial_parcels_sha256=legacy.fingerprint(parcels(model)), archive=str(archive),
              share_sha256=old.digest(share_path) if portfolio else None))

    def observe(current, year):
        old.write(year / "actual_plot_use.json", snapshot_plot_use(current))
        write_actor_trace(current, year)
        env = current.environment
        cfg = current.config
        last = env.labor_market_history[-1]
        old.write(year / "price_inputs.json", {
            "year": current.simulation_base_year + current.step_count,
            "base_crop_prices": cfg["environment"]["market"]["base_prices"],
            "base_land_rent": cfg["environment"]["market"]["base_land_rent"],
            "other_cost_per_mu": cfg["production"]["cost_structure"]["input_requirements"]["other_cost_per_mu"],
            "labor_shock_factor": last["shock_factor"],
        })
        old.write(year / "market_outcomes.json", dict(env.market_prices))

    print(path, flush=True)
    result = legacy.run_episode(model, budget, path, seed, sources, False, full_audit=True, observe_year=observe)
    assert not result["training"] and result["frozen_q_unchanged"]
    baseline = paired_old / "episode" if portfolio else archive
    checks = {}
    def compare(rel):
        lhs, rhs = old.digest(path / rel), old.digest(baseline / rel)
        checks[rel] = dict(match=lhs == rhs, actual=lhs, reference=rhs)
    for name in ("initial_households.json", "initial_operators.json", "initial_exogenous.json"):
        compare(name)
    for i in (1, 2, 3):
        year = f"year_{i:02d}/"
        prices = old.read(path / (year+"price_inputs.json"))
        scenario = expected_cfg["paper_b_flagship_scenario"]
        assert prices["year"] == 2010+i
        assert prices["base_crop_prices"] == expected_cfg["environment"]["market"]["base_prices"]
        assert prices["base_land_rent"] == scenario["initial_land_rent"]*(1+scenario["rental_growth_rate"])**(i-1)
        for crop, key in (("grain", "initial_grain_other_cost"), ("cash_crop", "initial_cash_other_cost")):
            assert prices["other_cost_per_mu"][crop] == scenario[key]*(1+scenario["input_cost_growth_rate"])**(i-1)
        if portfolio and i == 1:
            decisions = old.read(path / (year+"decision_trace.json"))
            observed = {r["operator_id"]: r["previous_grain_share"] for r in decisions
                        if r["operator_type"] == "household" and r["previous_grain_share"] is not None}
            assert observed == shares, "initial shares did not enter actual household decisions"
        assert old.read(path / (year+"government_sar.json"))["action"] == 0
        weather = old.read(path / (year+"exogenous_shocks.json"))
        assert set(weather["stream_state_sha256"]) == {"b19_environment","b19_spatial_climate"}
        assert set(weather["spatial_maps"]) == {"current_hazard_map","current_favorable_map"}
        for name in (("price_inputs.json",) if portfolio else FILES) + ("exogenous_shocks.json",):
            compare(year + name)
    ok = all(r["match"] for r in checks.values())
    if portfolio:
        ok = ok and old.read(output/"design.json")["initial_parcels_sha256"] == old.read(
            paired_old/"design.json")["initial_parcels_sha256"]
    old.write(output/"audit.json", dict(seed=seed, rule=rule, ok=ok, baseline=str(baseline), checks=checks,
              scope="Bound weather streams/maps; no separate crop-price innovation in the crop market. "
                    "Realized market prices and individual yields are not common exogenous draws. "
                    "Archived paths do not store the separate price-input observer."))
    if not ok:
        raise RuntimeError(f"pairing/reproduction incomplete: {output}/audit.json")
    print("verified", output, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("seed", type=int, choices=(103,109))
    p.add_argument("rule", choices=("old","new"))
    p.add_argument("--paired-old", type=Path)
    args = p.parse_args()
    run(args.seed, args.rule, paired_old=args.paired_old)
