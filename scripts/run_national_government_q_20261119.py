"""Four finite Q-training episodes and four paired held-out evaluations.

Run only after parent review: python -B -m scripts.run_national_government_q_20261119 OUTPUT
Income reference and reward weights are stipulated, not estimated welfare weights.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import yaml

from scripts import run_national_formal_revision_20261118 as old


BASE = old.ROOT / "results/diagnostics/national_formal_revision_20261118/run_01"
WEIGHTS = {"grain_security": .5, "farmer_welfare": .3, "fiscal_balance": .2}
PLAN = [("train_01", 17, None), ("train_02", 23, None),
        ("train_03", 17, None), ("train_04", 23, None),
        ("eval_q", 31, None), ("eval_fixed0", 31, 0),
        ("eval_fixed350", 31, 1), ("eval_fixed525", 31, 2)]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def q_rows(table):
    rows = [{"state": list(state), "values": [float(v) for v in values]}
            for state, values in sorted(table.items())]
    decode_q(rows)
    return rows


def decode_q(rows):
    table = {}
    for row in rows:
        state, values = row["state"], row["values"]
        if (not isinstance(state, list)
                or any(type(v) is not int for v in state)
                or not isinstance(values, list) or (len(state),len(values)) not in ((5,3),(6,7))
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
            raise ValueError("Q requires five integer state coordinates and three finite values, or V1 six/seven")
        key = tuple(state)
        if key in table:
            raise ValueError("duplicate Q state")
        table[key] = list(values)
    return table


def source_hashes():
    hashes = old.source_hashes()
    for path in (Path(__file__), old.ROOT / "scripts/paper_b_flagship_observation.py"):
        hashes[str(path.resolve())] = old.digest(path)
    return hashes


def check_sources(expected):
    if source_hashes() != expected:
        raise RuntimeError("government Q source changed during execution")


def check_only_government_q(cfg):
    switches = cfg.get("mechanism_switches", {})
    if any(switches.get(key, False) for key in (
            "use_learning_farmer", "use_learning_enterprise", "use_reputation_village",
            "use_extended_government_rl")):
        raise ValueError("other agent learning or extended government RL must be off")
    policy = cfg.get("policy", {})
    if (policy.get("rl_enabled", False) or policy.get("policy_mode", "rule") != "rule"
            or policy.get("government", {}).get("policy_mode", "rule") != "rule"
            or policy.get("government", {}).get("adaptive_adjustment_enabled", False)):
        raise ValueError("old government RL, PPO and adaptive policy must be off")


def load_base():
    manifest_path = BASE / "prepared_seed17_release1/preparation_manifest.json"
    release_path = BASE / "release_01.json"
    job = old.read(manifest_path)["jobs"][8]
    release = old.read(release_path)
    if (job["original_job"], job["mode"]) != ("b19_s0_m1_v1/seed_17", "budget_finance"):
        raise ValueError("base job8 identity changed")
    inputs = {job["prepared_config"]: job["prepared_config_sha256"],
              job["cohort"]: job["cohort_sha256"],
              job["fiscal_reference"]: job["fiscal_reference_sha256"],
              release["finance_overlay_path"]: release["finance_overlay_sha256"]}
    inputs.update({r["frozen_path"]: r["sha256"] for r in job["frozen_data"]})
    for path, expected in inputs.items():
        if old.digest(Path(path)) != expected:
            raise ValueError("base input changed: " + path)
    cfg = old.read(Path(job["prepared_config"]))
    old.merge(cfg, yaml.safe_load(Path(release["finance_overlay_path"]).read_text(encoding="utf-8")))
    check_only_government_q(cfg)
    if (cfg["scale"]["n_farmers"], cfg["scale"]["n_enterprises"], cfg["experiment"]["steps"]) != (600, 60, 30):
        raise ValueError("formal scale changed")
    inputs[str(manifest_path)] = old.digest(manifest_path)
    inputs[str(release_path)] = old.digest(release_path)
    return cfg, release["initial_economics"], old.read(Path(job["fiscal_reference"])), inputs


def configure_government(gov, table, seed, training, fixed_action=None, training_steps=0):
    gov.q_table = decode_q(q_rows(table))
    gov.np_random = np.random.RandomState(100000 + seed)
    gov.eval_unknown_state_count = 0
    gov.training_step_count = training_steps
    if training:
        gov.unfreeze_q_table()
        return
    gov.freeze_q_table()
    original = gov._choose_action_rl

    def choose(state, action_count=3):
        gov.eval_unknown_state_count += state not in gov.q_table
        return original(state, action_count=action_count) if fixed_action is None else fixed_action

    gov._choose_action_rl = choose


def build_model(base, economics, seed):
    from core.model_builder import ModelBuilder
    from core.reproducibility import RandomController
    from core.fixed_rule_runtime import initialize_initial_household_tenure, preproduction_cash_enabled
    from scripts.b19_pairing import bind_exogenous_streams
    from modules.production_succession_accounting import ProductionSuccessionAccount

    cfg = deepcopy(base)
    cfg["experiment"]["seed"] = seed
    from agents.government_agent import NATIONAL_V1_REWARD_WEIGHTS
    weights = NATIONAL_V1_REWARD_WEIGHTS if cfg.get("runtime_fixed_rule", {}).get("national_v1") else WEIGHTS
    cfg.setdefault("policy", {})["rl_reward_weights"] = dict(weights)
    cfg["policy"]["government_q_project"] = {
        "enabled": True, "episode_end_step": 30,
        "grain_reference_kg": 1.0, "h0_income_reference_yuan": 20000.0}
    RandomController.set_seed(seed)
    model = ModelBuilder(config=deepcopy(cfg)).build()
    old.check_built_config(cfg, model.config)
    check_only_government_q(model.config)
    if (any(model.config["policy"][key] != cfg["policy"][key]
            for key in ("government_q_project", "rl_reward_weights"))
            or not preproduction_cash_enabled(model.config)):
        raise RuntimeError("builder changed Q or production funding inputs")
    n_org = cfg["scale"]["n_enterprises"] if cfg["runtime_fixed_rule"].get("national_v1") else 60
    if (len(model.farmers), len(model.agricultural_enterprises)) != (600, n_org):
        raise RuntimeError("initial population changed")
    if model.government.rl_enabled or any(getattr(f, "learner", None) is not None for f in model.farmers):
        raise RuntimeError("non-project learning is active")
    old.initialize_economics(model, economics)
    initialize_initial_household_tenure(model)
    model.revision_initial_organizations = tuple(model.agricultural_enterprises)
    bind_exogenous_streams(model, seed)
    account = ProductionSuccessionAccount()
    model.production_succession_account = account
    model.population_manager.land_bank.production_succession_account = account
    # Annual actor grain_area is not populated before the first production decision.
    initial_grain_area = math.fsum(p.area for p in model.land_patches
        if p.current_operator is not None and not p.abandoned
        and getattr(getattr(p.current_operator, "planned_crop", None), "value", None) == "Grain")
    grain_ref = initial_grain_area * 549.913549
    if not math.isfinite(grain_ref) or grain_ref <= 0:
        raise ValueError("initial grain area must give a positive fixed reference")
    model.config["policy"]["government_q_project"]["grain_reference_kg"] = grain_ref
    return model


def reward_summary(rows, gamma):
    return dict(
        cumulative_actual_grain_kg=math.fsum(r["outcome_observations"]["grain_output_kg"] for r in rows),
        cumulative_h0_cash_income_yuan=math.fsum(r["outcome_observations"]["h0_mean_net_income_yuan"]
            * r["outcome_observations"]["h0_count"] for r in rows),
        cumulative_fiscal_cost_yuan=math.fsum(r["outcome_observations"]["fiscal_cost_yuan"] for r in rows),
        cumulative_reward=math.fsum(r["reward"] for r in rows),
        discounted_reward=math.fsum(gamma ** t * r["reward"] for t, r in enumerate(rows)),
        final_outcome=rows[-1]["outcome_observations"], final_reward=rows[-1]["reward"])


def replay_training_q(before, rows, parameters):
    """Reconstruct only recorded updates, including unvisited nonterminal next states."""
    table = decode_q(before)
    for index, row in enumerate(rows):
        state, nxt = tuple(row["state"]), tuple(row["next_state"])
        action = row["action"]
        if (row["step"] != index + 1 or row["done"] != (index == len(rows) - 1)
                or row["q_updated"] is not True or type(action) is not int or action not in (0, 1, 2)
                or (index and row["state"] != rows[index - 1]["next_state"])):
            raise ValueError("incomplete or inconsistent training SAR sequence")
        table.setdefault(state, [0.0] * 3)
        if not row["done"]:
            table.setdefault(nxt, [0.0] * 3)
        prior = table[state][action]
        future = 0.0 if row["done"] else max(table[nxt])
        expected = prior + parameters["learning_rate"] * (
            row["reward"] + parameters["discount_factor"] * future - prior)
        if (not math.isclose(prior, row["q_before"], rel_tol=0, abs_tol=1e-12)
                or not math.isfinite(row["q_after"])
                or not math.isclose(expected, row["q_after"], rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("SAR Q update does not reproduce recorded values")
        table[state][action] = row["q_after"]
    q_rows(table)
    return table


def validate_resume_sources(output, inputs):
    before, after = old.read(output / "source_hashes.json"), source_hashes()
    runner_path = str(Path(__file__).resolve())
    changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    if before.keys() != after.keys() or changed - {runner_path}:
        raise RuntimeError("resume permits only this runner to differ; economic sources changed")
    if old.read(output / "inputs.json") != inputs:
        raise RuntimeError("resume inputs differ from original run")
    for path, sha in inputs.items():
        if old.digest(Path(path)) != sha:
            raise RuntimeError("resume input content changed: " + path)
    record = {"before": before, "after": after, "allowed_changed_path": runner_path,
              "changed_paths": sorted(changed),
              "reason": "runner Q serialization and explicit completed-path recovery only"}
    path = output / "resume_source_hashes.json"
    if path.exists():
        if old.read(path) != record:
            raise RuntimeError("sources changed after resume source freeze")
    else:
        old.write(path, record)
    return after


def recover_serialization_failure(output):
    """Recover train_01's completed 30 years without building or stepping a model."""
    output = Path(output)
    _, _, _, inputs = load_base()
    expected = validate_resume_sources(output, inputs)
    path = output / "train_01"
    if old.read(path / "FAILED.json") != {
            "error": "Q requires five integer state coordinates and three finite values"}:
        raise ValueError("not the known post-year-30 serialization failure")
    if any((path / name).exists() for name in ("q_after.json", "COMPLETED.json", "episode_summary.json", "RECOVERY.json")):
        raise FileExistsError("recovery outputs already exist")
    config = old.read(path / "effective_config.json")
    if (config["experiment"]["seed"], config["scale"]["n_farmers"],
            config["scale"]["n_enterprises"], config["policy"]["government_q_project"]["episode_end_step"]) != (17, 600, 60, 30):
        raise ValueError("unexpected failed episode configuration")
    directories = [path / f"year_{step:02d}" for step in range(1, 31)]
    if sorted(path.glob("year_*")) != directories:
        raise ValueError("recovery requires all and only 30 recorded years")
    initial = old.read(path / "initial_households.json")
    h0 = {r["id"] for r in initial}
    if len(initial) != 600 or len(h0) != 600:
        raise ValueError("recovery initial cohort differs")
    rows, shocks = [], []
    for step, directory in enumerate(directories, 1):
        row = old.read(directory / "government_sar.json")
        checks = old.read(directory / "annual_checks.json")
        snapshot = old.read(directory / "operator_snapshot.json")
        fiscal = old.read(directory / "fiscal_account.json")
        residuals = [checks[k] for k in ("land_max_error_mu", "grain_vs_operator_snapshot_error_kg", "rent_max_error_yuan")]
        residuals += [r[k] for r in checks["cash_residuals"] for k in ("annual_residual_yuan", "harvest_residual_yuan")]
        residuals += [r[k] for r in checks["provider_residuals"] for k in ("balance_residual_yuan", "counterparty_residual_yuan")]
        observed = row["outcome_observations"]
        if (any(not math.isfinite(v) or abs(v) > 1e-5 for v in residuals)
                or not h0 <= {r["id"] for r in checks["cash_residuals"]}
                or observed["h0_count"] != 600 or row["reward_weights"] != WEIGHTS
                or row["grain_reference_kg"] != config["policy"]["government_q_project"]["grain_reference_kg"]
                or abs(observed["grain_output_kg"] - snapshot["grain_output_kg"]) > 1e-5
                or abs(observed["fiscal_cost_yuan"] - fiscal["total_fiscal_cost_yuan"]) > 1e-5):
            raise ValueError(f"year {step} does not pass saved account checks")
        rows.append(row)
        shocks.append(old.read(directory / "exogenous_shocks.json"))
    before = old.read(path / "q_before.json")
    if before != []:
        raise ValueError("first episode must start with an empty Q table")
    table = replay_training_q(before, rows, old.read(path / "learning_parameters.json"))
    original_hashes = {str(p.relative_to(output)): old.digest(p) for p in path.rglob("*.json")}
    result = dict(seed=17, years=30, training=True, elapsed_seconds=None,
        elapsed_including_initialization_seconds=None, episode="train_01",
        training_step_count_before=0, training_step_count_after=30, unknown_state_count=0,
        initial_sha256=fingerprint(initial),
        initial_exogenous_sha256=fingerprint(old.read(path / "initial_exogenous.json")),
        annual_exogenous_sha256=fingerprint(shocks), grain_reference_kg=rows[0]["grain_reference_kg"],
        frozen_q_unchanged=None, recovered_from_completed_sar=True)
    result.update(reward_summary(rows, old.read(path / "learning_parameters.json")["discount_factor"]))
    check_sources(expected)
    old.write(path / "q_after.json", q_rows(table))
    old.write(path / "episode_summary.json", result)
    old.write(path / "RECOVERY.json", {
        "reason": "post-year-30 Q serialization failure only; no model build or step rerun",
        "original_failed_preserved": True, "original_evidence_hashes": original_hashes,
        "q_reconstruction": "q_before plus sequential SAR; zero-init state/nonterminal next; assign validated q_after",
        "elapsed_time": "not recorded before failure; null, not estimated",
        "source_record": "../resume_source_hashes.json", "reconstructed_training_steps": 30})
    old.write(path / "COMPLETED.json", result)
    return result


def run_episode(model, reference, output, seed, expected_sources, training, full_audit=False,
                observe_year=None):
    started = time.monotonic()
    output.mkdir()
    gov = model.government
    frozen = q_rows(gov.q_table)
    opening_training_steps = gov.training_step_count
    old.write(output / "learning_parameters.json", {
        key: getattr(gov, key) for key in ("learning_rate", "discount_factor", "epsilon", "pretrain_steps")})
    old.write(output / "q_before.json", frozen)
    old.write(output / "effective_config.json", model.config)
    initial = [old.household_row(f) for f in model.initial_household_cohort]
    old.write(output / "initial_households.json", initial)
    old.write(output / "initial_operators.json", old.operator_snapshot(model))
    initial_external = old.exogenous_snapshot(model)
    old.write(output / "initial_exogenous.json", initial_external)
    land = {str(p.get_patch_id()): float(p.area) for p in model.land_patches}
    opening, captured, shocks = {}, [], []

    def capture(current):
        directory = output / f"year_{current.step_count:02d}"
        if full_audit:
            old.record_year(current, directory, opening, seed, land)
            if "budget_area_menu" in current.config["policy"]["government_q_project"]:
                old.write(directory / "plot_production_status.json", [{
                    "id": str(p.get_patch_id()), "area_mu": p.area,
                    "operator_id": str(p.current_operator.unique_id) if p.current_operator is not None else None,
                    "crop": getattr(p.current_crop, "value", None),
                    "cultivated_area_mu": getattr(p, "preproduction_cultivated_area_mu", None),
                    "release_event_id": getattr(p, "production_succession_release_event_id", None),
                } for p in current.land_patches])
        else:
            directory.mkdir()
            old.write(directory / "annual_checks.json", old.check_annual_accounts(
                current, opening, land, old.operator_snapshot(current)))
            old.write(directory / "operator_snapshot.json", old.operator_snapshot(current))
            old.write(directory / "fiscal_account.json", current.government.current_subsidy_runtime)
        shock = old.exogenous_snapshot(current)
        if not full_audit:
            old.write(directory / "exogenous_shocks.json", shock)
        shocks.append(shock)
        if observe_year is not None:
            observe_year(current, directory)
        captured.append(current.step_count)

    model._completed_production_year_snapshot_hook = capture
    years = model.config["policy"]["government_q_project"]["episode_end_step"]
    try:
        for index in range(years):
            check_sources(expected_sources)
            opening = {key: float(a.capital) for key, a in old.annual_actors(model).items()}
            old.apply_annual_exogenous_path(model, index)
            old.advance_program(model, index, reference)
            with old.observe_actual_harvests(model):
                model.step()
            if captured != list(range(1, index + 2)) or len(gov.government_q_sar) != index + 1:
                raise RuntimeError("annual audit or SAR missing/duplicated")
            sar = gov.government_q_sar[-1]
            v1 = model.config.get("runtime_fixed_rule", {}).get("national_v1") is True
            state_dim = 6 if v1 else 5
            expected_weights = model.config["policy"]["rl_reward_weights"] if v1 else WEIGHTS
            if (sar["step"] != index + 1 or sar["done"] != (index == years - 1)
                    or len(sar["state"]) != state_dim or len(sar["next_state"]) != state_dim
                    or sar["reward_weights"] != expected_weights or sar["q_updated"] != training):
                raise RuntimeError("invalid government transition")
            if not training and (q_rows(gov.q_table) != frozen or gov.training_step_count != opening_training_steps):
                raise RuntimeError("evaluation changed frozen Q")
            old.write(output / f"year_{index + 1:02d}/government_sar.json", sar)
        check_sources(expected_sources)
        old.write(output / "q_after.json", q_rows(gov.q_table))
        result = {"seed": seed, "years": years, "training": training,
                  "elapsed_seconds": time.monotonic() - started,
                  "training_step_count_before": opening_training_steps,
                  "training_step_count_after": gov.training_step_count,
                  "unknown_state_count": gov.eval_unknown_state_count,
                  "initial_sha256": fingerprint(initial),
                  "initial_exogenous_sha256": fingerprint(initial_external),
                  "annual_exogenous_sha256": fingerprint(shocks),
                  "grain_reference_kg": model.config["policy"]["government_q_project"]["grain_reference_kg"],
                  "frozen_q_unchanged": None if training else q_rows(gov.q_table) == frozen}
        result.update(reward_summary(gov.government_q_sar, gov.discount_factor))
        old.write(output / "COMPLETED.json", result)
        return result
    except BaseException as error:
        old.write(output / "FAILED.json", {"error": str(error)})
        raise
    finally:
        del model._completed_production_year_snapshot_hook


def verify_pairs(results):
    for seed in (17, 23, 31):
        group = [r for r in results if r["seed"] == seed]
        for key in ("initial_sha256", "initial_exogenous_sha256", "annual_exogenous_sha256", "grain_reference_kg"):
            if len({r[key] for r in group}) != 1:
                raise RuntimeError(f"paired seed {seed} differs: {key}")
    return {"paired_seeds": [17, 23, 31], "scope": "initial households and bound external streams, not individual draws"}


def reuse_completed(path, seed, training, table, training_steps, grain_ref):
    result = old.read(path / "COMPLETED.json")
    if (result["seed"] != seed or result["years"] != 30 or result["training"] != training
            or result["training_step_count_before"] != training_steps
            or result["training_step_count_after"] != training_steps + (30 if training else 0)
            or result["grain_reference_kg"] != grain_ref
            or decode_q(old.read(path / "q_before.json")) != table):
        raise ValueError("completed episode identity, reference or Q/count chain differs")
    if (path / "FAILED.json").exists() and not (path / "RECOVERY.json").exists():
        raise ValueError("failed episode lacks explicit recovery evidence")
    rows = [old.read(path / f"year_{step:02d}/government_sar.json") for step in range(1, 31)]
    after = decode_q(old.read(path / "q_after.json"))
    if training:
        replayed = replay_training_q(q_rows(table), rows, old.read(path / "learning_parameters.json"))
        if replayed != after:
            raise ValueError("completed Q differs from recorded SAR")
    elif after != table or any(r["q_updated"] for r in rows):
        raise ValueError("completed evaluation changed Q")
    initial = old.read(path / "initial_households.json")
    shocks = [old.read(path / f"year_{step:02d}/exogenous_shocks.json") for step in range(1, 31)]
    if (fingerprint(initial) != result["initial_sha256"]
            or fingerprint(shocks) != result["annual_exogenous_sha256"]
            or fingerprint(old.read(path / "initial_exogenous.json")) != result["initial_exogenous_sha256"]):
        raise ValueError("completed initial/exogenous evidence changed")
    for step in range(1, 31):
        old.read(path / f"year_{step:02d}/annual_checks.json")
    result["episode"] = path.name
    return result, after if training else table, result["training_step_count_after"]


def run(output, resume=False):
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(output)
    base, economics, reference, inputs = load_base()
    design = {"plan": PLAN, "weights": WEIGHTS,
        "income_reference_yuan": 20000, "reference_yield_kg_per_mu": 549.913549,
        "initial_economics": economics, "budget_reference": reference,
        "assumption": "stipulated reward references and weights; four episodes are not convergence",
        "observation_scope": "five discretized observations are partially observable, not a complete Markov state",
        "reward_reference": "first training initial grain area; same Gref for all eight episodes",
        "fixed0_meaning": "zero trial area rate; preserved 83 yuan rights-area support remains"}
    if resume:
        if (output / "COMPLETED.json").exists():
            raise FileExistsError("whole run already completed")
        expected = validate_resume_sources(output, inputs)
        if fingerprint(old.read(output / "design.json")) != fingerprint(design):
            raise ValueError("resume design changed")
    else:
        expected = source_hashes()
        output.mkdir(parents=True)
        old.write(output / "source_hashes.json", expected)
        old.write(output / "inputs.json", inputs)
        old.write(output / "design.json", design)
    table, results, training_steps, grain_ref = {}, [], 0, None
    if resume:
        grain_ref = old.read(output / "reward_reference.json")["grain_reference_kg"]
    for name, seed, action in PLAN:
        check_sources(expected)
        for path, sha in inputs.items():
            if old.digest(Path(path)) != sha:
                raise RuntimeError("frozen input changed: " + path)
        episode_path = output / name
        if resume and episode_path.exists():
            if not (episode_path / "COMPLETED.json").exists():
                raise RuntimeError("incomplete path; explicit recovery required, no automatic rerun: " + name)
            result, table, training_steps = reuse_completed(
                episode_path, seed, name.startswith("train"), table, training_steps, grain_ref)
            results.append(result)
            print(f"{name} reused: 30 recorded years; training steps {training_steps}", flush=True)
            continue
        episode_started = time.monotonic()
        model = build_model(base, economics, seed)
        qcfg = model.config["policy"]["government_q_project"]
        if grain_ref is None:
            grain_ref = qcfg["grain_reference_kg"]
            old.write(output / "reward_reference.json", {"grain_reference_kg": grain_ref,
                "source": "first_training_initial_grain_area_times_549.913549_fixed_for_all_eight",
                "income_reference_yuan": 20000, "weights": WEIGHTS})
        qcfg["grain_reference_kg"] = grain_ref
        configure_government(model.government, table, seed, name.startswith("train"), action,
                             training_steps=training_steps)
        results.append(run_episode(model, reference, output / name, seed, expected,
                                   name.startswith("train"), full_audit=name == "eval_q"))
        if name.startswith("train"):
            table = decode_q(q_rows(model.government.q_table))
            training_steps = model.government.training_step_count
        results[-1]["episode"] = name
        results[-1]["elapsed_including_initialization_seconds"] = time.monotonic() - episode_started
        old.write(output / name / "episode_summary.json", results[-1])
        print(f"{name} complete: {results[-1]['elapsed_including_initialization_seconds']:.1f}s", flush=True)
        del model
    check_sources(expected)
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] in ("core", "agents", "modules", "utils") and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(old.ROOT):
                raise RuntimeError("nonlocal model module: " + name)
    old.write(output / "paired_checks.json", verify_pairs(results))
    old.write(output / "trained_q.json", q_rows(table))
    old.write(output / "COMPLETED.json", {"episodes": results, "count": len(results)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", action="store_true", help="reuse completed episodes; never rerun incomplete paths")
    args = parser.parse_args()
    run(args.output.resolve(), resume=args.resume)
