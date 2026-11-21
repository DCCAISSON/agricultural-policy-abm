"""Prepare C-source revision inputs; never construct or execute a model."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("restricted_inputs")
BATCHES = ("b19_revised_formal240_20261023_r1", "p3_revised_formal80_20261023_r1")
MODES = ("legacy", "budget", "budget_finance")
BUDGET = ROOT / "config/experiments/national_crop_budget_20261118_real2024_high.yaml"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_jobs(seed=17):
    selected = []
    for name in BATCHES:
        batch = BASE / name
        manifest = read(batch / "batch_manifest.json")
        if digest(batch / "batch_manifest.json") != read(batch / "PREPARED.json")["manifest_sha256"]:
            raise ValueError("October manifest no longer matches preparation seal")
        jobs = [j for j in manifest["jobs"] if j["seed"] == seed]
        if len(jobs) != (12 if manifest["matrix"] == "b19" else 4):
            raise ValueError("one original seed must cover every original scenario")
        for job in jobs:
            path = batch / job["config"]
            if digest(path) != manifest["frozen_hashes"][str(path)]:
                raise ValueError("original config hash changed")
            cfg = read(path)
            if (cfg["experiment"]["steps"] != 30 or cfg["scale"]["n_farmers"] != 600
                    or cfg["scale"]["n_enterprises"] != 60):
                raise ValueError("original formal population/horizon changed")
            selected.append((manifest, job, path, cfg))
    return selected


def revision_config(original, mode, patch):
    if mode not in MODES:
        raise ValueError(mode)
    cfg = deepcopy(original)
    cfg["agents"]["farmer"]["consumption_payment"] = {
        "mode": "cash_limited_no_credit",
        "funds_basis": "capital_as_liquid_operating_balance",
    }
    if mode != "legacy":
        if patch.get("budget_basis", {}).get("output_unit", {}).get("grain") != "physical_kg":
            raise ValueError("grain budget requires physical output, not revenue-equivalent kilograms")
        from scripts.national_crop_budget_real2024_20261118 import apply_real2024_budget
        apply_real2024_budget(cfg, patch)
        cfg["budget_basis"] = deepcopy(patch["budget_basis"])
    # No finance settings are guessed. This is a prepared input, never a release.
    cfg["national_revision_20261118"] = {
        "mode": mode, "source_basis": "C_revised_source_plus_October_frozen_config_data",
        "is_original_E_execution": False,
        "finance_overlay_pending": mode == "budget_finance",
        "initial_economic_mapping": "inherit_E_liquid_stock_reserve_and_reference_consumption",
        "initial_core_readback_pending": True,
    }
    return cfg


def bind_frozen_data(cfg, manifest):
    fields = (
        "population_dynamics.macro_anchor.data_source",
        "population_dynamics.entry_mechanism.entrant_feature_library",
        "population_dynamics.entry_mechanism.feature_library",
        "initial_conditions.initial_feature_library", "initial_conditions.feature_library",
        "initial_conditions.calibration_file", "initial_conditions.cache_file",
        "calibration.empirical_moment_config", "calibration.calibrated_params_path",
        "production.cobb_douglas.estimation_file", "production.yield_stack.auxiliary_driver_panel.path",
        "monetary_representation.cpi_source", "monetary_representation.runtime_money_basis",
    )
    snapshots = {Path(k).resolve(): v for k, v in manifest["snapshots"].items()}
    records = []
    for field in fields:
        parts = field.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node[part]
        original = node[parts[-1]]
        relative = Path(original)
        if field == "population_dynamics.macro_anchor.data_source" and len(relative.parts) == 1:
            relative = Path("config") / relative
        candidates = [relative] if relative.is_absolute() else [BASE / "source" / relative, ROOT / relative]
        source = next((p.resolve() for p in candidates if p.resolve() in snapshots), None)
        if source is None:
            raise ValueError("no October frozen data snapshot: " + field)
        frozen = Path(manifest["root"]) / snapshots[source]
        expected = manifest["frozen_hashes"][str(frozen)]
        if digest(frozen) != expected:
            raise ValueError("October frozen data hash changed: " + field)
        node[parts[-1]] = str(frozen)
        records.append({"field": field, "original_value": original, "original_source": str(source),
                        "frozen_path": str(frozen), "sha256": expected})
    return records


def prepare(output, seed=17):
    if output.exists():
        raise FileExistsError(output)
    # Resolve every input and budget before creating the output tree.
    jobs = selected_jobs(seed)
    patch = yaml.safe_load(BUDGET.read_text(encoding="utf-8"))
    planned = [(m, j, p, mode, revision_config(c, mode, patch))
               for m, j, p, c in jobs for mode in MODES]
    output.mkdir(parents=True)
    records = []
    for manifest, job, source, mode, cfg in planned:
        data_records = bind_frozen_data(cfg, manifest)
        target = output / manifest["matrix"] / job["job"] / mode
        target.mkdir(parents=True)
        (target / "original_input_config.json").write_bytes(source.read_bytes())
        (target / "prepared_input_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        records.append({"matrix": manifest["matrix"], "original_job": job["job"],
            "mode": mode, "original_config": str(source), "original_config_sha256": digest(source),
            "prepared_config": str(target / "prepared_input_config.json"),
            "prepared_config_sha256": digest(target / "prepared_input_config.json"),
            "fiscal_reference": str(Path(manifest["root"]) / "fiscal_reference.json"),
            "fiscal_reference_sha256": digest(Path(manifest["root"]) / "fiscal_reference.json"),
            "frozen_data": data_records,
            "cohort": manifest["cohort"], "cohort_sha256": manifest["cohort_sha256"]})
    source_paths = [ROOT / "agents/farmer_agent.py", ROOT / "modules/agricultural_service.py",
        ROOT / "core/p3_service_settlement.py", ROOT / "core/model.py", ROOT / "core/fixed_rule_runtime.py",
        ROOT / "scripts/national_crop_budget_real2024_20261118.py", BUDGET, Path(__file__)]
    record = {"status": "PREPARED_NOT_RELEASED", "model_executions": 0,
        "seed": seed, "original_scenarios": len(jobs), "candidate_paths": len(records),
        "years": 30, "households": 600, "organizations": 60, "jobs": records,
        "blockers": ["finance overlay and production-funding interface pending",
                     "core readback of frozen L/R and once-only operating reserve pending",
                     "C-source annual recorder and revised checks must not invoke old cost-anchor contracts",
                     "final source/data hashes and parent run release pending"],
        "source_sha256_at_preparation_not_final_freeze": {str(p): digest(p) for p in source_paths}}
    (output / "preparation_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("status", "model_executions", "original_scenarios", "candidate_paths")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.seed)
