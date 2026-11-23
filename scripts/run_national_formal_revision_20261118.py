"""C-source annual execution, isolated from the old E-source formal contracts."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from scripts.prepare_national_formal_revision_20261118 import ROOT, read, digest
from scripts.paper_b_flagship_scenarios import apply_annual_exogenous_path
from core.p3_project_contracts import P3Program
from core.p3_project_payments import freeze_previous_claims
from scripts.national_cash_flow_observer import observe_national_cash_flows
from scripts.paper_b_flagship_observation import operator_snapshot


def write(path, value):
    def enum_value(item):
        if isinstance(item, Enum):
            return item.value
        raise TypeError(type(item).__name__)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=enum_value)


def source_hashes():
    paths = [p for folder in ("agents", "core", "modules", "utils")
             for p in (ROOT / folder).rglob("*.py") if "__pycache__" not in p.parts]
    paths += [Path(__file__), ROOT / "scripts/prepare_national_formal_revision_20261118.py",
              ROOT / "scripts/paper_b_flagship_scenarios.py", ROOT / "scripts/b19_pairing.py",
              ROOT / "scripts/national_cash_flow_observer.py"]
    return {str(p.resolve()): digest(p) for p in sorted(paths)}


def check_sources(expected):
    if source_hashes() != expected:
        raise RuntimeError("C source changed; stop before another production year")


def merge(target, overlay):
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = deepcopy(value)


def advance_program(model, index, reference):
    program = P3Program.model_validate(model.config["fiscal_system"]["p3_support"])
    pool = reference["trial_pool_year1"] * reference["annual_real_growth_factor"] ** index
    program = program.model_copy(update={"trial_budget_yuan": pool,
        "implementation_cost_yuan": pool * program.implementation_cost_rate})
    if index:
        program = freeze_previous_claims(model.government, program)
    model.config["fiscal_system"]["p3_support"] = program.model_dump()


def initialize_economics(model, specification):
    """Require an explicit parent decision; never silently reinterpret household capital."""
    method = specification["method"]
    if method != "inherit_E_liquid_stock_reserve_and_reference_consumption":
        raise ValueError("unknown initial economic mapping")
    for farmer in model.farmers:
        row = farmer._microdata_source
        liquid = float(row["initial_working_capital_yuan"])
        reserve = float(row["household_cash_reserve_yuan"])
        if not all(math.isfinite(x) and x >= 0 for x in (liquid, reserve)):
            raise ValueError("invalid frozen household liquidity/reserve")
        if (farmer.capital != liquid or farmer.initial_capital != liquid
                or farmer.household_cash_reserve_yuan != reserve):
            raise RuntimeError("core must read frozen L/R during construction; runner never injects capital")
        # This must be implemented by the approved core finance interface, not this driver.
        if not math.isclose(farmer.balance_sheet_snapshot()["working_capital"],
                            max(0.0, liquid - reserve), abs_tol=1e-7):
            raise RuntimeError("core does not implement the approved once-only household reserve")
        target = float(row["economic_basis_20261021"]["annual_all_sale_consumption_yuan2024"])
        consumption = farmer.consumption_model
        weights = consumption.marginal_propensity.copy()
        total = math.fsum(weights.values())
        consumption.subsistence_consumption = {k: target * v / total / farmer.household_size
                                             for k, v in weights.items()}
        consumption.marginal_propensity = {k: 0.0 for k in weights}
        farmer.economic_basis_20261021_consumption_target = target


def household_row(actor):
    return {"id": str(actor.unique_id), "capital": float(actor.capital),
        "area_mu": actor.get_operated_land_area(),
        "weight": getattr(actor, "representation_weight_q", None),
        "microdata_id": getattr(actor, "_microdata_source", {}).get("cohort_agent_id"),
        "frozen_initial_liquidity": getattr(actor, "_microdata_source", {}).get("initial_working_capital_yuan"),
        "frozen_household_reserve": getattr(actor, "_microdata_source", {}).get("household_cash_reserve_yuan"),
        "working_capital": actor.balance_sheet_snapshot()["working_capital"]}


def annual_actors(model):
    actors = {str(a.unique_id): a for a in (*model.initial_household_cohort, *model.farmers,
              *model.agricultural_enterprises, *getattr(model, "revision_initial_organizations", ()))}
    for row in getattr(model, "preproduction_rent_ledger", ()):
        for key in ("tenant", "recipient"):
            actor = row[key]
            if actor is not None:
                actors[str(actor.unique_id)] = actor
    for lease in getattr(model, "initial_paid_lease_accounts", ()):
        for actor in (lease.tenant, lease.recipient):
            if actor is not None:
                actors[str(actor.unique_id)] = actor
    return actors


@contextmanager
def observe_actual_harvests(model):
    """Observe successful real settlements, restoring methods after each annual step."""
    model.revision_harvest_records = {}
    restored = []
    def observer(actor, original):
        def call(*args, **kwargs):
            identity = str(actor.unique_id)
            if identity in model.revision_harvest_records:
                raise RuntimeError("duplicate actual harvest settlement: " + identity)
            before = float(actor.capital)
            value = original(*args, **kwargs)
            model.revision_harvest_records[identity] = {
                "step": model.step_count, "capital_before_harvest": before,
                "capital_after_harvest": float(actor.capital),
                "grain_physical_kg": float(actor.grain_production),
                "cash_output_model_units": float(actor.cash_crop_production),
            }
            return value
        return call
    for actor in annual_actors(model).values():
        original = getattr(actor, "harvest_and_settle", None)
        if callable(original):
            local = "harvest_and_settle" in actor.__dict__
            restored.append((actor, original, local))
            actor.harvest_and_settle = observer(actor, original)
    try:
        yield
    finally:
        for actor, original, local in restored:
            if local:
                actor.harvest_and_settle = original
            else:
                del actor.harvest_and_settle


def rental_rows(model):
    rows = []
    for row in getattr(model, "preproduction_rent_ledger", ()):
        rows.append({**{k: v for k, v in row.items() if k not in ("patch", "tenant", "recipient")},
            "patch_id": str(row["patch"].get_patch_id()),
            "tenant_id": str(row["tenant"].unique_id),
            "recipient_id": str(row["recipient"].unique_id) if row["recipient"] is not None else None})
    return rows


def exogenous_snapshot(model):
    environment = model.environment
    climate = getattr(model, "climate_manager", None)
    streams = {}
    for label, actor in (("b19_environment", environment), ("b19_spatial_climate", climate)):
        rng = getattr(actor, "np_random", None)
        if rng is not None:
            state = json.dumps(rng.get_state(), default=lambda array: array.tolist())
            streams[label] = hashlib.sha256(state.encode()).hexdigest()
    maps = {}
    for name in ("current_hazard_map", "current_favorable_map"):
        values = getattr(climate, name, None)
        if values is not None:
            maps[name] = {"shape": list(values.shape), "dtype": str(values.dtype),
                          "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                          "sum": float(values.sum()), "max": float(values.max())}
    return {"step": model.step_count, "stream_state_sha256": streams, "spatial_maps": maps,
        "environment": {k: getattr(environment, k, None) for k in
            ("current_year_climate_shock", "climate_shock_severity",
             "current_year_favorable_weather", "favorable_weather_gain")},
        "scope": "bound_exogenous_streams_only_not_all_individual_random_draws",
        "cross_path_equality_verified": False}


def check_built_config(expected, actual):
    sections = ("production", "paper_b_flagship_scenario", "fiscal_system",
                "runtime_fixed_rule", "initial_conditions", "monetary_representation")
    differences = [key for key in sections if expected.get(key) != actual.get(key)]
    if expected["agents"]["farmer"]["consumption_payment"] != actual["agents"]["farmer"].get("consumption_payment"):
        differences.append("agents.farmer.consumption_payment")
    if differences:
        raise RuntimeError("builder changed explicit revision inputs: " + ", ".join(differences))
    return {"matched_sections": list(sections) + ["agents.farmer.consumption_payment"]}


def check_annual_accounts(model, opening, initial_land, snapshot):
    land = {str(p.get_patch_id()): float(p.area) for p in model.land_patches}
    if len(land) != len(model.land_patches) or land.keys() != initial_land.keys():
        raise RuntimeError("annual land identity changed")
    land_error = max((abs(land[k] - initial_land[k]) for k in land), default=0.0)
    settlements = model.revision_harvest_records
    if any(r["step"] != model.step_count for r in settlements.values()):
        raise RuntimeError("stale production settlement")
    grain = math.fsum(r["grain_physical_kg"] for r in settlements.values())
    grain_error = abs(grain - snapshot["grain_output_kg"])
    residuals = []
    for identity, actor in annual_actors(model).items():
        if identity not in settlements:
            continue
        consumption = getattr(actor, "last_consumption_account", {}) or {}
        paid = 0.0
        if consumption.get("step") == model.step_count:
            paid = consumption["actual_cash_payment_yuan"]
            if paid is None:
                raise RuntimeError("all revision modes require actual consumption payment")
        if hasattr(actor, "crop_portfolio"):
            net = float(actor.total_revenue - actor.total_cost)
        else:
            net = float(actor.net_income)
        harvest_residual = float(actor.capital) - settlements[identity]["capital_before_harvest"] - net + paid
        annual_residual = float(actor.capital) - opening[identity] - net + paid
        row = {"id": identity, "harvest_residual_yuan": harvest_residual,
               "annual_residual_yuan": annual_residual, "actual_consumption_paid_yuan": paid,
               "cash_balance_change_yuan": float(actor.capital) - opening[identity],
               "cash_balance_change_before_consumption_yuan": float(actor.capital) - opening[identity] + paid}
        budget = getattr(actor, "last_preproduction_cash_account", {}) or {}
        if budget.get("step") == model.step_count:
            if (budget["status"] != "settled" or budget["actual_paid_yuan"] is None
                    or budget["actual_paid_yuan"] > budget["committed_yuan"] + 1e-6
                    or budget["committed_yuan"] > budget["opening_available_yuan"] + 1e-6):
                raise RuntimeError("invalid actual preproduction funding account: " + identity)
        residuals.append(row)
    provider_residuals = []
    contracts = [c for actor in annual_actors(model).values() for c in getattr(actor, "service_contracts", ())
                 if c.p3_application_step == model.step_count]
    for provider in model.service_providers:
        account = provider.last_settlement_account or {}
        if account.get("cash_mode") == "operating_cash":
            receipts = math.fsum(c.p3_paid_yuan for c in contracts if c.service_provider_id == provider.unique_id)
            residual = account["cash_open_yuan"] + account["cash_receipts_yuan"] - account["operating_cash_paid_yuan"] - account["cash_close_yuan"]
            provider_residuals.append({"id": str(provider.unique_id), "balance_residual_yuan": residual,
                "counterparty_residual_yuan": receipts - account["cash_receipts_yuan"]})
    rents = rental_rows(model)
    current_rents = [r for r in rents if r["step"] == model.step_count]
    rent_summary = {key: math.fsum(r[key] for r in current_rents) for key in
                    ("opening_arrears_yuan", "current_due_yuan", "payment_yuan", "arrears_yuan")}
    rent_errors = [abs(r["opening_arrears_yuan"] + r["current_due_yuan"]
                       - r["payment_yuan"] - r["arrears_yuan"])
                   for r in rents if r["step"] == model.step_count]
    errors = [land_error, grain_error, *rent_errors,
        *(abs(r[k]) for r in residuals for k in ("harvest_residual_yuan", "annual_residual_yuan")),
        *(abs(r[k]) for r in provider_residuals for k in ("balance_residual_yuan", "counterparty_residual_yuan"))]
    if any(not math.isfinite(x) or x > 1e-5 for x in errors):
        raise RuntimeError("annual land/grain/cash reconciliation failed: " + str(max(errors)))
    return {"land_max_error_mu": land_error, "grain_vs_operator_snapshot_error_kg": grain_error,
            "actual_settlement_count": len(settlements), "cash_residuals": residuals,
            "provider_residuals": provider_residuals, "rent_max_error_yuan": max(rent_errors, default=0.0),
            "ordinary_rent_payables": rent_summary,
            "ordinary_rent_ledger_count": len(current_rents),
            "accounting_basis": {
                "ordinary_rent_arrears": "unpaid_payable_stock_not_free_rent_or_new_cash_credit",
                "rent_identity": "opening_arrears_yuan + current_due_yuan - payment_yuan = arrears_yuan",
                "cash_balance_changes": "observed_liquid_balance_changes_not_accrual_profit_or_net_wealth",
                "actor_income_fields": "raw_model_fields_not_reclassified_as_accrual_profit",
                "rent_scope": "ordinary_rent_only_initial_paid_leases_excluded"}}


def record_year(model, directory, opening, seed, initial_land):
    directory.mkdir()
    year = model.simulation_base_year + model.step_count
    actors = annual_actors(model)
    write(directory / "exogenous_shocks.json", exogenous_snapshot(model))
    fields = ("capital", "gross_income", "net_income", "cash_net_income", "total_expense",
              "total_revenue", "total_cost", "cash_production_cost", "hired_labor_cost",
              "land_rental_income", "land_rental_expense", "land_rental_cost", "service_fee_expense",
              "subsidy_income", "subsidy_revenue", "grain_production", "cash_crop_production",
              "current_production_area", "total_consumption_expenditure")
    with (directory / "actors.jsonl").open("x", encoding="utf-8") as stream:
        for identity, actor in actors.items():
            row = {"year": year, "id": identity, "kind": type(actor).__name__,
                   "enterprise_type": getattr(getattr(actor, "enterprise_type", None), "value", None),
                   "in_current_farmers": actor in model.farmers,
                   "in_current_organizations": actor in model.agricultural_enterprises,
                   "opening_balance": opening.get(identity),
                   **{k: getattr(actor, k, None) for k in fields},
                   "income_account": getattr(actor, "last_income_settlement", None),
                   "preproduction_cash_account": getattr(actor, "last_preproduction_cash_account", None),
                   "consumption_account": getattr(actor, "last_consumption_account", None)}
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    write(directory / "service_providers.json", [{"provider_id": str(p.unique_id),
        "year": year, "capital": p.capital, "capacity": p.current_capacity,
        "account": p.last_settlement_account} for p in model.service_providers])
    write(directory / "service_tasks.json", [{"year": year, "operator_id": identity, **asdict(c)}
        for identity, actor in actors.items() for c in getattr(actor, "service_contracts", ())
        if c.p3_application_step == model.step_count])
    write(directory / "project_payments.json", model.government.p3_project_payments)
    write(directory / "preproduction_rent_ledger.json", rental_rows(model))
    write(directory / "fiscal_account.json", model.government.current_subsidy_runtime)
    write(directory / "cash_flow.json", observe_national_cash_flows(model,
        model.config["fiscal_system"]["p3_support"]["scenario_id"], seed))
    write(directory / "decision_trace.json", [r for r in model.v4_8_fixed_rule_trace if r["year"] == year])
    write(directory / "land.json", [{"id": str(p.get_patch_id()), "area": p.area,
        "operator_id": str(p.current_operator.unique_id) if p.current_operator is not None else None,
        "abandoned": p.abandoned} for p in model.land_patches])
    snapshot = operator_snapshot(model)
    write(directory / "operator_snapshot.json", snapshot)
    write(directory / "actual_harvests.json", model.revision_harvest_records)
    write(directory / "production.json", {"year": year,
        "grain_physical_kg": math.fsum(r["grain_physical_kg"] for r in model.revision_harvest_records.values()),
        "cash_output_model_units_not_physical": math.fsum(r["cash_output_model_units"]
                                                          for r in model.revision_harvest_records.values())})
    write(directory / "annual_checks.json", check_annual_accounts(model, opening, initial_land, snapshot))


def annual_loop(model, reference, output, seed, expected_sources, years=30):
    completed = []
    opening = {}
    initial_land = {str(p.get_patch_id()): float(p.area) for p in getattr(model, "land_patches", ())}
    def capture(current):
        record_year(current, output / f"year_{current.step_count:02d}", opening, seed, initial_land)
        completed.append(current.step_count)
    model._completed_production_year_snapshot_hook = capture
    for index in range(years):
        check_sources(expected_sources)
        opening = {identity: float(a.capital) for identity, a in annual_actors(model).items()}
        apply_annual_exogenous_path(model, index)
        advance_program(model, index, reference)
        write(output / f"anchors_{index + 1:02d}.json", {
            "index": index, "costs": model.config["production"]["cost_structure"]["input_requirements"],
            "land_rent": model.environment.land_rental_price,
            "p3_program": model.config["fiscal_system"]["p3_support"]})
        with observe_actual_harvests(model):
            model.step()
        if completed != list(range(1, index + 2)):
            raise RuntimeError("annual pre-demographic recorder missing or duplicated")
        print(f"Completed production year {index + 1}/{years}", flush=True)
    check_sources(expected_sources)
    return completed


def run_job(manifest_path, index, output, release_path):
    manifest = read(manifest_path)
    release = read(release_path)
    if (release.get("parent_release") is not True
            or release.get("preparation_manifest_sha256") != digest(manifest_path)
            or not release.get("initial_economics", {}).get("approved")
            or not release.get("finance_interface_ready")):
        raise PermissionError("parent release, initial economics and finance interface required")
    required_checks = ("p3_actual_production_area_payment", "exited_ordinary_tenant_settlement",
                       "initial_liquidity_and_reserve", "service_actual_plot_coverage",
                       "builder_preserves_preproduction_cash")
    if any(release.get("verified_checks", {}).get(key) is not True for key in required_checks):
        raise PermissionError("release requires verified production-area, exited-rent and initial-liquidity fixes")
    expected = release["source_hashes"]
    check_sources(expected)
    job = manifest["jobs"][index]
    config_path = Path(job["prepared_config"])
    if digest(config_path) != job["prepared_config_sha256"]:
        raise ValueError("prepared config changed")
    cfg = read(config_path)
    if job["mode"] == "budget_finance":
        finance = Path(release["finance_overlay_path"])
        if digest(finance) != release["finance_overlay_sha256"]:
            raise ValueError("finance overlay changed")
        import yaml
        merge(cfg, yaml.safe_load(finance.read_text(encoding="utf-8")))
    consumption = cfg["agents"]["farmer"]["consumption_payment"]
    if (consumption.get("mode") != "cash_limited_no_credit"
            or consumption.get("funds_basis") != "capital_as_liquid_operating_balance"):
        raise ValueError("all three modes require the same actual consumption payment rule")
    from core.fixed_rule_runtime import preproduction_cash_enabled
    if preproduction_cash_enabled(cfg) != (job["mode"] == "budget_finance"):
        raise ValueError("preproduction funding must be enabled only in budget_finance")
    if (cfg["experiment"]["steps"] != 30 or cfg["scale"]["n_farmers"] != 600
            or cfg["scale"]["n_enterprises"] != 60 or cfg["experiment"]["seed"] != manifest["seed"]):
        raise ValueError("formal population/seed/horizon changed")
    if digest(Path(job["cohort"])) != job["cohort_sha256"]:
        raise ValueError("frozen cohort changed")
    for item in job["frozen_data"]:
        if digest(Path(item["frozen_path"])) != item["sha256"]:
            raise ValueError("frozen input changed: " + item["field"])
    if digest(Path(job["fiscal_reference"])) != job["fiscal_reference_sha256"]:
        raise ValueError("frozen fiscal reference changed")
    if output.exists():
        raise FileExistsError(output)
    from core.model_builder import ModelBuilder
    from core.reproducibility import RandomController
    from core.fixed_rule_runtime import initialize_initial_household_tenure
    from scripts.b19_pairing import bind_exogenous_streams
    from modules.production_succession_accounting import ProductionSuccessionAccount
    output.mkdir(parents=True)
    started = time.monotonic()
    write(output / "source_hashes.json", expected)
    write(output / "input_config.json", cfg)
    write(output / "release.json", release)
    try:
        RandomController.set_seed(manifest["seed"])
        model = ModelBuilder(config=deepcopy(cfg)).build()
        write(output / "effective_built_config.json", model.config)
        write(output / "built_config_checks.json", check_built_config(cfg, model.config))
        if len(model.farmers) != 600 or len(model.agricultural_enterprises) != 60:
            raise RuntimeError("built initial population differs from frozen population")
        write(output / "initial_before_economic_mapping.json", [household_row(f) for f in model.farmers])
        initialize_economics(model, release["initial_economics"])
        write(output / "initial_after_economic_mapping.json", [household_row(f) for f in model.farmers])
        initialize_initial_household_tenure(model)
        model.revision_initial_organizations = tuple(model.agricultural_enterprises)
        bind_exogenous_streams(model, manifest["seed"])
        account = ProductionSuccessionAccount()
        model.production_succession_account = account
        model.population_manager.land_bank.production_succession_account = account
        write(output / "initial_exogenous_streams.json", exogenous_snapshot(model))
        reference = read(Path(job["fiscal_reference"]))
        completed = annual_loop(model, reference, output, manifest["seed"], expected)
        origins = {}
        for name, module in tuple(sys.modules.items()):
            if name.split(".")[0] not in ("core", "agents", "modules", "utils"):
                continue
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if not path.is_relative_to(ROOT):
                    raise RuntimeError("non-C core module loaded: " + str(path))
                origins[name] = {"path": str(path), "sha256": digest(path)}
        write(output / "module_provenance.json", origins)
        write(output / "COMPLETED.json", {"years": len(completed), "seed": manifest["seed"],
            "mode": job["mode"], "original_job": job["original_job"],
            "source_basis": "C_revision_not_E_October_execution",
            "elapsed_seconds": time.monotonic() - started, "formal_matrix_accepted": False})
    except BaseException as error:
        write(output / "FAILED.json", {"error": str(error), "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("job_index", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    run_job(args.manifest.resolve(), args.job_index, args.output.resolve(), args.release.resolve())
