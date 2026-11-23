import csv
from dataclasses import asdict
import json

import pytest

from core.fixed_rule_runtime import ActionIncomeAccount
from scripts import audit_national_inertia_pairs as audit


def decision(m, kind="household"):
    end = "exit" if kind == "household" else "contract"
    row = dict(operator_id=kind + "_1", year=2011, operator_type=kind,
               inertia_basis="yuan_per_mu", accounting_basis="action_conditioned_v1",
               previous_action="maintain_grain", inertia_yuan=m * 2,
               action="expand_cash" if m == 0 else "maintain_grain")
    for action, sales, risk, key in (("maintain_grain", 10, 0, "grain_return"),
                                      ("expand_cash", 130, 20, "cash_return"),
                                      (end, 200, 0, "exit_return")):
        fields = dict.fromkeys(audit.ACCOUNT_FIELDS, 0.0)
        fields.update(action=action, crop="none", sales_yuan=sales, risk_penalty_yuan=risk,
                      hired_labor_days=float(action == end))
        account = ActionIncomeAccount(**fields)
        row.update({f"{action}_{k}": v for k, v in asdict(account).items()})
        row[key] = account.net_income
    return row


def test_replay_feasibility_risk_threshold_and_unknown_area():
    row = decision(50)
    replay = audit.replay(row, 50)
    assert replay["operated_area_mu"] == 2 and not replay["terminal_feasible"]
    assert [replay[f"action_m{m}"] for m in audit.MARGINS] == [
        "expand_cash", "maintain_grain", "maintain_grain"]
    row["maintain_grain_management_cost_yuan"] = 1
    row["grain_return"] -= 1
    row["action"] = "expand_cash"
    assert not audit.replay(row, 50)["maintain_grain_feasible"]
    zero = audit.replay(decision(0), 0)
    assert zero["action_m0"] == "expand_cash" and zero["action_m75"] is None
    assert zero["operated_area_mu"] is None


def test_replay_includes_terminal_and_checks_recorded_action():
    row = decision(50, "scale")
    row["contract_hired_labor_days"] = 0
    with pytest.raises(ValueError, match="not reproduced"):
        audit.replay(row, 50)
    row["action"] = "contract"
    assert audit.replay(row, 50)["action_m75"] == "contract"
    row.update(maintain_grain_hired_labor_days=1, expand_cash_hired_labor_days=1, contract_hired_labor_days=1)
    assert audit.replay(row, 50)["replay_status"] == "paid_lease_fallback_not_verified"


def write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "run_01"
    root.mkdir()
    write(root / "design.json", {"observation_sources": {"observer": "recorded digest"}})
    manifest = []
    for seed in (103, 109):
        for m, p, market in audit.ARMS:
            path = root / f"s{seed}_m{m}_{p}_{market}"
            path.mkdir()
            h0, initial = [{"id": str(seed)}], {"stream": seed}
            shocks = [{"stream": seed + y} for y in audit.YEARS]
            completed = dict(seed=seed, years=3, training=False, frozen_q_unchanged=True,
                             initial_sha256=audit.fingerprint(h0), initial_exogenous_sha256=audit.fingerprint(initial),
                             annual_exogenous_sha256=audit.fingerprint(shocks))
            manifest.append(dict(completed, path=path.name, initial_land_sha256=audit.fingerprint(seed)))
            write(path / "COMPLETED.json", completed)
            write(path / "initial_households.json", h0)
            write(path / "initial_operators.json", {"grain_area_mu": 12})
            write(path / "initial_exogenous.json", initial)
            write(path / "effective_config.json", dict(experiment={"seed": seed},
                  runtime_fixed_rule=dict(inertia_basis="yuan_per_mu", minimum_inertia_margin_yuan_per_mu=m)))
            for y in audit.YEARS:
                year = path / f"year_{y:02d}"
                year.mkdir()
                action = 2 if p == "scale" or (p == "late" and y == 3) else 0
                grain = 100 + m + y + (5 * y if p == "scale" else 5 if action else 0)
                if market == "closed":
                    grain -= 2 if action else 1
                write(year / "exogenous_shocks.json", shocks[y-1])
                write(year / "production.json", dict(year=2010+y, grain_physical_kg=grain))
                write(year / "actual_plot_use.json", dict(year=2010+y, reconciliation=dict(
                    unreconciled_operators=[], harvest=dict(grain_output_kg=grain, grain_area_mu=grain/2))))
                write(year / "government_sar.json", dict(action=action, step=y, q_updated=False,
                      program_rule="scale_eligibility" if action else "area_universal",
                      outcome_observations=dict(grain_output_kg=grain, h0_mean_net_income_yuan=-grain, h0_count=1)))
                write(year / "fiscal_account.json", dict(payments_made=[
                    dict(layer="base", paid_yuan=20), dict(layer="trial", paid_yuan=action*10)]))
                write(year / "decision_trace.json", [dict(decision(m, kind), year=2010+y)
                                                     for kind in ("household", "scale")])
                if market == "closed":
                    write(year / "lease_transactions.json", [{"source": "village_contract"}])
    write(root / "COMPLETED.json", manifest)
    return root


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_matrix_contrasts_and_no_overwrite(run):
    manifest = audit.read(run / "COMPLETED.json")
    manifest[0]["initial_land_sha256"] = None
    write(run / "COMPLETED.json", manifest)
    write(run / "reuse.json", {"completed_open_only": [manifest[0]["path"]]})
    out = run / "audit_01"
    audit.audit(run, out)
    pairs = csv_rows(out / "pair_checks.csv")
    assert len(pairs) == 18 and pairs[0]["land_check"] == "not_recorded_reused_open"
    assert pairs[0]["initial_land_sha256"] == ""
    assert audit.read(out / "audit.json")["initial_land_hash_status"] == {
        "not_recorded_reused_open": 1, "matched_among_recorded_paths": 17}
    levels = csv_rows(out / "annual_levels.csv")
    assert len(levels) == 54
    assert {(r["clearing_house_count"], r["village_contract_count"])
            for r in levels if r["market"] == "closed"} == {("0", "1")}
    assert len(csv_rows(out / "fixed_account_replay.csv")) == 108
    contrasts = {r["contrast"]: r for r in csv_rows(out / "paired_differences.csv")
                 if (r["year"], r["margin"], r["seed"]) == ("3", "50", "103")}
    assert float(contrasts["continuous_scale_minus_late"]["grain_kg"]) == 10
    assert float(contrasts["late_minus_continuous_base"]["trial_paid_yuan"]) == 20
    d = contrasts["scale_minus_base_open_minus_closed"]
    assert [float(d[k]) for k in ("grain_kg", "grain_area_mu", "h0_mean_net_income_yuan")] == [1, .5, -1]
    with pytest.raises(FileExistsError):
        audit.audit(run, out)


@pytest.mark.parametrize("damage,match", [("land", "pairing"), ("shock", "annual_exogenous"),
                                          ("schedule", "action"), ("prefix", "pre-switch")])
def test_refuse_bad_pair_or_history(run, damage, match):
    path = run / "s109_m50_late_open" / "year_02"
    if damage == "land":
        file = run / "COMPLETED.json"
        data = audit.read(file)
        data[-1]["initial_land_sha256"] = "a" * 64
    elif damage == "shock":
        file, data = path / "exogenous_shocks.json", {"stream": 999}
    else:
        file = path / "government_sar.json"
        data = audit.read(file)
        if damage == "schedule":
            data["action"] = 2
        else:
            data["outcome_observations"]["h0_mean_net_income_yuan"] += 1
    write(file, data)
    out = run / "audit_01"
    with pytest.raises(ValueError, match=match):
        audit.audit(run, out)
    assert not out.exists()


def test_closed_requires_actual_zero_ordinary_transactions(run):
    ledger = run / "s103_m50_base_closed/year_02/lease_transactions.json"
    write(ledger, [{"source": "clearing_house"}, {"source": "village_contract"}])
    with pytest.raises(ValueError, match="closed market has clearing_house"):
        audit.audit(run, run / "audit_01")
    ledger.unlink()
    with pytest.raises(FileNotFoundError, match="lease_transactions"):
        audit.audit(run, run / "audit_01")
    assert not (run / "audit_01").exists()
