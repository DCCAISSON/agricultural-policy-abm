"""Check the four archived runs; release aggregates, not household or contract records."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from scripts.run_national_household_portfolio import ROOT as B, legacy, parcels

R = Path(__file__).resolve().parents[1]
O = R/"results/diagnostics/first_paper_20261122"

def audit_run(path, *, separate_rent_rights=False):
    """Read a completed baseline in place. Emit annual aggregates, never actor records."""
    run = (path.parent if path.name == "episode" else path).resolve()
    ep = run / "episode"
    read = lambda p: json.loads(p.read_bytes())
    design, audit = (read(run / name) for name in ("design.json", "audit.json"))
    cfg = read(ep / "effective_config.json")
    seed, rule = design["seed"], design["rule"]

    def check(ok, label):
        if not ok:
            raise ValueError(f"{run.name}/{year}: {label}")

    def equal(actual, expected, label):
        check(abs(actual - expected) <= 1e-6, label)

    year = "design"
    check(audit["ok"] is True and (audit["seed"], audit["rule"]) == (seed, rule), "run audit")
    check(design["policy"] == "base" and cfg["runtime_fixed_rule"].get("national_v1", False) is (rule == "v1"),
          "baseline or V1 switch mismatch")
    initial_n = cfg["scale"]["n_enterprises"]
    steps = cfg["policy"]["government_q_project"]["episode_end_step"]
    check({d.name for d in ep.glob("year_*")} == {f"year_{j:02}" for j in range(1, steps+1)}, "year coverage")
    result = []
    for j in range(1, steps+1):
        d = ep / f"year_{j:02}"
        load = lambda name: read(d / (name + ".json"))
        cash, fiscal, checks = (load(n) for n in ("cash_flow", "fiscal_account", "annual_checks"))
        year = cash["simulation_year"]
        check((cash["runtime_seed"], cash["step"], fiscal["step"]) == (seed, j, j), "ledger period")
        if j == 1 and rule == "v1":
            equal(cash["enterprise_count_pre_turnover"], initial_n, "opening organization count")
        plots = load("actual_plot_use")
        pp = {p["patch_id"]: p for p in plots["patches"]}
        land = load("land")
        check(len(pp) == len(plots["patches"]) == len(land) and set(pp) == {r["id"] for r in land}, "plot keys")
        for r in land:
            p = pp[r["id"]]
            equal(r["area"], p["area_mu"], "plot area")
            equal(r["area"], p["cultivated_area_mu"] + p["uncultivated_area_mu"], "land use balance")
        actors = {}
        with (d / "actors.jsonl").open(encoding="utf8") as f:
            for x in map(json.loads, f):
                check(x["id"] not in actors and x["year"] == year, "actor ledger")
                actors[x["id"]] = x
        roles = defaultdict(dict)
        bids = Counter()
        with (d / "actor_trace.jsonl").open(encoding="utf8") as f:
            for x in map(json.loads, f):
                for key, ids in x["links"].items():
                    if key.startswith("actual_plot_use.json:patch_id:"):
                        role = roles[key.rsplit(":", 1)[1]]
                        for pid in ids:
                            check(pid in pp and pid not in role, "duplicate or unknown parcel role")
                            role[pid] = x["actor_id"]
                if x["enterprise_type"]:
                    for b in x["land_bid_or_gate_records"]:
                        bids["submitted"] += b.get("submitted") is True
                        for reason in (b.get("reason") or "").split("|"):
                            bids[reason] += 1
        for pid, who in roles["plot_use_operator"].items():
            check(pp[pid]["operator_id"] == who, "plot operator")
        leases = load("lease_transactions")
        rents = load("preproduction_rent_ledger")
        rents_by_parties = {(r["patch_id"], r["tenant_id"], r["recipient_id"]): r for r in rents}
        missing = 0
        missing_mu = 0.
        split_roles = 0
        split_paid = 0.
        for l in leases:
            check(l["supplier_id"] != l["demander_id"] and l["supplier_id"] in actors and l["demander_id"] in actors
                  and l["area_mu"] > 0 and l["contract_rent_yuan"] >= 0, "transaction parties or value")
            ids = [l["patch_id"]] if l["patch_id"] is not None else l.get("contract_patch_ids")
            if ids is None:
                check(l["source"] == "village_contract" and "contract_patch_ids" in l["unknown"], "undocumented missing parcel")
                missing += 1
                missing_mu += l["area_mu"]
            else:
                check(bool(ids) and len(ids) == len(set(ids)), "contract parcel keys")
                equal(sum(pp[k]["area_mu"] for k in ids), l["area_mu"], "contract area")
                due = 0.
                for k in ids:
                    tenant, supplier = l["demander_id"], l["supplier_id"]
                    key = (k, tenant, supplier)
                    if separate_rent_rights and roles["rental_rights_holder"].get(k) != supplier:
                        holder = roles["rental_rights_holder"].get(k)
                        check(holder is not None and supplier == roles["owner"].get(k) and holder != supplier
                              and tenant == roles["tenant"].get(k), "unexplained contract/recipient mismatch")
                        matches = [r for r in rents if r["patch_id"] == k and r["tenant_id"] == tenant]
                        check(len(matches) == 1 and matches[0]["recipient_id"] == holder
                              and actors[holder]["land_rental_income"] is not None, "unverified rent beneficiary")
                        key = (k, tenant, holder)
                        split_roles += 1
                        split_paid += rents_by_parties[key]["payment_yuan"]
                    due += rents_by_parties[key]["current_due_yuan"]
                equal(due, l["contract_rent_yuan"], "contract rent vs tenant and recipient annual due")
        rent_in, rent_out = Counter(), Counter()
        for r in rents:
            check(r["patch_id"] in pp and r["tenant_id"] in actors and r["recipient_id"] in actors, "rent parties")
            equal(r["total_due_yuan"], r["opening_arrears_yuan"] + r["current_due_yuan"], "rent due")
            equal(r["total_due_yuan"], r["payment_yuan"] + r["arrears_yuan"], "rent paid")
            rent_in[r["recipient_id"]] += r["payment_yuan"]
            rent_out[r["tenant_id"]] += r["payment_yuan"]
        for r in load("initial_rent_settlements"):
            rent_in[r["recipient_id"]] += r["amount_paid_yuan"]
            rent_out[r["tenant_id"]] += r["amount_paid_yuan"]
        pay = load("project_payments")
        posted = {r["claim_id"]: r for r in fiscal["payments_made"]}
        check(len(set(r["claim_id"] for r in pay)) == len(pay) == len(posted) == len(fiscal["payments_made"]), "payment IDs")
        receipts = Counter()
        for r in pay:
            t = posted[r["claim_id"]]
            check(t["agent_id"] == r["recipient_id"] and r["recipient_id"] in actors and r["step"] == j, "payee")
            equal(t["cash_amount"], r["paid_yuan"], "payment amount")
            receipts[r["recipient_id"]] += r["paid_yuan"]
            pid = r["plot_id"]
            if pid is not None:
                role = "rental_rights_holder" if r["season"] == "annual_rights" else "plot_use_operator"
                check(pid in pp and roles[role].get(pid) == r["recipient_id"], "plot entitlement")
        for who, x in actors.items():
            equal(receipts[who], x["subsidy_revenue"] if x["enterprise_type"] else x["subsidy_income"], "recipient income")
            if not x["enterprise_type"]:
                equal(rent_in[who], x["land_rental_income"], "rent recipient income")
            equal(rent_out[who], x["land_rental_cost"] if x["enterprise_type"] else x["land_rental_expense"], "tenant expense")
        paid = sum(receipts.values())
        equal(paid, cash["government_cash_paid_yuan"], "government cash")
        equal(paid + fiscal["implementation_cost_yuan"], fiscal["total_fiscal_cost_yuan"], "fiscal cost")
        check(fiscal["total_fiscal_cost_yuan"] <= fiscal["fiscal_envelope_yuan"] + 1e-6, "budget exceeded")
        rec = plots["reconciliation"]
        check(not rec["unreconciled_operators"], "unreconciled harvest")
        for v in (*rec["residual_harvest_minus_allocated"].values(), checks["land_max_error_mu"],
                  checks["rent_max_error_yuan"], checks["grain_vs_operator_snapshot_error_kg"],
                  *(x["annual_residual_yuan"] for x in checks["cash_residuals"])):
            equal(v, 0, "annual balance")
        harvest = load("actual_harvests")
        org = [x for x in actors.values() if x["enterprise_type"] and x["id"] in harvest]
        for x in org:
            equal(x["grain_production"], harvest[x["id"]]["grain_physical_kg"], "organization grain")
            equal(x["cash_crop_production"], harvest[x["id"]]["cash_output_model_units"], "organization cash crop")
        sar = load("government_sar")
        check(sar["action"] == 0 and not sar["q_updated"], "baseline policy or training")
        equal(sum(r["paid_yuan"] for r in pay if r["layer"] == "trial"), 0, "trial payments")
        result.append(dict(run=run.name, rule=rule, seed=seed, initial_org_config=initial_n, year=year,
            plots=len(pp), missing_owner=len(pp)-len(roles["owner"]), missing_rights=len(pp)-len(roles["rental_rights_holder"]),
            missing_operator=len(pp)-len(roles["plot_use_operator"]), unknown_crop_mu=rec["unknown_crop_area_mu"],
            contracts=len(leases), contract_mu=sum(l["area_mu"] for l in leases), missing_contract_ids=missing,
            missing_contract_mu=missing_mu, contracted_yuan=sum(l["contract_rent_yuan"] for l in leases),
            ordinary_rent_paid=sum(r["payment_yuan"] for r in rents), ordinary_arrears=sum(r["arrears_yuan"] for r in rents),
            org_current=sum(x["in_current_organizations"] for x in actors.values()), org_settled=len(org),
            org_bid=bids["submitted"], org_capital_rejections=bids["working_capital"],
            org_income_rejections=bids["nonpositive_incremental_income"], org_new_contracts=sum(bool(actors[l["demander_id"]]["enterprise_type"]) for l in leases),
            org_grain_kg=sum(x["grain_production"] for x in org), org_net_yuan=sum(x["total_revenue"]-x["total_cost"] for x in org),
            claims=len(pay), missing_claim_plot=sum(r["plot_id"] is None for r in pay),
            unlocated_paid_yuan=sum(r["paid_yuan"] for r in pay if r["plot_id"] is None), paid_yuan=paid,
            fiscal_cost_yuan=fiscal["total_fiscal_cost_yuan"]))
        if separate_rent_rights:
            result[-1].update(contract_plots_paid_to_separate_holder=split_roles, separate_holder_paid_yuan=split_paid)
    return result

def main():
    evidence = O/"evidence"
    evidence.mkdir(exist_ok=True)
    hist = R/"results/diagnostics/national_succession_learning_20261121/run_01"
    for src in (*[B/n for n in ("behavior_2011.csv","behavior_2012.csv","baseline_accounts.csv","policy_comparison_status.csv","PROTOCOL.md")],
                *[B/"decision_audit/one_mu"/n for n in ("summary.csv","contrast.csv","labor_trace/median_labor.csv",
                  "labor_trace/survey_2011.csv","labor_trace/budget_year_crop.csv")],
                R/"config/policies/regional_grain_support.yaml",
                R/"results/diagnostics/national_org_register_coverage_20261121/partition_totals.csv"):
        shutil.copyfile(src,evidence/src.name)
    history=O/"历史对照_非正文证据"
    history.mkdir(exist_ok=True)
    for n in ("policy_summary.csv","state_action_coverage.csv","fixed_policy_comparison.csv"):
        shutil.copyfile(hist/n,history/n)
    base=pd.read_csv(B/"baseline_accounts.csv")
    summary=[]; inputs={}; obs=[]
    for row in pd.read_csv(B/"behavior_2011.csv").itertuples():
        run=B/"runs"/row.run
        read=lambda p:json.loads(p.read_bytes())
        design=read(run/"design.json")
        assert read(run/"audit.json")["ok"]
        for name,h in design["inputs"].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
            inputs[Path(name).name]=h
        positions=None
        if (row.seed,row.rule)==(103,"portfolio"):
            cfg,econ,*_=legacy.load_base()
            cfg["runtime_fixed_rule"]["household_crop_portfolio"]=True
            m=legacy.build_model(cfg,econ,103)
            assert legacy.fingerprint(parcels(m))==design["initial_parcels_sha256"]
            positions={str(p.get_patch_id()):(p.coordinates,p.area) for p in m.land_patches}
        previous=None
        for j in (1,2,3):
            year=2010+j
            d=run/"episode"/f"year_{j:02}"
            a=read(d/"annual_checks.json"); plots=read(d/"actual_plot_use.json")
            rec=plots["reconciliation"]; pay=read(d/"project_payments.json"); sar=read(d/"government_sar.json")
            actors=[json.loads(l) for l in (d/"actors.jsonl").read_text(encoding="utf8").splitlines()]
            org={x["id"] for x in actors if x["in_current_organizations"]}
            leases=read(d/"lease_transactions.json"); sizes={x["patch_id"]:x["area_mu"] for x in plots["patches"]}
            assert len(sizes)==len(plots["patches"])
            assert all(x["supplier_id"] and x["demander_id"] and x["supplier_id"]!=x["demander_id"] and x["area_mu"]>0 for x in leases)
            assert all((x["source"]=="village_contract" and "contract_patch_ids" in x["unknown"])
                       if x["patch_id"] is None else np.isclose(x["area_mu"],sizes[x["patch_id"]]) for x in leases)
            assert len({x["claim_id"] for x in pay})==len(pay)
            br=base.query("year==@year and seed==@row.seed and rule==@row.rule").iloc[0]
            assert np.isclose(rec["harvest"]["grain_output_kg"],br.grain_kg)
            assert np.isclose(rec["harvest"]["grain_area_mu"],br.grain_area_mu)
            assert not rec["unreconciled_operators"]
            assert all(abs(x)<1e-6 for x in rec["residual_harvest_minus_allocated"].values())
            residual=max(abs(x["annual_residual_yuan"]) for x in a["cash_residuals"])
            assert max(residual,a["land_max_error_mu"],abs(a["grain_vs_operator_snapshot_error_kg"]),a["rent_max_error_yuan"])<1e-6
            assert not sar["q_updated"] and sar["action"]==0 and br.trial_paid_yuan==0
            summary.append(dict(seed=row.seed,rule=row.rule,year=year,land_residual=a["land_max_error_mu"],
               cash_max_abs=residual,grain_residual=a["grain_vs_operator_snapshot_error_kg"],rent_residual=a["rent_max_error_yuan"],
               lease_records=len(leases),no_contract_plot_id=sum(x["patch_id"] is None for x in leases),
               leased_mu=sum(x["area_mu"] for x in leases),payment_ids=len(pay),
               organizations=len(org),added_since_previous=None if previous is None else len(org-previous),
               absent_since_previous=None if previous is None else len(previous-org),unknown_crop_mu=rec["unknown_crop_area_mu"],
               action=sar["action"],reward=sar["reward"],updated=sar["q_updated"]))
            previous=org
            if positions and j==1:
                for p in plots["patches"]:
                    (x,y),area=positions[p["patch_id"]]
                    assert np.isclose(area,p["area_mu"])
                    obs.append(dict(x=x,y=y,identified_mu=p["cultivated_area_mu"]-p["unknown_crop_area_mu"],
                                    unknown_mu=p["unknown_crop_area_mu"]))
    pd.DataFrame(summary).to_csv(evidence/"accounts_verified.csv",index=False)
    pd.DataFrame(obs).groupby(["x","y"]).sum().reset_index().to_csv(evidence/"synthetic_cells.csv",index=False)
    pd.DataFrame(inputs.items(),columns=["approved_input","sha256"]).to_csv(evidence/"input_hashes.csv",index=False)
    print("Audited",len(summary),"run-years;",len(obs),"plot records")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", help="Read completed baseline run(s) or episode directories; print annual aggregates as CSV.")
    parser.add_argument("--separate-rent-rights", action="store_true", help="Verify and count rent beneficiaries distinct from the contracting owner.")
    args = parser.parse_args()
    if args.runs:
        print(pd.DataFrame(row for path in args.runs for row in audit_run(path, separate_rent_rights=args.separate_rent_rights)).to_csv(index=False, lineterminator="\n"), end="")
    else:
        main()
