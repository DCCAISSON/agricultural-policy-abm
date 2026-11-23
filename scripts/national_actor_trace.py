"""Annual companion to record_year; call before demography, with full_audit=True."""

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

from core.fixed_rule_runtime import rental_rights_holder
from scripts.run_national_formal_revision_20261118 import annual_actors, write


def _alternatives(actor, decision):
    # Instance accounts survive into later years for operators who no longer decide.
    if decision is None:
        return None, None, ["no_current_decision"]
    household = decision["operator_type"] == "household"
    modes = getattr(actor, "p3_organization_accounts", ()) if household else ()
    selected = getattr(actor, "fixed_rule_selected_account", None)
    options = ([(item.account, item.mode, item.service_available) for item in modes]
               or [(account, None, None) for account in getattr(actor, "fixed_rule_action_accounts", ())])
    rows, picked = [], None
    for index, (account, mode, available) in enumerate(options):
        row = {"action": account.action, "mode": mode, "service_available": None if available is None else bool(available),
               "account_feasible": account.feasible, "constraint_reason": account.constraint_reason,
               "net_income_yuan": account.net_income, "score_yuan": account.score}
        if modes:
            row["expected_account"] = asdict(account)
        rows.append(row)
        if account is selected:
            if account.action != decision["action"]:
                raise ValueError("selected account differs from annual decision")
            picked = index
    unknown = ["self_outsource_comparison_not_saved"] if household and not modes else []
    if picked is None:
        unknown.append("selected_account_not_saved")
    return rows or None, picked, unknown


def write_actor_trace(model, directory):
    """Write additional evidence; index references are zero-based in this year's files."""
    directory = Path(directory)
    step = model.step_count
    year = model.simulation_base_year + step
    actors = annual_actors(model)
    links = defaultdict(lambda: defaultdict(list))

    def link(actor_id, role, key):
        if actor_id is not None:
            links[str(actor_id)][role].append(key)

    def saved(name):
        with (directory / name).open(encoding="utf-8") as stream:
            return json.load(stream)

    decisions = {}
    for index, row in enumerate(saved("decision_trace.json")):
        key = str(row["operator_id"])
        if row["year"] != year or key in decisions or key not in actors:
            raise ValueError(f"unmatched or duplicate decision: {year}/{key}")
        decisions[key] = row
        link(key, "decision_trace.json:operator", index)

    for patch in model.land_patches:
        key = str(patch.get_patch_id())
        for role, actor in (("owner", patch.owner), ("rental_rights_holder", rental_rights_holder(patch)),
                            ("tenant", patch.leased_to), ("current_operator", patch.current_operator),
                            ("plot_use_operator", patch.current_operator or patch.leased_to or patch.owner)):
            if actor is not None:
                link(actor.unique_id, "actual_plot_use.json:patch_id:" + role, key)

    transactions = []
    def transaction(row):
        index = len(transactions)
        transactions.append(row)
        for role in ("supplier", "demander"):
            link(row[role + "_id"], "lease_transactions.json:" + role, index)

    for row in model.land_clearing_house.transactions:
        transaction({"source": "clearing_house", "patch_id": str(row["patch"].get_patch_id()),
                     "supplier_id": str(row["supplier"].unique_id), "demander_id": str(row["demander"].unique_id),
                     "contract_rent_yuan": row["price"], "area_mu": row["patch"].area,
                     **{k: row[k] for k in ("supplier_reservation", "demander_bid", "transfer_channel")},
                     "is_acquaintance": bool(row["is_acquaintance"]), "is_contiguous": bool(row["is_contiguous"])})
    for village in model._get_village_collective_list():
        for row in village.land_transfer_records:
            if row["step"] == step:
                transaction({"source": "village_contract", "patch_id": None,
                             **({"contract_patch_ids": row["contract_patch_ids"]} if "contract_patch_ids" in row
                                else {"unknown": ["contract_patch_ids"]}),
                             "supplier_id": str(row["supplier_id"]), "demander_id": str(row["demander_id"]),
                             "contract_rent_yuan": row["rental_value"], "area_mu": row["area"],
                             **{k: row.get(k) for k in ("source_village_id", "receiver_village_id",
                                                       "transfer_channel", "contract_id")}})

    initial = []
    for lease in getattr(model, "initial_paid_lease_accounts", ()):
        if lease.settlement_step == step:
            row = {"patch_id": str(lease.patch.get_patch_id()),
                   "tenant_id": str(lease.tenant.unique_id),
                   "recipient_id": str(lease.recipient.unique_id) if lease.recipient is not None else None,
                   **{k: getattr(lease, k) for k in ("annual_rent_yuan", "end_step", "settlement_step",
                      "amount_due_yuan", "amount_paid_yuan", "opening_arrears_yuan", "arrears_yuan")}}
            for role in ("tenant", "recipient"):
                link(row[role + "_id"], "initial_rent_settlements.json:" + role, len(initial))
            initial.append(row)

    for index, row in enumerate(saved("preproduction_rent_ledger.json")):
        if row["step"] == step:
            for role in ("tenant", "recipient"):
                link(row[role + "_id"], "preproduction_rent_ledger.json:" + role, index)
    for index, row in enumerate(saved("project_payments.json")):
        if row["step"] == step:
            for role in ("producer", "recipient"):
                link(row[role + "_id"], "project_payments.json:" + role, index)
    for index, row in enumerate(saved("service_tasks.json")):
        if row["p3_application_step"] == step:
            link(row["operator_id"], "service_tasks.json:applicant", index)
            if row["p3_fulfilled"]:
                link(row["operator_id"], "service_tasks.json:fulfilled", index)
            if row["p3_payment_step"] == step:
                link(row["operator_id"], "service_tasks.json:paid", index)
    for index, row in enumerate(saved("annual_checks.json")["cash_residuals"]):
        link(row["id"], "annual_checks.json:cash_residuals", index)

    write(directory / "lease_transactions.json", transactions)
    write(directory / "initial_rent_settlements.json", initial)
    with (directory / "actor_trace.jsonl").open("x", encoding="utf-8") as stream:
        for index, (key, actor) in enumerate(actors.items()):
            decision = decisions.get(key)
            options, picked, unknown = _alternatives(actor, decision)
            harvest = model.revision_harvest_records.get(key)
            if harvest is None or harvest["step"] != step:
                unknown.append("no_current_harvest_settlement")
            else:
                link(key, "actual_harvests.json:operator_id", key)
            link(key, "actors.jsonl:account", index)
            bids = [{k: bid[k] for k in ("patch_id", "action", "bid_yuan", "record_type", "submitted",
                                         "lease_gate_passed", "reason") if k in bid}
                    for bid in getattr(actor, "fixed_rule_bid_accounts", ()) if bid["year"] == year]
            row = {"year": year, "step": step, "actor_id": key, "kind": type(actor).__name__,
                   "enterprise_type": getattr(getattr(actor, "enterprise_type", None), "value", None),
                   "alternatives": options, "selected_index": picked,
                   "selected_action": decision["action"] if decision else None,
                   "selected_mode": options[picked]["mode"] if picked is not None else None,
                   "land_bid_or_gate_records": bids, "links": links[key], "unknown": unknown}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
