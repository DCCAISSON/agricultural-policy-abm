"""Small original P3 execution path, not a formal national scenario rerun."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from tests.test_service_cash_closure_20261118 import setup_cash, demands
from core.p3_service_settlement import confirm_service_settlement
from core.p3_project_payments import freeze_previous_claims
from core.p3_project_contracts import configured_program

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/diagnostics/service_cash_closure_20261118/run_01"


def run_comparison():
    annual, tasks = [], []
    for label, mode, cash in (("legacy", "legacy", None),
                              ("cash_low", "operating_cash", 3640),
                              ("cash_ample", "operating_cash", 20000)):
        farmer, provider, market = setup_cash(cash, mode)
        model = farmer.model
        for step in range(3):
            model.step_count = step
            year = model.simulation_base_year + step
            model._reset_service_market_stats()
            if step:
                model.government.begin_annual_cycle()
            provider.active_contracts = [c for c in provider.active_contracts if not c.is_expired(year)]
            provider.service_area_used = sum(c.area for c in provider.active_contracts)
            matches = market.match_service_demands(demands(farmer), [provider])
            for sequence, match in enumerate(matches, 1):
                before = getattr(provider, "cash_balance", None)
                model._apply_service_contract(match)
                contract = match["contract"]
                tasks.append({"scenario": label, "farmer_id": str(farmer.unique_id),
                    "provider_id": str(provider.unique_id), "year": year,
                    "task_type": contract.service_type.value, "task_sequence": sequence,
                    "area_mu": contract.area, "fee_per_mu": contract.fee_per_mu,
                    "invoice_yuan": contract.area * contract.fee_per_mu,
                    "cash_before_provide": before,
                    "cash_after_provide": getattr(provider, "cash_balance", None),
                    "confirmed_paid_before_harvest": contract.p3_paid_yuan})
            farmer_before = farmer.capital
            farmer.harvest_and_settle()
            farmer_after = farmer.capital
            paid = sum(m["contract"].p3_paid_yuan for m in matches)
            assert abs(paid - farmer.service_fee_expense) < 1e-6
            cash_after = getattr(provider, "cash_balance", None)
            confirm_service_settlement(farmer)
            assert farmer.capital == farmer_after
            assert getattr(provider, "cash_balance", None) == cash_after
            provider.end_of_year_settlement(year)
            account = deepcopy(provider.last_settlement_account)
            if mode == "operating_cash":
                assert abs(account["cash_open_yuan"] + account["cash_receipts_yuan"]
                           - account["operating_cash_paid_yuan"] - account["cash_close_yuan"]) < 1e-6
                assert abs(paid - account["cash_receipts_yuan"]) < 1e-6
                assert account["cash_close_yuan"] >= 0
            annual.append({"scenario": label, "provider_id": str(provider.unique_id),
                "farmer_id": str(farmer.unique_id), "year": year,
                "demand_count": 2, "accepted_count": len(matches),
                "farmer_capital_before_harvest": farmer_before,
                "farmer_capital_after_harvest": farmer_after,
                "farmer_net_income": farmer.net_income,
                "farmer_service_fee_expense": farmer.service_fee_expense,
                "confirmed_service_payment": paid, "account": account})
            model.government.verify_subsidy_claims()
            model.government.distribute_subsidies()
            model.config["fiscal_system"]["p3_support"] = freeze_previous_claims(
                model.government, configured_program(model)).model_dump()
    assert [r["accepted_count"] for r in annual if r["scenario"] == "cash_low"] == [1, 0, 0]
    for left, right in zip(annual[:3], annual[6:]):
        assert left["accepted_count"] == right["accepted_count"] == 2
        assert left["confirmed_service_payment"] == right["confirmed_service_payment"]
        assert left["account"]["profit_yuan"] == right["account"]["profit_yuan"]
    return annual, tasks


def main():
    annual, tasks = run_comparison()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "verification.json").exists():
        raise FileExistsError("preserve existing run_01; rerun tests instead")
    for name, rows in (("annual_accounts.json", annual), ("task_execution.json", tasks)):
        (OUT / name).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    sources = ["modules/agricultural_service.py", "core/p3_service_settlement.py",
               "tests/service_cash_closure_20261118.py", "tests/test_service_cash_closure_20261118.py",
               "tests/test_p3_project_payments_20260909.py", "scripts/n1_income_fixtures.py",
               "core/model.py", "agents/farmer_agent.py", "config/master_config.yaml"]
    verification = {"status": "PASS", "provider_years": len(annual), "tasks": len(tasks),
        "source": "current original P3 apply/harvest/confirm methods using existing hand_farmer fixture",
        "boundary": "constructed three-year service-path test; not September/October archived orders or national full-model run",
        "hypotheses": {"initial_cash_yuan": {"cash_low": 3640, "cash_ample": 20000},
            "annual_contracts_only": True, "fixed_equipment": True,
            "maintenance": "reserve annual cost; pay on first fulfillment or funded idle year close; unfunded maintenance not performed, no debt, no operation",
            "labor_fuel": "original 45+25 yuan/mu, paid at fulfillment to external sector",
            "receipts": "only after original farmer harvest expense posting is reconciled",
            "credit_or_exit": "not implemented"},
        "sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}}
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "provider_years": len(annual), "tasks": len(tasks)}))


if __name__ == "__main__":
    main()
