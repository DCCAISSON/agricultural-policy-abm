import pytest

from core.fixed_rule_runtime import prepare_preproduction_cash, production_area
from core.p3_project_claims import production_records
from modules.agricultural_service import ServiceMarket, ServiceProvider
from tests.test_p3_project_payments_20260909 import prepared, program


def test_reduced_cultivation_service_coverage_harvest_and_family_service_payment():
    spec = program("family_service")
    farmer = prepared(spec)
    model = farmer.model
    service_config = model.config["production"]["agricultural_service"]
    service_config.setdefault("provider_economics", {}).update(
        cash_mode="operating_cash", initial_cash_yuan=20000,
    )
    service_config["enabled"] = True
    provider = ServiceProvider("cash_provider", model, initial_capacity=100)
    model.service_providers = [provider]
    model.service_market = ServiceMarket(model, service_config)
    model.config["runtime_fixed_rule"]["preproduction_cash"] = {
        "mode": "reserved_no_credit",
        "funds_basis": "capital_as_liquid_operating_balance",
    }
    farmer.capital = 5000
    farmer.total_labor_units = 100
    farmer.age = 60
    farmer.p3_service_mode = "outsource"
    assert len(model.land_patches) == 2
    for patch in model.land_patches:
        patch.area = 5
        patch.owner = farmer
        patch.leased_to = None
        patch.current_operator = farmer
        patch.abandoned = False
        patch.rental_price = 0
    farmer.land_patches = list(model.land_patches)
    farmer.leased_in_patches = []
    model.invalidate_land_tenure_caches()

    prepare_preproduction_cash(model)
    budget = farmer.last_preproduction_cash_account
    cultivated = production_area(farmer)
    assert farmer.get_operated_land_area() == 10
    assert cultivated == pytest.approx(100 / 29)
    assert budget["uncommitted_yuan"] > 0
    actual_by_plot = {
        str(p.patch_id): p.preproduction_cultivated_area_mu for p in model.land_patches
    }
    assert all(p.preproduction_cash_step == model.step_count for p in model.land_patches)
    assert all(area == pytest.approx(cultivated / 2) for area in actual_by_plot.values())

    # Use the original demand and matching chain, not a supplied demand stub.
    demand = model._get_farmer_service_demand(farmer)
    assert demand is not None
    assert demand["area"] == pytest.approx(cultivated)
    model._conduct_service_market()
    assert len(farmer.service_contracts) == 1
    contract = farmer.service_contracts[0]
    assert contract.area == pytest.approx(cultivated)
    assert list(contract.plot_areas_mu) == sorted(actual_by_plot)
    assert contract.plot_areas_mu == pytest.approx(actual_by_plot)
    invoice = contract.area * contract.fee_per_mu
    assert invoice == pytest.approx(675.8620689655173)
    assert contract.p3_paid_yuan == 0
    assert provider._cash_receipts == 0
    assert budget["service_reserved_yuan"] == pytest.approx(invoice)
    assert budget["committed_yuan"] <= budget["opening_available_yuan"] + 1e-8

    model._conduct_labor_market()
    farmer.make_production_decisions()
    provider_before_receipt = provider.cash_balance
    farmer.harvest_and_settle()
    verified = {r.plot_id: r.area_mu for r in production_records(farmer)}
    assert verified == pytest.approx(actual_by_plot)
    assert all(area <= verified[key] + 1e-8 for key, area in contract.plot_areas_mu.items())
    assert farmer.service_fee_expense == pytest.approx(invoice)
    assert contract.p3_paid_yuan == pytest.approx(invoice)
    assert provider.cash_balance - provider_before_receipt == pytest.approx(invoice)
    assert farmer.capital == pytest.approx(5000 + farmer.gross_income - farmer.total_expense)
    assert budget["actual_paid_yuan"] <= budget["committed_yuan"] + 1e-6

    before_support = farmer.capital
    model.government.verify_subsidy_claims()
    model.government.distribute_subsidies()
    payments = model.government.p3_project_payments
    service_payments = [row for row in payments if row["pool"] == "service"]
    assert len(service_payments) == 2
    expected_support = min(invoice * 0.3, cultivated * 100 * spec.service_2019_to_model)
    assert sum(row["paid_yuan"] for row in service_payments) == pytest.approx(expected_support)
    assert expected_support > 0
    assert farmer.capital - before_support == pytest.approx(sum(row["paid_yuan"] for row in payments))
    assert provider.cash_balance == pytest.approx(provider_before_receipt + invoice)
