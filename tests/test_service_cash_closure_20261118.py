from copy import deepcopy

import pytest

from core.p3_project_contracts import P3AccountingError
from core.p3_service_settlement import confirm_service_settlement
from modules.agricultural_service import ServiceMarket, ServiceProvider, ServiceType
from tests.test_p3_project_payments_20260909 import prepared, program


def setup_cash(cash=3640, mode="operating_cash"):
    farmer = prepared(program())
    cfg = farmer.model.config["production"]["agricultural_service"]
    cfg.setdefault("provider_economics", {}).update(cash_mode=mode, initial_cash_yuan=cash)
    provider = ServiceProvider("cash_provider", farmer.model, initial_capacity=100)
    farmer.model.service_providers = [provider]
    return farmer, provider, ServiceMarket(farmer.model, cfg)


def demands(farmer, count=2):
    return [{"farmer": farmer, "area": 2, "service_type": ServiceType.TILLAGE,
             "max_price_per_mu": 600} for _ in range(count)]


@pytest.mark.parametrize("cash", [None, -1, float("nan"), float("inf")])
def test_initial_cash_is_explicit(cash):
    with pytest.raises(ValueError):
        setup_cash(cash)


def test_non_p3_fails_at_initialization():
    farmer, _, _ = setup_cash()
    del farmer.model.config["fiscal_system"]["p3_support"]
    with pytest.raises(ValueError, match="P3"):
        ServiceProvider("unsupported", farmer.model)


def test_batch_and_repeated_batch_reservations():
    farmer, provider, market = setup_cash()
    matches = market.match_service_demands(demands(farmer), [provider])
    assert len(matches) == 1
    assert market.match_service_demands(demands(farmer), [provider]) == []
    assert provider.cash_balance == 3640
    assert provider.annual_revenue == 0
    farmer.model._apply_service_contract(matches[0])
    assert provider.cash_balance == 0
    assert not provider.can_serve(None, 2)
    invoice = matches[0]["contract"].area * matches[0]["contract"].fee_per_mu
    assert provider.annual_revenue == invoice
    assert matches[0]["contract"].p3_paid_yuan == 0
    before = (provider.cash_balance, provider.annual_revenue, provider.service_area_used)
    provider.provide_service(matches[0]["contract"])
    provider.add_service_contract(matches[0]["contract"])
    assert before == (provider.cash_balance, provider.annual_revenue, provider.service_area_used)
    farmer.harvest_and_settle()
    assert provider.cash_balance == pytest.approx(invoice)
    assert farmer.service_fee_expense == invoice
    capital = farmer.capital
    confirm_service_settlement(farmer)
    provider.receive_confirmed_payment(matches[0]["contract"])
    assert farmer.capital == capital
    assert provider.cash_balance == pytest.approx(invoice)
    provider.end_of_year_settlement(2010)
    account = deepcopy(provider.last_settlement_account)
    assert account["cash_open_yuan"] + account["cash_receipts_yuan"] - account["operating_cash_paid_yuan"] == pytest.approx(account["cash_close_yuan"])
    assert account["operating_cash_paid_yuan"] == 3640
    provider.end_of_year_settlement(2010)
    assert provider.last_settlement_account == account


def test_unfunded_maintenance_stops_work_without_debt():
    farmer, provider, market = setup_cash(3499)
    for year in (2010, 2011):
        assert market.match_service_demands(demands(farmer), [provider]) == []
        provider.end_of_year_settlement(year)
        assert provider.cash_balance == 3499
        assert provider.last_settlement_account["maintenance_not_performed_yuan"] == 3500
        assert provider.last_settlement_account["operating_cash_paid_yuan"] == 0
        assert provider.last_settlement_profit == -35000
    assert provider.capital == 30000


def test_direct_p3_rejects_before_service_effects():
    farmer, provider, market = setup_cash(10000)
    match = market.match_service_demands(demands(farmer, 1), [provider])[0]
    provider._cash_orders.clear()
    provider.cash_balance = 0
    before = (farmer.service_fee_expense, len(farmer.service_contracts), provider.annual_revenue)
    with pytest.raises(P3AccountingError, match="cash"):
        farmer.model._apply_service_contract(match)
    assert before == (farmer.service_fee_expense, len(farmer.service_contracts), provider.annual_revenue)


def test_close_requires_fulfillment_and_payment():
    farmer, provider, market = setup_cash()
    match = market.match_service_demands(demands(farmer, 1), [provider])[0]
    with pytest.raises(P3AccountingError):
        provider.end_of_year_settlement(2010)
    farmer.model._apply_service_contract(match)
    with pytest.raises(P3AccountingError):
        provider.end_of_year_settlement(2010)
    with pytest.raises(P3AccountingError):
        provider.receive_confirmed_payment(match["contract"])


def test_expansion_is_frozen_even_above_original_threshold():
    farmer, provider, market = setup_cash(10000)
    provider.expansion_investment_threshold = -100000
    for match in market.match_service_demands(demands(farmer), [provider]):
        farmer.model._apply_service_contract(match)
    farmer.harvest_and_settle()
    provider.end_of_year_settlement(2010)
    assert provider.current_capacity == 100
    assert provider.large_tractors == 1
    assert provider.last_settlement_account["equipment_investment_paid_yuan"] == 0


def test_capacity_reservation_is_not_double_counted():
    farmer, provider, market = setup_cash(100000)
    provider.current_capacity = 6
    matches = market.match_service_demands(demands(farmer, 4), [provider])
    assert len(matches) == 3
    assert market.match_service_demands(demands(farmer, 1), [provider]) == []


def test_receipt_validation_precedes_any_posting():
    farmer, provider, market = setup_cash(10000)
    matches = market.match_service_demands(demands(farmer), [provider])
    for match in matches:
        farmer.model._apply_service_contract(match)
    matches[-1]["contract"].p3_fulfilled = False
    with pytest.raises(P3AccountingError):
        farmer.harvest_and_settle()
    assert provider._cash_receipts == 0
    assert all(m["contract"].p3_payment_step is None for m in matches)


def test_three_year_original_p3_path():
    from tests.service_cash_closure_20261118 import run_comparison
    annual, tasks = run_comparison()
    assert len(annual) == 9
    assert len(tasks) == 13


def test_annual_contract_and_prepared_entry_required():
    farmer, provider, market = setup_cash()
    contract = market.match_service_demands(demands(farmer, 1), [provider])[0]["contract"]
    with pytest.raises(P3AccountingError, match="prepare"):
        provider.provide_service(contract)
    contract.duration = 2
    with pytest.raises(P3AccountingError, match="annual"):
        provider.reserve_operating_cash(contract)
