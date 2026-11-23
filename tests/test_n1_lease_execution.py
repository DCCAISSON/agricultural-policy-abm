from copy import copy
from dataclasses import asdict, replace
from decimal import Decimal

import pytest

from agents.farmer_agent import FarmerAgent
from agents.village_collective import CollectiveStrategy, VillageCollective
from core.fixed_rule_runtime import enact_fixed_rule_action
from modules.land_patch import LandPatch
from scripts.n1_income_fixtures import HAND_CASES, hand_farmer


@pytest.fixture
def lease_case():
    farmer = hand_farmer(HAND_CASES[0])
    parcel = farmer.model.land_patches[1]
    parcel.end_lease()
    parcel.area = 2.0
    parcel.owner.land_patches = [parcel]
    farmer.model.transfer_rate_control.update({"total_land_area": 1000.0, "baseline_total_land_area": 1000.0, "annual_new_transferred_area": 0.0, "observed_current_transferred_area": 0.0, "max_annual_rate": 1.0})
    farmer.model.village_collectives = [VillageCollective("n1_local", farmer.model, config=farmer.model.config)]
    farmer.model.invalidate_land_tenure_caches()
    enact_fixed_rule_action(farmer, "maintain_grain", "household")
    farmer.fixed_rule_bid_accounts = []
    return farmer, parcel


@pytest.mark.parametrize("total_rent", [0.0, 400.0])
def test_direct_lease_rejects_unfunded_full_production_even_when_rent_is_small(lease_case, total_rent):
    farmer, parcel = lease_case
    farmer.capital = 9500.0
    assert farmer.expected_fixed_rule_account("maintain_grain").feasible
    applied = parcel.set_leased(farmer, rental_price=total_rent, transfer_channel="village_collective")
    assert applied is False
    assert parcel.leased_to is None


def test_zero_rent_is_allowed_when_complete_production_is_funded(lease_case):
    farmer, parcel = lease_case
    farmer.capital = 12000.0
    assert parcel.set_leased(farmer, rental_price=0.0, transfer_channel="village_collective") is True
    assert farmer.expected_fixed_rule_account("maintain_grain").feasible


def test_total_rent_is_not_multiplied_by_area_again_in_the_financial_gate(lease_case):
    farmer, parcel = lease_case
    parcel.area = 10.0
    farmer.capital = 30000.0
    assert parcel.set_leased(farmer, rental_price=3000.0, transfer_channel="village_collective") is True
    assert farmer.expected_fixed_rule_account("maintain_grain").rent_in_yuan == 3000.0


def test_repeated_direct_leases_share_the_same_remaining_working_capital(lease_case):
    farmer, first = lease_case
    second = copy(first)
    second.patch_id = "n1_second_parcel"
    farmer.model.land_patches.append(second)
    farmer.capital = 13000.0
    assert first.set_leased(farmer, rental_price=800.0, transfer_channel="village_collective") is True
    assert second.set_leased(farmer, rental_price=800.0, transfer_channel="village_collective") is False
    assert second.leased_to is None


@pytest.mark.parametrize("v1", [False, True])
def test_local_village_passes_total_annual_rent_to_the_patch(lease_case, v1):
    farmer, parcel = lease_case
    farmer.model.config["runtime_fixed_rule"]["national_v1"] = v1
    if v1:
        farmer.set_v1_labor(2, False)
    village = farmer.model.village_collectives[0]
    village.current_strategy = CollectiveStrategy.BALANCED_WELFARE
    village.expected_rental_price = 400.0
    area = village._record_transfer({"farmer": parcel.owner, "patches": [parcel]}, {"agent": farmer, "type": "farmer"})
    assert area == 2.0
    assert village.land_transfer_records[-1].get("contract_patch_ids") == ([str(parcel.get_patch_id())] if v1 else None)
    assert parcel.rental_price == 800.0
    assert parcel.contracted_rental_price == 800.0
    farmer.calculate_land_rental_expense()
    parcel.owner.calculate_land_rental_income()
    assert farmer.land_rental_expense == parcel.owner.land_rental_income == 800.0


@pytest.mark.parametrize("v1", [False, True])
def test_cross_village_passes_total_annual_rent_to_the_patch(lease_case, v1):
    farmer, parcel = lease_case
    farmer.model.config["runtime_fixed_rule"]["national_v1"] = v1
    if v1:
        farmer.set_v1_labor(2, False)
    village = farmer.model.village_collectives[0]
    receiver = VillageCollective("n1_receiver", farmer.model, config=farmer.model.config)
    village.expected_rental_price = receiver.expected_rental_price = 400.0
    record = village._execute_cross_village_subcontract(
        receiver, {"farmer": parcel.owner, "patches": [parcel]}, {"agent": farmer, "type": "farmer"}, {"rent_markup": 0.0},
    )
    assert record["rental_value"] == 800.0
    assert record.get("contract_patch_ids") == ([str(parcel.get_patch_id())] if v1 else None)
    assert parcel.rental_price == 800.0


def test_scale_operator_cannot_receive_zero_rent_land_without_production_funds(lease_case):
    farmer, parcel = lease_case
    enterprise = farmer.model.agricultural_enterprises[0]
    enterprise.leased_patches = []
    enterprise.capital = 0.0
    farmer.model.invalidate_land_tenure_caches()
    enact_fixed_rule_action(enterprise, "maintain_grain", "scale")
    assert parcel.set_leased(enterprise, rental_price=0.0, transfer_channel="village_collective") is False


def test_chosen_exit_cannot_receive_new_land_through_direct_assignment(lease_case):
    farmer, parcel = lease_case
    enact_fixed_rule_action(farmer, "exit", "household")
    assert parcel.set_leased(farmer, rental_price=0.0, transfer_channel="village_collective") is False


@pytest.mark.parametrize(("capital", "accepted"), [(9500.0, False), (13000.0, True)])
def test_lease_log_preserves_complete_accounts_for_accepted_and_rejected_candidates(
    lease_case: tuple[FarmerAgent, LandPatch], capital: float, accepted: bool,
) -> None:
    farmer, parcel = lease_case
    farmer.capital = capital
    before = farmer.expected_fixed_rule_account("maintain_grain")
    candidate = replace(
        farmer.expected_fixed_rule_account("maintain_grain", before.area_mu + parcel.area),
        rent_in_yuan=before.rent_in_yuan + 800.0,
    )

    applied = parcel.set_leased(farmer, rental_price=800.0, transfer_channel="village_collective")

    assert applied is accepted
    record = farmer.fixed_rule_bid_accounts[-1]
    cash_fields = ("production_cost_yuan", "service_cost_yuan", "rent_in_yuan", "management_cost_yuan", "conversion_cost_yuan")
    for prefix, account in (("before", before), ("after", candidate)):
        total = sum((Decimal(str(record[f"{prefix}_{field}"])) for field in cash_fields), Decimal(0))
        assert abs(total - Decimal(str(record[f"{prefix}_working_capital_required"]))) < Decimal("1e-8")
        assert {field: record[f"{prefix}_{field}"] for field in asdict(account)} == asdict(account)
    assert record["candidate_land_cost_yuan"] == 800.0
    assert record["replaced_patch_rent_yuan"] == 0.0
    assert record["after_rent_in_yuan"] == record["before_rent_in_yuan"] - record["replaced_patch_rent_yuan"] + record["candidate_land_cost_yuan"]
