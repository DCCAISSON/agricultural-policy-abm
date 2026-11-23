from __future__ import annotations

import pytest

from core.p3_project_contracts import P3AccountingError, P3InputsMissing, PreservedProject
from modules.agricultural_service import ServiceContract, ServiceProvider, CustodyType
from tests.test_p3_project_payments_20260909 import prepared, program, service


@pytest.mark.parametrize("cost_rate", [0.0, .008027, .016054])
def test_public_cost_is_separate_and_charged_on_paid_trial_support(cost_rate: float) -> None:
    from core.p3_project_contracts import P3Program
    spec = P3Program.model_validate(program(budget=7000).model_dump() | {
        "implementation_cost_rate": cost_rate,
        "implementation_cost_yuan": 7000 * cost_rate,
    })
    farmer = prepared(spec)
    government = farmer.model.government

    farmer.submit_subsidy_claim()
    government.verify_subsidy_claims()
    government.distribute_subsidies()

    runtime = government.current_subsidy_runtime
    assert government.total_subsidy_paid == pytest.approx(3600)
    assert government.p3_implementation_cost_yuan == pytest.approx(3500 * cost_rate)
    assert runtime["implementation_reserved_yuan"] == pytest.approx(7000 * cost_rate)
    assert runtime["implementation_unspent_yuan"] == pytest.approx(3500 * cost_rate)
    assert runtime["total_fiscal_cost_yuan"] == pytest.approx(3600 + 3500 * cost_rate)
    assert runtime["fiscal_envelope_yuan"] == pytest.approx(8000 + 7000 * cost_rate)
    assert runtime["trial_unspent_yuan"]["area"] == pytest.approx(3500)


def test_formal_pending_does_not_inherit_fixture_claim_forecasts() -> None:
    import ast
    import json
    from pathlib import Path
    from core.p3_project_contracts import P3Program

    root = Path(__file__).resolve().parents[1]
    source = ast.parse((root / "scripts/p3_project_payment_smoke_20260909.py").read_text(encoding="utf-8"))
    assignment = next(node for node in ast.walk(source) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "pending" for target in node.targets))
    assert isinstance(assignment.value, ast.BinOp)
    overrides = ast.literal_eval(assignment.value.right)
    output = root / "results/empirical/three_paper_execution_20260908/national/p3_project_payments_20260909_r1"
    pending = json.loads((output / "formal_inputs_pending.json").read_text(encoding="utf-8"))
    keys = ("expected_area_claims_yuan", "expected_service_claims_yuan")
    for key in keys:
        assert overrides[key] is None
        assert pending[key] is None
    otherwise_ready = program().model_dump() | {key: pending[key] for key in keys}

    with pytest.raises(P3InputsMissing) as error:
        P3Program.model_validate(otherwise_ready).require_ready()

    assert str(error.value) == ", ".join(keys)
    receipt = json.loads((output / "E_P3_ENTRY_EVIDENCE.json").read_text(encoding="utf-8"))
    with pytest.raises(P3InputsMissing) as all_missing:
        P3Program.model_validate(pending).require_ready()
    assert receipt["formal_gate"]["missing"] == str(all_missing.value)


def test_p3_rejects_legacy_actorless_capitalization_payment_path() -> None:
    farmer = prepared(program())
    before = farmer.capital
    with pytest.raises(P3AccountingError, match="project recipient"):
        farmer.model.environment.calculate_subsidy_for_actor(10, actor_type="farmer")
    assert farmer.capital == before


def test_abandoned_rights_land_is_not_automatically_eligible() -> None:
    from core.p3_project_claims import expected_support
    farmer = prepared(program(budget=0))
    farmer.model.land_patches[0].abandoned = True
    assert expected_support(farmer, 0, program(budget=0)) == 0
    farmer.submit_subsidy_claim()
    farmer.model.government.verify_subsidy_claims()
    farmer.model.government.distribute_subsidies()
    assert farmer.model.government.total_subsidy_paid == 0


@pytest.mark.parametrize("legacy_target", ["operator", "owner"])
def test_non_p3_enterprise_keeps_legacy_claim_target(legacy_target: str) -> None:
    from scripts.n1_income_fixtures import HAND_CASES, hand_farmer
    farmer = hand_farmer(HAND_CASES[0])
    model = farmer.model
    enterprise = model.agricultural_enterprises[0]
    model.land_patches[0].current_operator = enterprise
    enterprise.grain_area = 10
    model.environment.manual_subsidy_dispatch = True
    model.environment.active_policy["subsidy_target"] = legacy_target

    claim = enterprise.build_subsidy_claim()

    assert (claim is None) == (legacy_target == "owner")
    if claim is not None:
        assert claim["agent"] is enterprise
        assert claim["grain_area"] == 10


def test_ecological_service_bonus_is_a_separate_active_channel() -> None:
    qualities = []
    for bonus in (0.0, 0.0035):
        farmer = prepared(program("family_service"))
        service(farmer, duration=5)
        patch = farmer.model.land_patches[0]
        patch.soil_quality, patch.initial_soil_quality = .6, .8
        patch.lease_contract_years = 5
        farmer.model.config["production"]["ecological"] = {
            "enabled": True, "service_management_recovery_bonus": bonus}
        farmer.model._update_ecological_feedback()
        qualities.append(patch.soil_quality)
    assert qualities[1] - qualities[0] == pytest.approx(.0035)


def test_six_author_acceptance_cases_run_through_actual_agents() -> None:
    from scripts.p3_project_acceptance_cases_20260909 import run_cases
    results = run_cases(program())
    assert len(results) == 6
    assert sum(len(case.project_payments) for case in results) >= 6


def test_failed_service_is_not_silently_fulfilled_or_charged(monkeypatch: pytest.MonkeyPatch) -> None:
    farmer = prepared(program("family_service"))
    provider = ServiceProvider("failed_provider", farmer.model, initial_capacity=100)
    farmer.model.service_providers = [provider]
    contract = ServiceContract(str(farmer.unique_id), provider.unique_id, 2, "grain", CustodyType.PARTIAL, 200)

    def failed_execution(contract: ServiceContract) -> None:
        raise P3AccountingError("constructed service execution failure")

    monkeypatch.setattr(provider, "provide_service", failed_execution)
    with pytest.raises(P3AccountingError, match="execution failure"):
        farmer.model._apply_service_contract({"farmer": farmer, "provider": provider, "contract": contract})
    assert farmer.service_contracts == []
    assert farmer.service_fee_expense == 0
    assert not contract.p3_fulfilled


def test_order_expected_income_requires_explicit_forecast_not_full_harvest() -> None:
    from core.fixed_rule_runtime import expected_support
    farmer = prepared(program().model_copy(update={"preserved_projects": (
        PreservedProject(project_id="order", basis="order_delivery", rate=.2,
            price_year=2010, to_model=1, source="constructed"),)}))
    before = farmer.capital
    with pytest.raises(P3InputsMissing, match="forecast"):
        expected_support(farmer, 10, "farmer")
    farmer.p3_expected_project_claims = ({"project_id": "order", "expected_model_yuan": 400,
                                        "source": "constructed_2000kg_delivery_forecast"},)
    assert expected_support(farmer, 10, "farmer") == pytest.approx(3900)
    assert farmer.capital == before
