from __future__ import annotations

from dataclasses import dataclass, replace

from core.p3_project_contracts import P3Program, PreservedProject, ProductionRecord, required_value
from core.p3_project_payments import PaymentRow
from core.model import MainModel
from agents.government_agent import GovernmentAgent
from scripts.n1_income_fixtures import HAND_CASES, hand_farmer


@dataclass(frozen=True, slots=True)
class AcceptanceResult:
    case: str
    land_stock_mu: float
    seasonal_grain_area_mu: float
    annual_rent_expense_yuan: float
    annual_rent_income_yuan: float
    original_recipient_yuan: float
    successor_recipient_yuan: float
    project_payments: tuple[PaymentRow, ...]


def run_cases(template: P3Program) -> tuple[AcceptanceResult, ...]:
    results: list[AcceptanceResult] = []
    names = ("rights_owner_enterprise_operator", "30_early_30_late",
             "same_plot_two_seasons_one_annual_rent", "harvest_exceeds_qualified_delivery",
             "zero_trial_preserves_base", "exit_pending_payment_and_next_year_successor")
    for name in names:
        hand = HAND_CASES[0]
        if name == "same_plot_two_seasons_one_annual_rent":
            hand = replace(HAND_CASES[1], rented_mu=60)
        farmer = hand_farmer(hand)
        model = farmer.model
        assert isinstance(model, MainModel)
        enterprise = model.agricultural_enterprises[0]
        model.farmers, model.agricultural_enterprises = [farmer], [enterprise]
        patch = model.land_patches[1] if hand.rented_mu > 0 else model.land_patches[0]
        spec = template.model_dump() | {"scenario_id": name, "trial_budget_yuan": 10000,
                                       "base_budget_yuan": 50000}
        records: tuple[ProductionRecord, ...] = ()
        rent_expense = rent_income = 0.0
        if name in ("30_early_30_late", "same_plot_two_seasons_one_annual_rent"):
            if name == "30_early_30_late":
                patch.area = 30
            records = tuple(ProductionRecord(plot_id=str(patch.patch_id), season=season,
                area_mu=patch.area, harvest_kg=patch.area * 500,
                source="constructed_season_input_not_observed_policy") for season in ("early", "late"))
            setattr(farmer, "p3_production_records", tuple(r.model_dump() for r in records))
            spec["trial_budget_yuan"] = 0
            spec["preserved_projects"] = [
                PreservedProject(project_id="province_120", basis="grain_area", rate=120,
                    price_year=2010, to_model=1, source="constructed_common_price_example",
                    threshold_mu=50, eligible_seasons=("early", "late")).model_dump(),
                PreservedProject(project_id="local_150", basis="grain_area", rate=150,
                    price_year=2010, to_model=1, source="constructed_common_price_example",
                    threshold_mu=50, qualification_period="season",
                    eligible_seasons=("early", "late")).model_dump(),
            ]
        if name == "same_plot_two_seasons_one_annual_rent":
            patch.initial_rental_rights_holder = patch.owner
            patch.owner.land_patches = [patch]
            farmer.calculate_land_rental_expense()
            patch.owner.calculate_land_rental_income()
            rent_expense, rent_income = farmer.land_rental_expense, patch.owner.land_rental_income
            assert rent_expense == rent_income == 60 * 400, (rent_expense, rent_income)
        if name == "rights_owner_enterprise_operator":
            patch.current_operator = patch.leased_to = enterprise
            patch.rental_price = 10 * 400
            enterprise.grain_area = 10
            enterprise.grain_production = 5000
            farmer.calculate_land_rental_income()
            enterprise.calculate_land_rental_cost()
            rent_expense, rent_income = enterprise.land_rental_cost, farmer.land_rental_income
            assert rent_expense == rent_income == 4000
        if name == "harvest_exceeds_qualified_delivery":
            records = (ProductionRecord(plot_id=str(patch.patch_id), season="annual_grain",
                area_mu=10, harvest_kg=5000, qualified_delivery_kg=2000,
                source="constructed_delivery_subset_of_total_sales"),)
            setattr(farmer, "p3_production_records", tuple(r.model_dump() for r in records))
            spec["trial_budget_yuan"] = 0
            spec["preserved_projects"] = [PreservedProject(project_id="order_delivery", basis="order_delivery",
                rate=.6, price_year=2010, to_model=1, source="constructed_common_price_example").model_dump()]
        if name == "zero_trial_preserves_base":
            spec["trial_budget_yuan"] = 0
        program = P3Program.model_validate(spec)
        if records:
            forecasts = []
            for project in program.preserved_projects or ():
                quantity = sum(r.area_mu for r in records if
                    (project.qualification_period == "annual" or r.area_mu >= project.threshold_mu))
                if project.basis == "order_delivery":
                    quantity = 2000
                forecasts.append({"project_id": project.project_id,
                    "expected_model_yuan": quantity * required_value(project.rate),
                    "source": "constructed_preproduction_forecast_not_realized_data_estimate"})
            setattr(farmer, "p3_expected_project_claims", tuple(forecasts))
        model.config["fiscal_system"] = {"p3_support": program.model_dump()}
        government = getattr(model, "government")
        assert isinstance(government, GovernmentAgent)
        government.begin_annual_cycle()
        if name == "harvest_exceeds_qualified_delivery":
            farmer.expected_yield = 5000
            farmer.current_production_area = 10
            model.config["production"]["random_noise"] = {"settlement_std": 0}
            farmer.harvest_and_settle()
            assert farmer.grain_production == 5000
        else:
            farmer.submit_subsidy_claim()
        if name == "rights_owner_enterprise_operator":
            enterprise.submit_subsidy_claim()
        government.verify_subsidy_claims()
        before, enterprise_before = farmer.capital, enterprise.capital
        sales_before = farmer.agricultural_income
        if name == "exit_pending_payment_and_next_year_successor":
            model.farmers = []
            patch.current_operator = patch.leased_to = enterprise
        government.distribute_subsidies()
        received, successor = farmer.capital - before, enterprise.capital - enterprise_before
        rows = tuple(row.copy() for row in government.p3_project_payments)
        assert farmer.agricultural_income == sales_before
        expected = {"rights_owner_enterprise_operator": 100, "30_early_30_late": 7200,
                    "same_plot_two_seasons_one_annual_rent": 32400,
                    "harvest_exceeds_qualified_delivery": 1200, "zero_trial_preserves_base": 100,
                    "exit_pending_payment_and_next_year_successor": 3600}
        assert abs(received - expected[name]) < 1e-6
        if name == "rights_owner_enterprise_operator":
            assert successor == 3500
        if name == "exit_pending_payment_and_next_year_successor":
            assert successor == 0
            from core.p3_project_payments import freeze_previous_claims
            model.config["fiscal_system"]["p3_support"] = freeze_previous_claims(government, program).model_dump()
            model.step_count += 1
            enterprise.grain_area, enterprise.grain_production = 10, 5000
            government.begin_annual_cycle()
            farmer.submit_subsidy_claim()
            enterprise.submit_subsidy_claim()
            government.verify_subsidy_claims()
            before_next = enterprise.capital
            government.distribute_subsidies()
            successor = enterprise.capital - before_next
            assert successor == 3500
            rows += tuple(row.copy() for row in government.p3_project_payments)
        results.append(AcceptanceResult(name, sum(p.area for p in model.land_patches),
            sum(r.area_mu for r in records) if records else 10,
            rent_expense, rent_income, received, successor, rows))
    return tuple(results)
