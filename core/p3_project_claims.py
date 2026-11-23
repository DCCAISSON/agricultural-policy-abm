from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, assert_never

from core.p3_project_contracts import (
    P3AccountingError, P3InputsMissing, P3Program, ProductionRecord, ExpectedProjectClaim, required_value, ration,
)

if TYPE_CHECKING:
    from agents.farmer_agent import FarmerAgent
    from agents.enterprise_agent import AgriculturalEnterpriseAgent


@dataclass(frozen=True, slots=True)
class ProjectClaim:
    recipient: FarmerAgent | AgriculturalEnterpriseAgent
    producer_id: str
    project_id: str
    layer: Literal["base", "trial"]
    pool: Literal["base", "area", "service"]
    plot_id: str | None
    season: str
    nominal_claim_yuan: float
    expected_yuan: float | None
    source: str
    source_price_year: int
    to_model: float
    statutory_rate: float
    quantity: float
    quantity_unit: str


def production_records(actor: FarmerAgent | AgriculturalEnterpriseAgent) -> tuple[ProductionRecord, ...]:
    from core.fixed_rule_runtime import preproduction_cash_enabled
    from core.model import MainModel
    assert isinstance(actor.model, MainModel)
    explicit = getattr(actor, "p3_production_records", None)
    patches = {str(p.patch_id): p for p in actor.model.land_patches
               if p.current_operator is actor and not p.abandoned and p.area > 0}
    cash_constrained = preproduction_cash_enabled(actor.model.config)
    areas = {key: (float(p.preproduction_cultivated_area_mu)
                   if cash_constrained and getattr(p, 'preproduction_cash_step', None) == actor.model.step_count
                   else float(p.area)) for key, p in patches.items()}
    if explicit is not None:
        records = tuple(ProductionRecord.model_validate(r) for r in explicit)
        if getattr(actor, "household_planned_grain_share", None) is not None and abs(
                sum(r.area_mu for r in records if r.grain) - actor.grain_area) > 1e-8:
            raise P3AccountingError("explicit grain records disagree with household harvest")
    else:
        area = sum(areas.values())
        share = getattr(actor, "household_planned_grain_share", None)
        if share is not None and 0 < share < 1:
            if actor.grain_area + actor.cash_crop_area > area + 1e-8:
                raise P3AccountingError("household production exceeds cultivated land")
            # The annual area is known, but its distribution over plots is not.
            return (ProductionRecord(plot_id=None, season="annual_grain", area_mu=actor.grain_area,
                    harvest_kg=actor.grain_production, source="household_harvest_plot_unknown"),) if actor.grain_area > 0 else ()
        grain_area = area if str(getattr(getattr(actor, "planned_crop", None), "value", "")).lower() == "grain" else 0.0
        if hasattr(actor, "enterprise_type") or hasattr(actor, "household_actual_grain_share"):
            grain_area = float(getattr(actor, "grain_area", 0.0))
        if grain_area > area + 1e-8:
            raise P3AccountingError("grain production area exceeds operated land")
        output = max(0.0, float(getattr(actor, "grain_production", 0.0)))
        records = tuple(ProductionRecord(
            plot_id=key, season="annual_grain", area_mu=areas[key] * grain_area / area,
            harvest_kg=output * areas[key] / area, source="simulated_annual_production_not_observed_seasons",
        ) for key in patches if grain_area > 0 and areas[key] > 0)
    keys = [(r.plot_id, r.season) for r in records]
    if len(keys) != len(set(keys)):
        raise P3AccountingError("duplicate plot-season production")
    for record in records:
        if record.plot_id not in patches or record.area_mu > areas[record.plot_id] + 1e-8:
            raise P3AccountingError("production exceeds producer's physical plot area")
    return records


def area_offer(actor, program, area):
    """The same announced rate and forecast apply to production decisions and claims."""
    actor_id = str(actor.unique_id)
    shares = program.regional_shares()
    if not shares:
        rate = program.nominal_rate
        nominal = area * rate * required_value(program.trial_to_model)
        return rate, program.expected_payment_ratio(actor_id, "area", nominal)
    government = actor.model.government
    region = government.v1_applicant_regions.get(actor_id, 'unknown')
    share = shares.get(region, 0.)
    rate = program.nominal_rate * (4 * share)
    nominal = area * rate * required_value(program.trial_to_model)
    own = (program.previous_actor_pool_nominal or {}).get(actor_id, {}).get('area', 0.)
    competing = government.v1_previous_area_claims.get(region, 0.) - own
    return rate, ration(required_value(program.trial_budget_yuan) * share, max(0., competing) + nominal)


def project_claims(actor: FarmerAgent | AgriculturalEnterpriseAgent, program: P3Program) -> list[ProjectClaim]:
    from core.fixed_rule_runtime import rental_rights_holder, preproduction_cash_enabled
    from core.model import MainModel
    assert isinstance(actor.model, MainModel)

    records = production_records(actor)
    forecasts = {r.project_id: r.expected_model_yuan for r in
                 (ExpectedProjectClaim.model_validate(raw) for raw in
                  getattr(actor, "p3_expected_project_claims", ()))}
    grains = tuple(r for r in records if r.grain)
    annual_area = sum(r.area_mu for r in grains)
    claims: list[ProjectClaim] = []
    for project in program.preserved_projects or ():
        rate = required_value(project.rate) * required_value(project.to_model)
        match project.basis:
            case "rights_area":
                for patch in actor.model.land_patches:
                    if (rental_rights_holder(patch) is actor and patch.area > 0
                            and (not project.requires_cultivation or
                                 (not patch.abandoned and patch.current_operator is not None))):
                        eligible_area = float(patch.area)
                        if (project.requires_cultivation and preproduction_cash_enabled(actor.model.config)
                                and getattr(patch, 'preproduction_cash_step', None) == actor.model.step_count):
                            eligible_area = float(patch.preproduction_cultivated_area_mu)
                        if eligible_area <= 0:
                            continue
                        value = rate * eligible_area
                        claims.append(ProjectClaim(actor, str(actor.unique_id), project.project_id,
                            "base", "base", str(patch.patch_id), "annual_rights", value, value,
                            str(project.source), required_value(project.price_year),
                            required_value(project.to_model), required_value(project.rate),
                            eligible_area, "mu_year"))
            case "grain_area" | "order_delivery":
                for record in grains:
                    qualification_area = annual_area
                    if project.qualification_period == "season":
                        qualification_area = sum(r.area_mu for r in grains if r.season == record.season)
                    if qualification_area < project.threshold_mu:
                        continue
                    if project.eligible_seasons and record.season not in project.eligible_seasons:
                        continue
                    quantity = record.area_mu
                    if project.basis == "order_delivery":
                        if record.qualified_delivery_kg is None:
                            raise P3InputsMissing(f"{project.project_id}: qualified delivery missing")
                        quantity = record.qualified_delivery_kg
                    value = rate * quantity
                    expected = value
                    if project.basis == "order_delivery" or project.eligible_seasons or project.qualification_period == "season":
                        forecast = forecasts.get(project.project_id)
                        forecast_area = sum(r.area_mu for r in grains
                                            if not project.eligible_seasons or r.season in project.eligible_seasons)
                        expected = forecast * record.area_mu / forecast_area if forecast is not None else None
                    claims.append(ProjectClaim(actor, str(actor.unique_id), project.project_id,
                        "base", "base", record.plot_id, record.season, value, expected, str(project.source),
                        required_value(project.price_year), required_value(project.to_model),
                        required_value(project.rate), quantity,
                        "kg_qualified_delivery" if project.basis == "order_delivery" else "mu_season"))
            case unreachable:
                assert_never(unreachable)
    area_eligible = program.rule in ("area_universal", "area_service_mix") or (
        program.rule == "scale_eligibility" and annual_area >= 80)
    if area_eligible:
        rate, factor = area_offer(actor, program, annual_area)
        for record in grains:
            value = record.area_mu * rate * required_value(program.trial_to_model)
            claims.append(ProjectClaim(actor, str(actor.unique_id), "trial_area", "trial", "area",
                record.plot_id, record.season, value, value * factor, program.rate_origin,
                required_value(program.trial_price_year), required_value(program.trial_to_model),
                rate, record.area_mu, "mu_season"))
    if program.rule in ("family_service", "area_service_mix") and not hasattr(actor, "enterprise_type"):
        grouped: dict[tuple[str | None, str], tuple[float, float]] = {}
        production = {(r.plot_id, r.season): r.area_mu for r in grains}
        from modules.agricultural_service import ServiceContract
        contracts: list[ServiceContract] = getattr(actor, 'service_contracts', [])
        for contract in contracts:
            if contract.crop_type not in ("grain", "mixed"):
                continue
            if contract.p3_paid_yuan <= 0:
                continue
            if not contract.p3_fulfilled:
                raise P3AccountingError("paid service claim has no fulfilled contract")
            if contract.p3_payment_step != actor.model.step_count:
                continue
            if not contract.production_season or not contract.plot_areas_mu:
                raise P3InputsMissing("paid service lacks plot-season coverage")
            covered = sum(contract.plot_areas_mu.values())
            if contract.crop_type == "mixed":
                cultivated = actor.grain_area + actor.cash_crop_area
                if cultivated > 0 and abs(covered - cultivated) <= 1e-8 and annual_area > 0:
                    key = (None, "annual_grain")
                    fees, _ = grouped.get(key, (0.0, 0.0))
                    grouped[key] = (fees + contract.p3_paid_yuan * annual_area / cultivated, annual_area)
                continue
            for plot_id, area in contract.plot_areas_mu.items():
                key = (plot_id, contract.production_season)
                if key not in production or area > production[key] + 1e-8:
                    raise P3AccountingError("paid service exceeds verified grain coverage")
                fees, prior_area = grouped.get(key, (0.0, 0.0))
                grouped[key] = (fees + contract.p3_paid_yuan * area / covered, max(prior_area, area))
        values = {key: min(0.3 * fees, 100 * required_value(program.service_2019_to_model) * area)
                  for key, (fees, area) in grouped.items()}
        factor = program.expected_payment_ratio(str(actor.unique_id), "service", sum(values.values()))
        for (plot_id, season), value in values.items():
            claims.append(ProjectClaim(actor, str(actor.unique_id), "trial_service", "trial", "service",
                plot_id, season, value, value * factor, "2019_rule_simulated_prefix_coverage",
                2019, required_value(program.service_2019_to_model), .3,
                grouped[(plot_id, season)][0], "yuan_actual_service_fee_capped"))
    return claims


def expected_support(actor: FarmerAgent | AgriculturalEnterpriseAgent, grain_area: float,
                     program: P3Program) -> float:
    from core.fixed_rule_runtime import rental_rights_holder
    from core.model import MainModel
    assert isinstance(actor.model, MainModel)

    base = 0.0
    forecasts = {r.project_id: r.expected_model_yuan for r in
                 (ExpectedProjectClaim.model_validate(raw) for raw in
                  getattr(actor, "p3_expected_project_claims", ()))}
    for project in program.preserved_projects or ():
        match project.basis:
            case "rights_area":
                patches = getattr(actor, "_fixed_rule_bid_land_candidates", actor.model.land_patches)
                area = sum(p.area for p in patches if rental_rights_holder(p) is actor
                           and (not project.requires_cultivation or
                                (not p.abandoned and p.current_operator is not None)))
                base += area * required_value(project.rate) * required_value(project.to_model)
            case "grain_area":
                if project.eligible_seasons or project.qualification_period == "season":
                    if grain_area > 0 and project.project_id not in forecasts:
                        raise P3InputsMissing(f"{project.project_id}: seasonal forecast missing")
                    base += forecasts[project.project_id] if grain_area > 0 else 0.0
                elif grain_area >= project.threshold_mu:
                    base += grain_area * required_value(project.rate) * required_value(project.to_model)
            case "order_delivery":
                if grain_area > 0 and project.project_id not in forecasts:
                    raise P3InputsMissing(f"{project.project_id}: delivery forecast missing")
                base += forecasts[project.project_id] if grain_area > 0 else 0.0
            case unreachable:
                assert_never(unreachable)
    if program.rule in ("area_universal", "area_service_mix") or (
        program.rule == "scale_eligibility" and grain_area >= 80):
        rate, factor = area_offer(actor, program, grain_area)
        nominal = grain_area * rate * required_value(program.trial_to_model)
        base += nominal * factor
    return base
