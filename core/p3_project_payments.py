from __future__ import annotations

from dataclasses import dataclass, field
from math import fsum
from typing import TYPE_CHECKING, Literal, TypedDict

from core.p3_project_claims import ProjectClaim, project_claims
from core.p3_project_contracts import P3AccountingError, P3Program, ration, required_value

if TYPE_CHECKING:
    from agents.government_agent import GovernmentAgent
    from agents.farmer_agent import FarmerAgent
    from agents.enterprise_agent import AgriculturalEnterpriseAgent


class PaymentRow(TypedDict):
    claim_id: str
    step: int
    scenario_id: str
    recipient_id: str
    producer_id: str
    project_id: str
    layer: str
    pool: str
    plot_id: str | None
    season: str
    nominal_claim_yuan: float
    expected_yuan: float | None
    verified_payable_yuan: float
    paid_yuan: float
    payment_ratio: float
    model_price_year: int
    rate_origin: str
    nominal_rate: float
    source: str
    source_price_year: int
    to_model: float
    statutory_rate: float
    quantity: float
    quantity_unit: str
    price_conversion_source: str
    scenario_rate_source: str | None


@dataclass(slots=True)
class PaymentBatch:
    program: P3Program
    step: int
    claims: list[ProjectClaim] = field(default_factory=list)
    submitted_ids: set[str] = field(default_factory=set)
    verified: bool = False
    paid: bool = False


def freeze_previous_claims(government: GovernmentAgent, program: P3Program) -> P3Program:
    previous = government.p3_batch
    if not previous.verified or not previous.paid or previous.step != government._get_current_step():
        raise P3AccountingError("expectation snapshot requires the just-completed annual batch")
    by_actor: dict[str, dict[Literal["area", "service"], float]] = {}
    for claim in previous.claims:
        if claim.layer != "trial" or claim.pool == "base":
            continue
        values = by_actor.setdefault(str(claim.recipient.unique_id), {"area": 0.0, "service": 0.0})
        values[claim.pool] += claim.nominal_claim_yuan
    return program.model_copy(update={
        "expected_area_claims_yuan": fsum(values["area"] for values in by_actor.values()),
        "expected_service_claims_yuan": fsum(values["service"] for values in by_actor.values()),
        "previous_actor_pool_nominal": by_actor,
        "expectation_source_step": previous.step,
    })


def begin(government: GovernmentAgent, program: P3Program) -> None:
    from core.model import MainModel
    assert isinstance(government.model, MainModel)
    assert government.model.environment is not None
    program.require_ready()
    if government.total_arrears_outstanding > 0:
        raise P3AccountingError("legacy arrears need an explicit separate disposition")
    previous = getattr(government, "p3_batch", None)
    if previous is not None and previous.step == government._get_current_step():
        raise P3AccountingError("annual project batch already opened")
    if previous is not None and previous.claims and not previous.paid:
        raise P3AccountingError("unsettled prior project batch")
    if previous is not None and (program.previous_actor_pool_nominal is None
                                or program.expectation_source_step != previous.step
                                or previous.step != government._get_current_step() - 1):
        raise P3AccountingError("next year requires the previous-year actor-pool expectation snapshot")
    government.p3_batch = PaymentBatch(program, government._get_current_step())
    government.p3_project_payments = []
    government.total_subsidy_paid = 0.0
    government.subsidy_records = []
    government.verified_subsidy_queue = []
    government.current_annual_subsidy_budget = required_value(program.base_budget_yuan) + required_value(program.trial_budget_yuan)
    government.annual_subsidy_budget = government.current_annual_subsidy_budget
    government.remaining_budget = government.current_annual_subsidy_budget
    government.p3_implementation_cost_yuan = required_value(program.implementation_cost_yuan)
    government.p3_implementation_reserved_yuan = required_value(program.implementation_cost_yuan)
    government.model.environment.manual_subsidy_dispatch = True
    government._reset_annual_subsidy_runtime()
    government.current_subsidy_runtime["p3_program"] = program.model_dump()
    government.announce_subsidy_program()


def submit(government: GovernmentAgent, actor: FarmerAgent | AgriculturalEnterpriseAgent) -> None:
    batch = government.p3_batch
    if batch.verified or batch.paid:
        raise P3AccountingError("project application period is closed")
    key = str(actor.unique_id)
    if key in batch.submitted_ids:
        raise P3AccountingError("duplicate annual claimant")
    claims = project_claims(actor, batch.program)
    batch.claims.extend(claims)
    batch.submitted_ids.add(key)
    government.current_subsidy_runtime["claims_submitted"] = [
        {"agent_id": key, "step": batch.step} for key in sorted(batch.submitted_ids)
    ]


def verify(government: GovernmentAgent) -> list[PaymentRow]:
    batch = government.p3_batch
    if batch.verified:
        return government.p3_project_payments
    totals: dict[Literal["base", "area", "service"], float] = {"base": 0.0, "area": 0.0, "service": 0.0}
    keys: set[tuple[str, str | None, str, str | None]] = set()
    for claim in batch.claims:
        key = (claim.project_id, claim.plot_id, claim.season, claim.producer_id if claim.plot_id is None else None)
        if key in keys:
            raise P3AccountingError("duplicate project-plot-season claim across recipients")
        keys.add(key)
        totals[claim.pool] += claim.nominal_claim_yuan
    gap = max(0.0, totals["base"] - batch.program.pool_budget("base"))
    government.current_subsidy_runtime["base_funding_gap_yuan"] = gap
    if gap > 1e-8:
        raise P3AccountingError(f"preserved-project funding gap: {gap}")
    ratios = {pool: ration(batch.program.pool_budget(pool), total) for pool, total in totals.items()}
    ratios["base"] = 1.0
    rows: list[PaymentRow] = []
    for index, claim in enumerate(batch.claims):
        ratio = ratios[claim.pool]
        rows.append(PaymentRow(
            claim_id=f"p3_{batch.step}_{index}", step=batch.step,
            scenario_id=batch.program.scenario_id, recipient_id=str(claim.recipient.unique_id),
            producer_id=claim.producer_id, project_id=claim.project_id, layer=claim.layer,
            pool=claim.pool, plot_id=claim.plot_id, season=claim.season,
            nominal_claim_yuan=claim.nominal_claim_yuan, expected_yuan=claim.expected_yuan,
            verified_payable_yuan=claim.nominal_claim_yuan * ratio, paid_yuan=0.0,
            payment_ratio=ratio, model_price_year=required_value(batch.program.model_price_year),
            rate_origin=batch.program.rate_origin, nominal_rate=batch.program.nominal_rate,
            source=claim.source,
            source_price_year=claim.source_price_year, to_model=claim.to_model,
            statutory_rate=claim.statutory_rate, quantity=claim.quantity,
            quantity_unit=claim.quantity_unit,
            price_conversion_source=required_value(batch.program.price_conversion_source),
            scenario_rate_source=batch.program.scenario_rate_source,
        ))
    government.p3_project_payments = rows
    government.current_subsidy_runtime["claims_verified"] = [dict(row) for row in rows]
    batch.verified = True
    return rows


def pay(government: GovernmentAgent) -> None:
    batch = government.p3_batch
    if batch.paid:
        return
    if not batch.verified:
        raise P3AccountingError("project payments require a verified complete batch")
    for claim, row in zip(batch.claims, government.p3_project_payments, strict=True):
        amount = row["verified_payable_yuan"]
        claim.recipient.apply_verified_subsidy_payment(
            cash_amount=amount, basic_cash=amount, scale_cash=0.0,
            payment_record=row,
        )
        row["paid_yuan"] = amount
    paid = sum(row["paid_yuan"] for row in government.p3_project_payments)
    government.total_subsidy_paid = paid
    government.remaining_budget = government.current_annual_subsidy_budget - paid
    government.subsidy_records = [dict(row, agent_id=row["recipient_id"],
        amount=row["paid_yuan"], cash_amount=row["paid_yuan"], in_kind_value=0.0)
        for row in government.p3_project_payments]
    government.current_subsidy_runtime["payments_made"] = government.subsidy_records
    if batch.program.implementation_cost_rate is not None:
        trial_paid = sum(row["paid_yuan"] for row in government.p3_project_payments if row["layer"] == "trial")
        government.p3_implementation_cost_yuan = trial_paid * batch.program.implementation_cost_rate
    reserved = government.p3_implementation_reserved_yuan
    government.current_subsidy_runtime["implementation_reserved_yuan"] = reserved
    government.current_subsidy_runtime["implementation_cost_yuan"] = government.p3_implementation_cost_yuan
    government.current_subsidy_runtime["implementation_unspent_yuan"] = reserved - government.p3_implementation_cost_yuan
    government.current_subsidy_runtime["implementation_cost_basis"] = (
        "actual_paid_trial_support_author_condition_not_observed_cost"
        if batch.program.implementation_cost_rate is not None else "fixed_cost_input")
    government.current_subsidy_runtime["total_fiscal_cost_yuan"] = paid + government.p3_implementation_cost_yuan
    government.current_subsidy_runtime["fiscal_envelope_yuan"] = (
        government.current_annual_subsidy_budget + reserved)
    government.current_subsidy_runtime["trial_unspent_yuan"] = {
        pool: batch.program.pool_budget(pool) - sum(row["paid_yuan"] for row in
            government.p3_project_payments if row["pool"] == pool)
        for pool in ("area", "service")
    }
    batch.paid = True
