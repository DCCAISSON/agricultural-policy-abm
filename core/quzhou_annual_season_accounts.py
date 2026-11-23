from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Final, Literal


@dataclass(frozen=True, slots=True)
class SeasonAccount:
    crop: Literal["early_rice", "late_rice"]
    area_mu: float
    harvest_kg: float
    sold_kg: float
    price_yuan_per_kg: float
    cost_yuan: float = 0.0
    service_fee_yuan: float = 0.0
    labor_days: float = 0.0
    own_premium_yuan: float = 0.0
    insured: bool = False
    order_enabled: bool = True
    verified_order_kg: float | None = None
    conditional_order_share: float | None = None
    area_verified: bool = False
    quantity_basis: str = "conditional_production"
    part: Literal["original", "expansion"] = "original"
    order_rate_yuan_per_kg: float | None = None
    order_cap_yuan_per_mu: float | None = None


@dataclass(frozen=True, slots=True)
class AnnualAccount:
    operator_id: str
    source_year: int
    year: int
    policy_year: int
    policy_reference_only: bool
    local_jurisdiction: str
    tenure_basis: str
    owned_operated_mu: float
    owned_entitlement_mu: float
    fixed_rent_yuan_per_mu: float | None
    market_rent_yuan_per_mu: float | None
    lessor_id: str = ""
    other_province_mu: float = 0.0
    other_local_mu: Mapping[str, float] = field(default_factory=dict[str, float])
    other_seasons: tuple[SeasonAccount, ...] = ()
    completed_early: SeasonAccount | None = None
    verified_order_kg: float | None = None
    conditional_order_share: float | None = None
    fertility_yuan: float | None = None
    fertility_recipient: str = ""
    service_support: Mapping[str, float] = field(default_factory=dict[str, float])
    premium_subsidy: Mapping[str, float] = field(default_factory=dict[str, float])
    experimental_claim_yuan: float = 0.0
    experimental_allocation_yuan: float = 0.0
    paid: Mapping[str, float] = field(default_factory=dict[str, float])
    verified_due: Mapping[str, float] = field(default_factory=dict[str, float])
    matched_out_mu: float = 0.0
    matched_out_rate: float = 0.0
    matched_out_payer: str = ""
    released_at_year_end_mu: float = 0.0
    year_end_reason: str = "none"
    season_coverage_basis: str = "unknown"
    support_amount_basis: str = "unknown"
    early_sold_kg: float | None = None
    opening_inventory_kg: Mapping[str, float] = field(default_factory=dict[str, float])
    prior_payments: tuple[ProjectPayment, ...] = ()
    paid_basis: str = "unrecorded"


@dataclass(frozen=True, slots=True)
class ProjectPayment:
    project: str
    season: str
    part: str
    payer: str
    recipient: str
    eligibility_area_mu: float
    payment_base: float
    rate: float
    nominal_claim_yuan: float | None
    expected_payment_yuan: float | None
    verified_due_yuan: float | None
    actual_paid_yuan: float | None
    basis: str
    claim_year: int = 0
    payment_year: int | None = None


@dataclass(frozen=True, slots=True)
class AnnualSettlement:
    original: Mapping[str, float]
    expansion: Mapping[str, float]
    payments: tuple[ProjectPayment, ...]
    seasons: tuple[SeasonAccount, ...]
    next_matching_year: int
    released_at_year_end_mu: float
    closing_inventory_kg: Mapping[str, float]


ACCOUNT_PARTS: Final = ("net", "base_grain_revenue", "extra_subsidy", "order_revenue", "scale_subsidy",
    "insurance_payout", "rent_component", "land_rent_paid", "nonfarm_component", "cost_component",
    "production_cost_component", "service_fee", "insurance_premium", "risk_penalty", "output_kg",
    "required_capital_yuan", "required_labor_days", "fertility_subsidy", "service_support",
    "experimental_support", "annual_crop_output_kg", "unresolved_support_count", "prior_cash_received_yuan", "prior_cash_paid_yuan")


def settle_year(account: AnnualAccount, seasons: tuple[SeasonAccount, ...], land_mu: float,
                expansion_mu: float, params: Mapping[str, float]) -> AnnualSettlement:
    assert account.policy_year == 2024, "implemented policy rules apply to 2024 only"
    assert account.local_jurisdiction in ("Qujiang", "conditional_Qujiang", "outside_Qujiang"), "unknown local jurisdiction"
    assert account.season_coverage_basis in ("observed_full_year", "conditional_single_early", "conditional_listed_seasons"), "annual season coverage requires explicit basis"
    assert account.season_coverage_basis != "conditional_single_early" or all(s.crop == "early_rice" for s in seasons), "single early condition contains other seasons"
    assert account.year == account.policy_year or account.policy_reference_only, "policy year requires explicit conditional reference"
    assert account.market_rent_yuan_per_mu is None or account.year == 2026, "550/900 market scenarios are 2026 only"
    assert 0 <= account.owned_operated_mu <= land_mu and account.owned_entitlement_mu >= account.owned_operated_mu
    assert 0 <= account.matched_out_mu <= account.owned_entitlement_mu - account.owned_operated_mu + 1e-8
    assert account.matched_out_mu == 0 or account.matched_out_payer, "matched rental receipt requires a counterparty"
    assert 0 <= account.released_at_year_end_mu <= land_mu + expansion_mu
    assert len({(s.part, s.crop) for s in seasons}) == len(seasons), "duplicate plot-part/crop season"
    assert all(0 <= s.area_mu <= (land_mu if s.part == "original" else expansion_mu) + 1e-8 for s in seasons)
    parts: dict[str, dict[str, float]] = {part: dict.fromkeys(ACCOUNT_PARTS, 0.0) for part in ("original", "expansion")}
    payments: list[ProjectPayment] = []

    def emit(project: str, season: str, part: str, payer: str, recipient: str, quantity: float, rate: float,
             value: float | None, verified: bool, basis: str, qualification: float = 0.0,
             nominal_claim: float | None = None) -> float:
        key = f"{part}:{season}:{project}"
        paid = account.paid.get(key)
        assert paid is None or paid >= 0, "negative recorded payment"
        assert paid is None or account.paid_basis != "unrecorded", "recorded or conditional payment basis required"
        payments.append(ProjectPayment(project, season, part, payer, recipient, qualification, quantity, rate,
            value if nominal_claim is None else nominal_claim, value,
            account.verified_due.get(key, value if verified else None), paid,
            basis if paid is None else f"{basis};{account.paid_basis}", account.year, account.year if paid is not None else None))
        if value is None:
            parts[part]["unresolved_support_count"] += 1
        return value or 0.0

    annual_qual = account.other_province_mu + sum(s.area_mu for s in seasons)
    for s in seasons:
        assert 0 <= s.sold_kg <= s.harvest_kg and min(s.cost_yuan, s.service_fee_yuan, s.labor_days, s.own_premium_yuan) >= 0
        if s.verified_order_kg is not None:
            assert 0 <= s.verified_order_kg <= s.sold_kg, "verified order exceeds sales"
            order_kg, order_verified = s.verified_order_kg, True
        else:
            assert not s.order_enabled or s.conditional_order_share is not None, "missing order requires explicit condition"
            share = s.conditional_order_share or 0.0
            assert 0 <= share <= 1, "order conditional share outside [0,1]"
            order_kg, order_verified = s.sold_kg * share, False
        if not s.order_enabled:
            order_kg = 0.0
        local_qual = account.other_local_mu.get(s.crop, 0.0) + sum(c.area_mu for c in seasons if c.crop == s.crop)
        b = parts[s.part]
        b["scale_subsidy"] += emit("province_scale", s.crop, s.part, "fiscal_province", account.operator_id,
            s.area_mu, 120, s.area_mu * 120 if annual_qual >= 50 else 0.0, s.area_verified,
            "Zhejiang_2024_6_verified_rule;annual_rice_replant_qualification", annual_qual)
        b["scale_subsidy"] += emit("local_scale", s.crop, s.part, "fiscal_local", account.operator_id,
            s.area_mu, 150, s.area_mu * 150 if local_qual >= 50 and account.local_jurisdiction != "outside_Qujiang" else 0.0, s.area_verified,
            "author_authorized_Qujiang_2024_single_season_rule;original_pending", local_qual)
        order_rate = params["extra_subsidy_yuan_per_jin"] * 2 if s.crop == "early_rice" else s.order_rate_yuan_per_kg
        reward = order_kg * order_rate if order_rate is not None else 0.0 if order_kg == 0 else None
        if reward is not None and s.order_cap_yuan_per_mu is not None:
            assert s.order_cap_yuan_per_mu >= 0
            reward = min(reward, s.area_mu * s.order_cap_yuan_per_mu)
        b["extra_subsidy"] += emit("order_reward", s.crop, s.part, "fiscal_order", account.operator_id,
            order_kg, order_rate or 0.0, reward, order_verified and order_rate is not None,
            "verified_order_delivery" if order_verified else "explicit_conditional_share_not_verified")
        sales = s.sold_kg * s.price_yuan_per_kg
        b["base_grain_revenue"] += sales
        b["order_revenue"] += sales + (reward or 0)
        loss = (sales + (reward or 0)) * params["disaster_loss_rate"]
        fixed = params.get("expected_insurance_payout_yuan_per_mu")
        payout = (s.area_mu * fixed if fixed is not None else loss * params["indemnity_coverage_rate"]) if s.insured else 0.0
        b["insurance_payout"] += emit("insurance_payout", s.crop, s.part, "insurer", account.operator_id,
            s.area_mu, 0.0, payout, False, "model_expected_indemnity_not_fiscal_payment")
        b["risk_penalty"] += (loss - payout) * params["risk_aversion_weight"]
        b["cost_component"] += s.cost_yuan
        b["production_cost_component"] += s.cost_yuan
        b["service_fee"] += s.service_fee_yuan
        b["insurance_premium"] += s.own_premium_yuan
        b["required_labor_days"] += s.labor_days
        b["output_kg"] += s.harvest_kg if s.crop == "early_rice" else 0.0
        b["annual_crop_output_kg"] += s.harvest_kg
        supported = account.service_support.get(f"{s.part}:{s.crop}") if s.service_fee_yuan > 0 else 0.0
        b["service_support"] += emit("service_support", s.crop, s.part, "fiscal_service", account.operator_id,
            s.area_mu, 0, supported, supported is not None and s.area_verified, "explicit_project_amount_or_unknown_not_automatic_30pct")
        subsidy = account.premium_subsidy.get(f"{s.part}:{s.crop}") if s.insured else 0.0
        _ = emit("insurance_premium_subsidy", s.crop, s.part, "fiscal_insurance", "insurer", s.area_mu, 0,
            subsidy, subsidy is not None and s.area_verified, "separate_from_operator_self_paid_premium")
        _ = emit("operator_insurance_premium", s.crop, s.part, account.operator_id, "insurer", s.area_mu, 0,
            s.own_premium_yuan, False, "self_paid_premium_condition_not_total_premium")
    original_rate = account.fixed_rent_yuan_per_mu if account.fixed_rent_yuan_per_mu is not None else account.market_rent_yuan_per_mu
    new_rate = account.market_rent_yuan_per_mu if account.market_rent_yuan_per_mu is not None else original_rate
    for part, rented, rate in (("original", land_mu - account.owned_operated_mu, original_rate), ("expansion", expansion_mu, new_rate)):
        assert rented == 0 or rate is not None, "leased land requires annual rent terms"
        assert rented == 0 or account.lessor_id, "leased land requires known or explicit conditional counterparty"
        if rented > 0:
            parts[part]["land_rent_paid"] = emit("annual_rent", "annual", part, account.operator_id,
                account.lessor_id, rented, rate or 0, rented * (rate or 0), False,
                "fixed_contract" if part == "original" and account.fixed_rent_yuan_per_mu is not None else "author_2026_new_or_renewed_annual_rent_condition")
        parts[part]["rent_component"] = -parts[part]["land_rent_paid"]
    recipient = account.fertility_recipient or (account.operator_id if account.owned_entitlement_mu > 0 else account.lessor_id)
    fertility = emit("land_fertility", "annual", "original", "fiscal_base", recipient, account.owned_entitlement_mu, 0,
        account.fertility_yuan, False, "explicit_entitlement_and_amount_or_unknown;not_in_rent")
    if recipient == account.operator_id:
        parts["original"]["fertility_subsidy"] = fertility
    assert min(account.experimental_claim_yuan, account.experimental_allocation_yuan) >= 0
    experiment = min(account.experimental_claim_yuan, account.experimental_allocation_yuan)
    parts["original"]["experimental_support"] = emit("experimental_support", "annual", "original", "fiscal_experiment",
        account.operator_id, 0, 0, experiment, False, "recipient_allocation_envelope_not_base_policy_budget",
        nominal_claim=account.experimental_claim_yuan)
    if account.matched_out_mu > 0:
        parts["original"]["rent_component"] += emit("matched_rent_receipt", "annual", "original", account.matched_out_payer,
            account.operator_id, account.matched_out_mu, account.matched_out_rate,
            account.matched_out_mu * account.matched_out_rate, False, "matched_owned_land_only_not_past_rented_in_area")
    for b in parts.values():
        b["required_capital_yuan"] = sum(b[k] for k in ("cost_component", "service_fee", "insurance_premium", "land_rent_paid"))
        b["net"] = sum(b[k] for k in ("base_grain_revenue", "extra_subsidy", "scale_subsidy", "insurance_payout",
            "rent_component", "nonfarm_component", "fertility_subsidy", "service_support", "experimental_support")) - sum(b[k] for k in ("cost_component", "service_fee", "insurance_premium", "risk_penalty"))
    closing_inventory = dict(account.opening_inventory_kg)
    assert all(value >= 0 for value in closing_inventory.values()), "negative opening inventory"
    for season in seasons:
        closing_inventory[season.crop] = closing_inventory.get(season.crop, 0.0) + season.harvest_kg - season.sold_kg
    for prior in account.prior_payments:
        assert prior.claim_year < account.year and account.operator_id in (prior.payer, prior.recipient), "prior claim belongs to another period or party"
        key = f"{prior.claim_year}:{prior.part}:{prior.season}:{prior.project}"
        paid = account.paid.get(key)
        due = None if prior.verified_due_yuan is None or prior.actual_paid_yuan is None else prior.verified_due_yuan - prior.actual_paid_yuan
        assert paid is None or (due is not None and 0 <= paid <= due), "prior payment requires verified outstanding claim"
        assert paid is None or account.paid_basis != "unrecorded", "recorded or conditional payment basis required"
        expected = due if due is not None else None if prior.expected_payment_yuan is None or prior.actual_paid_yuan is None else max(0.0, prior.expected_payment_yuan - prior.actual_paid_yuan)
        payments.append(replace(prior, expected_payment_yuan=expected, verified_due_yuan=due, actual_paid_yuan=paid,
            payment_year=account.year if paid is not None else None, basis=f"{prior.basis};carried_prior_claim;{account.paid_basis}"))
        flow = "prior_cash_received_yuan" if prior.recipient == account.operator_id else "prior_cash_paid_yuan"
        parts["original"][flow] += paid or 0.0
    return AnnualSettlement(parts["original"], parts["expansion"], tuple(payments), seasons, account.year + 1,
        account.released_at_year_end_mu, closing_inventory)
