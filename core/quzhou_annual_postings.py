from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from core.quzhou_annual_season_accounts import AnnualAccount, AnnualSettlement


@dataclass(frozen=True, slots=True)
class CounterpartyPosting:
    party_id: str
    counterparty_id: str
    project: str
    season: str
    part: str
    side: Literal["income", "expense"]
    nominal_yuan: float | None
    expected_yuan: float | None
    verified_due_yuan: float | None
    actual_yuan: float | None
    party_expected_balance_yuan: float | None
    identity_basis: str
    claim_year: int
    payment_year: int | None


def post_counterparty_accounts(settlement: AnnualSettlement) -> tuple[CounterpartyPosting, ...]:
    balances: dict[tuple[str, int], float | None] = {}
    postings: list[CounterpartyPosting] = []
    for payment in settlement.payments:
        sides: tuple[tuple[str, str, Literal["income", "expense"], float], ...] = (
            (payment.payer, payment.recipient, "expense", -1.0),
            (payment.recipient, payment.payer, "income", 1.0))
        for party, counterparty, side, sign in sides:
            assert party and counterparty, "bilateral accounts require explicit counterparties"
            previous = balances.get((party, payment.claim_year), 0.0)
            balance = previous + sign * payment.expected_payment_yuan if previous is not None and payment.expected_payment_yuan is not None else None
            balances[party, payment.claim_year] = balance
            postings.append(CounterpartyPosting(party, counterparty, payment.project, payment.season, payment.part,
                side, payment.nominal_claim_yuan, payment.expected_payment_yuan, payment.verified_due_yuan,
                payment.actual_paid_yuan, balance,
                "conditional_receivable_account_not_historical_person" if party.startswith("conditional_counterparty:") else "supplied_party_identifier",
                payment.claim_year, payment.payment_year))
    return tuple(postings)


@dataclass(frozen=True, slots=True)
class InventoryAccount:
    operator_id: str
    year: int
    crop: str
    opening_inventory_kg: float
    harvest_kg: float
    sold_kg: float
    closing_inventory_kg: float
    disposition: str = "unsold_stock_retained_by_producer;no_storage_loss_or_stock_sale_condition"


def inventory_accounts(account: AnnualAccount, settlement: AnnualSettlement) -> tuple[InventoryAccount, ...]:
    rows: list[InventoryAccount] = []
    for crop, closing in sorted(settlement.closing_inventory_kg.items()):
        opening = account.opening_inventory_kg.get(crop, 0.0)
        harvest = sum(s.harvest_kg for s in settlement.seasons if s.crop == crop)
        sold = sum(s.sold_kg for s in settlement.seasons if s.crop == crop)
        assert abs(opening + harvest - sold - closing) < 1e-7, "inventory mass balance"
        rows.append(InventoryAccount(account.operator_id, account.year, crop, opening, harvest, sold, closing))
    return tuple(rows)


def carry_account(previous: AnnualAccount, settled: AnnualSettlement, next_account: AnnualAccount) -> AnnualAccount:
    assert previous.operator_id == next_account.operator_id, "inventory and prior claims remain with original party"
    assert next_account.year == settled.next_matching_year, "next account period must consume settlement year"
    claims = tuple(p for p in settled.payments if previous.operator_id in (p.payer, p.recipient)
        for amount in (p.verified_due_yuan if p.verified_due_yuan is not None else p.expected_payment_yuan,)
        if (amount is None or amount > 0)
        and (p.actual_paid_yuan is None or amount is None or amount > p.actual_paid_yuan))
    return replace(next_account, opening_inventory_kg=settled.closing_inventory_kg, prior_payments=claims)
