from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from core.fixed_rule_runtime import ActionIncomeAccount

if TYPE_CHECKING:
    from agents.farmer_agent import FarmerAgent


@dataclass(frozen=True, slots=True)
class OrganizationAccount:
    mode: Literal["self", "outsource", "transfer"]
    account: ActionIncomeAccount
    service_available: bool


def organization_accounts(farmer: FarmerAgent, actions=("maintain_grain", "expand_cash", "exit")) -> tuple[OrganizationAccount, ...]:
    alternatives: list[OrganizationAccount] = []
    previous_mode = getattr(farmer, "p3_service_mode", None)
    # Keep the real rights-holder identity while the existing method previews costs.
    try:
        for action in actions:
            if action == "exit":
                alternatives.append(OrganizationAccount("transfer", farmer.expected_fixed_rule_account("exit"), True))
                continue
            for mode in ("self", "outsource"):
                setattr(farmer, "p3_service_mode", mode)
                account = farmer.expected_fixed_rule_account(action)
                alternatives.append(OrganizationAccount(mode, account,
                    mode == "self" or account.service_area_mu > 0))
    finally:
        setattr(farmer, "p3_service_mode", previous_mode)
    return tuple(alternatives)


def best_crop_accounts(alternatives: tuple[OrganizationAccount, ...]) -> tuple[ActionIncomeAccount, ...]:
    selected: list[ActionIncomeAccount] = []
    for action in dict.fromkeys(item.account.action for item in alternatives):
        choices = [item.account for item in alternatives if item.account.action == action and item.service_available]
        feasible = [account for account in choices if account.feasible]
        selected.append(max(feasible or choices, key=lambda account: account.score))
    return tuple(selected)
