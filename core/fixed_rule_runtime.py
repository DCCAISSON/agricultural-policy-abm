from __future__ import annotations
def crop_history_group(g,c):
    if not (g>=0 and c>=0): return 'unknown'
    if g+c<1e-7: return 'unplanted'
    if g<1e-7: return 'cash'
    if c<1e-7: return 'grain'
    return 'mixed_lower' if g<c else 'mixed_upper'


def continuous_crop_target(history, prior, share, u):
    """Conditional share among historical planters; leaving production is not a zero grain share."""
    import numpy as np
    group='mixed' if prior.startswith('mixed') else prior
    if group not in history: return None
    x,y=history[group]
    if group=='mixed':
        distance=np.abs(x-share)
        k=int(np.ceil(np.sqrt(len(x))))
        y=y[distance<=np.partition(distance,k-1)[k-1]]
    planted=np.sort(y[np.isfinite(y)])
    return float(planted[min(int(u*len(planted)),len(planted)-1)]) if len(planted) else None


def continuous_farm_plan(a, land, rights, cash, rent, years_inputs, trading):
    """Annual operating surplus on physical mu and crop-mu; no sale proceeds finance planting."""
    import numpy as np
    from scipy.optimize import linprog
    grain_sale, cash_sale, kg, kc, lg, lc, wage, aid = years_inputs
    ci = a.sowing_intensity
    # grain, cash crop, hired days, land out, land in, eligible cultivated rights
    gain = np.array([grain_sale-kg, cash_sale-kc, -wage, rent, -rent, aid])
    bounds = [(0,None),(0,None),(0,None),(0,rights if trading else 0),
              (0,a.area_limit if trading else 0),(0,None)]
    rows = [[1/ci,1/ci,0,1,-1,0], [lg,lc,-1,0,0,0],
            [kg,kc,wage,0,rent,0], [-1/ci,-1/ci,0,0,0,1],
            [0,0,0,1,0,1],[1/ci,1/ci,0,0,0,0]]
    target=getattr(a,'crop_mix_target',None)
    composition={} if target is None else dict(A_eq=[[1-target,-target,0,0,0,0]],b_eq=[0])
    penalty=getattr(a,'crop_adjustment_yuan',None)
    if target is not None and penalty is not None:
        if not np.isfinite(penalty) or penalty<0: raise ValueError('invalid crop adjustment cost')
        # The historical reference is a choice cost, not a cash payment or a target equality.
        gain=np.r_[gain,-penalty,-penalty]
        rows=[r+[0,0] for r in rows]
        bounds.extend([(0,None),(0,None)])
        composition=dict(A_eq=[[1-target,-target,0,0,0,0,-1,1]],b_eq=[0])
    sol = linprog(-gain,**composition,A_ub=rows,b_ub=[land,a.family_days,max(0,cash),0,rights,a.area_limit],
                  bounds=bounds,method='highs')
    if not sol.success:
        raise ValueError(sol.message)
    g,c,h,out,inside,eligible = np.maximum(sol.x[:6],0)
    return dict(grain=g,crop=c,hired=h,offer=max(0,out-inside),demand=max(0,inside-out),
                cultivated=(g+c)/ci,eligible=eligible,cost=g*kg+c*kc+h*wage,
                expected=-sol.fun, land_value=max(0,-sol.ineqlin.marginals[0]),
                cash_shadow=max(0,-sol.ineqlin.marginals[2]),labor_shadow=max(0,-sol.ineqlin.marginals[1]))


from dataclasses import asdict, dataclass, replace
from copy import copy
import math
from pathlib import Path
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Final, Literal

import yaml

if TYPE_CHECKING:
    from agents.farmer_agent import FarmerAgent
    from agents.enterprise_agent import AgriculturalEnterpriseAgent
    from core.model import MainModel
    from modules.land_patch import LandPatch


MARKET_ORDER: Final = (
    "government",
    "environment_price_update",
    "technology_update",
    "climate_update",
    "land_market_planning",
    "village_collective",
    "land_rental_market",
    "labor_planning",
    "service_market",
    "labor_market",
    "agent_production",
    "annual_collection",
    "population_dynamics",
    "enterprise_demography",
)

OperatorKind = Literal["household", "scale"]

# 2009-10 frozen NFPS cohort; endpoint-gap proxy, not an estimated structural rate.
NFPS_HOUSEHOLD_ADJUSTMENT_PROXY: Final = 0.201342766


@dataclass(frozen=True, slots=True)
class ActionIncomeAccount:
    action: str
    crop: str
    area_mu: float
    output_kg: float
    sales_yuan: float
    support_yuan: float
    rent_out_yuan: float
    off_farm_yuan: float
    common_income_yuan: float
    production_cost_yuan: float
    hired_labor_cost_yuan: float
    service_cost_yuan: float
    rent_in_yuan: float
    management_cost_yuan: float
    conversion_cost_yuan: float
    risk_penalty_yuan: float
    family_labor_days: float
    hired_labor_days: float
    off_farm_days: float
    service_area_mu: float
    working_capital_available: float
    hired_labor_available_days: float
    grain_area_mu: float = 0.0
    grain_output_kg: float = 0.0
    cash_output_kg: float = 0.0

    @property
    def cash_cost(self) -> float:
        return (self.production_cost_yuan + self.service_cost_yuan + self.rent_in_yuan
                + self.management_cost_yuan + self.conversion_cost_yuan)

    @property
    def working_capital_required(self) -> float:
        return self.cash_cost

    @property
    def net_income(self) -> float:
        return (self.sales_yuan + self.support_yuan + self.rent_out_yuan
                + self.off_farm_yuan + self.common_income_yuan - self.cash_cost)

    @property
    def score(self) -> float:
        return self.net_income - self.risk_penalty_yuan

    @property
    def constraint_reason(self) -> str:
        reasons = []
        if self.cash_cost > self.working_capital_available + 1e-8:
            reasons.append("working_capital")
        if self.hired_labor_days > self.hired_labor_available_days + 1e-8:
            reasons.append("hired_labor")
        return "|".join(reasons)

    @property
    def feasible(self) -> bool:
        return not self.constraint_reason


def action_income_enabled(config) -> bool:
    return config.get("runtime_fixed_rule", {}).get("accounting_basis") == "action_conditioned_v1"


def preproduction_cash_enabled(config) -> bool:
    cfg = config.get("runtime_fixed_rule", {}).get("preproduction_cash", {})
    mode = cfg.get("mode", "off")
    if mode == "off":
        return False
    if mode != "reserved_no_credit":
        raise ValueError("Unknown preproduction_cash mode")
    if not action_income_enabled(config):
        raise ValueError("preproduction_cash requires action_conditioned_v1")
    if cfg.get("funds_basis") != "capital_as_liquid_operating_balance":
        raise ValueError("preproduction_cash requires an explicit liquid-capital assumption")
    if not config.get("production", {}).get("cost_structure", {}).get("use_detailed_cost", True):
        raise ValueError("preproduction_cash requires quantity-based detailed costs")
    return True


def production_area(operator) -> float:
    """Seasonal cultivation is separate from ownership and leased area."""
    if hasattr(operator, "_fixed_rule_preview_area"):
        return float(operator._fixed_rule_preview_area)
    account = getattr(operator, "last_preproduction_cash_account", {})
    if account.get("step") == operator.model.step_count and preproduction_cash_enabled(operator.model.config):
        return account["cultivated_area_mu"]
    return operator.get_operated_land_area()


def production_cash_quote(operator, area):
    """Current inputs and labor at the bid ceiling, without future receipts."""
    preview = copy(operator)
    preview._fixed_rule_preview_area = area
    preview.fixed_rule_labor_settled = False
    preview.hired_labor_cost = preview.hired_labor_days = 0.0
    market = operator.model.labor_market
    if hasattr(operator, "crop_portfolio"):
        nonlabor = days = 0.0
        for crop, weight in operator.crop_portfolio.items():
            cost = preview._calculate_detailed_cost(crop, area * weight)
            nonlabor += cost - preview.hired_labor_cost
            days += preview.hired_labor_days
        management = operator.model.config.get("agents", {}).get("enterprise", {}).get("diseconomy", {}).get("supervision_cost_base", 50.0)
        nonlabor += area * float(management or 50.0)
        wage_ceiling = market.base_agricultural_wage * operator.efficiency_factor * 1.2
    else:
        if operator.planned_crop is None or area <= 0:
            nonlabor = days = 0.0
        else:
            cost = preview._calculate_detailed_cost(operator.planned_crop, area)
            nonlabor = cost["cash_total_cost"] - cost["hired_labor_cost"]
            days = preview.requested_hired_labor_days
        wage_ceiling = market.base_agricultural_wage * (0.9 + 0.2 * operator.farming_ability)
    values = (nonlabor, days, wage_ceiling)
    if any(not math.isfinite(float(v)) or v < 0 for v in values):
        raise ValueError("Non-finite or negative production funding quote")
    return float(nonlabor), float(days), float(wage_ceiling)


def prepare_preproduction_cash(model):
    if not preproduction_cash_enabled(model.config):
        return
    for patch in model.land_patches:
        if getattr(patch, 'preproduction_cash_step', None) != model.step_count:
            patch.preproduction_cash_step = model.step_count
            patch.preproduction_cultivated_area_mu = 0.0
            patch.preproduction_uncultivated_area_mu = patch.area
    initial_patches = {id(a.patch) for a in getattr(model, "initial_paid_lease_accounts", ())}
    rent_ledger = getattr(model, "preproduction_rent_ledger", [])
    for patch in model.land_patches:
        tenant = patch.leased_to
        if id(patch) in initial_patches or tenant is None or tenant is rental_rights_holder(patch):
            continue
        if not any(row['patch'] is patch and row['tenant'] is tenant for row in rent_ledger):
            rent_ledger.append(dict(patch=patch, tenant=tenant, recipient=rental_rights_holder(patch),
                                    arrears_yuan=0.0, step=None))
    for row in rent_ledger:
        if row['step'] == model.step_count:
            continue
        patch = row['patch']
        current = max(0.0, float(patch.rental_price)) if patch.leased_to is row['tenant'] else 0.0
        row.update(step=model.step_count, opening_arrears_yuan=row['arrears_yuan'],
                   current_due_yuan=current, total_due_yuan=row['arrears_yuan'] + current,
                   payment_yuan=0.0, status='unallocated')
    model.preproduction_rent_ledger = rent_ledger
    operators = [*model.farmers, *model.agricultural_enterprises]
    for household in getattr(model, 'initial_household_cohort', ()):
        if household not in operators:
            operators.append(household)
    for lease in getattr(model, "initial_paid_lease_accounts", ()):
        if lease.tenant not in operators:
            operators.append(lease.tenant)
    for row in rent_ledger:
        if row['total_due_yuan'] > 0 and row['tenant'] not in operators:
            operators.append(row['tenant'])
    for operator in operators:
        previous = getattr(operator, "last_preproduction_cash_account", {})
        if previous.get("step") == model.step_count:
            continue
        opening = float(operator.capital)
        if not math.isfinite(opening):
            raise ValueError("preproduction capital must be finite")
        reserve = float(getattr(operator, 'household_cash_reserve_yuan', 0.0))
        if not math.isfinite(reserve) or reserve < 0:
            raise ValueError('household_cash_reserve_yuan must be finite and nonnegative')
        available = max(0.0, opening - reserve)
        held_area = float(operator.get_operated_land_area())
        if hasattr(operator, "crop_portfolio"):
            grain = max(next(v for k, v in operator.crop_portfolio.items() if k.value == "Grain"), operator._get_village_grain_requirement())
            operator.crop_portfolio = {k: grain if k.value == "Grain" else 1.0 - grain for k in operator.crop_portfolio}
        initial = [a for a in getattr(model, "initial_paid_lease_accounts", ()) if a.tenant is operator]
        current_initial = sum(a.annual_rent_yuan for a in initial if model.step_count <= a.end_step)
        initial_due = sum(a.arrears_yuan + (a.annual_rent_yuan if model.step_count <= a.end_step else 0.0) for a in initial)
        ordinary_rows = [row for row in rent_ledger if row['tenant'] is operator]
        ordinary_rent = sum(row['total_due_yuan'] for row in ordinary_rows)
        account = {
            "step": model.step_count, "operator_id": str(operator.unique_id),
            "opening_signed_balance_yuan": opening, "opening_available_yuan": available,
            "household_cash_reserve_yuan": reserve,
            "held_area_mu": held_area, "cultivated_area_mu": held_area,
            "ordinary_rent_due_yuan": ordinary_rent, "initial_rent_due_yuan": initial_due,
            "current_rent_accrual_yuan": current_initial + sum(row['current_due_yuan'] for row in ordinary_rows),
            "opening_rent_arrears_yuan": initial_due - current_initial + sum(row['opening_arrears_yuan'] for row in ordinary_rows),
            "ordinary_rent_funding_gap_yuan": max(0.0, ordinary_rent - available),
            "status": "planning", "actual_paid_yuan": None,
        }
        operator.last_preproduction_cash_account = account
        rent_due = ordinary_rent + initial_due
        rent_ratio = min(1.0, available / rent_due) if rent_due > 0 else 0.0
        ordinary_reserved = ordinary_rent * rent_ratio
        initial_reserved = initial_due * rent_ratio
        rent_reserved = ordinary_reserved + initial_reserved
        for row in ordinary_rows:
            paid = row['total_due_yuan'] * rent_ratio
            row.update(payment_yuan=paid, arrears_yuan=row['total_due_yuan'] - paid, status='reserved')
        conversion = float(getattr(operator, "fixed_rule_conversion_expense", 0.0))
        planned_conversion = conversion
        area = held_area
        if not hasattr(operator, "crop_portfolio") and operator.planned_crop is None:
            area = 0.0
        if operator not in model.farmers and operator not in model.agricultural_enterprises:
            area = 0.0
        if getattr(operator, '_skip_remaining_step_this_cycle', False):
            area = 0.0
        nonlabor, days, wage = production_cash_quote(operator, area)
        full_requirement = rent_reserved + conversion + nonlabor + days * wage
        # Proportional contraction, re-quoting scale discontinuities each time.
        # This is a feasible plan, not a global optimum across areas or crops.
        for _ in range(64):
            if rent_reserved + conversion + nonlabor + days * wage <= available + 1e-8:
                break
            if available <= rent_reserved + conversion or area <= 1e-10:
                area = conversion = 0.0
            else:
                area *= (available - rent_reserved - conversion) / (nonlabor + days * wage)
            nonlabor, days, wage = production_cash_quote(operator, area)
        else:
            area = conversion = 0.0
            nonlabor, days, wage = production_cash_quote(operator, area)
        operator.fixed_rule_conversion_expense = conversion
        reserved = rent_reserved + conversion + nonlabor + days * wage
        account.update(
            cultivated_area_mu=area, uncultivated_area_mu=held_area - area,
            initial_rent_reserved_yuan=initial_reserved, rent_reserved_yuan=rent_reserved,
            ordinary_rent_reserved_yuan=ordinary_reserved,
            ordinary_rent_arrears_yuan=ordinary_rent - ordinary_reserved,
            closing_rent_arrears_yuan=rent_due - rent_reserved,
            planned_conversion_yuan=planned_conversion, cancelled_conversion_yuan=planned_conversion - conversion,
            full_plan_required_yuan=full_requirement, input_management_reserved_yuan=nonlabor,
            conversion_reserved_yuan=conversion, labor_reserved_yuan=days * wage,
            hired_days_planned=days, wage_ceiling_yuan=wage, service_reserved_yuan=0.0,
            initial_reserved_yuan=reserved, committed_yuan=reserved,
            uncommitted_yuan=max(0.0, available - reserved), status="reserved",
        )
        for patch in model.land_patches:
            controls_patch = patch.current_operator is operator or (
                patch.current_operator is None and (patch.leased_to is operator or (
                    patch.leased_to is None and patch.owner is operator)))
            if controls_patch:
                patch.preproduction_cash_step = model.step_count
                patch.preproduction_cultivated_area_mu = patch.area * area / held_area if held_area > 0 else 0.0
                patch.preproduction_uncultivated_area_mu = patch.area - patch.preproduction_cultivated_area_mu


def settle_preproduction_cash(operator, actual_expense):
    if not preproduction_cash_enabled(operator.model.config):
        return
    account = getattr(operator, "last_preproduction_cash_account", {})
    if account.get("step") != operator.model.step_count:
        raise RuntimeError("Production settlement without preproduction funding")
    if account.get("status") == "settled":
        raise RuntimeError("Duplicate production cash settlement")
    expense = float(actual_expense)
    if not math.isfinite(expense) or expense < 0:
        raise ValueError("Invalid actual production expense")
    if expense > account["committed_yuan"] + 1e-6:
        raise RuntimeError("Actual cost exceeds its preproduction commitment")
    account.update(
        actual_paid_yuan=expense, released_yuan=account["committed_yuan"] - expense,
        total_released_yuan=account["initial_reserved_yuan"] + account['service_reserved_yuan'] - expense,
        expected_actual_difference_yuan=expense - account["initial_reserved_yuan"],
        capital_before_settlement_yuan=float(operator.capital),
        unreceived_income_used_yuan=0.0, status="settled",
    )
    account['actual_hired_labor_yuan'] = float(getattr(operator, 'hired_labor_cost', 0.0))
    account['actual_rent_yuan'] = float(getattr(operator, 'land_rental_cost',
                                              getattr(operator, 'land_rental_expense', 0.0)))
    account['actual_service_yuan'] = float(getattr(operator, 'service_fee_expense',
                                                 getattr(operator, 'service_expense', 0.0)))
    account['actual_conversion_yuan'] = float(getattr(operator, 'fixed_rule_conversion_expense', 0.0))
    account['actual_input_management_yuan'] = expense - sum(account[k] for k in (
        'actual_hired_labor_yuan', 'actual_rent_yuan', 'actual_service_yuan', 'actual_conversion_yuan'))
    for row in getattr(operator.model, 'preproduction_rent_ledger', []):
        if row['tenant'] is operator and row['step'] == operator.model.step_count:
            row['status'] = 'paid'


@dataclass(frozen=True, slots=True)
class InitialTenureAccount:
    operated_area_mu: float
    representation_weight_q: float
    observed_rental_income_yuan: float
    observed_rental_expense_yuan: float
    observed_production_cost_yuan: float
    observed_net_income_yuan: float
    assigned_rental_income_yuan: float = 0.0
    assigned_rental_expense_yuan: float = 0.0
    rights_area_mu: float = 0.0


@dataclass(frozen=True, slots=True)
class InitialPaidLeaseAccount:
    patch: LandPatch
    tenant: AgriculturalEnterpriseAgent
    recipient: FarmerAgent
    annual_rent_yuan: float
    end_step: int
    settlement_step: int = 0
    amount_due_yuan: float = 0.0
    amount_paid_yuan: float = 0.0
    opening_arrears_yuan: float = 0.0
    arrears_yuan: float = 0.0


def settle_initial_paid_rent(enterprise: AgriculturalEnterpriseAgent, available_cash: float) -> float:
    """Allocate available cash across due rent and arrears, never future instalments."""
    model = enterprise.model
    step = int(model.step_count)
    accounts = tuple(getattr(model, "initial_paid_lease_accounts", ()))
    pending = [a for a in accounts if a.tenant is enterprise and a.settlement_step < step]
    payable = sum(a.arrears_yuan + (a.annual_rent_yuan if step <= a.end_step else 0.0) for a in pending)
    ratio = min(1.0, max(0.0, available_cash) / payable) if payable > 0 else 0.0
    updated = []
    paid_total = 0.0
    for account in accounts:
        if account.tenant is not enterprise or account.settlement_step >= step:
            updated.append(account)
            continue
        due = account.annual_rent_yuan if step <= account.end_step else 0.0
        debt = account.arrears_yuan + due
        paid = debt * ratio
        paid_total += paid
        updated.append(replace(account, settlement_step=step, amount_due_yuan=due,
                               amount_paid_yuan=paid, opening_arrears_yuan=account.arrears_yuan,
                               arrears_yuan=max(0.0, debt - paid)))
    model.initial_paid_lease_accounts = tuple(updated)
    own = [a for a in updated if a.tenant is enterprise]
    enterprise.initial_rent_due_yuan = sum(a.amount_due_yuan for a in own)
    enterprise.initial_rent_paid_yuan = sum(a.amount_paid_yuan for a in own)
    enterprise.initial_rent_arrears_yuan = sum(a.arrears_yuan for a in own)
    return paid_total


def initialize_initial_household_tenure(model: MainModel) -> None:
    """Assign initial household rights and annual obligations before year one."""
    if model.step_count != 0 or hasattr(model, "initial_household_cohort"):
        message = "Initial household tenure must be assigned once before simulation."
        raise RuntimeError(message)
    mode = model.config.get("runtime_fixed_rule", {}).get("initial_tenure_mode", "legacy")
    if mode not in ("legacy", "A", "B_star_1y", "B_star_5y"):
        raise ValueError("Unknown initial_tenure_mode")
    paid = mode in ("B_star_1y", "B_star_5y")
    for patch in model.land_patches:
        if patch.owner in model.farmers:
            if patch.leased_to not in (None, patch.owner) or patch.current_operator is not patch.owner:
                if paid:
                    base_rent = float(model.config.get("agents", {}).get("farmer", {}).get("land_transfer", {}).get("base_rent_price", 420.0))
                    if (patch.leased_to in model.agricultural_enterprises
                            and patch.current_operator is patch.leased_to
                            and getattr(patch, "operational_transfer_channel", "") == "author_paid_initial_lease"
                            and patch.lease_contract_years == (1 if mode == "B_star_1y" else 5)
                            and patch.rental_price > 0
                            and abs(patch.rental_price - float(patch.area) * base_rent) < 1e-8
                            and patch.contracted_rental_price == patch.rental_price):
                        continue
                    raise RuntimeError("Invalid paid initial household lease")
                message = "Initial self-operation baseline requires no historical household leases."
                raise RuntimeError(message)
    model.initial_household_cohort = tuple(model.farmers)
    for farmer in model.initial_household_cohort:
        area = float(farmer.get_operated_land_area())
        rights_area = sum(float(p.area) for p in model.land_patches if p.owner is farmer)
        rent_income = sum(float(p.rental_price) for p in model.land_patches
                          if p.owner is farmer and p.leased_to not in (None, farmer))
        farmer.initial_tenure_account = InitialTenureAccount(
            area, float(getattr(farmer, "representation_weight_q", 1.0)),
            float(farmer.land_rental_income), float(farmer.land_rental_expense),
            float(farmer.production_cost), float(farmer.cash_net_income),
            assigned_rental_income_yuan=rent_income, rights_area_mu=rights_area,
        )
        farmer.initial_rental_rights_area_mu = rights_area
        farmer.land_rental_income = rent_income
        farmer.land_rental_expense = 0.0
    for patch in model.land_patches:
        if patch.owner in model.initial_household_cohort:
            patch.initial_rental_rights_holder = patch.owner
    if paid:
        model.initial_paid_lease_accounts = tuple(
            InitialPaidLeaseAccount(patch, patch.leased_to, patch.initial_rental_rights_holder,
                                    float(patch.contracted_rental_price), int(patch.lease_contract_years))
            for patch in model.land_patches
            if getattr(patch, "operational_transfer_channel", "") == "author_paid_initial_lease"
        )
        for enterprise in model.agricultural_enterprises:
            enterprise.calculate_land_rental_cost()


def rental_rights_holder(patch: LandPatch) -> FarmerAgent | AgriculturalEnterpriseAgent | None:
    return getattr(patch, "initial_rental_rights_holder", patch.owner)


def expected_support(operator, grain_area: float, actor_type: str,
                     *, service_cost_yuan: float = 0.0, service_area_mu: float = 0.0) -> float:
    from core.p3_project_contracts import configured_program
    from core.p3_project_claims import expected_support as project_expected_support
    program = configured_program(operator.model)
    if program is not None:
        value = project_expected_support(operator, grain_area, program)
        if actor_type == "farmer" and grain_area > 0 and program.rule in ("family_service", "area_service_mix"):
            nominal = min(0.3 * service_cost_yuan,
                          100 * float(program.service_2019_to_model) * min(grain_area, service_area_mu))
            value += nominal * program.expected_payment_ratio(str(operator.unique_id), "service", nominal)
        return value
    environment = copy(operator.model.environment)
    base = environment.calculate_subsidy_for_actor(grain_area, actor_type=actor_type)
    scale = environment.calculate_scale_subsidy(grain_area, actor_type=actor_type)
    return float(base["cash"] + scale["scale_subsidy"])


def expected_land_cash(operator, area: float, terminal: bool) -> tuple[float, float]:
    patches = getattr(operator, "_fixed_rule_bid_land_candidates", operator.model.land_patches)
    rent_out = sum(p.rental_price for p in patches if rental_rights_holder(p) is operator and p.leased_to not in (None, operator))
    rent_in = sum(p.rental_price for p in patches if p.leased_to is operator and rental_rights_holder(p) is not operator)
    rent_per_mu = float(operator.model.environment.land_rental_price)
    if terminal:
        own_operated = sum(p.area for p in patches if rental_rights_holder(p) is operator and p.current_operator is operator and p.leased_to in (None, operator))
        return float(rent_out + own_operated * rent_per_mu), 0.0
    added = max(0.0, area - operator.get_operated_land_area())
    return float(rent_out), float(rent_in + added * rent_per_mu)


def choose_action_income(operator, accounts: tuple[ActionIncomeAccount, ...]) -> str:
    feasible = [account for account in accounts if account.feasible]
    if not feasible and getattr(operator.model, "initial_paid_lease_accounts", ()):
        terminal = next((a for a in accounts if a.action == "contract" and a.area_mu == 0), None)
        if terminal is not None and any(a.tenant is operator for a in operator.model.initial_paid_lease_accounts):
            return terminal.action
    if not feasible:
        raise RuntimeError(f"No feasible annual action for {operator.unique_id}")
    best = max(feasible, key=lambda account: account.score)
    prior_action = getattr(operator, "fixed_rule_previous_action", "maintain_grain")
    shares = getattr(operator, "household_candidate_grain_shares", None)
    if shares is not None and operator.household_previous_grain_share is not None:
        prior_action = next(key for key, share in shares.items() if share == operator.household_previous_grain_share)
    previous = next((a for a in feasible if a.action == prior_action), None)
    margin = fixed_rule_operator_inertia(operator)
    if previous is not None and best.score - previous.score <= margin:
        return previous.action
    return best.action


def submit_action_income_bids(operator, max_added_area: float) -> None:
    market = operator.model.land_clearing_house
    action = operator.fixed_rule_current_action
    if action in {"exit", "contract"} or max_added_area <= 0:
        return
    # Tenure is unchanged until clear_market, after all operators submit bids.
    operator._fixed_rule_bid_land_candidates = tuple(
        p for p in operator.model.land_patches
        if rental_rights_holder(p) is operator or p.leased_to is operator
    )
    try:
        base = operator.expected_fixed_rule_account(action)
        current = base
        for offer in sorted(market.supply_offers, key=lambda row: str(row['patch'].get_patch_id())):
            parcel = offer['patch']
            if parcel.owner is operator or parcel.leased_to is not None:
                continue
            new_area = current.area_mu + parcel.area
            if new_area > base.area_mu + max_added_area + 1e-8:
                continue
            candidate = operator.expected_fixed_rule_account(action, new_area)
            candidate = replace(candidate, rent_in_yuan=current.rent_in_yuan + offer['reservation_price'])
            reason = candidate.constraint_reason
            if candidate.score <= current.score:
                reason = '|'.join(filter(None, (reason, 'nonpositive_incremental_income')))
            count = len(market.demand_offers)
            if not reason:
                market.add_demand_offer(parcel, operator, offer['reservation_price'])
                if len(market.demand_offers) == count:
                    reason = 'clearing_house_scale_gate'
            row = {
                'year': operator.model.simulation_base_year + operator.model.step_count,
                'operator_id': str(operator.unique_id), 'action': action,
                'patch_id': str(parcel.get_patch_id()), 'bid_yuan': offer['reservation_price'],
                'submitted': len(market.demand_offers) > count, 'reason': reason,
            }
            for prefix, account in (('before', current), ('after', candidate)):
                row.update({f'{prefix}_{key}': value for key, value in asdict(account).items()})
                row[f'{prefix}_net_income'] = account.net_income
                row[f'{prefix}_working_capital_required'] = account.working_capital_required
            operator.fixed_rule_bid_accounts.append(row)
            if not reason:
                current = candidate
        operator.fixed_rule_desired_area = current.area_mu
    finally:
        del operator._fixed_rule_bid_land_candidates


@dataclass(frozen=True, slots=True)
class FixedRuleThresholds:
    min_land_capacity: float
    min_labor_capacity: float
    min_service_capacity: float


@dataclass(frozen=True, slots=True)
class FixedRuleState:
    operator_id: str
    operator_type: OperatorKind
    previous_action: str
    grain_return: float
    cash_return: float
    exit_return: float
    land_capacity: float
    labor_capacity: float
    service_capacity: float


def load_experiment_overlay(path: str | Path) -> dict[str, Any]:  # noqa: DICT_OK
    overlay_path = Path(path)
    with overlay_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"experiment overlay must be a mapping: {overlay_path}")  # noqa: GENERIC_ERR_OK
    return payload


def apply_experiment_overlay(config: dict[str, Any], overlay: dict[str, Any]) -> None:
    for section in ("experiment", "policy", "mechanism_switches"):
        value = overlay.get(section)
        if isinstance(value, dict):
            config.setdefault(section, {}).update(value)
    data = overlay.get("data")
    if isinstance(data, dict):
        entrant_library = data.get("entrant_feature_library")
        if isinstance(entrant_library, str) and entrant_library:
            entry = config.setdefault("population_dynamics", {}).setdefault("entry_mechanism", {})
            entry["entrant_feature_library"] = entrant_library
    decision = overlay.get("decision_rule")
    if isinstance(decision, dict) and decision.get("type") == "fixed_expected_net_return":
        config["runtime_fixed_rule"] = decision
        config["runtime_smoke"] = overlay.get("runtime_smoke", {})
        config["market_sequence"] = overlay.get("market_sequence", {})
        config["disabled_mechanisms"] = overlay.get("disabled_mechanisms", [])


def thresholds_from_rule(rule: dict[str, Any]) -> FixedRuleThresholds:
    raw = rule.get("feasible_thresholds", {})
    if not isinstance(raw, dict):
        raw = {}
    return FixedRuleThresholds(
        min_land_capacity=float(raw.get("min_land_capacity", 0.5) or 0.5),
        min_labor_capacity=float(raw.get("min_labor_capacity", 0.35) or 0.35),
        min_service_capacity=float(raw.get("min_service_capacity", 0.5) or 0.5),
    )


def choose_fixed_rule_action(
    state: FixedRuleState,
    inertia_margin: float,
    thresholds: FixedRuleThresholds,
) -> str:
    cash_feasible = (
        state.land_capacity >= thresholds.min_land_capacity
        and state.labor_capacity >= thresholds.min_labor_capacity
        and state.service_capacity >= thresholds.min_service_capacity
    )
    cash_return = state.cash_return if cash_feasible else float("-inf")
    best_action = "expand_cash" if cash_return > state.grain_return else "maintain_grain"
    best_return = max(state.grain_return, cash_return)
    terminal_action = "contract" if state.operator_type == "scale" else "exit"
    if state.exit_return > best_return:
        best_action = terminal_action
        best_return = state.exit_return
    previous_return = state.grain_return if state.previous_action == "maintain_grain" else cash_return
    if best_return - previous_return <= inertia_margin:
        return state.previous_action
    return best_action


def fixed_rule_enabled(config: dict[str, Any]) -> bool:
    rule = config.get("runtime_fixed_rule", {})
    return isinstance(rule, dict) and rule.get("type") == "fixed_expected_net_return"


def fixed_rule_inertia_margin(config: dict[str, Any]) -> float:
    rule = config.get("runtime_fixed_rule", {})
    if not isinstance(rule, dict):
        return 0.0
    return float(rule.get("minimum_inertia_margin_yuan_per_mu", 0.0) or 0.0)


def fixed_rule_operator_inertia(operator) -> float:
    config = operator.model.config
    rule = config.get("runtime_fixed_rule", {})
    if rule.get("national_v1") is True and hasattr(operator, "crop_portfolio"):
        # Organization persistence is a separate scenario assumption, not a household estimate.
        margin = rule.get("organization_inertia_margin_yuan_per_mu", 50.0)
        if type(margin) not in (float, int) or not math.isfinite(margin) or margin < 0:
            raise ValueError("organization_inertia_margin_yuan_per_mu must be finite and nonnegative")
        return float(margin) * operator.get_operated_land_area()
    margin = fixed_rule_inertia_margin(config)
    if config.get("runtime_fixed_rule", {}).get("inertia_basis") == "yuan_per_mu":
        return margin * operator.get_operated_land_area()
    return margin


def state_from_operator(operator: Any, operator_type: OperatorKind) -> FixedRuleState:
    operator_id = str(getattr(operator, "unique_id", ""))
    previous_action = str(getattr(operator, "fixed_rule_previous_action", "maintain_grain"))
    return FixedRuleState(
        operator_id=operator_id,
        operator_type=operator_type,
        previous_action=previous_action,
        grain_return=float(getattr(operator, "planting_net_income", 0.0) or 0.0),
        cash_return=float(getattr(operator, "cash_net_income", 0.0) or 0.0),
        exit_return=float(getattr(operator, "fixed_rule_exit_return", 0.0) or 0.0),
        land_capacity=float(getattr(operator, "fixed_rule_land_capacity", 1.0) or 0.0),
        labor_capacity=float(getattr(operator, "fixed_rule_labor_capacity", 1.0) or 0.0),
        service_capacity=float(getattr(operator, "fixed_rule_service_capacity", 1.0) or 0.0),
    )


def enact_fixed_rule_action(operator: Any, action: str, operator_type: OperatorKind) -> bool:
    if hasattr(operator, "model") and action_income_enabled(operator.model.config):
        paid_initial_mode = operator.model.config.get("runtime_fixed_rule", {}).get("initial_tenure_mode") in {"B_star_1y", "B_star_5y"}
        operator.fixed_rule_previous_action = operator.fixed_rule_current_action = action
        operator._skip_remaining_step_this_cycle = False
        operator.fixed_rule_exit_pending = action == "exit"
        operator.fixed_rule_participating = action != "exit"
        if hasattr(operator, "household_candidate_grain_shares"):
            operator.household_planned_grain_share = operator.household_candidate_grain_shares.get(action)
        if action in {"exit", "contract"}:
            for patch in list(operator.model.land_patches):
                if (paid_initial_mode and patch.leased_to is operator and patch.lease_contract_years > 0
                        and getattr(patch, "operational_transfer_channel", "") == "author_paid_initial_lease"):
                    patch.current_operator = None
                    patch.current_crop = None
                    patch.last_yield = 0.0
                    continue
                if hasattr(patch, "initial_rental_rights_holder"):
                    if rental_rights_holder(patch) is not operator and (
                        patch.leased_to is operator or patch.current_operator is operator
                    ):
                        patch.end_lease()
                elif patch.leased_to is operator and patch.owner is not operator:
                    patch.end_lease()
        elif paid_initial_mode:
            for patch in operator.model.land_patches:
                if (patch.leased_to is operator and patch.current_operator is None and patch.lease_contract_years > 0
                        and getattr(patch, "operational_transfer_channel", "") == "author_paid_initial_lease"):
                    patch.current_operator = operator
        if paid_initial_mode:
            operator.model.invalidate_land_tenure_caches()
        if action == "exit" and (operator_type == "household" or not paid_initial_mode):
            _set_household_transfer_out(operator)
            operator.planned_crop = None
            operator.agricultural_labor = 0.0
            operator.off_farm_labor = operator.total_labor_units
        else:
            if operator_type == "household":
                from agents.farmer_agent import CropType
                operator.planned_crop = CropType.CASH_CROP if action == "expand_cash" else CropType.GRAIN
                if action.startswith("portfolio_"):
                    operator.planned_crop = CropType.MIXED
                operator.last_crop_choice = operator.planned_crop
            else:
                _set_planned_crop(operator, "cash" if action == "expand_cash" else "grain")
            if operator_type == "scale":
                operator.fixed_rule_scale_decision = "contract" if action in {"contract", "exit"} else "maintain"
                operator.target_expansion_area = 0.0
                if action == "expand_cash":
                    enum = _resolve_crop_enum(operator)
                    share = operator._get_village_grain_requirement()
                    operator.crop_portfolio = {enum.GRAIN: share, enum.CASH_CROP: 1.0 - share}
        return False
    setattr(operator, "fixed_rule_previous_action", action)
    setattr(operator, "fixed_rule_current_action", action)
    if action == "maintain_grain":
        _set_planned_crop(operator, "grain")
        setattr(operator, "fixed_rule_participating", True)
        return False
    if action == "expand_cash":
        _set_planned_crop(operator, "cash")
        setattr(operator, "fixed_rule_participating", True)
        if operator_type == "scale":
            setattr(operator, "fixed_rule_scale_decision", "expand")
            _set_scale_expansion(operator, 1.0)
        return False
    if action == "contract":
        _set_planned_crop(operator, "grain")
        setattr(operator, "fixed_rule_participating", True)
        setattr(operator, "fixed_rule_scale_decision", "contract")
        _set_scale_expansion(operator, 0.0)
        return False
    if action == "exit":
        setattr(operator, "fixed_rule_participating", False)
        setattr(operator, "fixed_rule_exit_pending", True)
        setattr(operator, "_skip_remaining_step_this_cycle", True)
        _set_household_transfer_out(operator)
        return False
    return False


def build_fixed_rule_trace_row(state: FixedRuleState, action: str, year: int) -> dict[str, Any]:  # noqa: DICT_OK
    return {
        "year": int(year),
        "operator_id": state.operator_id,
        "operator_type": state.operator_type,
        "previous_action": state.previous_action,
        "action": action,
        "grain_return": state.grain_return,
        "cash_return": state.cash_return,
        "exit_return": state.exit_return,
        "land_capacity": state.land_capacity,
        "labor_capacity": state.labor_capacity,
        "service_capacity": state.service_capacity,
    }


def _national_v1_household_choice(operator, rule, previous):
    from core.p3_organization import best_crop_accounts, organization_accounts
    from core.p3_project_contracts import configured_program

    rate = rule.get("household_crop_adjustment_rate", NFPS_HOUSEHOLD_ADJUSTMENT_PROXY)
    if type(rate) not in (float, int) or not 0 <= rate <= 1:
        raise ValueError("household_crop_adjustment_rate must be in [0,1]")
    shares = {"maintain_grain": 1.0, "expand_cash": 0.0}
    if previous is not None:
        if not math.isfinite(previous) or not 0 <= previous <= 1:
            raise ValueError(f"Invalid previous grain share for {operator.unique_id}: {previous}")
        if previous not in shares.values():
            shares["portfolio_keep"] = previous
        for i in range(1, 10):
            if i / 10 not in shares.values():
                shares[f"portfolio_target_{i}"] = i / 10
    operator.household_previous_grain_share = previous
    operator.household_candidate_grain_shares = shares
    p3 = configured_program(operator.model) is not None
    modes = []

    def evaluate(actions):
        if p3:
            alternatives = organization_accounts(operator, actions)
            modes.extend(alternatives)
            return best_crop_accounts(alternatives)
        return tuple(operator.expected_fixed_rule_account(a) for a in actions)

    actions = ("maintain_grain", "expand_cash", "exit", *list(shares)[2:])
    accounts = evaluate(actions)
    feasible = [a for a in accounts if a.feasible]
    if not feasible:
        raise RuntimeError(f"No feasible annual action for {operator.unique_id}")
    keep = next((a for a in feasible if previous is not None and shares.get(a.action) == previous), None)
    target = max(feasible, key=lambda a: (a.score, a is keep))
    selected = target
    step_share = shares.get(target.action)
    if previous is not None and target.action != "exit":
        goal = shares[target.action]
        step_share = goal if rate == 1 else previous + rate * (goal - previous)
        # A previously priced identical share has the same feasible service modes and cash bill.
        step = next((a for a in accounts if shares.get(a.action) == step_share), None)
        if step is None:
            shares["portfolio_adjusted"] = step_share
            step, = evaluate(("portfolio_adjusted",))
            accounts += (step,)
        terminal = next(a for a in accounts if a.action == "exit")
        actual = [a for a in (keep, step, terminal) if a is not None and a.feasible]
        if not actual:
            raise RuntimeError(f"No feasible adjusted or exit action for {operator.unique_id}")
        selected = max(actual, key=lambda a: a.score)
    operator.fixed_rule_action_accounts = accounts
    if p3:
        operator.p3_organization_accounts = tuple(modes)
    return selected.action, dict(target_action=target.action, target_grain_share=shares.get(target.action),
                                 adjusted_candidate_grain_share=step_share, household_crop_adjustment_rate=rate,
                                 inertia_yuan=0.0, inertia_basis="household_partial_adjustment")


def evaluate_fixed_rule_operators(
    config: dict[str, Any],
    farmers: Iterable[Any],
    enterprises: Iterable[Any],
    year: int,
) -> tuple[list[dict[str, Any]], list[Any]]:
    if action_income_enabled(config):
        from core.p3_project_contracts import configured_program
        from core.p3_organization import best_crop_accounts, organization_accounts
        operators = [(f, "household") for f in sorted(farmers, key=_operator_sort_key)]
        operators += [(e, "scale") for e in sorted(enterprises, key=_operator_sort_key)
                      if not (getattr(e.model, "initial_paid_lease_accounts", ())
                              and getattr(e, "fixed_rule_exit_pending", False))]
        rule = config["runtime_fixed_rule"]
        v1 = rule.get("national_v1", False)
        if type(v1) is not bool:
            raise ValueError("national_v1 must be boolean")
        household_choices = {}
        for operator, kind in operators:
            actions = ("maintain_grain", "expand_cash", "exit" if kind == "household" else "contract")
            if kind == "household" and (v1 or rule.get("household_crop_portfolio", False)):
                share = getattr(operator, "household_actual_grain_share", None)
                if not hasattr(operator, "household_actual_grain_share") and operator.model.step_count <= 1:
                    share = getattr(operator.model, "household_initial_grain_shares", {}).get(str(operator.unique_id))
                if v1:
                    household_choices[id(operator)] = _national_v1_household_choice(operator, rule, share)
                    operator.fixed_rule_bid_accounts = []
                    continue
                shares = {"maintain_grain": 1.0, "expand_cash": 0.0}
                if share is not None:
                    if not math.isfinite(share) or not 0 <= share <= 1:
                        raise ValueError(f"Invalid initial grain share for {operator.unique_id}: {share}")
                    for name, value in (("keep", share), ("down", max(0.0, share - 0.10)),
                                        ("up", min(1.0, share + 0.10))):
                        if value not in shares.values():
                            shares[f"portfolio_{name}"] = value
                operator.household_previous_grain_share = share
                operator.household_candidate_grain_shares = shares
                actions += tuple(a for a in shares if a.startswith("portfolio_"))
            if kind == "household" and configured_program(operator.model) is not None:
                operator.p3_organization_accounts = organization_accounts(operator, actions)
                operator.fixed_rule_action_accounts = best_crop_accounts(operator.p3_organization_accounts)
            else:
                operator.fixed_rule_action_accounts = tuple(operator.expected_fixed_rule_account(a) for a in actions)
            operator.fixed_rule_bid_accounts = []
        rows = []
        for operator, kind in operators:
            accounts = operator.fixed_rule_action_accounts
            action, extra = (household_choices[id(operator)] if id(operator) in household_choices
                             else (choose_action_income(operator, accounts), {}))
            previous = getattr(operator, "fixed_rule_previous_action", "maintain_grain")
            operator.fixed_rule_selected_account = next(a for a in accounts if a.action == action)
            if kind == "household" and configured_program(operator.model) is not None:
                operator.p3_service_mode = next(item.mode for item in operator.p3_organization_accounts
                    if item.account is operator.fixed_rule_selected_account)
            operator.fixed_rule_conversion_expense = operator.fixed_rule_selected_account.conversion_cost_yuan
            rows.append({
                "year": year, "operator_id": str(operator.unique_id), "operator_type": kind,
                "previous_action": previous, "action": action,
                "grain_return": accounts[0].net_income, "cash_return": accounts[1].net_income,
                "exit_return": accounts[2].net_income,
                "inertia_yuan": fixed_rule_operator_inertia(operator),
                "inertia_basis": ("yuan_per_mu" if v1 and kind == "scale" else
                                  rule.get("inertia_basis", "legacy_annual_yuan")),
                "accounting_basis": "action_conditioned_v1",
                "farming_ability_parameter": getattr(operator, "farming_ability", None),
                **extra,
            })
            if hasattr(operator, "household_candidate_grain_shares"):
                rows[-1].update(previous_grain_share=operator.household_previous_grain_share,
                                selected_grain_share=operator.household_candidate_grain_shares.get(action))
            for account in accounts:
                prefix = account.action
                for field, value in asdict(account).items():
                    rows[-1][f"{prefix}_{field}"] = value
                rows[-1][f"{prefix}_grain_area_mu"] = account.grain_area_mu
                if kind == "household":
                    free_days = max(0.0, operator.total_labor_units - account.family_labor_days)
                    rows[-1][f"{prefix}_free_labor_days"] = free_days
                    rows[-1][f"{prefix}_off_farm_days"] = account.off_farm_days
                    rows[-1][f"{prefix}_unused_free_labor_days"] = max(0.0, free_days - account.off_farm_days)
        for (operator, kind), row in zip(operators, rows):
            enact_fixed_rule_action(operator, row["action"], kind)
        return rows, []
    rule = config.get("runtime_fixed_rule", {})
    thresholds = thresholds_from_rule(rule if isinstance(rule, dict) else {})
    inertia = fixed_rule_inertia_margin(config)
    trace_rows = []
    exited_farmers = []
    for farmer in sorted(farmers, key=_operator_sort_key):
        state = state_from_operator(farmer, "household")
        actor_inertia = fixed_rule_operator_inertia(farmer) if rule.get('inertia_basis') == 'yuan_per_mu' else inertia
        action = choose_fixed_rule_action(state, actor_inertia, thresholds)
        if enact_fixed_rule_action(farmer, action, "household"):
            exited_farmers.append(farmer)
        trace_rows.append(build_fixed_rule_trace_row(state, action, year))
    for enterprise in sorted(enterprises, key=_operator_sort_key):
        state = state_from_operator(enterprise, "scale")
        actor_inertia = fixed_rule_operator_inertia(enterprise) if rule.get('inertia_basis') == 'yuan_per_mu' else inertia
        action = choose_fixed_rule_action(state, actor_inertia, thresholds)
        enact_fixed_rule_action(enterprise, action, "scale")
        trace_rows.append(build_fixed_rule_trace_row(state, action, year))
    return trace_rows, exited_farmers


def _operator_sort_key(operator: Any) -> str:
    return str(getattr(operator, "unique_id", operator))


def _set_planned_crop(operator: Any, crop: Literal["grain", "cash"]) -> None:
    enum_type = _resolve_crop_enum(operator)
    if enum_type is None:
        setattr(operator, "planned_crop", "Grain" if crop == "grain" else "Cash_Crop")
    else:
        setattr(operator, "planned_crop", enum_type.GRAIN if crop == "grain" else enum_type.CASH_CROP)
    setattr(operator, "last_crop_choice", getattr(operator, "planned_crop", None))
    if hasattr(operator, "crop_portfolio"):
        enum_type = _resolve_crop_enum(operator)
        if enum_type is not None:
            if crop == "grain":
                operator.crop_portfolio = {enum_type.GRAIN: 1.0, enum_type.CASH_CROP: 0.0}
            else:
                operator.crop_portfolio = {enum_type.GRAIN: 0.0, enum_type.CASH_CROP: 1.0}


def _resolve_crop_enum(operator: Any) -> Any | None:
    current_crop = getattr(operator, "planned_crop", None)
    enum_type = getattr(current_crop, "__class__", None)
    if enum_type is not None and hasattr(enum_type, "GRAIN") and hasattr(enum_type, "CASH_CROP"):
        return enum_type
    portfolio = getattr(operator, "crop_portfolio", None)
    if isinstance(portfolio, dict):
        for key in portfolio:
            enum_type = getattr(key, "__class__", None)
            if enum_type is not None and hasattr(enum_type, "GRAIN") and hasattr(enum_type, "CASH_CROP"):
                return enum_type
    return None


def _set_household_transfer_out(operator: Any) -> None:
    setattr(operator, "land_transfer_willingness", 1.0)
    owned_area = 0.0
    for patch in getattr(operator, "land_patches", []) or []:
        if getattr(patch, "owner", None) is operator and getattr(patch, "leased_to", None) is None:
            owned_area += float(getattr(patch, "area", 0.0) or 0.0)
    setattr(operator, "land_pending_transfer", owned_area)
    if hasattr(operator, "agricultural_labor_days"):
        setattr(operator, "agricultural_labor_days", 0.0)


def _set_scale_expansion(operator: Any, willingness: float) -> None:
    setattr(operator, "expansion_willingness", float(willingness))
    if willingness <= 0.0:
        setattr(operator, "target_expansion_area", 0.0)
