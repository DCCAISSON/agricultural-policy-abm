"""Native Mesa 2.3.2 Agent Classes for Agricultural Succession & Government RL (V2).

Defines distinct Mesa Agent classes:
- `HouseholdMesaAgent`: 600 empirical NFPS survey households (with subsistence/marketed grain split)
- `FamilyFarmAgent`: Commercial family farm organization (family labor + hired labor + custom service)
- `CooperativeAgent`: Agricultural cooperative organization (contiguous service coordination, separate account)
- `AgriCompanyAgent`: Corporate farm organization (paid labor + management overhead + solvency/return exit)
- `GovernmentMesaAgent`: Sole RL / policy-making central fiscal agent managing 4 policy regions
"""

import mesa

from mesa_agri_abm.calibration import ActorBlueprint, RegionTechV2
from mesa_agri_abm.config import V2Design
from mesa_agri_abm.pmp_production import (
    PMPPlanV2,
    SubsidySchedule,
    compute_shadow_land_bid_v2,
    solve_actor_pmp_v2,
)


class OperatorMesaAgent(mesa.Agent):
    """Base Mesa Agent for all agricultural land operators (households and scale organizations)."""

    def __init__(self, unique_id: str, model: "mesa.Model", blueprint: ActorBlueprint, tech: RegionTechV2):
        super().__init__(unique_id, model)
        self.bp = blueprint
        self.tech = tech
        self.region = blueprint.region
        self.province = blueprint.province
        self.submarket = blueprint.submarket
        self.actor_category = blueprint.actor_category
        self.is_survey_household = blueprint.is_survey_household

        # Dynamic intertemporal economic states
        self.working_cap: float = blueprint.working_capital_init
        self.wealth: float = blueprint.savings_stock
        self.ref_grain_share: float = blueprint.ref_grain_share
        self.active_in_market: bool = True
        self.consecutive_loss_years: int = 0

        # Current-year realized runtime registers (populated during model.step())
        self.current_opp_wage: float = blueprint.offfarm_wage_day
        self.current_fam_labor: float = blueprint.ag_days_year
        self.current_wc_open: float = blueprint.working_capital_init
        self.last_year_summary: dict = {}

    def clone_into(self, new_model: "mesa.Model") -> "OperatorMesaAgent":
        cls = type(self)
        copy_ag = cls(self.unique_id, new_model, self.bp, self.tech)
        copy_ag.working_cap = self.working_cap
        copy_ag.wealth = self.wealth
        copy_ag.ref_grain_share = self.ref_grain_share
        copy_ag.active_in_market = self.active_in_market
        copy_ag.consecutive_loss_years = self.consecutive_loss_years
        copy_ag.current_opp_wage = self.current_opp_wage
        copy_ag.current_fam_labor = self.current_fam_labor
        copy_ag.current_wc_open = self.current_wc_open
        copy_ag.last_year_summary = self.last_year_summary.copy()
        return copy_ag

    def evaluate_production(
        self,
        operated_phys_mu: float,
        new_takeover_phys_mu: float,
        design: V2Design,
        sched: SubsidySchedule,
        working_cap_avail: float,
    ) -> PMPPlanV2:
        return solve_actor_pmp_v2(
            operated_phys_mu,
            new_takeover_phys_mu,
            self.bp,
            self.tech,
            design,
            sched,
            family_labor_avail=self.current_fam_labor,
            opp_wage_day=self.current_opp_wage,
            working_cap_avail=working_cap_avail,
            ref_share=self.ref_grain_share,
        )

    def evaluate_land_bid(
        self,
        current_phys_mu: float,
        current_takeover_mu: float,
        lot_area_mu: float,
        is_new_contract_group: bool,
        design: V2Design,
        sched: SubsidySchedule,
        working_cap_avail: float,
        base_plan: PMPPlanV2 | None = None,
    ) -> float:
        if not self.active_in_market:
            return 0.0
        return compute_shadow_land_bid_v2(
            current_phys_mu,
            current_takeover_mu,
            lot_area_mu,
            is_new_contract_group,
            self.bp,
            self.tech,
            design,
            sched,
            family_labor_avail=self.current_fam_labor,
            opp_wage_day=self.current_opp_wage,
            working_cap_avail=working_cap_avail,
            ref_share=self.ref_grain_share,
            base_plan=base_plan,
        )

    def step(self) -> None:
        """Mesa standard agent step hook (staged execution coordinated by AgriSuccessionMesaModel)."""
        return None


class HouseholdMesaAgent(OperatorMesaAgent):
    """Empirical NFPS survey household agent (one of the 600 cohort households)."""

    def decide_land_retention(
        self,
        held_mu: float,
        held_new_tk_mu: float,
        own_trad_parcels: list,
        nontrad_mu: float,
        expected_rent_per_mu: float,
        design: V2Design,
        sched: SubsidySchedule,
        working_cap_after_held: float,
    ) -> tuple[list, list, PMPPlanV2]:
        """Unified AHM-PMP comparison across full retention, full release, and partial release."""
        trad_mu = sum(p.area for p in own_trad_parcels)
        full_mu = held_mu + trad_mu + nontrad_mu

        plan_keep_all = self.evaluate_production(
            full_mu, held_new_tk_mu, design, sched, working_cap_after_held
        )
        best_keep = own_trad_parcels
        best_rel: list = []
        best_plan = plan_keep_all
        best_val = plan_keep_all.net_economic_surplus_yuan

        if trad_mu > 1e-9:
            rem_mu = held_mu + nontrad_mu
            plan_rel_all = self.evaluate_production(
                rem_mu, held_new_tk_mu, design, sched, working_cap_after_held
            )
            val_rel_all = plan_rel_all.net_economic_surplus_yuan + expected_rent_per_mu * trad_mu
            if val_rel_all > best_val + 1e-6:
                best_keep = []
                best_rel = list(own_trad_parcels)
                best_plan = plan_rel_all
                best_val = val_rel_all

            if len(own_trad_parcels) >= 2:
                half_n = len(own_trad_parcels) // 2
                keep_sub = own_trad_parcels[:half_n]
                rel_sub = own_trad_parcels[half_n:]
                keep_mu = sum(p.area for p in keep_sub)
                rel_mu = sum(p.area for p in rel_sub)
                plan_half = self.evaluate_production(
                    rem_mu + keep_mu, held_new_tk_mu, design, sched, working_cap_after_held
                )
                val_half = plan_half.net_economic_surplus_yuan + expected_rent_per_mu * rel_mu
                if val_half > best_val + 1e-6:
                    best_keep = keep_sub
                    best_rel = rel_sub
                    best_plan = plan_half

        return best_keep, best_rel, best_plan


class ScaleOrgMesaAgent(OperatorMesaAgent):
    """Base class for non-survey scale agricultural organizations (segregated accounting)."""

    def update_organizational_viability(self, net_profit: float, capital_base: float, design: V2Design) -> None:
        if self.working_cap < 1_000.0:
            self.active_in_market = False


class FamilyFarmAgent(ScaleOrgMesaAgent):
    """Commercial family farm agent (family management/labor + seasonal hiring + scale eligibility)."""


class CooperativeAgent(ScaleOrgMesaAgent):
    """Agricultural cooperative agent (contiguous machinery service coordination, zero fabricated patronage)."""


class AgriCompanyAgent(ScaleOrgMesaAgent):
    """Corporate agricultural enterprise agent (100% paid labor + overhead + return-on-capital exit rule)."""

    def update_organizational_viability(self, net_profit: float, capital_base: float, design: V2Design) -> None:
        super().update_organizational_viability(net_profit, capital_base, design)
        ror = net_profit / max(10_000.0, capital_base)
        if ror < design.company_min_return_rate:
            self.consecutive_loss_years += 1
        else:
            self.consecutive_loss_years = 0
        if self.consecutive_loss_years >= 2:
            self.active_in_market = False


def build_operator_agent(unique_id: str, model: "mesa.Model", bp: ActorBlueprint, tech: RegionTechV2) -> OperatorMesaAgent:
    if bp.actor_category == "survey_household":
        return HouseholdMesaAgent(unique_id, model, bp, tech)
    if bp.actor_category == "family_farm":
        return FamilyFarmAgent(unique_id, model, bp, tech)
    if bp.actor_category == "cooperative":
        return CooperativeAgent(unique_id, model, bp, tech)
    if bp.actor_category == "agri_company":
        return AgriCompanyAgent(unique_id, model, bp, tech)
    raise ValueError(f"Unknown actor_category: {bp.actor_category}")


class GovernmentMesaAgent(mesa.Agent):
    """Central Fiscal & Reinforcement Learning Agent managing the 4 policy regions."""

    def __init__(
        self,
        unique_id: str,
        model: "mesa.Model",
        *,
        policy_mode: str = "p0",
        fixed_action_idx: int = 0,
        learned_weights: dict[int, list[list[float]]] | None = None,
    ):
        super().__init__(unique_id, model)
        self.policy_mode = policy_mode
        self.fixed_action_idx = fixed_action_idx
        self.learned_weights = learned_weights
        self.decision_trace: list[dict] = []

    def clone_into(self, new_model: "mesa.Model") -> "GovernmentMesaAgent":
        g = GovernmentMesaAgent(
            self.unique_id,
            new_model,
            policy_mode=self.policy_mode,
            fixed_action_idx=self.fixed_action_idx,
            learned_weights=self.learned_weights,
        )
        g.decision_trace = list(self.decision_trace)
        return g

    def step(self) -> None:
        return None
