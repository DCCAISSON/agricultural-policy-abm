"""One fiscal authority over province-local markets in four sampled regions."""

from dataclasses import dataclass, replace
from math import floor, fsum

from core.land_succession import CROPS, Scenario, Succession

LocalAction = tuple[str, int]
Action = tuple[LocalAction, ...]
OFF: LocalAction = ("none", 0)


@dataclass(frozen=True, slots=True)
class RegionalBudget:
    common: int
    regions: int
    instruments: tuple[str, ...]

    def menu(self) -> tuple[Action, ...]:
        q, r = divmod(self.common, self.regions)
        off = (OFF,)*self.regions
        concentrations = tuple(
            tuple((policy,self.common) if j==i else OFF for j in range(self.regions))
            for i in range(self.regions) for policy in self.instruments
        )
        equal = tuple(tuple((policy,q+int(j<r)) for j in range(self.regions)) for policy in self.instruments)
        return (off,*concentrations,*equal)

    def step(self, worlds: tuple[Succession, ...], action: Action) -> tuple[dict, ...]:
        if len(worlds)!=self.regions or len({w.year for w in worlds})!=1 or action not in self.menu():
            raise ValueError("Regional allocation is not a feasible annual budget action")
        rows = tuple(self.local_step(w,a) for w,a in zip(worlds,action,strict=True))
        if fsum(row["committed"] for row in rows)>self.common+1e-7:
            raise ValueError("Common fiscal commitments exceed opening reservations")
        return rows

    @staticmethod
    def local_step(world: Succession, action: LocalAction) -> dict:
        policy, cap = action
        original = world.scenario
        assert 0 <= cap and (cap == 0)==(policy == "none")
        world.scenario = replace(original,budget=cap)
        try:
            row = world.step(policy)
        finally:
            world.scenario = original
        assert 0 <= row["spending"] <= row["committed"]+1e-7 <= cap+2e-7
        guarantee = min(original.rate, cap/world.land, CROPS["grain"][2])
        return dict(row, policy=policy,reservation=cap,rate_mu=guarantee)


def make_budget(assets: tuple, design: dict) -> RegionalBudget:
    return RegionalBudget(floor(design["budget_yuan_per_sample_mu"]*fsum(a.area for a in assets)),
                          len(assets),tuple(design["policies"][1:]))


def worlds_at_start(seed: int, assets: tuple, design: dict) -> tuple[Succession, ...]:
    scenario = Scenario(owners=0,entrants=0,recovery=False,contract_years=2,
                        rate=design["nominal_rate_yuan_per_mu"])
    return tuple(Succession.from_assets(len(assets)*seed+i,scenario,r.households,r.plots,r.grain_kg)
                 for i,r in enumerate(assets))


def public_state(worlds: tuple[Succession, ...], design: dict, budget: RegionalBudget) -> tuple[int, ...]:
    reference_fraction = design["grain_bin_fraction"]*CROPS["grain"][0]
    return (design["main_years"]-worlds[0].year,
            *(int(w.observed["grain_kg"]>=reference_fraction*w.land) for w in worlds),budget.common)


def fork_menu(worlds: tuple[Succession, ...], design: dict, budget: RegionalBudget,
              menu: tuple[Action, ...]) -> tuple[list[tuple], list[tuple[Succession, ...]]]:
    """Combine exact local branches of every joint action without double simulation."""
    local = []
    for i,w in enumerate(worlds):
        outcomes = {}
        for a in dict.fromkeys(move[i] for move in menu):
            next_world = w.fork()
            result = budget.local_step(next_world,a)
            outcomes[a] = next_world,result["grain_kg"]
        local.append(outcomes)
    rewards, successors = [], []
    for move in menu:
        v = tuple(local[i][a][0] for i,a in enumerate(move))
        rewards.append((fsum(local[i][a][1] for i,a in enumerate(move)),public_state(v,design,budget),0.))
        successors.append(v)
    return rewards,successors
