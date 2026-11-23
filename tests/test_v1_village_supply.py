from types import SimpleNamespace

import pytest

from agents.village_collective import VillageCollective


@pytest.mark.parametrize("v1", [True, False])
def test_village_supply_respects_v1_household_decision(v1):
    farmers = []
    for action in ("maintain_grain", "expand_cash", "portfolio_adjusted", "exit", None):
        f = SimpleNamespace(fixed_rule_current_action=action, get_transfer_out_willingness=lambda: 1.)
        f.land_patches = [SimpleNamespace(owner=f, leased_to=None), SimpleNamespace(owner=f, leased_to=object())]
        farmers.append(f)
    village = SimpleNamespace(model=SimpleNamespace(config={"runtime_fixed_rule": {"national_v1": v1}}),
                              registered_farmers=farmers)
    offers = VillageCollective._identify_potential_suppliers(village)
    assert [o["farmer"].fixed_rule_current_action for o in offers] == (
        ["exit"] if v1 else [f.fixed_rule_current_action for f in farmers])
    assert all(len(o["patches"]) == 1 and o["patches"][0].leased_to is None for o in offers)
