from dataclasses import replace

import pytest

from core.land_succession import AREA, POLICIES, Plot, Scenario, Succession, choose_crop
from scripts.run_land_succession import EVAL_SEEDS, TRAIN_STARTS, designs, simulate
from scripts.succession_q import MIN_ACTION_VISITS, LearnedPolicy, observation


def test_cash_surplus_does_not_override_nonfarm_option():
    p=Plot("p","h",1.,"grain")
    assert choose_crop(p,10000,30,0,30)[0] is not None
    assert choose_crop(p,10000,30,400,30)==(None,0.)


@pytest.mark.parametrize("policy", POLICIES)
def test_annual_renewal_cash_and_trade_accounts(policy: str) -> None:
    w=Succession(301,Scenario(cash=.65,labor=.65))
    old=None
    for _ in range(8):
        a,p,h,t=w.step(policy,ledger=True)
        assert sum(z["physical_area"] for z in p)==w.land
        assert len({z["plot"] for z in p})==len(p)
        assert all(z["crop"]=="idle" or z["operator"] is not None for z in p)
        assert all(z["operator"] in w.operators or z["operator"] is None for z in p)
        assert all(z["planted_area"]<=z["physical_area"] for z in p)
        assert a["grain_area"]==sum(z["planted_area"] for z in p if z["crop"]=="grain")
        assert a["grain_kg"]==pytest.approx(sum(z["harvest"] for z in p if z["crop"]=="grain"))
        assert a["offered_area"]==a["leased_area"]+a["unfilled_area"]
        assert a["spending"]<=w.scenario.budget+1e-8
        assert sum(x["support"] for x in p)==pytest.approx(a["spending"])
        assert all(x["post_support" if policy.endswith("_pre") else "pre_support"]==0 for x in h)
        if policy=="none":
            assert a["spending"]==0.
        if policy.startswith("takeover_"):
            assert all(z["new_lease"] for z in p if z["support"] > 0)
        assert sum(z["rent_paid"] for z in h)==pytest.approx(sum(z["rent_income"] for z in h))
        assert sum(z["rent"] for z in t)==pytest.approx(sum(z["rent_paid"] for z in h))
        assert all(z["ask"] <= z["rent"] <= z["bid"] + 1e-8 for z in t if z["matched"])
        for z in h:
            assert z["closing"]>=0 and z["farm_days"]+z["job_days"]<=z["labor_capacity"]+1e-8
            assert z["opening"]+z["income"]-z["consumption"]==pytest.approx(z["closing"],abs=1e-7)
            if old is not None:
                assert z["opening"]==old[z["id"]]
        old={z["id"]:z["closing"] for z in h}


def test_full_and_zero_search_failure():
    s=Scenario(search_failure=0.,fee=0.)
    active=Succession(301,s).step("none",ledger=True)
    assert active[0]["leased_area"]>0 and all(x["search_open"] for x in active[3])
    closed=Succession(301,replace(s,search_failure=1.)).step("none")
    assert closed["offered_area"]>0 and closed["unfilled_area"]==closed["offered_area"]


def test_household_can_rent_and_qualify_on_another_owners_plot() -> None:
    scenario = Scenario(
        owners=2, entrants=0, fee=0, start_cost=0, outside_wage=0,
        search_failure=0, labor_realization=1, cash_available=1, harvest=1,
    )
    world = Succession(301, scenario)
    world.cash.update(H00=0, H01=20_000)
    world.operators["H01"] = replace(world.operators["H01"], labor=60)
    world.plots[:] = [replace(p, quality=1.12) for p in world.plots]
    _, plots, actors, _ = world.step("takeover_pre", ledger=True)
    takeover = [p for p in plots if p["eligible"]]
    assert takeover
    assert all(p["new_lease"] and p["operator"] == "H01" and p["owner"] == "H00" for p in takeover)
    household = next(a for a in actors if a["id"] == "H01")
    assert household["kind"] == "owner" and household["support"] > 0
    assert all(p["support"] == 0 for p in plots if p["owner"] == "H01")


@pytest.mark.parametrize("condition",["labor_realization","cash_available"])
def test_a_valid_lease_can_fail_to_produce(condition):
    w=Succession(301,replace(Scenario(),**{condition:0.}))
    a,p,_,_=w.step("none",ledger=True)
    assert a["leased_area"]>0 and a["leased_idle"]==a["leased_area"]
    expected="labor" if condition=="labor_realization" else "cash"
    assert {x["failure"] for x in p if x["lease"]}=={expected}


def test_no_release_and_no_trade():
    w=Succession(15,Scenario(owners=1,labor_realization=1.,cash_available=1.))
    owner=w.operators["H00"]
    w.operators[owner.id]=replace(owner,labor=100.,wage=0.)
    w.cash[owner.id]=10000.
    a=w.step("none")
    assert a["offered_area"]==a["leased_area"]==0 and a["idle_area"]==0.


def test_beginnings_repeat_and_annual_state_is_real():
    a,b=(Succession(301,Scenario()) for _ in range(2))
    assert a.step("takeover_pre",ledger=True)==b.step("takeover_pre",ledger=True)
    paid,plots,actors,_=a.step("takeover_pre",ledger=True)
    assert paid["grain_area"]+paid["cash_crop_area"]+paid["idle_area"]==AREA*len(plots)
    assert any(x["opening"]!=b.operators[x["id"]].cash for x in actors)
    static=Succession(301,Scenario(feedback=False))
    static.step("none")
    _,_,a,_=static.step("none",ledger=True)
    assert all(x["opening"]==static.operators[x["id"]].cash for x in a)


def test_policy_reads_previous_observation_and_evaluation_does_not_learn() -> None:
    design = designs()
    assert set(design["factorial"].values()).isdisjoint(design["evaluation"].values())
    training = {
        seed for start in TRAIN_STARTS for offset in (0, 5000)
        for seed in range(start + offset, start + offset + 4096)
    }
    assert training.isdisjoint(EVAL_SEEDS)
    scenario = next(iter(design["evaluation"].values()))
    world = Succession(EVAL_SEEDS[0], scenario)
    initial = observation(world)
    action = POLICIES.index("takeover_pre")
    table = {initial: tuple(2.0 if i == action else 0.0 for i in range(len(POLICIES)))}
    visits = {initial: (MIN_ACTION_VISITS,) * len(POLICIES)}
    learned = LearnedPolicy("test", "base", False, table, visits, True)
    before = table.copy(), visits.copy()

    rows = simulate(world.seed, scenario, "test", learned, lambda *_: None, "evaluation", "heldout")

    assert (learned.table, learned.visits) == before
    assert rows[0]["policy"] == "takeover_pre" and rows[0]["policy_fallback"] == ""
    for year, row in enumerate(rows, 1):
        assert row["previous_state"] == str(observation(world))
        assert row["year"] == year
        world.step(row["policy"])


def test_runner_fiscal_rows_reconcile_commitments_and_actual_payouts() -> None:
    entries = {}

    def collect(label, rows) -> None:
        entries.setdefault(label, []).extend(rows)

    annual = simulate(301, Scenario(), "grain_pre", None, collect, "duration", "test", active=1)
    plots = {(p["year"], p["plot"]): p for p in entries["plots"]}
    fiscal = entries["fiscal"]
    assert any(p["support"] > 0 for p in fiscal)
    assert any(p["commitment"] > 0 and p["support"] == 0 for p in fiscal)
    assert {p["year"] for p in fiscal} == {1}
    keys = {(p["year"], p["plot"]) for p in fiscal}
    assert len(keys) == len(fiscal)
    assert keys == {key for key, p in plots.items() if p["eligible"]}
    for entry in fiscal:
        plot = plots[entry["year"], entry["plot"]]
        for field in ("operator", "crop", "eligible", "commitment", "expected_support", "support"):
            assert entry[field] == plot[field]
    assert sum(p["support"] for p in fiscal) == pytest.approx(annual[0]["spending"])
    assert sum(p["commitment"] for p in fiscal) == pytest.approx(annual[0]["committed"])
    assert all(a["spending"] == 0 for a in annual[1:])
