from dataclasses import replace
from itertools import product
from math import fsum
from random import Random
from typing import Final

import pytest

from core.land_succession import AREA, CROPS, POLICIES, Scenario, Succession
from core.succession_allocation import CropOffer, joint_allocation

SEED: Final = 301
GRAIN_COST: Final = AREA * CROPS["grain"][2]
GRAIN_DAYS: Final = AREA * CROPS["grain"][3]


def _owner_world() -> Succession:
    world = Succession(
        SEED,
        Scenario(
            owners=1,
            entrants=0,
            labor_realization=1,
            cash_available=1,
            harvest=1,
            search_failure=0,
        ),
    )
    world.operators["H00"] = replace(world.operators["H00"], labor=40, wage=0)
    world.cash["H00"] = 20_000
    world.plots[:] = [replace(p, quality=1.1) for p in world.plots]
    return world


def _rental_world(scenario: Scenario) -> Succession:
    world = Succession(SEED, scenario)
    world.cash["H00"] = 0
    for key, actor in world.operators.items():
        if actor.tenant:
            world.operators[key] = replace(actor, labor=GRAIN_DAYS)
    return world


@pytest.fixture
def rental() -> Scenario:
    return Scenario(
        owners=1,
        entrants=1,
        search_failure=0,
        fee=0,
        start_cost=0,
        ask_per_mu=100,
        cash=4,
        labor_realization=1,
        cash_available=1,
        harvest=1,
        outside_wage=0,
    )


def test_zero_budget_policies_have_common_baseline() -> None:
    assert POLICIES == (
        "none",
        "grain_pre",
        "grain_post",
        "takeover_pre",
        "takeover_post",
    )
    baseline = Succession(SEED, Scenario(budget=0))
    alternatives = [Succession(SEED, baseline.scenario) for _ in POLICIES]
    for _ in range(4):
        annual, plots, actors, trades = baseline.step("none", ledger=True)
        assert annual["spending"] == 0
        for policy, world in zip(POLICIES, alternatives, strict=True):
            actual, land, accounts, contracts = world.step(policy, ledger=True)
            assert actual == annual
            assert accounts == actors
            assert contracts == trades
            assert [
                {key: value for key, value in row.items() if key != "eligible"}
                for row in land
            ] == [
                {key: value for key, value in row.items() if key != "eligible"}
                for row in plots
            ]


def test_fixed_none_replays_baseline_across_years(rental: Scenario) -> None:
    scenario = replace(rental, contract_years=3)
    baseline, replay = (_rental_world(scenario) for _ in range(2))
    records = []
    for _ in range(4):
        annual, plots, actors, trades = baseline.step("none", ledger=True)
        records.extend(plots)
        actual, land, accounts, fixed_trades = replay.step("none", ledger=True, fixed=plots)
        assert actual == annual and fixed_trades == trades
        # A fixed replay bypasses auction planning, but not production and settlement.
        assert [dict(p, planned_crop=None) for p in land] == [
            dict(p, planned_crop=None) for p in plots
        ]
        diagnostics = dict.fromkeys(("planned_cash", "planned_days", "quoted_plots", "highest_bid_per_mu"), 0)
        assert [a | diagnostics for a in accounts] == [a | diagnostics for a in actors]
    assert any(row["matched"] for row in records)
    assert any(row["lease"] and not row["matched"] for row in records)


def test_fixed_pre_payment_can_cover_planting_but_post_cannot(rental: Scenario) -> None:
    baseline = _rental_world(rental)
    _, market_plots, _, _ = baseline.fork().step("none", ledger=True)
    [market_lease] = [p for p in market_plots if p["matched"]]
    cash_after_rent = baseline.cash["T00"] - market_lease["rent"]
    baseline.scenario = replace(rental, cash_available=(GRAIN_COST - 100) / cash_after_rent)
    pre, post = baseline.fork(), baseline.fork()
    _, frozen, _, trades = baseline.step("none", ledger=True)

    early, pre_plots, pre_actors, pre_trades = pre.step("takeover_pre", ledger=True, fixed=frozen)
    late, post_plots, post_actors, post_trades = post.step("takeover_post", ledger=True, fixed=frozen)
    assert early["grain_area"] == AREA and late["grain_area"] == 0
    assert early["spending"] == sum(a["pre_support"] for a in pre_actors) == 85 * AREA
    assert late["spending"] == sum(a["post_support"] for a in post_actors) == 0
    assert pre_trades == post_trades == trades
    [pre_lease] = [p for p in pre_plots if p["lease"]]
    [post_lease] = [p for p in post_plots if p["lease"]]
    for key in ("operator", "contract_start", "contract_end", "rent_per_mu"):
        assert pre_lease[key] == post_lease[key] == market_lease[key]
    for lease in (pre_lease, post_lease):
        assert lease["eligible"] and lease["commitment"] == lease["expected_support"] == 85 * AREA
    assert post_lease["failure"] == "cash"


def test_joint_allocation_plants_two_grain_plots_instead_of_one_cash_crop() -> None:
    world = _owner_world()
    world.cash["H00"] = 2 * GRAIN_COST
    world.operators["H00"] = replace(world.operators["H00"], labor=2 * GRAIN_DAYS)
    _, plots, actors, _ = world.step("none", ledger=True)

    # At this quality one cash plot wins alone, but loses to two grain plots.
    def margin(crop: str) -> float:
        yield_mu, price, input_mu, _ = CROPS[crop]
        return AREA * (1.1 * yield_mu * price - input_mu)

    assert margin("grain") < margin("cash") < 2 * margin("grain")
    assert [row["crop"] for row in plots] == ["grain", "grain"]
    assert actors[0]["inputs"] == 2 * GRAIN_COST
    assert actors[0]["farm_days"] == 2 * GRAIN_DAYS


def test_joint_allocation_compares_different_offer_groups_against_one_budget() -> None:
    offers = [
        CropOffer("p0", 1.15, grant=340, credit=340),
        CropOffer("p1", 1.12, forgone_rent=500),
        CropOffer("p2", 1.06, restart_cash=50, restart_days=1),
        CropOffer("p3", 1.09, grant=340, credit=340),
    ]
    budget, labor, wage, job_days = 7_000, 34, 70, 20
    best = 0.0
    for crops in product((None, *CROPS), repeat=len(offers)):
        cash = days = value = 0.0
        for offer, crop in zip(offers, crops, strict=True):
            if crop is None:
                continue
            output, price, inputs, required = CROPS[crop]
            cost = AREA * inputs + offer.restart_cash
            cash += cost - (offer.credit if crop == "grain" else 0)
            days += AREA * required + offer.restart_days
            value += AREA * output * price * offer.quality - cost - offer.forgone_rent
            value += offer.grant if crop == "grain" else 0
        if cash <= budget and days <= labor:
            best = max(best, value - wage * (job_days - min(labor - days, job_days)))
    plan = joint_allocation(offers, budget, labor, wage, job_days)
    assert plan.cash <= budget and plan.days <= labor
    assert best > 0 and plan.value == pytest.approx(best)


@pytest.mark.parametrize("term", [1, 3])
def test_same_operator_does_not_requalify_when_contract_continues_or_renews(
    rental: Scenario,
    term: int,
) -> None:
    world = _rental_world(replace(rental, contract_years=term))
    world.plots[:] = [replace(p, quality=1.12) for p in world.plots]
    initial = world.plots[0]
    world.crop[initial.id] = "grain"
    world.plots[:] = [initial]
    world.land = AREA
    first, plots, _, _ = world.step("takeover_pre", ledger=True)
    plot = plots[0]
    assert plot["matched"] and plot["operator"] != plot["owner"]
    assert plot["eligible"] and plot["support"] > 0
    assert (plot["contract_start"], plot["contract_end"]) == (1, term)
    assert first["continuity_area"] == AREA

    second, plots, actors, _ = world.step("takeover_pre", ledger=True)
    next_plot = plots[0]
    assert next_plot["operator"] == plot["operator"]
    assert next_plot["crop"] == "grain"
    assert not next_plot["eligible"] and next_plot["support"] == 0
    assert second["continuity_area"] == 0
    assert next_plot["matched"] is (term == 1)
    if term == 3:
        assert not next_plot["posted"]
        assert (next_plot["contract_start"], next_plot["contract_end"]) == (1, 3)
        assert next_plot["rent"] == plot["rent"]
    assert sum(actor["rent_paid"] for actor in actors) == next_plot["rent"] > 0


def test_new_operator_on_recently_fallow_plot_can_qualify_without_strict_continuity(
    rental: Scenario,
) -> None:
    world = _rental_world(replace(rental, search_failure=1))
    _, fallow, _, _ = world.step("none", ledger=True)
    assert all(row["operator"] is None and row["crop"] == "idle" for row in fallow)
    world.scenario = rental
    annual, plots, _, _ = world.step("takeover_post", ledger=True)
    revived = [row for row in plots if row["support"] > 0]
    assert revived
    assert all(row["eligible"] and row["previous_operator"] is None for row in revived)
    assert annual["continuity_area"] == 0


@pytest.mark.parametrize("boundary", ["search", "cash", "labor", "no_entrants"])
def test_market_and_production_constraints_bind(
    rental: Scenario, boundary: str
) -> None:
    unconstrained, _, _, _ = _rental_world(rental).step("none", ledger=True)
    assert unconstrained["leased_area"] > 0 and unconstrained["grain_area"] > 0
    scenario = replace(
        rental,
        **{
            "search": {"search_failure": 1},
            "cash": {"cash": 0},
            "labor": {"labor_realization": 0},
            "no_entrants": {"entrants": 0},
        }[boundary],
    )
    constrained, plots, _, _ = _rental_world(scenario).step("none", ledger=True)
    assert constrained["grain_area"] == 0
    if boundary == "labor":
        assert constrained["leased_idle"] == constrained["leased_area"] > 0
    else:
        assert constrained["leased_area"] == 0
    if boundary == "search":
        assert any(plot["released"] for plot in plots)
        assert all(not plot["posted"] for plot in plots)


def test_no_release_when_owner_can_operate_every_plot() -> None:
    annual, plots, _, _ = _owner_world().step("none", ledger=True)
    assert annual["offered_area"] == annual["leased_area"] == annual["idle_area"] == 0
    assert all(p["operator"] == p["previous_operator"] == p["owner"] for p in plots)


def test_support_stops_without_resetting_cash_and_plot_history() -> None:
    world = _owner_world()
    world.operators["H00"] = replace(world.operators["H00"], labor=GRAIN_DAYS)
    paid, previous, settled, _ = world.step("grain_pre", ledger=True)
    assert paid["spending"] > 0
    stopped, plots, actors, _ = world.step("none", ledger=True)
    assert stopped["spending"] == 0
    assert actors[0]["opening"] == settled[0]["closing"]
    previous_by_id = {p["plot"]: p for p in previous}
    for plot in plots:
        assert plot["previous_operator"] == previous_by_id[plot["plot"]]["operator"]
        assert plot["previous_crop"] == previous_by_id[plot["plot"]]["crop"]
        assert plot["support"] == 0


def test_cash_carry_switch_changes_next_opening_balance() -> None:
    carried, reset = (
        Succession(SEED, Scenario(carry_cash=carry)) for carry in (True, False)
    )
    _, _, previous, _ = carried.step("none", ledger=True)
    reset.step("none")
    _, _, carried_actors, _ = carried.step("none", ledger=True)
    _, _, reset_actors, _ = reset.step("none", ledger=True)
    assert {a["id"]: a["opening"] for a in carried_actors} == {
        a["id"]: a["closing"] for a in previous
    }
    assert {a["id"]: a["opening"] for a in reset_actors} == {
        key: a.cash for key, a in reset.operators.items()
    }
    assert any(
        left["opening"] != right["opening"]
        for left, right in zip(carried_actors, reset_actors, strict=True)
    )


@pytest.mark.parametrize("budget", [24.0, 100_000.0])
@pytest.mark.parametrize(
    "policy", ["grain_pre", "grain_post", "takeover_pre", "takeover_post"]
)
def test_land_fiscal_and_operator_accounts_close(policy: str, budget: float) -> None:
    scenario = Scenario(budget=budget, contract_years=3)
    world = Succession(SEED, scenario)
    spending = 0.0
    for _ in range(4):
        annual, plots, actors, trades = world.step(policy, ledger=True)
        assert len({row["plot"] for row in plots}) == len(world.plots)
        assert fsum(row["physical_area"] for row in plots) == world.land
        for crop, key in (("grain", "grain_area"), ("cash", "cash_crop_area")):
            assert (
                fsum(p["planted_area"] for p in plots if p["crop"] == crop)
                == annual[key]
            )
        assert (
            fsum(p["physical_area"] for p in plots if p["crop"] == "idle")
            == annual["idle_area"]
        )
        assert fsum(
            p["harvest"] for p in plots if p["crop"] == "grain"
        ) == pytest.approx(annual["grain_kg"])
        assert fsum(p["support"] for p in plots) == pytest.approx(annual["spending"])
        assert fsum(p["commitment"] for p in plots) == pytest.approx(annual["committed"])
        assert annual["unused_budget"] + annual["spending"] == pytest.approx(budget)
        assert 0 <= annual["spending"] <= budget + 1e-8
        spending += annual["spending"]
        limit = min(85 * AREA, budget / len(plots))
        paid = [p for p in plots if p["support"] > 0]
        for plot in paid:
            assert plot["eligible"] and plot["crop"] == "grain"
            assert plot["support"] == pytest.approx(limit)
            assert plot["expected_support"] == pytest.approx(plot["support"])
        for plot in plots:
            assert 0 <= plot["support"] <= plot["expected_support"] == plot["commitment"]
            if plot["planted_area"] == 0:
                assert plot["harvest"] == plot["support"] == 0
            else:
                assert plot["operator"] is not None
                assert plot["planted_area"] == plot["physical_area"]
            if plot["lease"]:
                assert plot["operator"] != plot["owner"]
                assert plot["rent"] == pytest.approx(
                    plot["rent_per_mu"] * plot["physical_area"]
                )
            assert not plot["posted"] or plot["released"]
            assert not plot["matched"] or plot["posted"]
        by_plot = {p["plot"]: p for p in plots}
        for trade in trades:
            if trade["buyer"] is not None:
                assert trade["buyer"] == by_plot[trade["plot"]]["operator"]
                assert trade["area"] == by_plot[trade["plot"]]["physical_area"]
                assert trade["rent"] == pytest.approx(by_plot[trade["plot"]]["rent"])
        for row in actors:
            own = [p for p in plots if p["operator"] == row["id"]]
            assert row["support"] == pytest.approx(fsum(p["support"] for p in own))
            phase = "post_support" if policy.endswith("_pre") else "pre_support"
            assert row[phase] == 0
            assert row["inputs"] == pytest.approx(
                fsum(
                    p["planted_area"] * CROPS[p["crop"]][2]
                    for p in own
                    if p["planted_area"]
                )
            )
            assert row["farm_days"] == pytest.approx(
                fsum(
                    p["planted_area"] * CROPS[p["crop"]][3]
                    for p in own
                    if p["planted_area"]
                )
            )
            assert row["rent_paid"] == pytest.approx(fsum(p["rent"] for p in own))
            assert row["rent_income"] == pytest.approx(
                fsum(p["rent"] for p in plots if p["owner"] == row["id"])
            )
            assert row["farm_days"] + row["job_days"] <= row["labor_capacity"] + 1e-8
            assert row["inputs"] + row["recovery_cost"] <= row["liquidity"] + row["pre_support"] + 1e-8
            assert row["opening"] + row["income"] - row["consumption"] == pytest.approx(
                row["closing"]
            )
            assert row["closing"] >= 0
    assert spending > 0


@pytest.mark.parametrize("permutation", [None, 11, 39, 93])
def test_seeded_path_is_invariant_to_plot_order(permutation: int | None) -> None:
    scenario = Scenario(contract_years=3)
    baseline, reordered = (Succession(SEED, scenario) for _ in range(2))
    if permutation is None:
        reordered.plots.reverse()
    else:
        Random(permutation).shuffle(reordered.plots)

    for policy in ("takeover_pre", "grain_post", "none", "takeover_post"):
        control, plots, actors, trades = baseline.step(policy, ledger=True)
        actual, shuffled_plots, shuffled_actors, shuffled_trades = reordered.step(
            policy, ledger=True
        )
        assert control == actual
        assert {p["plot"]: p for p in plots} == {p["plot"]: p for p in shuffled_plots}
        assert {a["id"]: a for a in actors} == {a["id"]: a for a in shuffled_actors}
        assert {t["plot"]: t for t in trades} == {t["plot"]: t for t in shuffled_trades}
