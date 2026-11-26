import pytest

from scripts.run_succession_annual_allocation import design_inputs
from scripts.run_succession_shared_budget import OFF, setup
from scripts.succession_annual_learning import branch_opening, fit, observation, roots, run_path


@pytest.fixture
def design():
    return design_inputs()[0]


def test_all_joint_moves_mask_competing_reservations(design):
    budget, _ = setup(design)
    moves = budget.feasible_moves()
    assert len(moves) == len(set(moves)) == 13
    assert all(a[1] + b[1] <= 7200 for a, b in moves)
    with pytest.raises(ValueError, match="joint cap"):
        budget.advance(roots(5, design), (("grain_pre", 7200), ("takeover_pre", 3600)))


def test_actual_payment_tiers_change_intensity(design):
    budget, _ = setup(design)
    a, b = roots(5, design)
    for i, who in enumerate((a, b)):
        for policy in ("grain_pre", "takeover_pre"):
            for cap, expected in ((3600, 18.75), (7200, 37.5)):
                action = ((policy, cap), OFF) if i == 0 else (OFF, (policy, cap))
                row = budget.advance((a.fork(), b.fork()), action)[i]
                assert row["guarantee_mu"] == expected == cap/who.land
                assert 0 <= row["spending"] <= row["committed"] <= cap


def test_factorized_opening_matches_direct_actions_and_stop(design):
    budget, _ = setup(design)
    worlds = a, b = roots(21, design)
    for _ in range(design["active_years"]-1):
        budget.advance(worlds, budget.feasible_moves()[2])
    factored, states = branch_opening(worlds, budget, budget.feasible_moves(), design)
    for i, move in enumerate(budget.feasible_moves()):
        direct = a.fork(), b.fork()
        rows = budget.advance(direct, move)
        assert factored[i][:2] == (sum(r["grain_kg"] for r in rows), observation(direct, design))
        for j, w in enumerate(states[i]):
            assert w.cash == direct[j].cash
            assert w.crop == direct[j].crop
            assert w.contracts == direct[j].contracts
            assert w.operator == direct[j].operator
        expected = sum(sum(r["grain_kg"] for r in budget.advance(direct,(OFF,OFF))) for _ in range(2))
        assert factored[i][2] == pytest.approx(expected, abs=1e-9)


def test_fitted_q_uses_later_production_but_myopic_does_not(design):
    d = dict(design, active_years=2, horizon=4)
    initial, middle, terminal = (2,0,0,7200), (1,0,0,7200), (0,0,0,7200)
    samples = [(s, initial, [(5.,middle,0.), (6.,middle,0.)]) for s in range(5)]
    samples += [(s, middle, [(5.,terminal,100.), (6.,terminal,0.)]) for s in range(5)]
    tab, coverage = fit(samples,d,{2:0.})
    assert tab["1:0:0:7200"]["q"] == 0
    assert tab["1:0:0:7200"]["myopic"] == 1
    assert tab["2:0:0:7200"]["q_values"] == [110.,111.]
    assert coverage["coverage"] == 1


def test_unseen_and_sparse_same_fallback_and_two_year_stop(design):
    state = observation(roots(21,design),design)
    key = ":".join(map(str,state))
    table = {key: {"supported":False, "q":2, "myopic":3}}
    rows = run_path(21,design,"q_1",table=table)
    assert all(r["reason"] == "sparse" for r in rows[:2])
    assert all(r["reason"] == "unseen" for r in rows[2:12])
    assert all(r["reason"] == "withdrawal" for r in rows[12:])
    assert all(r["reservation"] == r["committed"] == r["spending"] == 0 for r in rows)
    assert [r["grain_kg"] for r in rows] == [
        r["grain_kg"] for r in run_path(21,design,"fixed_00",fixed=0)
    ]


def test_cash_reset_only_changes_post_initial_year(design):
    carry = run_path(432, design, "fixed_02",fixed=2)
    reset = run_path(432, design, "fixed_02",fixed=2,cash_reset=True)
    fields = ("grain_kg", "tenant_opening_cash", "owner_closing_cash", "committed", "spending")
    assert all(a[k] == b[k] for a,b in zip(carry[:2],reset[:2]) for k in fields)
    assert any(a["owner_opening_cash"] != b["owner_opening_cash"] for a,b in zip(carry[2:],reset[2:]))
    assert sum(r["spending"] for r in carry[12:]+reset[12:]) == 0
