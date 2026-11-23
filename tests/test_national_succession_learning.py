import pytest

from scripts.run_national_succession_learning import fit, frozen_choice, signal, threshold
from scripts.summarize_national_plot_destinations import release_chains


def terminal(action, reward):
    return dict(step=1, action=action, reward=reward, done=True, grain_reference_kg=100.,
                state_observations={"grain_output_kg": 0.}, outcome_observations={"grain_output_kg": 90.})


def test_known_best_negative_value_beats_an_unobserved_zero():
    rows = fit([terminal(0, -3.), terminal(1, -2.), terminal(1, -1.)])
    assert next(r for r in rows if r["action"] == 1)["q"] == pytest.approx(-1.5)
    assert frozen_choice("initial", rows, 2) == (1, "covered_greedy", 2)
    assert frozen_choice("gap", rows, 2) == (2, "rule_fallback", 0)


def test_supported_zero_counts_as_trained_and_unknown_state_uses_same_rule():
    rows = fit([terminal(0, -1.), terminal(2, 0.)])
    assert frozen_choice("initial", rows, 1) == (2, "covered_greedy", 1)
    assert signal(1, 0., 100.) == "initial"
    assert signal(2, 59., 100.) == "gap"
    assert signal(2, 60., 100.) == "other"
    assert threshold("gap", 1) == 1 and threshold("other", 1) == 0


def test_multiple_release_events_count_one_patch_year_but_preserve_the_chain():
    events = [dict(release_year=2011, patch_ids=["p"], source_operator_id=owner)
              for owner in ("original", "first_recipient")]
    first, chains = release_chains(events)
    assert first == events[:1]
    assert chains[2011, "p"] == events
