import json
from types import SimpleNamespace

import pytest


def _model(*, next_year=False):
    step = 2 if next_year else 1
    actor = SimpleNamespace(
        unique_id="new" if next_year else "old",
        crop_portfolio={},
        planned_crop="Grain",
        grain_area=0.0 if next_year else 3.0,
        cash_crop_area=4.0 if next_year else 1.0,
        last_preproduction_cash_account={"step": step, "cultivated_area_mu": 4.0},
    )
    patches = [
        SimpleNamespace(
            get_patch_id=lambda i=i: i, area=area, current_operator=actor,
            owner=actor, leased_to=None, current_crop="Grain", preproduction_cash_step=step,
            preproduction_cultivated_area_mu=area / 2,
        )
        for i, area in (("a", 2.0), ("b", 6.0))
    ]
    return SimpleNamespace(
        step_count=step, simulation_base_year=2010, land_patches=patches,
        farmers=[], agricultural_enterprises=[actor], initial_household_cohort=[],
        revision_harvest_records={
            actor.unique_id: {"step": step, "grain_physical_kg": 0.0 if next_year else 1800.0,
                              "cash_output_model_units": 1000.0 if next_year else 250.0},
        },
    )


def test_account_allocation_reconciles_without_marking_a_plot_as_grain():
    from scripts import national_plot_use_observer as evidence

    model = _model()
    before = [vars(patch).copy() for patch in model.land_patches]
    result = evidence.snapshot_plot_use(model)
    small, large = result["patches"]
    assert result["year"] == 2011
    assert small["cultivated_area_mu"] == small["unknown_crop_area_mu"] == 1.0
    assert small["grain_area_mu"] is None and small["economic_crop_area_mu"] is None
    assert small["allocated_grain_area_mu"] == 0.75
    assert small["allocated_grain_output_kg"] == 450.0
    assert large["allocated_economic_crop_area_mu"] == 0.75
    assert large["allocated_economic_output_model_units"] == 187.5
    assert result["reconciliation"]["harvest"] == result["reconciliation"]["allocated"]
    assert [vars(patch) for patch in model.land_patches] == before
    json.dumps(result, allow_nan=False)


def test_inconsistent_funded_area_is_not_renormalized():
    from scripts import national_plot_use_observer as evidence

    model = _model()
    model.land_patches[0].preproduction_cultivated_area_mu = 0.5
    result = evidence.snapshot_plot_use(model)
    assert result["reconciliation"]["unreconciled_operators"]["old"]["cultivated_area_mu"] == 0.5
    assert result["reconciliation"]["residual_harvest_minus_allocated"]["cultivated_area_mu"] == 0.5
    assert all(p["crop_evidence"] == "unreconciled" and p["grain_area_mu"] is None
               for p in result["patches"])


def test_single_crop_identified_from_area_even_when_planned_crop_disagrees():
    from scripts import national_plot_use_observer as evidence

    result = evidence.snapshot_plot_use(_model(next_year=True))
    for plot, expected in zip(result["patches"], (1.0, 3.0), strict=True):
        assert plot["crop_evidence"] == "area_conservation_single_crop"
        assert plot["economic_crop_area_mu"] == expected
        assert plot["grain_area_mu"] == plot["unknown_crop_area_mu"] == 0.0


def test_mixed_harvest_identifies_both_crops_only_with_one_cultivated_plot():
    from scripts import national_plot_use_observer as evidence

    model = _model()
    model.land_patches[0].preproduction_cultivated_area_mu = 0.0
    model.land_patches[1].preproduction_cultivated_area_mu = 4.0
    idle, mixed = evidence.snapshot_plot_use(model)["patches"]
    assert idle["crop_evidence"] == "not_cultivated"
    assert mixed["crop_evidence"] == "area_conservation_single_cultivated_patch"
    assert mixed["grain_area_mu"] == 3.0 and mixed["economic_crop_area_mu"] == 1.0
    assert mixed["unknown_crop_area_mu"] == 0.0


def test_paired_release_keeps_unidentified_origin_and_missing_next_year():
    from scripts import national_plot_use_observer as evidence

    first = evidence.snapshot_plot_use(_model())
    second = evidence.snapshot_plot_use(_model(next_year=True))
    released = {"release_year": 2011, "patch_ids": ["a"], "source_operator_id": "old",
                "release_event_id": "ref-1", "release_reason": "urbanization"}
    last = dict(released, release_year=2012, source_operator_id="new", release_event_id="cmp-2")
    result = evidence.paired_release_destinations(
        [first, second], [first, second],
        reference_releases=[released], comparison_releases=[last], seed=31,
    )
    matched, terminal = result
    assert matched["seed"] == 31 and matched["patch_id"] == "a"
    assert matched["reference"]["release_event_id"] == "ref-1"
    assert matched["reference"]["source_matches_prior_operator"] is True
    assert matched["reference"]["origin"]["grain_area_mu"] is None
    assert matched["reference"]["origin"]["allocated_grain_area_mu"] == 0.75
    assert matched["reference"]["destination"]["operator_id"] == "new"
    assert matched["reference"]["destination"]["economic_crop_area_mu"] == 1.0
    assert matched["comparison"]["release_event_id"] is None
    assert terminal["comparison"]["destination"] is None
    json.dumps(result, allow_nan=False)


def test_frozen_national_first_year_snapshot_at_runner_hook(tmp_path):
    from scripts import national_plot_use_observer as evidence
    from scripts import run_national_government_q_20261119 as runner

    seed = 31
    base, economics, budget, _ = runner.load_base()
    model = runner.build_model(base, economics, seed)
    assert (len(model.farmers), len(model.agricultural_enterprises)) == (600, 60)
    model.config["policy"]["government_q_project"]["episode_end_step"] = 1
    runner.configure_government(model.government, {}, seed, training=False, fixed_action=0)

    def observe(current, directory):
        assert (directory / "annual_checks.json").is_file()
        runner.old.write(directory / "actual_plot_use.json", evidence.snapshot_plot_use(current))

    output = tmp_path / "episode"
    runner.run_episode(model, budget, output, seed, runner.source_hashes(), False, observe_year=observe)
    directory = output / "year_01"
    result = runner.old.read(directory / "actual_plot_use.json")
    accounts = runner.old.read(directory / "operator_snapshot.json")
    reconciled = result["reconciliation"]
    assert result["year"] == 2011
    assert reconciled["settlement_count"] >= 600
    assert reconciled["harvest"]["grain_output_kg"] == pytest.approx(accounts["grain_output_kg"])
    assert reconciled["harvest"]["grain_area_mu"] == pytest.approx(accounts["grain_area_mu"])
    assert reconciled["harvest"]["economic_crop_area_mu"] == pytest.approx(accounts["cash_crop_area_mu"])
    assert reconciled["allocated"] == pytest.approx(reconciled["harvest"])
    assert any(p["crop_evidence"] == "area_conservation_single_crop" for p in result["patches"])
