from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunnerSpec:
    """Metadata-only description of an experiment runner.

    The registry intentionally does not import runner modules. Importing root
    scripts can trigger plotting setup, CLI parsing, or model initialization.
    """

    script_name: str
    run_type: str
    question: str
    defensible_claim: str
    claim_boundary: str
    result_subdir: str


CANONICAL_RUNNERS: tuple[RunnerSpec, ...] = (
    RunnerSpec(
        script_name="run_integrated_workflow_v2.py",
        run_type="integrated_workflow_v2",
        question="Can the calibrated model produce a stable baseline worldline and collect indicators?",
        defensible_claim="Model health check, trend inspection, and indicator-pipeline audit.",
        claim_boundary="Not a policy net-effect identification design.",
        result_subdir="results/canonical/integrated_workflow_v2",
    ),
    RunnerSpec(
        script_name="run_counterfactual_analysis_v2.py",
        run_type="counterfactual_v2",
        question="Where do policy and no-policy worldlines diverge under the same model mechanism?",
        defensible_claim="Internal policy-channel direction, rent capitalization, and actor restructuring.",
        claim_boundary="Single-path differences are not real-world sampling inference.",
        result_subdir="results/canonical/counterfactual_v2",
    ),
    RunnerSpec(
        script_name="run_subsidy_evaluation_v2.py",
        run_type="subsidy_evaluation_v2",
        question="How do subsidy intensity gradients affect production, income, fiscal efficiency, and transfer?",
        defensible_claim="Within-model intensity gradients, nonlinear response, and fiscal-efficiency comparison.",
        claim_boundary="Not an empirical optimal subsidy standard.",
        result_subdir="results/canonical/subsidy_evaluation_v2",
    ),
    RunnerSpec(
        script_name="run_policy_optimization_v2.py",
        run_type="policy_optimization_v2",
        question="How do policy structures trade off under a fixed budget or bounded policy space?",
        defensible_claim="Policy-structure comparison and Pareto exploration inside the model.",
        claim_boundary="Not a real-world optimal policy package or local Quzhou prescription.",
        result_subdir="results/canonical/policy_optimization_v2",
    ),
)


LEGACY_RUNNERS: tuple[str, ...] = (
    "run_abm_empirical_validation.py",
    "run_abm_predictive_validation.py",
    "run_full_validation.py",
    "run_government_decision_support.py",
    "run_mechanism_ablation.py",
    "run_mechanism_ablation_baseline_shock.py",
    "run_nature_workflow.py",
    "run_paired_worldlines.py",
    "run_rl_learning_experiment.py",
    "run_rl_train_and_evaluate.py",
    "run_sensitivity_analysis.py",
    "run_shock_recovery_experiment.py",
    "run_tipping_surface_experiment.py",
)


def canonical_runner_names() -> tuple[str, ...]:
    return tuple(spec.script_name for spec in CANONICAL_RUNNERS)


def legacy_runner_names() -> tuple[str, ...]:
    return LEGACY_RUNNERS


def root_runner_names(root: Path | None = None) -> tuple[str, ...]:
    project_root = Path(root) if root is not None else PROJECT_ROOT
    return tuple(sorted(path.name for path in project_root.glob("run_*.py") if path.is_file()))


def classify_runner(script_name: str) -> str:
    if script_name in canonical_runner_names():
        return "canonical"
    if script_name in LEGACY_RUNNERS:
        return "legacy"
    return "unregistered"


def unregistered_root_runner_names(root: Path | None = None) -> tuple[str, ...]:
    canonical = set(canonical_runner_names())
    legacy = set(LEGACY_RUNNERS)
    return tuple(name for name in root_runner_names(root) if name not in canonical and name not in legacy)
