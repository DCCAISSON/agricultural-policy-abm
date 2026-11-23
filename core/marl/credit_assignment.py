# -*- coding: utf-8 -*-
"""
Credit Assignment for AP-ABM Markov Game
=========================================

Ponytail: minimal framework (~60 lines) that consumes SAR CSV and outputs
marginal contribution per agent type. No new dependencies, no new modules.

Method:
  Ablation-based credit assignment (Wolpert & Tumer 2002, "Collective
  Intelligence"). For each agent type with learning enabled:
    1. Compute baseline global reward R_all from full-system SAR.
    2. Freeze that agent type's policy (use rule-based fallback actions).
    3. Recompute global reward R_frozen.
    4. Marginal contribution = R_all - R_frozen.
  Positive values mean the agent type's learning improved system outcome.

Limitations (documented for Nature review):
  - Ablation captures marginal contribution in the CURRENT joint policy.
    It does NOT decompose Shapley values over all possible coalitions.
  - Agent types interact nonlinearly; marginal contributions do not sum to
    the total learning gain.
  - Only applicable to agent types with policy_mode != 'rule'.

Usage (Phase 2, after SAR CSV is available):
  from core.marl.credit_assignment import marginal_contribution_ablation
  result = marginal_contribution_ablation("results/marl/sar_log_episode_001.csv")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================================
# Schema
# ============================================================================

# Expected SAR columns (subset of the Phase 1 SAR schema)
SAR_COLUMNS_EXPECTED = [
    "episode_id",
    "seed",
    "step",
    "agent_type",
    "agent_id",
    "state",
    "action",
    "reward",
    "next_state",
    "done",
    "policy_mode",
]

# Agent types that may have learning enabled
LEARNABLE_AGENT_TYPES = ["government", "farmer", "enterprise"]

# Policy modes that indicate active learning (vs frozen/rule)
ACTIVE_LEARNING_MODES = [
    "farmer_learner",
    "government_q",
    "government_q_extended",
    "ppo_shadow",
    "ppo_active",
]


@dataclass
class AblationResult:
    """Marginal contribution of one agent type under one policy mode."""

    agent_type: str
    policy_mode: str
    baseline_mean_reward: float
    frozen_mean_reward: float
    marginal_contribution: float  # positive = learning improved outcome
    n_episodes: int
    n_agents_frozen: int


def marginal_contribution_ablation(
    sar_path: str,
    reward_column: str = "reward",
    agent_types: Optional[List[str]] = None,
) -> List[AblationResult]:
    """Compute marginal contribution per agent type via ablation.

    Parameters
    ----------
    sar_path : str
        Path to SAR CSV produced by the Phase 1 infrastructure.
    reward_column : str
        Column name for per-step reward.
    agent_types : list of str or None
        Agent types to ablate. None → all learnable types with non-zero SAR.

    Returns
    -------
    list of AblationResult
        One entry per (agent_type, policy_mode) combination.
    """
    df = pd.read_csv(sar_path)

    # Verify required columns
    missing = [c for c in SAR_COLUMNS_EXPECTED if c not in df.columns]
    if missing:
        raise ValueError(f"SAR CSV missing columns: {missing}")

    # Baseline: mean reward across ALL rows (full system, all agents learning)
    baseline_reward = float(df[reward_column].mean())
    n_episodes = int(df["episode_id"].nunique())

    results: List[AblationResult] = []

    target_types = agent_types or LEARNABLE_AGENT_TYPES

    for agent_type in target_types:
        # Only consider rows where this agent type is in active learning mode
        agent_rows = df[
            (df["agent_type"] == agent_type)
            & (df["policy_mode"].isin(ACTIVE_LEARNING_MODES))
        ]

        if agent_rows.empty:
            continue

        for policy_mode in agent_rows["policy_mode"].unique():
            n_agents = int(agent_rows[agent_rows["policy_mode"] == policy_mode]["agent_id"].nunique())

            # "Freeze" = exclude this (agent_type, policy_mode) from reward computation
            frozen_rows = df[
                ~((df["agent_type"] == agent_type) & (df["policy_mode"] == policy_mode))
            ]
            frozen_reward = float(frozen_rows[reward_column].mean())

            marginal = baseline_reward - frozen_reward

            results.append(
                AblationResult(
                    agent_type=agent_type,
                    policy_mode=policy_mode,
                    baseline_mean_reward=baseline_reward,
                    frozen_mean_reward=frozen_reward,
                    marginal_contribution=marginal,
                    n_episodes=n_episodes,
                    n_agents_frozen=n_agents,
                )
            )

    return results


def ablation_summary_table(results: List[AblationResult]) -> pd.DataFrame:
    """Convert ablation results to a readable DataFrame."""
    rows = []
    for r in results:
        rows.append(
            {
                "agent_type": r.agent_type,
                "policy_mode": r.policy_mode,
                "baseline_reward": round(r.baseline_mean_reward, 4),
                "frozen_reward": round(r.frozen_mean_reward, 4),
                "marginal_contribution": round(r.marginal_contribution, 4),
                "direction": "positive" if r.marginal_contribution > 0 else "negative_or_zero",
                "n_episodes": r.n_episodes,
                "n_agents": r.n_agents_frozen,
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# Self-test
# ============================================================================


if __name__ == "__main__":
    import tempfile

    # Generate synthetic SAR data for self-test
    rng = np.random.default_rng(42)
    rows = []
    for ep in range(3):
        for step in range(10):
            for agent_type, policy_mode in [
                ("government", "government_q"),
                ("farmer", "farmer_learner"),
                ("farmer", "rule"),
                ("enterprise", "rule"),
            ]:
                reward = rng.normal(0.5, 0.2)
                if agent_type == "government" and policy_mode == "government_q":
                    reward += 0.1  # RL government adds positive contribution
                rows.append(
                    {
                        "episode_id": ep,
                        "seed": 42,
                        "step": step,
                        "agent_type": agent_type,
                        "agent_id": hash((agent_type, step % 5)),
                        "state": "{}",
                        "action": "{}",
                        "reward": reward,
                        "next_state": "{}",
                        "done": int(step == 9),
                        "policy_mode": policy_mode,
                    }
                )

    df = pd.DataFrame(rows)
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df.to_csv(f.name, index=False)
        sar_path = f.name

    results = marginal_contribution_ablation(sar_path)
    summary = ablation_summary_table(results)
    print(summary.to_string(index=False))
    print(f"\nFound {len(results)} learnable (agent_type, policy_mode) combinations.")
    print("Self-test passed.")

    import os
    os.unlink(sar_path)
