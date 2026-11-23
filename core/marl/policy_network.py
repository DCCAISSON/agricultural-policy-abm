# -*- coding: utf-8 -*-
"""
PPO Policy Network for Government Subsidy Optimisation
======================================================

Ponytail design:
  - numpy-only: zero framework dependencies (no torch, no tf, no jax).
  - ~3000 parameters: small enough for CPU training on 300-agent rollouts.
  - Standalone: imports nothing from the AP-ABM codebase.
  - Compatible with government_agent.py's _get_rl_state_extended() (8-dim obs)
    and 3 continuous actions: (grain_subsidy_per_mu, scale_subsidy_rate,
    grain_price_floor).

Architecture:
  obs (8) -> FC 64 ReLU -> FC 32 ReLU -> mean (3) + log_std (3)

PPO specifics:
  - Clipped surrogate objective (epsilon=0.2).
  - Generalised Advantage Estimation (GAE, lambda=0.95).
  - Hand-rolled Adam optimiser.
  - Mini-batch updates over collected rollouts.

Usage (Phase 2, after SAR logging and Q-learning baseline):
  from core.marl.policy_network import PPOPolicy, PPOHyperparams
  policy = PPOPolicy(obs_dim=8, act_dim=3)
  # ... collect rollout from MainModel.step() ...
  policy.update(observations, actions, advantages, old_log_probs)

References:
  Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
  Schulman et al. (2016) "High-Dimensional Continuous Control Using GAE"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Hyperparameters
# ============================================================================


@dataclass
class PPOHyperparams:
    """Default PPO hyperparameters tuned for ~300-agent, 30-year ABM rollouts."""

    # Network
    hidden_sizes: Tuple[int, ...] = (64, 32)
    activation: str = "relu"

    # PPO clipping
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Optimisation
    learning_rate: float = 3e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    # Training loop
    n_epochs: int = 10
    mini_batch_size: int = 64
    max_grad_norm: float = 0.5

    # Action bounds (will be scaled to match government policy ranges)
    action_low: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))
    action_high: np.ndarray = field(default_factory=lambda: np.array([400.0, 300.0, 3.2], dtype=np.float64))
    # ^ grain_subsidy_per_mu ∈ [0, 400], scale_subsidy_rate ∈ [0, 300], grain_price_floor ∈ [0, 3.2]


# ============================================================================
# Neural network helpers (numpy-only)
# ============================================================================


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _softplus(x: np.ndarray) -> np.ndarray:
    """log(1 + exp(x)), clamped for numerical stability."""
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -20.0, 20.0))))


# ============================================================================
# PPO Policy Network
# ============================================================================


class PPOPolicy:
    """Gaussian policy network with PPO clipped-objective training.

    Parameters
    ----------
    obs_dim : int
        Observation dimension (8 for extended government RL state).
    act_dim : int
        Action dimension (3 for grain_subsidy, scale_subsidy, price_floor).
    hp : PPOHyperparams
        Hyperparameters.
    seed : int or None
        Random seed for weight initialisation.
    """

    def __init__(
        self,
        obs_dim: int = 8,
        act_dim: int = 3,
        hp: Optional[PPOHyperparams] = None,
        seed: Optional[int] = None,
    ):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hp = hp or PPOHyperparams()

        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        # --- Build layers ---
        sizes = [self.obs_dim] + list(self.hp.hidden_sizes)
        self.weights: List[Dict[str, np.ndarray]] = []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            fan_out = sizes[i + 1]
            # He initialisation
            std = np.sqrt(2.0 / fan_in)
            self.weights.append({
                "W": rng.normal(0.0, std, (fan_in, fan_out)).astype(np.float64),
                "b": np.zeros(fan_out, dtype=np.float64),
            })

        # --- Output heads ---
        last_hidden = sizes[-1]
        std_out = np.sqrt(2.0 / last_hidden)
        self.mean_head_W = rng.normal(0.0, std_out, (last_hidden, act_dim)).astype(np.float64)
        self.mean_head_b = np.zeros(act_dim, dtype=np.float64)
        self.log_std = np.zeros(act_dim, dtype=np.float64)  # state-independent log-std

        # --- Value head (critic) ---
        self.value_head_W = rng.normal(0.0, std_out, (last_hidden, 1)).astype(np.float64)
        self.value_head_b = np.zeros(1, dtype=np.float64)

        # --- Adam state ---
        self._adam_m: Dict[str, np.ndarray] = {}
        self._adam_v: Dict[str, np.ndarray] = {}
        self._adam_t = 0

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward_hidden(self, obs: np.ndarray) -> np.ndarray:
        """obs: (batch, obs_dim) or (obs_dim,) -> hidden: (batch, last_hidden)."""
        x = np.atleast_2d(obs.astype(np.float64))
        for layer in self.weights:
            x = x @ layer["W"] + layer["b"]
            x = _relu(x)
        return x

    def forward(
        self, obs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (action_mean, action_std, value) for the given observation(s).

        Parameters
        ----------
        obs : ndarray, shape (obs_dim,) or (batch, obs_dim)

        Returns
        -------
        mean : ndarray, shape (..., act_dim)
            Action means in [-1, 1] (pre-scaling).
        std : ndarray, shape (act_dim,)
            Action standard deviations.
        value : ndarray, shape (..., 1)
            State-value estimate.
        """
        hidden = self._forward_hidden(obs)
        mean = hidden @ self.mean_head_W + self.mean_head_b
        std = np.exp(np.clip(self.log_std, -20.0, 2.0))
        # ^ exp(log_std) is always positive; clipping prevents inf
        value = hidden @ self.value_head_W + self.value_head_b
        return mean, std, value

    def sample_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample an action from the Gaussian policy.

        Returns
        -------
        action : ndarray, shape (act_dim,) — in raw space (clipped to bounds).
        log_prob : ndarray, shape (act_dim,) — log-prob of sampled action.
        value : ndarray, shape (1,) — state value.
        """
        mean, std, value = self.forward(obs)
        if deterministic:
            raw_action = mean
        else:
            raw_action = mean + std * np.random.randn(*mean.shape)
        raw_action = np.atleast_1d(raw_action.squeeze())

        # Scale from [-inf, +inf] to [action_low, action_high]
        # We use tanh-squashing: a = action_low + (action_high - action_low) * (tanh(raw) + 1) / 2
        # But simpler: just clip to bounds for now
        action = np.clip(raw_action, self.hp.action_low, self.hp.action_high)

        # Log-prob under diagonal Gaussian
        log_prob = -0.5 * (
            np.log(2.0 * np.pi * std**2) + ((action - mean.squeeze()) / std) ** 2
        ).sum()

        return action, np.array(log_prob), value.squeeze()

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(
        self,
        observations: np.ndarray,       # (n_steps, obs_dim)
        actions: np.ndarray,            # (n_steps, act_dim)
        old_log_probs: np.ndarray,      # (n_steps,)
        advantages: np.ndarray,         # (n_steps,)
        returns: np.ndarray,            # (n_steps,)
    ) -> Dict[str, float]:
        """PPO clipped-objective update over collected rollouts.

        Returns a dict of diagnostic scalars (loss, approx_kl, etc.).
        """
        n_samples = len(observations)
        if n_samples == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Normalise advantages
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.hp.n_epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, self.hp.mini_batch_size):
                batch_idx = indices[start:start + self.hp.mini_batch_size]
                batch_obs = observations[batch_idx]
                batch_act = actions[batch_idx]
                batch_old_lp = old_log_probs[batch_idx]
                batch_adv = adv[batch_idx]
                batch_ret = returns[batch_idx]

                # Forward
                mean, std, values = self.forward(batch_obs)
                values = values.squeeze(-1)

                # New log-probs
                diff = (batch_act - mean.squeeze()) / (std + 1e-8)
                new_log_probs = -0.5 * (np.log(2.0 * np.pi * std**2) + diff**2).sum(axis=1)

                # Ratio
                ratio = np.exp(new_log_probs - batch_old_lp)

                # Clipped surrogate
                clip_eps = self.hp.clip_epsilon
                surr1 = ratio * batch_adv
                surr2 = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch_adv
                policy_loss = -np.mean(np.minimum(surr1, surr2))

                # Value loss
                value_loss = np.mean((batch_ret - values) ** 2)

                # Entropy bonus
                entropy = np.mean(0.5 * np.log(2.0 * np.pi * np.e * std**2))

                # Total loss
                loss = (
                    policy_loss
                    + self.hp.value_coef * value_loss
                    - self.hp.entropy_coef * entropy
                )

                # Gradient step (Adam)
                self._adam_step(loss, batch_obs, batch_act, batch_old_lp, batch_adv, batch_ret)

                total_policy_loss += float(policy_loss)
                total_value_loss += float(value_loss)
                total_entropy += float(entropy)
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    # ------------------------------------------------------------------
    # Hand-rolled Adam (ponytail: no torch dependency)
    # ------------------------------------------------------------------

    def _adam_step(
        self,
        loss: np.ndarray,
        obs: np.ndarray,
        act: np.ndarray,
        old_lp: np.ndarray,
        adv: np.ndarray,
        ret: np.ndarray,
    ) -> None:
        """Compute gradients via finite differences and apply Adam update.

        IMPORTANT: This is a *placeholder* finite-difference gradient.
        Production use should compute analytical gradients. For a network
        with ~3000 parameters and rollouts of ~30 steps, finite differences
        are tractable but slow. Replace with analytical backprop before
        claiming PPO in a Nature figure.
        """
        # Store current param values
        params = self._get_params()
        eps = 1e-6
        grads = {}

        for key in params:
            original = params[key].copy()
            grad = np.zeros_like(original, dtype=np.float64)
            flat = original.ravel()
            grad_flat = grad.ravel()

            for i in range(len(flat)):
                old_val = flat[i]
                flat[i] = old_val + eps
                self._set_params(params)
                loss_plus = self._compute_loss(obs, act, old_lp, adv, ret)
                flat[i] = old_val - eps
                self._set_params(params)
                loss_minus = self._compute_loss(obs, act, old_lp, adv, ret)
                flat[i] = old_val
                grad_flat[i] = (loss_plus - loss_minus) / (2.0 * eps)

            self._set_params({key: original})  # restore
            grads[key] = grad.reshape(original.shape)

        # Adam update
        self._adam_t += 1
        t = self._adam_t
        lr = self.hp.learning_rate
        beta1, beta2, eps_adam = self.hp.adam_beta1, self.hp.adam_beta2, self.hp.adam_eps

        for key, grad in grads.items():
            if key not in self._adam_m:
                self._adam_m[key] = np.zeros_like(grad)
                self._adam_v[key] = np.zeros_like(grad)

            self._adam_m[key] = beta1 * self._adam_m[key] + (1.0 - beta1) * grad
            self._adam_v[key] = beta2 * self._adam_v[key] + (1.0 - beta2) * (grad**2)

            m_hat = self._adam_m[key] / (1.0 - beta1**t)
            v_hat = self._adam_v[key] / (1.0 - beta2**t)

            params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)

        # Gradient clipping
        for key in params:
            norm = np.sqrt(np.sum(params[key] ** 2))
            if norm > self.hp.max_grad_norm:
                params[key] *= self.hp.max_grad_norm / norm

        self._set_params(params)

    def _compute_loss(
        self, obs, act, old_lp, adv, ret
    ) -> float:
        """Compute total loss for current parameters (used in finite-diff grad)."""
        mean, std, values = self.forward(obs)
        values = values.squeeze(-1)
        diff = (act - mean.squeeze()) / (std + 1e-8)
        new_lp = -0.5 * (np.log(2.0 * np.pi * std**2) + diff**2).sum(axis=1)
        ratio = np.exp(new_lp - old_lp)
        surr1 = ratio * adv
        surr2 = np.clip(ratio, 1.0 - self.hp.clip_epsilon, 1.0 + self.hp.clip_epsilon) * adv
        policy_loss = -np.mean(np.minimum(surr1, surr2))
        value_loss = np.mean((ret - values) ** 2)
        entropy = np.mean(0.5 * np.log(2.0 * np.pi * np.e * std**2))
        return float(
            policy_loss + self.hp.value_coef * value_loss - self.hp.entropy_coef * entropy
        )

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def _get_params(self) -> Dict[str, np.ndarray]:
        params = {}
        for i, layer in enumerate(self.weights):
            params[f"layer_{i}_W"] = layer["W"]
            params[f"layer_{i}_b"] = layer["b"]
        params["mean_head_W"] = self.mean_head_W
        params["mean_head_b"] = self.mean_head_b
        params["log_std"] = self.log_std
        params["value_head_W"] = self.value_head_W
        params["value_head_b"] = self.value_head_b
        return params

    def _set_params(self, params: Dict[str, np.ndarray]) -> None:
        for i in range(len(self.weights)):
            self.weights[i]["W"] = params[f"layer_{i}_W"]
            self.weights[i]["b"] = params[f"layer_{i}_b"]
        self.mean_head_W = params["mean_head_W"]
        self.mean_head_b = params["mean_head_b"]
        self.log_std = params["log_std"]
        self.value_head_W = params["value_head_W"]
        self.value_head_b = params["value_head_b"]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save policy weights to a .npz file."""
        np.savez_compressed(path, **self._get_params())

    @classmethod
    def load(cls, path: str, obs_dim: int = 8, act_dim: int = 3) -> "PPOPolicy":
        """Load policy weights from a .npz file."""
        policy = cls(obs_dim=obs_dim, act_dim=act_dim)
        data = np.load(path)
        params = {key: data[key] for key in data.files}
        policy._set_params(params)
        return policy

    @property
    def n_params(self) -> int:
        return sum(int(p.size) for p in self._get_params().values())


# ============================================================================
# GAE computation (standalone, uses only numpy)
# ============================================================================


def compute_gae(
    rewards: np.ndarray,        # (n_steps,)
    values: np.ndarray,         # (n_steps,)
    dones: np.ndarray,          # (n_steps,)
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and returns.

    Parameters
    ----------
    rewards : ndarray, shape (n_steps,)
    values : ndarray, shape (n_steps,) — V(s_t) for each step
    dones : ndarray, shape (n_steps,) — 1.0 if episode terminated, else 0.0
    gamma : float
    gae_lambda : float

    Returns
    -------
    advantages : ndarray, shape (n_steps,)
    returns : ndarray, shape (n_steps,) — advantages + values
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float64)
    last_advantage = 0.0
    last_value = 0.0  # V(s_{T+1}) = 0 for terminal

    for t in reversed(range(n)):
        delta = rewards[t] + gamma * last_value * (1.0 - dones[t]) - values[t]
        last_advantage = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_advantage
        advantages[t] = last_advantage
        last_value = values[t]

    returns = advantages + values
    return advantages, returns
