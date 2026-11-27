"""Shared Four-Region Fiscal Authority & Two-Level Ridge-Regularized Fitted Q-Iteration."""

from dataclasses import dataclass
from math import floor, fsum
import numpy as np

from competing_pmp.calibration_2010 import CalibratedRegion
from competing_pmp.config import CompetingDesign
from competing_pmp.economy import RegionalPMPEconomy

LocalAllocation = tuple[str, int]
JointAction = tuple[LocalAllocation, ...]
OFF_ALLOC: LocalAllocation = ("none", 0)


@dataclass(frozen=True, slots=True)
class SharedRegionalBudget:
    """Common annual budget (98,953 yuan) across the 4 regions with area-proportional & focused menu."""

    common_yuan: int
    region_areas: tuple[float, ...]
    tradable_areas: tuple[float, ...]
    instruments: tuple[str, ...]
    nominal_cap_mu: float
    overflow_reallocation: bool

    def _area_proportional_caps(self, total_yuan: int, indices: tuple[int, ...]) -> dict[int, int]:
        tot_area = fsum(self.region_areas[i] for i in indices)
        raw = [total_yuan * (self.region_areas[i] / tot_area) for i in indices]
        floors = [floor(x) for x in raw]
        rem = total_yuan - sum(floors)
        order = sorted(range(len(indices)), key=lambda k: (-(raw[k] - floors[k]), indices[k]))
        for k in order[:rem]:
            floors[k] += 1
        return {indices[k]: floors[k] for k in range(len(indices))}

    def menu(self) -> tuple[JointAction, ...]:
        """21 actions matching cardinality of 476b492:
        - Action 0: P0 (all zero)
        - Actions 1..16: Region-focused allocation (4 regions x 4 instruments), respecting 85 yuan/mu cap
                         and routing statutory-cap overflow proportionally to other regions (if enabled)
        - Actions 17..20: National area-proportional allocation across all 4 regions (4 instruments)
        """
        n_reg = len(self.region_areas)
        actions: list[JointAction] = [(OFF_ALLOC,) * n_reg]

        for r_idx in range(n_reg):
            for inst in self.instruments:
                max_useful = floor(self.nominal_cap_mu * self.region_areas[r_idx])
                if not self.overflow_reallocation or self.common_yuan <= max_useful:
                    row = tuple((inst, self.common_yuan) if j == r_idx else OFF_ALLOC for j in range(n_reg))
                else:
                    primary = max_useful
                    excess = self.common_yuan - primary
                    others = tuple(j for j in range(n_reg) if j != r_idx)
                    spill = self._area_proportional_caps(excess, others)
                    row = tuple(
                        (inst, primary) if j == r_idx else ((inst, spill[j]) if spill[j] > 0 else OFF_ALLOC)
                        for j in range(n_reg)
                    )
                actions.append(row)

        prop_caps = self._area_proportional_caps(self.common_yuan, tuple(range(n_reg)))
        for inst in self.instruments:
            actions.append(tuple((inst, prop_caps[j]) for j in range(n_reg)))

        assert len(actions) == 21
        for act in actions:
            assert sum(cap for _, cap in act) in (0, self.common_yuan)
        return tuple(actions)

    def step(self, worlds: tuple[RegionalPMPEconomy, ...], action: JointAction, *, ledger: bool = False) -> tuple[dict, ...]:
        rows = []
        for w, (pol, cap) in zip(worlds, action, strict=True):
            s, _, _ = w.step(pol, float(cap), ledger=ledger)
            rows.append(s)
        if fsum(r["committed"] for r in rows) > self.common_yuan + 1e-5:
            raise ValueError("Shared fiscal commitment exceeded annual common budget")
        return tuple(rows)


def make_shared_budget(regions: tuple[CalibratedRegion, ...], design: CompetingDesign) -> SharedRegionalBudget:
    tot_mu = fsum(r.physical_area for r in regions)
    return SharedRegionalBudget(
        common_yuan=floor(design.budget_yuan_per_sample_mu * tot_mu),
        region_areas=tuple(r.physical_area for r in regions),
        tradable_areas=tuple(r.verified_tradable_area for r in regions),
        instruments=design.instruments,
        nominal_cap_mu=design.nominal_rate_yuan_per_mu,
        overflow_reallocation=design.overflow_reallocation,
    )


def create_worlds(seed: int, regions: tuple[CalibratedRegion, ...], design: CompetingDesign) -> tuple[RegionalPMPEconomy, ...]:
    return tuple(
        RegionalPMPEconomy(len(regions) * seed + i, reg, design)
        for i, reg in enumerate(regions)
    )


def extract_planner_features(worlds: tuple[RegionalPMPEconomy, ...], design: CompetingDesign) -> np.ndarray:
    """Noise-filtered normalized continuous Markov state vector (1 + 4*4 = 17 features):
    bias=1.0, and for each region: [grain_cult_rate, unabsorbed_rate, leased_rate, tight_liquidity_share].
    """
    feats = [1.0]
    for w in worlds:
        obs = w.observed
        feats.extend([
            float(obs["grain_cult_rate"]),
            float(obs["unabsorbed_rate"]),
            float(obs["leased_rate"]),
            float(obs["tight_liquidity_share"]),
        ])
    return np.asarray(feats, dtype=np.float64)


def discrete_state_code(worlds: tuple[RegionalPMPEconomy, ...], design: CompetingDesign) -> str:
    """Compact human-readable structural state label for audit logs."""
    rem = design.main_years - worlds[0].year
    tags = []
    for w in worlds:
        g_bin = int(w.observed["grain_cult_rate"] >= 0.62)
        u_bin = int(w.observed["unabsorbed_rate"] >= 0.08)
        t_bin = int(w.observed["tight_liquidity_share"] >= 0.25)
        tags.append(f"{g_bin}{u_bin}{t_bin}")
    return f"h{rem}_" + "_".join(tags)


def fork_all_actions(
    worlds: tuple[RegionalPMPEconomy, ...],
    budget: SharedRegionalBudget,
    menu: tuple[JointAction, ...],
) -> tuple[np.ndarray, np.ndarray, list[tuple[RegionalPMPEconomy, ...]]]:
    """Evaluate all 21 joint actions from identical opening states without double-simulating local actions."""
    local_cache: list[dict[LocalAllocation, tuple[RegionalPMPEconomy, float, float]]] = []
    for i, w in enumerate(worlds):
        distinct_local = list(dict.fromkeys(act[i] for act in menu))
        res_i: dict[LocalAllocation, tuple[RegionalPMPEconomy, float, float]] = {}
        for loc_act in distinct_local:
            w_copy = w.fork()
            row, _, _ = w_copy.step(loc_act[0], float(loc_act[1]), ledger=False)
            res_i[loc_act] = (w_copy, float(row["grain_kg"]), float(row["spending"]))
        local_cache.append(res_i)

    grains = np.zeros(len(menu), dtype=np.float64)
    spends = np.zeros(len(menu), dtype=np.float64)
    next_worlds: list[tuple[RegionalPMPEconomy, ...]] = []
    for a_idx, act in enumerate(menu):
        nw = tuple(local_cache[i][loc][0] for i, loc in enumerate(act))
        grains[a_idx] = fsum(local_cache[i][loc][1] for i, loc in enumerate(act))
        spends[a_idx] = fsum(local_cache[i][loc][2] for i, loc in enumerate(act))
        next_worlds.append(nw)
    return grains, spends, next_worlds


def fit_backward_fqi(
    rollouts_by_year: dict[int, list[tuple[np.ndarray, np.ndarray, list[np.ndarray]]]],
    design: CompetingDesign,
    n_actions: int,
) -> dict:
    """Backward Finite-Horizon Ridge-Regularized Fitted Q-Iteration (gamma=1 for Q, gamma=0 for Myopic).

    rollouts_by_year[year] contains tuples (phi_t, r_vec, [phi_{t+1}^{(a)} for a in 0..A-1]).
    Returns per-year weight matrices W_q[year] and W_myopic[year] of shape (n_features, n_actions).
    """
    q_weights: dict[int, list[list[float]]] = {}
    myopic_weights: dict[int, list[list[float]]] = {}
    fit_r2: dict[str, float] = {}

    next_w_q: np.ndarray | None = None

    for year in range(design.main_years, 0, -1):
        samples = rollouts_by_year[year]
        x_mat = np.vstack([s[0] for s in samples])  # (N, d)
        r_mat = np.vstack([s[1] for s in samples])  # (N, A)
        n_samples, d_dim = x_mat.shape

        # Ridge normal matrix (do not penalize intercept column 0)
        reg = design.ridge_alpha * np.eye(d_dim)
        reg[0, 0] = 1e-8
        gram_inv = np.linalg.solve(x_mat.T @ x_mat + reg, x_mat.T)  # (d, N)

        # Myopic (gamma = 0): immediate expected grain_kg
        w_m = gram_inv @ r_mat  # (d, A)
        myopic_weights[year] = w_m.tolist()

        # Full horizon Q (gamma = 1): immediate grain_kg + max_{a'} Q_{t+1}(s'_{t+1}, a')
        if next_w_q is None:
            targets_q = r_mat
        else:
            cont = np.zeros((n_samples, n_actions), dtype=np.float64)
            for idx_s, (_, _, next_phis) in enumerate(samples):
                next_mat = np.vstack(next_phis)  # (A, d)
                q_next = next_mat @ next_w_q     # (A, A)
                # Action 0 (P0) excluded only if strictly dominated; standard max over all 21 legal actions
                cont[idx_s, :] = np.max(q_next, axis=1)
            targets_q = r_mat + cont

        w_q = gram_inv @ targets_q
        q_weights[year] = w_q.tolist()
        next_w_q = w_q

        pred_q = x_mat @ w_q
        ss_res = float(np.sum((targets_q - pred_q) ** 2))
        ss_tot = float(np.sum((targets_q - np.mean(targets_q, axis=0, keepdims=True)) ** 2))
        fit_r2[f"year_{year}"] = round(1.0 - ss_res / max(1e-9, ss_tot), 4)

    return {
        "q_weights": {str(k): v for k, v in q_weights.items()},
        "myopic_weights": {str(k): v for k, v in myopic_weights.items()},
        "fallback_action": 17,  # Neutral area-proportional grain_pre (never punitive P0!)
        "fit_r2": fit_r2,
    }


def select_planner_action(
    strategy: str,
    year: int,
    worlds: tuple[RegionalPMPEconomy, ...],
    design: CompetingDesign,
    frozen_model: dict,
    *,
    stopped: bool = False,
) -> tuple[int, str, float, float]:
    """Select annual joint fiscal action; returns (action_idx, reason, pred_q_kg, pred_myopic_kg)."""
    if stopped:
        return 0, "forced_stop", 0.0, 0.0
    if strategy.startswith("fixed_"):
        act = int(strategy.split("_")[1])
        return act, "fixed", 0.0, 0.0

    is_repeat = strategy.endswith("_repeat")
    base_name = strategy.replace("_repeat", "")
    block = frozen_model["repeat_fit" if is_repeat else "main_fit"]

    phi = extract_planner_features(worlds, design)
    w_q = np.asarray(block["q_weights"][str(year)], dtype=np.float64)
    w_m = np.asarray(block["myopic_weights"][str(year)], dtype=np.float64)

    q_vals = phi @ w_q
    m_vals = phi @ w_m

    if not (np.all(np.isfinite(q_vals)) and np.all(np.isfinite(m_vals))):
        fb = int(block["fallback_action"])
        return fb, "fallback_proportional", 0.0, 0.0

    if base_name == "q":
        chosen = int(np.argmax(q_vals))
    elif base_name == "myopic":
        chosen = int(np.argmax(m_vals))
    else:
        raise ValueError(f"Unknown planner strategy: {strategy}")

    return chosen, "fqi_supported", float(q_vals[chosen]), float(m_vals[chosen])
