"""Central Government Shared Fiscal Authority & 6-Way Policy Solver (V2 / V2.2).

Implements fair same-environment, same-budget, same-lagged-information comparison across:
1. `p0`: Baseline institutional environment (background fertility protection only, zero incremental budget)
2. `uniform_fixed`: Best national uniform-instrument area-proportional policy (selected on dev seeds)
3. `regional_fixed`: Best regionally differentiated/tilted fixed policy (selected on dev seeds)
4. `transparent_rule`: Explicit zero-black-box domain-expert dynamic rule
5. `myopic`: Same lagged public state (17-dim in V2.2 30-Action clean menu; 21-dim in legacy V2.0 menu) & disjoint train seeds, gamma = 0 (single-year horizon)
6. `gov_rl`: Finite-horizon Backward Ridge-Regularized Fitted Q-Iteration (gamma = 0.96 in V2.2 32 train seeds; 8-year horizon)
"""

from concurrent.futures import ProcessPoolExecutor
from math import fsum
import os
from pathlib import Path
import pickle
import warnings
import numpy as np

from mesa_agri_abm.calibration import CalibratedRegionV2
from mesa_agri_abm.config import V2Design
from mesa_agri_abm.model import (
    AgriSuccessionMesaModel,
    LocalAllocV2,
    RegionalSubEconomy,
    _predict_q_values,
)


def fork_all_menu_actions(
    model: AgriSuccessionMesaModel,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], dict[int, dict[LocalAllocV2, RegionalSubEconomy]]]:
    """Evaluate all joint menu actions from `model`'s opening state by caching distinct regional pairs."""
    next_year = model.year + 1
    reg_cache: dict[int, dict[LocalAllocV2, tuple[RegionalSubEconomy, float, float, float, list[float]]]] = {}

    for r_idx, reg in enumerate(model.regions):
        distinct_pairs = list(dict.fromkeys(act[r_idx] for act in model.menu))
        sub_map: dict[LocalAllocV2, tuple[RegionalSubEconomy, float, float, float, list[float]]] = {}
        for pair in distinct_pairs:
            r_clone = reg.clone_into(model)
            r_sum, _ = r_clone.step_region(next_year, pair[0], float(pair[1]), model.design)
            f_vec = r_clone.public_feature_list(model.design)
            rent_excess = max(0.0, float(r_sum["unit_rent_yuan_mu"]) - reg.tech.ask_rent_yuan_phys_mu * 2.4) * float(r_sum["leased_area_mu"])
            sub_map[pair] = (r_clone, float(r_sum["grain_kg"]), float(r_sum["spending"]), rent_excess, f_vec)
        reg_cache[r_idx] = sub_map

    n_act = len(model.menu)
    grains = np.zeros(n_act, dtype=np.float64)
    spends = np.zeros(n_act, dtype=np.float64)
    rent_penalties = np.zeros(n_act, dtype=np.float64)
    next_phis: list[np.ndarray] = []

    for a_idx, act in enumerate(model.menu):
        grains[a_idx] = fsum(reg_cache[i][act[i]][1] for i in range(len(model.regions)))
        spends[a_idx] = fsum(reg_cache[i][act[i]][2] for i in range(len(model.regions)))
        rent_penalties[a_idx] = fsum(reg_cache[i][act[i]][3] for i in range(len(model.regions))) / 100_000.0
        if model.design.menu_mode == "v2_2_intertemporal":
            avg_rem = fsum(reg_cache[i][act[i]][0].observed.get("village_pool_rem_frac", 0.0) for i in range(len(model.regions))) / max(1, len(model.regions))
            feat = [
                1.0,
                next_year / max(1.0, float(model.design.main_years)),
                avg_rem,
                1.0 if avg_rem <= 1e-9 else 0.0,
            ]
        else:
            feat = [1.0]
        for i in range(len(model.regions)):
            feat.extend(reg_cache[i][act[i]][4])
        next_phis.append(np.asarray(feat, dtype=np.float64))

    clone_lookup = {
        i: {pair: tup[0] for pair, tup in sub_map.items()}
        for i, sub_map in reg_cache.items()
    }
    return grains, spends, rent_penalties, next_phis, clone_lookup


def advance_model_from_cache(
    model: AgriSuccessionMesaModel,
    chosen_action_idx: int,
    clone_lookup: dict[int, dict[LocalAllocV2, RegionalSubEconomy]],
) -> AgriSuccessionMesaModel:
    """Assemble the next-year AgriSuccessionMesaModel state for `chosen_action_idx` from cached clones."""
    nxt = AgriSuccessionMesaModel.__new__(AgriSuccessionMesaModel)
    import mesa
    mesa.Model.__init__(nxt)
    nxt.master_seed = model.master_seed
    nxt.design = model.design
    nxt.year = model.year + 1
    nxt.running = True
    nxt.menu = model.menu
    nxt.government = model.government.clone_into(nxt)
    act = model.menu[chosen_action_idx]
    nxt.regions = tuple(clone_lookup[i][act[i]].clone_into(nxt) for i in range(len(model.regions)))
    for r in nxt.regions:
        r.model = nxt
        for op in r.operators.values():
            op.model = nxt
    nxt.all_mesa_agents = [nxt.government]
    for r in nxt.regions:
        nxt.all_mesa_agents.extend(r.operators.values())
    nxt.last_joint_summary = {}
    nxt.last_region_summaries = ()
    nxt.history_joint = []
    nxt.history_regions = []
    nxt.last_interprov_pair_flows = model.last_interprov_pair_flows.copy()
    nxt.datacollector = model.datacollector
    return nxt


def _rollout_single_train_seed(args: tuple) -> dict[int, tuple[np.ndarray, np.ndarray, list[np.ndarray]]]:
    s, calibrated_regions, design, seed_offset = args
    exp_rng = np.random.default_rng(seed_offset + s)
    out: dict[int, tuple[np.ndarray, np.ndarray, list[np.ndarray]]] = {}
    m = AgriSuccessionMesaModel(s, calibrated_regions, design, policy_mode="p0")
    n_actions = len(m.menu)
    for yr in range(1, design.main_years + 1):
        phi_t = m.extract_public_state_features()
        grains_kg, spends_yuan, rent_pen, next_phis, clone_lookup = fork_all_menu_actions(m)
        base_tons = grains_kg[0] / 1000.0
        base_rent_pen = rent_pen[0]
        rewards = (
            (grains_kg / 1000.0 - base_tons)
            - design.gov_fiscal_cost_weight * (spends_yuan / 100_000.0)
            - design.gov_rent_cap_penalty_weight * np.maximum(0.0, rent_pen - base_rent_pen)
        )
        out[yr] = (phi_t, rewards, next_phis)
        if yr < design.main_years:
            u_draw = exp_rng.random()
            a_next = int(exp_rng.integers(0, n_actions)) if u_draw < 0.65 else int(np.argmax(rewards))
            m = advance_model_from_cache(m, a_next, clone_lookup)
    return out


def fit_v2_government_fqi(
    train_seeds: list[int],
    calibrated_regions: tuple[CalibratedRegionV2, ...],
    design: V2Design,
    *,
    exploration_seed: int = 20261127,
    max_workers: int | None = None,
) -> dict:
    """Collect multi-year exploratory rollouts on `train_seeds` and fit Backward Ridge-FQI (`gov_rl` & `myopic`)."""
    rollouts_by_year: dict[int, list[tuple[np.ndarray, np.ndarray, list[np.ndarray]]]] = {
        yr: [] for yr in range(1, design.main_years + 1)
    }
    probe = AgriSuccessionMesaModel(train_seeds[0], calibrated_regions, design, policy_mode="p0")
    n_actions = len(probe.menu)

    tasks = [(s, calibrated_regions, design, exploration_seed) for s in train_seeds]
    workers = max_workers if max_workers is not None else min(8, max(1, (os.cpu_count() or 4) - 2))
    if workers > 1 and len(tasks) >= 8:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            seed_results = list(pool.map(_rollout_single_train_seed, tasks))
    else:
        seed_results = [_rollout_single_train_seed(t) for t in tasks]

    for sr in seed_results:
        for yr, tup in sr.items():
            rollouts_by_year[yr].append(tup)

    q_weights: dict[int, list[list[float]]] = {}
    myopic_weights: dict[int, list[list[float]]] = {}
    fit_r2_q: dict[int, float] = {}
    fit_r2_myopic: dict[int, float] = {}
    next_w_q: np.ndarray | None = None

    for yr in range(design.main_years, 0, -1):
        batch = rollouts_by_year[yr]
        x_mat = np.vstack([row[0] for row in batch])
        r_mat = np.vstack([row[1] for row in batch])
        n_feat = x_mat.shape[1]

        if design.menu_mode == "v2_25":
            reg_eye = design.fqi_ridge_penalty * np.eye(n_feat, dtype=np.float64)
            reg_eye[0, 0] = 1e-6  # Unpenalized intercept
            xtx_inv_xt = np.linalg.solve(x_mat.T @ x_mat + reg_eye, x_mat.T)
            w_myo = xtx_inv_xt @ r_mat
        else:
            mu_x = np.mean(x_mat[:, 1:], axis=0)
            sd_x = np.maximum(1e-3, np.std(x_mat[:, 1:], axis=0))
            z_mat = np.column_stack([np.ones(len(x_mat), dtype=np.float64), (x_mat[:, 1:] - mu_x) / sd_x])
            reg_z = design.fqi_ridge_penalty * np.eye(n_feat, dtype=np.float64)
            reg_z[0, 0] = 1e-6
            ztz_inv_zt = np.linalg.solve(z_mat.T @ z_mat + reg_z, z_mat.T)
            wz_myo = ztz_inv_zt @ r_mat
            w_myo = np.zeros_like(wz_myo)
            w_myo[1:] = wz_myo[1:] / sd_x[:, None]
            w_myo[0] = wz_myo[0] - (mu_x / sd_x) @ wz_myo[1:]

        # Myopic (gamma = 0): fits immediate single-year reward surface
        myopic_weights[yr] = w_myo.tolist()

        # Gov-RL (gamma = design.fqi_discount_gamma): fits immediate reward + discounted optimal continuation value
        if next_w_q is None:
            y_q = r_mat
        else:
            cont = np.zeros_like(r_mat)
            for idx, (_, _, n_phis) in enumerate(batch):
                for a_i in range(n_actions):
                    q_next = n_phis[a_i] @ next_w_q
                    q_next[0] -= 1e-6
                    cont[idx, a_i] = float(np.max(q_next))
            y_q = r_mat + design.fqi_discount_gamma * cont

        if design.menu_mode == "v2_25":
            w_q = xtx_inv_xt @ y_q
        else:
            wz_q = ztz_inv_zt @ y_q
            w_q = np.zeros_like(wz_q)
            w_q[1:] = wz_q[1:] / sd_x[:, None]
            w_q[0] = wz_q[0] - (mu_x / sd_x) @ wz_q[1:]
        q_weights[yr] = w_q.tolist()
        next_w_q = w_q

        for target_mat, w_fit, store in ((y_q, w_q, fit_r2_q), (r_mat, w_myo, fit_r2_myopic)):
            pred = x_mat @ w_fit
            ss_res = float(np.sum((target_mat - pred) ** 2))
            ss_tot = float(np.sum((target_mat - np.mean(target_mat, axis=0, keepdims=True)) ** 2))
            store[yr] = round(1.0 - ss_res / max(1e-9, ss_tot), 4)

    return {
        "n_train_seeds": len(train_seeds),
        "n_actions": n_actions,
        "q_weights": q_weights,
        "myopic_weights": myopic_weights,
        "fit_r2_q": fit_r2_q,
        "fit_r2_myopic": fit_r2_myopic,
    }


def _rollout_intertemporal_seed_worker(args: tuple) -> list[tuple[int, dict[int, tuple[np.ndarray, np.ndarray, list[np.ndarray]]]]]:
    """Collect multi-pass exploratory rollouts on seed `s` using exogenous P0 baseline rewards.

    Crucial invariance fix: subtracting state-dependent `grains_kg[0]` (`G_t(s_t, a=0)`) erases the
    continuation value of prior 3-year village consolidation campaigns (because `a=0` already inherits
    the active pool boost). Subtracting a single purely exogenous `p0` reference trajectory `p0_ref_tons[yr]`
    per seed is 100% policy-invariant and preserves exact Bellman continuation values.
    """
    s, calibrated_regions, design, seed_offset, n_rollout_passes = args

    # 1) Exogenous P0 trajectory on seed `s` (strictly independent of RL/exploratory state s_t)
    m_p0 = AgriSuccessionMesaModel(s, calibrated_regions, design, policy_mode="p0")
    p0_ref_tons: dict[int, float] = {}
    p0_rent_pen: dict[int, float] = {}
    for yr in range(1, design.main_years + 1):
        j_p0, r_p0, _ = m_p0.step(explicit_action_idx=0)
        p0_ref_tons[yr] = float(j_p0["grain_kg"]) / 1000.0
        if design.gov_rent_cap_penalty_weight > 0.0:
            p0_rent_pen[yr] = fsum(
                max(0.0, float(rs["unit_rent_yuan_mu"]) - reg.tech.ask_rent_yuan_phys_mu * 2.4) * float(rs["leased_area_mu"])
                for reg, rs in zip(m_p0.regions, r_p0)
            ) / 100_000.0
        else:
            p0_rent_pen[yr] = 0.0

    # 2) Shared Year-1 fork from exact identical initial state on seed `s`
    m_init = AgriSuccessionMesaModel(s, calibrated_regions, design, policy_mode="p0")
    n_actions = len(m_init.menu)
    vc_idxs = [i for i, act in enumerate(m_init.menu) if act[0][0].startswith("vc+")]
    sub_idxs = [i for i in range(1, n_actions) if i not in vc_idxs]

    phi_1 = m_init.extract_public_state_features()
    grains_1, spends_1, rent_pen_1, next_phis_1, clone_lookup_1 = fork_all_menu_actions(m_init)
    rewards_1 = (
        (grains_1 / 1000.0 - p0_ref_tons[1])
        - design.gov_fiscal_cost_weight * (spends_1 / 100_000.0)
        - design.gov_rent_cap_penalty_weight * np.maximum(0.0, rent_pen_1 - p0_rent_pen[1])
    )

    pass_results: list[tuple[int, dict[int, tuple[np.ndarray, np.ndarray, list[np.ndarray]]]]] = []
    for pass_idx in range(n_rollout_passes):
        exp_rng = np.random.default_rng(seed_offset + s * 10 + pass_idx)
        out: dict[int, tuple[np.ndarray, np.ndarray, list[np.ndarray]]] = {
            1: (phi_1, rewards_1, next_phis_1)
        }
        if design.main_years <= 1:
            pass_results.append((s, out))
            continue

        # Select Year-1 action to diversify opening into pool_active vs pool_empty regimes at Year 2
        u1 = exp_rng.random()
        if pass_idx == 0:
            if vc_idxs and u1 < 0.55:
                a1 = int(exp_rng.choice(vc_idxs))
            elif u1 < 0.85:
                a1 = int(exp_rng.choice(sub_idxs)) if sub_idxs else int(exp_rng.integers(0, n_actions))
            else:
                a1 = int(np.argmax(rewards_1))
        else:
            if u1 < 0.60:
                a1 = int(exp_rng.choice(sub_idxs)) if sub_idxs else int(exp_rng.integers(0, n_actions))
            elif u1 < 0.85:
                a1 = int(np.argmax(rewards_1))
            else:
                a1 = int(exp_rng.choice(vc_idxs)) if vc_idxs else int(exp_rng.integers(0, n_actions))

        m = advance_model_from_cache(m_init, a1, clone_lookup_1)
        for yr in range(2, design.main_years + 1):
            phi_t = m.extract_public_state_features()
            grains_kg, spends_yuan, rent_pen, next_phis, clone_lookup = fork_all_menu_actions(m)
            rewards = (
                (grains_kg / 1000.0 - p0_ref_tons[yr])
                - design.gov_fiscal_cost_weight * (spends_yuan / 100_000.0)
                - design.gov_rent_cap_penalty_weight * np.maximum(0.0, rent_pen - p0_rent_pen[yr])
            )
            out[yr] = (phi_t, rewards, next_phis)
            if yr < design.main_years:
                pool_empty = phi_t[3] > 0.5
                u = exp_rng.random()
                if pool_empty:
                    if vc_idxs and u < 0.45:
                        a_next = int(exp_rng.choice(vc_idxs))
                    elif u < 0.80:
                        a_next = int(exp_rng.choice(sub_idxs)) if sub_idxs else int(exp_rng.integers(0, n_actions))
                    else:
                        a_next = int(np.argmax(rewards))
                else:
                    sub_share = 0.60 if pass_idx == 0 else 0.55
                    if u < sub_share:
                        a_next = int(exp_rng.choice(sub_idxs)) if sub_idxs else int(exp_rng.integers(0, n_actions))
                    elif u < 0.90:
                        a_next = int(np.argmax(rewards))
                    else:
                        a_next = int(exp_rng.choice(vc_idxs)) if vc_idxs else int(exp_rng.integers(0, n_actions))
                m = advance_model_from_cache(m, a_next, clone_lookup)
        pass_results.append((s, out))
    return pass_results


def _fit_mlp_ensemble(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    *,
    seed_base: int,
    n_ensemble: int = 2,
    hidden_layers: tuple[int, ...] = (32, 16),
    alpha: float = 0.5,
) -> tuple[dict, float]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    x_in = x_mat[:, 1:]
    scaler = StandardScaler()
    z_in = scaler.fit_transform(x_in)
    mu = scaler.mean_.tolist()
    sd = np.maximum(1e-6, scaler.scale_).tolist()

    members: list[dict] = []
    preds = np.zeros_like(y_mat, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for k in range(n_ensemble):
            reg = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation="relu",
                solver="adam",
                alpha=alpha,
                learning_rate_init=0.003,
                max_iter=800,
                tol=1e-4,
                random_state=seed_base + k,
            )
            reg.fit(z_in, y_mat)
            preds += reg.predict(z_in) / float(n_ensemble)
            members.append({
                "kind": "mlp",
                "mu": mu,
                "sd": sd,
                "coefs": [c.tolist() for c in reg.coefs_],
                "intercepts": [b.tolist() for b in reg.intercepts_],
            })

    ss_res = float(np.sum((y_mat - preds) ** 2))
    ss_tot = float(np.sum((y_mat - np.mean(y_mat, axis=0, keepdims=True)) ** 2))
    r2 = round(1.0 - ss_res / max(1e-9, ss_tot), 4)
    return {"kind": "mlp_ensemble", "members": members}, r2


def _serialize_extra_trees(et_models: list) -> dict:
    """Serialize one or more fitted ExtraTreesRegressor estimators into pure-JSON tree structures."""
    trees: list[dict] = []
    for est_group in et_models:
        for est in est_group.estimators_:
            tr = est.tree_
            trees.append({
                "feature": [int(f) for f in tr.feature.tolist()],
                "threshold": [round(float(t), 6) for t in tr.threshold.tolist()],
                "left": [int(c) for c in tr.children_left.tolist()],
                "right": [int(c) for c in tr.children_right.tolist()],
                "value": [
                    [round(float(v), 5) for v in row]
                    for row in tr.value[:, :, 0].tolist()
                ],
            })
    return {"kind": "tree_ensemble", "trees": trees}


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2))
    return round(1.0 - ss_res / max(1e-9, ss_tot), 4)


def fit_v2_government_mlp_fqi(
    train_seeds: list[int],
    calibrated_regions: tuple[CalibratedRegionV2, ...],
    design: V2Design,
    *,
    exploration_seed: int = 20261128,
    n_rollout_passes: int = 2,
    max_workers: int | None = None,
    rollout_cache_path: Path | None = None,
) -> dict:
    """Fit Cross-Fitted Double-Q Tree-Based Batch FQI (Ernst, Geurts & Wehenkel, 2005) + matched Ridge-FQI."""
    from sklearn.ensemble import ExtraTreesRegressor

    probe = AgriSuccessionMesaModel(train_seeds[0], calibrated_regions, design, policy_mode="p0")
    n_actions = len(probe.menu)

    if rollout_cache_path is not None and rollout_cache_path.exists():
        with rollout_cache_path.open("rb") as fh:
            cached = pickle.load(fh)
        rollouts_by_year = cached["rollouts_by_year"]
        rollout_seeds_by_year = cached["rollout_seeds_by_year"]
        n_episodes = int(cached["n_rollout_episodes"])
    else:
        rollouts_by_year = {yr: [] for yr in range(1, design.main_years + 1)}
        rollout_seeds_by_year = {yr: [] for yr in range(1, design.main_years + 1)}
        tasks = [
            (s, calibrated_regions, design, exploration_seed, n_rollout_passes)
            for s in train_seeds
        ]
        workers = max_workers if max_workers is not None else min(16, max(2, (os.cpu_count() or 4) - 4))
        if workers > 1 and len(tasks) >= 8:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                nested_results = list(pool.map(_rollout_intertemporal_seed_worker, tasks))
        else:
            nested_results = [_rollout_intertemporal_seed_worker(t) for t in tasks]

        n_episodes = 0
        for seed_passes in nested_results:
            for s_id, yr_dict in seed_passes:
                n_episodes += 1
                for yr, tup in yr_dict.items():
                    rollouts_by_year[yr].append(tup)
                    rollout_seeds_by_year[yr].append(s_id)

        if rollout_cache_path is not None:
            rollout_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with rollout_cache_path.open("wb") as fh:
                pickle.dump({
                    "rollouts_by_year": rollouts_by_year,
                    "rollout_seeds_by_year": rollout_seeds_by_year,
                    "n_rollout_episodes": n_episodes,
                }, fh)

    tree_q_weights: dict[int, dict] = {}
    tree_myopic_weights: dict[int, dict] = {}
    ridge_q_weights: dict[int, list[list[float]]] = {}
    ridge_myopic_weights: dict[int, list[list[float]]] = {}
    fit_r2_tree_q: dict[int, float] = {}
    fit_r2_tree_myopic: dict[int, float] = {}
    fit_r2_ridge_q: dict[int, float] = {}
    oos_r2_tree_q: dict[int, float] = {}
    oos_r2_ridge_q: dict[int, float] = {}

    next_tree_a: dict | None = None
    next_tree_b: dict | None = None
    next_ridge_a: np.ndarray | None = None
    next_ridge_b: np.ndarray | None = None

    for yr in range(design.main_years, 0, -1):
        batch = rollouts_by_year[yr]
        batch_seeds = np.asarray(rollout_seeds_by_year[yr], dtype=np.int64)
        x_mat = np.vstack([row[0] for row in batch])
        r_mat = np.vstack([row[1] for row in batch])
        n_rows, n_feat = x_mat.shape

        # Double-Q continuation targets (vectorized over batch x n_actions next states)
        if next_tree_a is None or next_tree_b is None:
            y_q_tree = r_mat
            y_q_ridge = r_mat
        else:
            all_next_phis = np.vstack([n_phis[a_i] for (_, _, n_phis) in batch for a_i in range(n_actions)])
            row_grid = np.arange(n_rows)[:, None]
            act_grid = np.arange(n_actions)[None, :]

            qa_t = _predict_q_values(next_tree_a, all_next_phis).reshape(n_rows, n_actions, n_actions)
            qb_t = _predict_q_values(next_tree_b, all_next_phis).reshape(n_rows, n_actions, n_actions)
            qa_t[:, :, 0] -= 1e-6
            qb_t[:, :, 0] -= 1e-6
            best_a_t = np.argmax(qa_t, axis=2)
            best_b_t = np.argmax(qb_t, axis=2)
            cont_t = 0.5 * (qb_t[row_grid, act_grid, best_a_t] + qa_t[row_grid, act_grid, best_b_t])
            y_q_tree = r_mat + design.fqi_discount_gamma * cont_t

            qa_r = (all_next_phis @ next_ridge_a).reshape(n_rows, n_actions, n_actions)
            qb_r = (all_next_phis @ next_ridge_b).reshape(n_rows, n_actions, n_actions)
            qa_r[:, :, 0] -= 1e-6
            qb_r[:, :, 0] -= 1e-6
            best_a_r = np.argmax(qa_r, axis=2)
            best_b_r = np.argmax(qb_r, axis=2)
            cont_r = 0.5 * (qb_r[row_grid, act_grid, best_a_r] + qa_r[row_grid, act_grid, best_b_r])
            y_q_ridge = r_mat + design.fqi_discount_gamma * cont_r

        # Zero-variance opening state (yr == 1): exact empirical sample mean
        if float(np.max(np.std(x_mat[:, 1:], axis=0))) < 1e-9:
            tree_myopic_weights[yr] = {"kind": "constant", "values": np.mean(r_mat, axis=0).tolist()}
            tree_q_weights[yr] = {"kind": "constant", "values": np.mean(y_q_tree, axis=0).tolist()}
            fit_r2_tree_myopic[yr] = 0.0
            fit_r2_tree_q[yr] = 0.0
            oos_r2_tree_q[yr] = 0.0

            w_myo_c = np.zeros((n_feat, n_actions), dtype=np.float64)
            w_myo_c[0, :] = np.mean(r_mat, axis=0)
            ridge_myopic_weights[yr] = w_myo_c.tolist()

            w_q_c = np.zeros((n_feat, n_actions), dtype=np.float64)
            w_q_c[0, :] = np.mean(y_q_ridge, axis=0)
            ridge_q_weights[yr] = w_q_c.tolist()
            fit_r2_ridge_q[yr] = 0.0
            oos_r2_ridge_q[yr] = 0.0
            next_tree_a = tree_q_weights[yr]
            next_tree_b = tree_q_weights[yr]
            next_ridge_a = w_q_c
            next_ridge_b = w_q_c
            continue

        # Seed-disjoint 50/50 split for Cross-Fitted Double-Q (fallback to parity rows on tiny seed tests)
        if len(set(batch_seeds.tolist())) >= 4:
            mask_a = (batch_seeds % 2) == 0
            mask_b = ~mask_a
        else:
            mask_a = (np.arange(n_rows) % 2) == 0
            mask_b = ~mask_a
        if not np.any(mask_a) or not np.any(mask_b):
            mask_a = np.ones(n_rows, dtype=bool)
            mask_b = np.ones(n_rows, dtype=bool)

        # 1. Myopic ExtraTrees (gamma = 0)
        min_leaf = min(4, max(1, n_rows // 8))
        et_myo = ExtraTreesRegressor(
            n_estimators=80, max_depth=3, min_samples_leaf=min_leaf, random_state=exploration_seed + yr * 100 + 1
        ).fit(x_mat[:, 1:], r_mat)
        myo_entry = _serialize_extra_trees([et_myo])
        tree_myopic_weights[yr] = myo_entry
        fit_r2_tree_myopic[yr] = _r2_score(r_mat, _predict_q_values(myo_entry, x_mat))

        # 2. Cross-Fitted Double-Q Intertemporal ExtraTrees (gamma = design.fqi_discount_gamma)
        min_leaf_f = min(4, max(1, int(np.sum(mask_a)) // 6))
        et_a = ExtraTreesRegressor(
            n_estimators=60, max_depth=3, min_samples_leaf=min_leaf_f, random_state=exploration_seed + yr * 100 + 40
        ).fit(x_mat[mask_a, 1:], y_q_tree[mask_a])
        et_b = ExtraTreesRegressor(
            n_estimators=60, max_depth=3, min_samples_leaf=min_leaf_f, random_state=exploration_seed + yr * 100 + 70
        ).fit(x_mat[mask_b, 1:], y_q_tree[mask_b])
        ser_a = _serialize_extra_trees([et_a])
        ser_b = _serialize_extra_trees([et_b])
        q_entry = _serialize_extra_trees([et_a, et_b])
        tree_q_weights[yr] = q_entry
        fit_r2_tree_q[yr] = _r2_score(y_q_tree, _predict_q_values(q_entry, x_mat))

        pred_ab = _predict_q_values(ser_a, x_mat[mask_b])
        pred_ba = _predict_q_values(ser_b, x_mat[mask_a])
        oos_r2_tree_q[yr] = round(0.5 * (_r2_score(y_q_tree[mask_b], pred_ab) + _r2_score(y_q_tree[mask_a], pred_ba)), 4)
        next_tree_a = ser_a
        next_tree_b = ser_b

        # 3. Matched Cross-Fitted Linear Ridge-FQI ablation on exact same rewards & folds
        mu_x = np.mean(x_mat[:, 1:], axis=0)
        sd_x = np.maximum(1e-3, np.std(x_mat[:, 1:], axis=0))
        z_mat = np.column_stack([np.ones(n_rows, dtype=np.float64), (x_mat[:, 1:] - mu_x) / sd_x])
        reg_z = design.fqi_ridge_penalty * np.eye(n_feat, dtype=np.float64)
        reg_z[0, 0] = 1e-6

        def _solve_raw_ridge(z_sub: np.ndarray, y_sub: np.ndarray) -> np.ndarray:
            wz = np.linalg.solve(z_sub.T @ z_sub + reg_z, z_sub.T) @ y_sub
            w_raw = np.zeros_like(wz)
            w_raw[1:] = wz[1:] / sd_x[:, None]
            w_raw[0] = wz[0] - (mu_x / sd_x) @ wz[1:]
            return w_raw

        w_myo = _solve_raw_ridge(z_mat, r_mat)
        ridge_myopic_weights[yr] = w_myo.tolist()

        w_qa = _solve_raw_ridge(z_mat[mask_a], y_q_ridge[mask_a])
        w_qb = _solve_raw_ridge(z_mat[mask_b], y_q_ridge[mask_b])
        w_q_merged = 0.5 * (w_qa + w_qb)
        ridge_q_weights[yr] = w_q_merged.tolist()
        next_ridge_a = w_qa
        next_ridge_b = w_qb

        fit_r2_ridge_q[yr] = _r2_score(y_q_ridge, x_mat @ w_q_merged)
        oos_r2_ridge_q[yr] = round(
            0.5 * (_r2_score(y_q_ridge[mask_b], x_mat[mask_b] @ w_qa) + _r2_score(y_q_ridge[mask_a], x_mat[mask_a] @ w_qb)),
            4,
        )

    return {
        "n_train_seeds": len(train_seeds),
        "n_rollout_episodes": n_episodes,
        "n_actions": n_actions,
        "tree_q_weights": tree_q_weights,
        "tree_myopic_weights": tree_myopic_weights,
        "mlp_q_weights": tree_q_weights,
        "mlp_myopic_weights": tree_myopic_weights,
        "ridge_q_weights": ridge_q_weights,
        "ridge_myopic_weights": ridge_myopic_weights,
        "fit_r2_tree_q": fit_r2_tree_q,
        "fit_r2_tree_myopic": fit_r2_tree_myopic,
        "fit_r2_mlp_q": fit_r2_tree_q,
        "fit_r2_mlp_myopic": fit_r2_tree_myopic,
        "fit_r2_ridge_q": fit_r2_ridge_q,
        "oos_r2_tree_q": oos_r2_tree_q,
        "oos_r2_ridge_q": oos_r2_ridge_q,
    }


def _eval_dev_action_worker(args: tuple) -> dict:
    a_idx, spec_str, dev_seeds, calibrated_regions, design = args
    tot_grain_tons = []
    tot_spend_wan = []
    for s in dev_seeds:
        m = AgriSuccessionMesaModel(
            s, calibrated_regions, design, policy_mode="fixed", fixed_action_idx=a_idx
        )
        g_sum = 0.0
        sp_sum = 0.0
        for _ in range(design.main_years):
            j_row, _, _ = m.step()
            g_sum += j_row["grain_kg"] / 1000.0
            sp_sum += j_row["spending"] / 10_000.0
        tot_grain_tons.append(g_sum)
        tot_spend_wan.append(sp_sum)
    return {
        "action_idx": a_idx,
        "menu_spec": spec_str,
        "dev_mean_8yr_grain_tons": round(float(np.mean(tot_grain_tons)), 4),
        "dev_mean_8yr_spending_wan": round(float(np.mean(tot_spend_wan)), 4),
    }


def select_dev_fixed_policies(
    dev_seeds: list[int],
    calibrated_regions: tuple[CalibratedRegionV2, ...],
    design: V2Design,
    *,
    max_workers: int | None = None,
) -> dict:
    """Evaluate all menu actions strictly on `dev_seeds` to lock `uniform_fixed` and `regional_fixed`."""
    probe = AgriSuccessionMesaModel(dev_seeds[0], calibrated_regions, design)
    tasks = [(a_idx, str(probe.menu[a_idx]), dev_seeds, calibrated_regions, design) for a_idx in range(len(probe.menu))]
    workers = max_workers if max_workers is not None else min(8, max(1, (os.cpu_count() or 4) - 2))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            leaderboard = list(pool.map(_eval_dev_action_worker, tasks))
    else:
        leaderboard = [_eval_dev_action_worker(t) for t in tasks]

    # Actions 1..5 (and 28 if uniform order_farming exists) are national uniform-instrument policies
    uniform_idxs = [i for i in range(1, min(6, len(leaderboard)))]
    if len(leaderboard) > 28:
        uniform_idxs.append(28)
    regional_idxs = [i for i in range(6, len(leaderboard)) if i not in uniform_idxs]

    score_fn = lambda r: (
        r["dev_mean_8yr_grain_tons"] - design.gov_fiscal_cost_weight * (r["dev_mean_8yr_spending_wan"] / 10.0),
        -r["dev_mean_8yr_spending_wan"],
    )
    best_uniform = max((leaderboard[i] for i in uniform_idxs), key=score_fn)
    best_regional = max((leaderboard[i] for i in regional_idxs), key=score_fn)

    return {
        "best_uniform_fixed_idx": int(best_uniform["action_idx"]),
        "best_regional_fixed_idx": int(best_regional["action_idx"]),
        "leaderboard": leaderboard,
    }


def run_policy_episode(
    seed: int,
    calibrated_regions: tuple[CalibratedRegionV2, ...],
    design: V2Design,
    *,
    policy_mode: str,
    policy_label: str,
    fixed_action_idx: int = 0,
    learned_weights: dict[int, list[list[float]]] | None = None,
    branch_withdrawal: bool = True,
) -> dict:
    """Run an 8-year main policy trajectory + optional Year 7-8 subsidy-withdrawal fork on `AgriSuccessionMesaModel`."""
    m = AgriSuccessionMesaModel(
        seed,
        calibrated_regions,
        design,
        policy_mode=policy_mode,
        fixed_action_idx=fixed_action_idx,
        learned_weights=learned_weights,
    )

    main_joint_rows: list[dict] = []
    main_reg_rows: list[dict] = []
    parcel_rents_by_year: dict[int, dict[str, tuple[float, float]]] = {}
    withdrawal_fork_model: AgriSuccessionMesaModel | None = None

    for yr in range(1, design.main_years + 1):
        j_row, r_rows, _ = m.step()
        j_out = {"seed": seed, "window": "main", "policy_label": policy_label, **j_row}
        main_joint_rows.append(j_out)
        for rr in r_rows:
            main_reg_rows.append({"seed": seed, "window": "main", "policy_label": policy_label, **rr})

        merged_rents: dict[str, tuple[float, float]] = {}
        for r_obj in m.regions:
            merged_rents.update(r_obj.last_parcel_rents)
        parcel_rents_by_year[yr] = merged_rents

        if branch_withdrawal and yr == design.withdrawal_after:
            withdrawal_fork_model = m.fork()
            withdrawal_fork_model.government.policy_mode = "p0"
            withdrawal_fork_model.government.fixed_action_idx = 0

    wd_joint_rows: list[dict] = []
    wd_reg_rows: list[dict] = []
    if branch_withdrawal and withdrawal_fork_model is not None:
        for _ in range(design.withdrawal_after + 1, design.main_years + 1):
            wj, wr_list, _ = withdrawal_fork_model.step(explicit_action_idx=0)
            wd_joint_rows.append({"seed": seed, "window": "withdrawal", "policy_label": policy_label, **wj})
            for rr in wr_list:
                wd_reg_rows.append({"seed": seed, "window": "withdrawal", "policy_label": policy_label, **rr})

    return {
        "seed": seed,
        "policy_label": policy_label,
        "main_joint": main_joint_rows,
        "main_regions": main_reg_rows,
        "withdrawal_joint": wd_joint_rows,
        "withdrawal_regions": wd_reg_rows,
        "parcel_rents_by_year": parcel_rents_by_year,
    }
