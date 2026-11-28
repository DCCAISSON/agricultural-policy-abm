"""Central Government 25-Action Shared Fiscal Authority & 6-Way Policy Solver (V2).

Implements fair same-environment, same-budget, same-lagged-information comparison across:
1. `p0`: Baseline institutional environment (background fertility protection only, zero incremental budget)
2. `uniform_fixed`: Best national uniform-instrument area-proportional policy (selected on dev seeds)
3. `regional_fixed`: Best regionally differentiated/tilted fixed policy (selected on dev seeds)
4. `transparent_rule`: Explicit zero-black-box domain-expert dynamic rule
5. `myopic`: Same 21-dim lagged public state & 96 training rollouts, gamma = 0 (single-year horizon)
6. `gov_rl`: Finite-horizon Backward Ridge-Regularized Fitted Q-Iteration (gamma = 1, 8-year horizon)
"""

from concurrent.futures import ProcessPoolExecutor
from math import fsum
import os
import numpy as np

from mesa_agri_abm.calibration import CalibratedRegionV2
from mesa_agri_abm.config import V2Design
from mesa_agri_abm.model import (
    AgriSuccessionMesaModel,
    LocalAllocV2,
    RegionalSubEconomy,
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
    """Evaluate all 25 fixed menu actions strictly on `dev_seeds` to lock `uniform_fixed` and `regional_fixed`."""
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
