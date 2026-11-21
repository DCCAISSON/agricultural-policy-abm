# -*- coding: utf-8 -*-
"""
数据驱动的初始条件加载器：
- 从 CLES 数据计算农户/企业的初始分布
- 将结果写入配置（内存中），供模型构建时使用
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import yaml

from analysis.derive_initial_conditions import (
    derive_initial_conditions,
    load_household_table,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _get_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _ensure_positive(value: float, fallback: float = 1e-3) -> float:
    return max(float(value), fallback)


def apply_data_driven_initial_conditions(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ic_cfg = config.get("initial_conditions")
    if not ic_cfg or not ic_cfg.get("enabled", False):
        return config, {}

    dataset_path = _get_project_path(ic_cfg.get("dataset"))
    year = ic_cfg.get("year", 2019)
    drop_desc = not ic_cfg.get("keep_description_row", False)
    cache_file = ic_cfg.get("cache_file")
    cache_path = _get_project_path(cache_file) if cache_file else None

    stats = None
    if cache_path and cache_path.exists():
        stats = yaml.safe_load(cache_path.read_text(encoding="utf-8"))

    if stats is None:
        df = load_household_table(dataset_path, drop_description_row=drop_desc)
        stats = derive_initial_conditions(df, year=year)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                yaml.safe_dump(stats, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

    _patch_farmer_config(config, stats)
    _patch_enterprise_config(config, stats)
    return config, stats


def _patch_farmer_config(config: Dict[str, Any], stats: Dict[str, Any]) -> None:
    farmer_cfg = config.get("agents", {}).get("farmer", {})
    farmer_stats = stats.get("farmer", {})
    metadata = stats.get("metadata", {})

    ability_summary = farmer_stats.get("farming_ability", {})
    if ability_summary:
        farmer_cfg["farming_ability_dist"] = {
            "type": "normal",
            "mean": round(float(ability_summary.get("mean", 1.0)), 4),
            "std": round(_ensure_positive(ability_summary.get("std", 0.05)), 4),
        }

    # 户主年龄分布（用于初始化年龄以及土地流转意愿函数中的 m_age / n_age）
    age_summary = farmer_stats.get("age", {})
    if age_summary:
        mean_age = float(age_summary.get("mean", 50.0))
        std_age = _ensure_positive(age_summary.get("std", 10.0))

        farmer_cfg["age_dist"] = {
            "type": "normal",
            "mean": round(mean_age, 2),
            "std": round(std_age, 2),
        }

        land_cfg = farmer_cfg.get("land_transfer", {})

        # m_age：土地流入意愿最高的年龄，取 CLES 户主平均年龄
        if "m_age" not in land_cfg:
            land_cfg["m_age"] = round(mean_age, 1)

        # n_age：土地流出意愿最高的年龄，取平均年龄减去约 1.3 个标准差（参照 NetLogo 中 45/30 的差距），并不低于 20 岁
        if "n_age" not in land_cfg:
            n_val = max(20.0, mean_age - 1.3 * std_age)
            land_cfg["n_age"] = round(n_val, 1)

        # optimal_scale：如未显式指定，则用 CLES 的 p80 作为目标经营规模
        area_quantiles = metadata.get("area_quantiles", {})
        if "base_target_operated_area" not in land_cfg and area_quantiles:
            p80 = float(area_quantiles.get("p80", 9.0))
            land_cfg["base_target_operated_area"] = round(p80, 2)

        farmer_cfg["land_transfer"] = land_cfg

    capital_dist = farmer_stats.get("initial_capital_dist", {})
    if capital_dist:
        farmer_cfg["initial_capital_dist"] = {
            "type": "lognormal",
            "mean": round(float(capital_dist.get("log_mean", 9.2)), 4),
            "std": round(_ensure_positive(capital_dist.get("log_std", 0.5)), 4),
        }

    config.setdefault("agents", {})["farmer"] = farmer_cfg


def _patch_enterprise_config(config: Dict[str, Any], stats: Dict[str, Any]) -> None:
    enterprise_cfg = config.get("agents", {}).get("enterprise", {})
    type_cfg = enterprise_cfg.get("types", {})
    cat_stats = stats.get("enterprise_categories", {})

    # 更新类型占比
    shares = {
        cat: entry.get("share_of_households", 0)
        for cat, entry in cat_stats.items()
    }
    if shares:
        total_share = sum(shares.values())
        if total_share > 0:
            enterprise_cfg["type_distribution"] = {
                cat: round(val / total_share, 4) if total_share else 0.0
                for cat, val in shares.items()
            }

    for cat, entry in cat_stats.items():
        cfg = type_cfg.get(cat)
        if not cfg:
            continue
        capital_dist = entry.get("initial_capital_dist", {})
        if capital_dist:
            cfg.setdefault("initial_capital_dist", {})
            cfg["initial_capital_dist"]["type"] = "lognormal"
            cfg["initial_capital_dist"]["mean"] = round(float(capital_dist.get("log_mean", cfg["initial_capital_dist"].get("mean", 10.0))), 4)
            cfg["initial_capital_dist"]["std"] = round(_ensure_positive(capital_dist.get("log_std", cfg["initial_capital_dist"].get("std", 0.5))), 4)

        ability = entry.get("farming_ability", {})
        if ability:
            cfg.setdefault("efficiency_factor_dist", {})
            base = float(ability.get("mean", cfg["efficiency_factor_dist"].get("mean", 1.2)))
            cfg["efficiency_factor_dist"]["mean"] = round(max(1.0, base * 1.05), 4)
            cfg["efficiency_factor_dist"]["std"] = round(_ensure_positive(ability.get("std", cfg["efficiency_factor_dist"].get("std", 0.1))), 4)

        type_cfg[cat] = cfg

    enterprise_cfg["types"] = type_cfg
    config.setdefault("agents", {})["enterprise"] = enterprise_cfg

