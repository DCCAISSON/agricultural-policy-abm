# -*- coding: utf-8 -*-
"""
从 CLES 户级数据中提取初始化所需的分布参数，用于配置
farmers / enterprises 的初始取值。

Usage:
    python analysis/derive_initial_conditions.py \
        --input 校准用数据源/CLES_2020_2021_household_panel_mapping_label.xlsx \
        --year 2019 \
        --output config/initial_conditions_2019.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 复用综合指标构建函数
_scripts_dir = PROJECT_ROOT / "scripts"
if str(_scripts_dir) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_scripts_dir))

from build_composite_indicators import (  # type: ignore  # pylint: disable=import-error
    build_livelihood_strategy,
    build_farming_ability_index,
    compute_income_components,
)

__all__ = [
    "derive_initial_conditions",
    "load_household_table",
]


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def load_household_table(path: Path, drop_description_row: bool = True) -> pd.DataFrame:
    df = pd.read_excel(path, header=0)
    if drop_description_row and len(df) > 0:
        df = df.iloc[1:].reset_index(drop=True)
    df = _coerce_numeric(df)
    return df


def _summary(series: pd.Series) -> Dict[str, float]:
    if series.empty:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _lognormal_params(values: pd.Series) -> Dict[str, float]:
    positive = values[values > 0]
    if positive.empty:
        return {"mean": 0.0, "std": 0.0}
    logs = np.log(positive)
    return {
        "log_mean": float(logs.mean()),
        "log_std": float(logs.std(ddof=1)) if len(logs) > 1 else 0.0,
    }


def derive_initial_conditions(df: pd.DataFrame, year: int) -> Dict[str, Any]:
    if "year" in df.columns:
        df = df[pd.to_numeric(df["year"], errors="coerce") == year]
        df = df.copy()
    if df.empty:
        raise ValueError(f"数据中过滤年份 {year} 后无记录，请确认输入。")

    # 构建综合指标
    livelihood_strategy, agri_income_ratio = build_livelihood_strategy(df)
    farming_ability = build_farming_ability_index(df)
    income_components = compute_income_components(df)

    df = df.assign(
        livelihood_strategy=livelihood_strategy.values,
        agri_income_ratio=agri_income_ratio.values,
        farming_ability=farming_ability.values,
        total_income=income_components["total_income"].values,
    )

    # 土地经营面积：e101a* (亩)
    area_cols = [col for col in df.columns if col.lower().startswith("e101a")]
    if area_cols:
        farmland_area = df[area_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    else:
        farmland_area = pd.Series([0.0] * len(df))

    df["farmland_area"] = farmland_area

    # 户主（受访者）年龄：a203a1（过滤不合理值）
    ages = pd.to_numeric(df.get("a203a1"), errors="coerce")
    ages = ages[(ages > 0) & (ages < 120)]

    # 计算策略占比
    strategy_counts = df["livelihood_strategy"].value_counts()
    strategy_dist = {
        k: float(v) / len(df)
        for k, v in strategy_counts.sort_index().items()
    }

    # 农户统计
    farmer_stats = {
        "n_households": int(len(df)),
        "strategy_distribution": strategy_dist,
        "avg_agri_income_ratio": float(df["agri_income_ratio"].mean()),
        "farming_ability": _summary(df["farming_ability"]),
        "farmland_area": _summary(df["farmland_area"]),
        "agri_income_ratio_summary": _summary(df["agri_income_ratio"]),
        "total_income": _summary(df["total_income"]),
        "initial_capital_dist": _lognormal_params(df["total_income"]),
        "age": _summary(ages) if not ages.empty else {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
    }

    # 按土地规模赋予企业类别（基于分位数）
    quantiles = df["farmland_area"].quantile([0.6, 0.8, 0.95]).to_dict()
    p60 = quantiles.get(0.6, 0)
    p80 = quantiles.get(0.8, 0)
    p95 = quantiles.get(0.95, 0)

    def classify(row):
        area = row["farmland_area"]
        if area >= p95:
            return "AGRI_CORPORATION"
        if area >= p80:
            return "FAMILY_FARM_LARGE"
        if area >= p60:
            return "COOPERATIVE"
        return None

    df["enterprise_category"] = df.apply(classify, axis=1)

    enterprise_stats: Dict[str, Dict[str, Any]] = {}
    for category in ["FAMILY_FARM_LARGE", "AGRI_CORPORATION", "COOPERATIVE"]:
        subset = df[df["enterprise_category"] == category]
        if subset.empty:
            continue
        share = float(len(subset)) / len(df)
        total_income = pd.to_numeric(subset["total_income"], errors="coerce").fillna(0)
        enterprise_stats[category] = {
            "share_of_households": share,
            "farmland_area": _summary(subset["farmland_area"]),
            "farming_ability": _summary(subset["farming_ability"]),
            "total_income": _summary(total_income),
            "initial_capital_dist": _lognormal_params(total_income),
        }

    latitudes = pd.to_numeric(df.get("a102"), errors="coerce")
    longitudes = pd.to_numeric(df.get("a103"), errors="coerce")

    return {
        "metadata": {
            "source": "CLES household panel",
            "input_rows": int(len(df)),
            "year": year,
            "area_quantiles": {
                "p60": float(p60),
                "p80": float(p80),
                "p95": float(p95),
            },
            "spatial": {
                "latitude": _summary(latitudes.dropna()) if latitudes.notna().any() else None,
                "longitude": _summary(longitudes.dropna()) if longitudes.notna().any() else None,
            },
        },
        "farmer": farmer_stats,
        "enterprise_categories": enterprise_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Derive initial parameter stats from CLES household file.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path("校准用数据源") / "CLES_2020_2021_household_panel_mapping_label.xlsx"),
        help="CLES 户级数据文件（Excel / CSV）",
    )
    parser.add_argument("--year", type=int, required=True, help="筛选年份，例如 2019")
    parser.add_argument("--output", type=str, default="config/initial_conditions.yaml", help="输出 YAML/JSON 文件路径")
    parser.add_argument("--format", choices=["yaml", "json"], default="yaml", help="输出格式")
    parser.add_argument("--no-drop-description", action="store_true", help="不要自动移除第二行变量说明")
    args = parser.parse_args()

    input_path = (PROJECT_ROOT / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_path}")

    df = load_household_table(input_path, drop_description_row=not args.no_drop_description)
    initial_conditions = derive_initial_conditions(df, year=args.year)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "yaml":
        output_path.write_text(yaml.safe_dump(initial_conditions, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        output_path.write_text(json.dumps(initial_conditions, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] 已写入初始条件统计: {output_path}")


if __name__ == "__main__":
    main()

