from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUAL_SOURCE_TARGETS_PATH = PROJECT_ROOT / "config" / "calibration_targets_dual_source.json"
COMPREHENSIVE_TARGETS_PATH = PROJECT_ROOT / "config" / "calibration_targets_comprehensive.json"
CALIBRATION_RESULTS_PATH = PROJECT_ROOT / "config" / "calibration_results_2009_2015.json"
NATIONAL_RESULTS_PATH = PROJECT_ROOT / "config" / "national_calibration_results.json"
NFPS_HOUSEHOLD_2009_2015 = PROJECT_ROOT / "校准用数据源" / "全国农村固定观察点" / "2009-2015household.csv"

# @new 2026-05-02: 真实分位数缓存
# 之前 get_validation_target_moments 的 p25/p50/p75 由 mean × 固定倍数构造，
# 现在必须从 NFPS 真实数据计算。结果在第一次调用时缓存以避免重复 IO。
_AGRI_INCOME_SHARE_DISTRIBUTION_CACHE: Optional[Dict[str, float]] = None


class ValidationTargetUnavailableError(RuntimeError):
    """Raised when an empirical validation target cannot be resolved."""


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _degraded_payload(source: str, reason: str, **values: Any) -> Dict[str, Any]:
    payload = {key: float("nan") for key in values}
    payload.update(values)
    payload.update({
        "status": "degraded",
        "source": source,
        "reason": reason,
    })
    return payload


def _weighted_mean(items: Iterable[Dict[str, Any]], value_path: tuple[str, ...]) -> Optional[float]:
    weighted_sum = 0.0
    total_weight = 0.0

    for item in items:
        cursor: Any = item
        for key in value_path:
            if not isinstance(cursor, dict):
                cursor = None
                break
            cursor = cursor.get(key)

        value = _safe_float(cursor)
        if value is None:
            continue

        weight = _safe_float(item.get("sample_size")) or 1.0
        weighted_sum += value * weight
        total_weight += weight

    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def get_livelihood_reference(year: int = 2020) -> Dict[str, Any]:
    dual_source = _load_json(DUAL_SOURCE_TARGETS_PATH)
    livelihood_cfg = dual_source.get("livelihood") or {}
    year_cfg = livelihood_cfg.get(f"crosssection_targets_{int(year)}") or {}

    pure = _safe_float(year_cfg.get("pure_farmer_rate"))
    pluri = _safe_float(year_cfg.get("pluriactive_rate"))
    non = _safe_float(year_cfg.get("non_farmer_rate"))
    if pure is not None and pluri is not None and non is not None:
        return {
            "pure_farmer_share": pure,
            "pluriactive_share": pluri,
            "non_farmer_share": non,
            "status": "available",
            "source": f"config/calibration_targets_dual_source.json livelihood crosssection_targets_{int(year)}",
        }

    comprehensive = _load_json(COMPREHENSIVE_TARGETS_PATH)
    evolution_cfg = ((comprehensive.get("livelihood") or {}).get("evolution") or {}).get(str(int(year))) or {}
    pure = _safe_float(evolution_cfg.get("pure_farmer"))
    pluri = _safe_float(evolution_cfg.get("pluriactive"))
    non = _safe_float(evolution_cfg.get("non_farmer"))
    if pure is not None and pluri is not None and non is not None:
        return {
            "pure_farmer_share": pure,
            "pluriactive_share": pluri,
            "non_farmer_share": non,
            "status": "available",
            "source": f"config/calibration_targets_comprehensive.json livelihood evolution {int(year)}",
        }

    return _degraded_payload(
        "missing empirical livelihood targets",
        f"no resolved livelihood target for year {int(year)}",
        pure_farmer_share=float("nan"),
        pluriactive_share=float("nan"),
        non_farmer_share=float("nan"),
    )


def get_household_income_reference() -> Dict[str, Any]:
    dual_source = _load_json(DUAL_SOURCE_TARGETS_PATH)
    income_cfg = (dual_source.get("income") or {}).get("total_income") or {}

    mean_income = _safe_float(income_cfg.get("mean_2015"))
    income_gini = _safe_float((dual_source.get("income") or {}).get("gini_coefficient"))
    annual_growth_rate = _safe_float(income_cfg.get("annual_growth_rate"))

    if mean_income is not None:
        degraded = income_gini is None or annual_growth_rate is None
        return {
            "mean_household_income": mean_income,
            "income_gini": income_gini if income_gini is not None else float("nan"),
            "annual_growth_rate": annual_growth_rate if annual_growth_rate is not None else float("nan"),
            "status": "degraded" if degraded else "available",
            "source": "config/calibration_targets_dual_source.json income.total_income.mean_2015",
            "reason": "income_gini or annual_growth_rate unavailable" if degraded else None,
        }

    calibration_results = _load_json(CALIBRATION_RESULTS_PATH)
    weighted_income = _weighted_mean(
        (calibration_results.get("income_structure") or {}).values(),
        ("income_stats", "mean"),
    )
    if weighted_income is not None:
        degraded = income_gini is None or annual_growth_rate is None
        return {
            "mean_household_income": weighted_income,
            "income_gini": income_gini if income_gini is not None else float("nan"),
            "annual_growth_rate": annual_growth_rate if annual_growth_rate is not None else float("nan"),
            "status": "degraded" if degraded else "available",
            "source": "config/calibration_results_2009_2015.json income_structure.income_stats.mean",
            "reason": "income_gini or annual_growth_rate unavailable" if degraded else None,
        }
    return _degraded_payload(
        "missing empirical household income targets",
        "no resolved household income target",
        mean_household_income=float("nan"),
        income_gini=float("nan"),
        annual_growth_rate=float("nan"),
    )


def get_agricultural_income_share_reference() -> Dict[str, Any]:
    national_results = _load_json(NATIONAL_RESULTS_PATH)
    weighted_share = _weighted_mean(
        (national_results.get("income_structure") or {}).values(),
        ("income_structure", "agricultural_ratio"),
    )
    if weighted_share is not None:
        return {
            "agricultural_income_share": weighted_share,
            "status": "available",
            "source": "config/national_calibration_results.json income_structure.agricultural_ratio",
        }

    return _degraded_payload(
        "missing empirical agricultural income share",
        "no resolved agricultural income share target",
        agricultural_income_share=float("nan"),
    )


def get_land_transfer_reference() -> Dict[str, Any]:
    calibration_results = _load_json(CALIBRATION_RESULTS_PATH)
    transfer_cfg = calibration_results.get("land_transfer_rate") or {}

    participation_rate = _weighted_mean(transfer_cfg.values(), ("participation_rate",))
    rent_out_rate = _weighted_mean(transfer_cfg.values(), ("rent_out_rate",))
    rent_in_rate = _weighted_mean(transfer_cfg.values(), ("rent_in_rate",))

    if participation_rate is not None:
        degraded = rent_out_rate is None or rent_in_rate is None
        return {
            "land_transfer_rate": participation_rate,
            "rent_out_rate": rent_out_rate if rent_out_rate is not None else float("nan"),
            "rent_in_rate": rent_in_rate if rent_in_rate is not None else float("nan"),
            "status": "degraded" if degraded else "available",
            "source": "config/calibration_results_2009_2015.json land_transfer_rate.weighted participation_rate",
            "reason": "rent_out_rate or rent_in_rate unavailable" if degraded else None,
        }

    return _degraded_payload(
        "missing empirical land transfer targets",
        "no resolved land transfer target",
        land_transfer_rate=float("nan"),
        rent_out_rate=float("nan"),
        rent_in_rate=float("nan"),
    )


def get_urbanization_reference(year: int = 2022) -> Dict[str, Any]:
    comprehensive = _load_json(COMPREHENSIVE_TARGETS_PATH)
    livelihood_cfg = comprehensive.get("livelihood") or {}
    evolution_cfg = (livelihood_cfg.get("evolution") or {}).get(str(int(year))) or {}
    urbanization = _safe_float(evolution_cfg.get("urbanized"))
    if urbanization is None:
        urbanization = _safe_float((livelihood_cfg.get("terminal_2022") or {}).get("urbanized"))

    if urbanization is not None:
        return {
            "urbanization_rate": urbanization,
            "status": "available",
            "source": f"config/calibration_targets_comprehensive.json livelihood evolution {int(year)}",
        }

    return _degraded_payload(
        "missing empirical urbanization target",
        f"no resolved urbanization target for year {int(year)}",
        urbanization_rate=float("nan"),
    )


def get_agricultural_income_share_distribution() -> Dict[str, Any]:
    """
    @new 2026-05-02: 从 NFPS 2009-2015 真实数据计算 agricultural-income-share 分位数。

    取代之前 get_validation_target_moments 中
        p25 = mean * 0.5
        p50 = mean * 0.8
        p75 = mean * 1.4
    的伪分位数（数据上完全错误，违反 Nature 级数据真实性要求）。

    口径：
      - 仅保留 inc_total > 0 且 inc_c 非缺失的农户
      - agri_share = inc_c / inc_total，clip 到 [0, 1]
      - 计算 25/50/75 经验分位数

    Returns:
        Dict 含 p25, p50, p75, p90, n, source, status, reason 字段。
        若数据文件缺失则返回 status='degraded' + 全 NaN。
    """
    global _AGRI_INCOME_SHARE_DISTRIBUTION_CACHE
    if _AGRI_INCOME_SHARE_DISTRIBUTION_CACHE is not None:
        return dict(_AGRI_INCOME_SHARE_DISTRIBUTION_CACHE)

    if not NFPS_HOUSEHOLD_2009_2015.exists():
        result = _degraded_payload(
            f"NFPS file missing: {NFPS_HOUSEHOLD_2009_2015}",
            "raw NFPS household file not found; cannot compute real quantiles",
            p25=float("nan"), p50=float("nan"), p75=float("nan"), p90=float("nan"), n=0,
        )
        _AGRI_INCOME_SHARE_DISTRIBUTION_CACHE = result
        return dict(result)

    try:
        # 局部导入以避免顶层强依赖 pandas
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        result = _degraded_payload(
            "pandas unavailable",
            f"pandas import failed: {exc}",
            p25=float("nan"), p50=float("nan"), p75=float("nan"), p90=float("nan"), n=0,
        )
        _AGRI_INCOME_SHARE_DISTRIBUTION_CACHE = result
        return dict(result)

    try:
        df = pd.read_csv(NFPS_HOUSEHOLD_2009_2015, low_memory=False, encoding="utf-8")
    except Exception as exc:
        result = _degraded_payload(
            "NFPS file read error",
            f"failed to read NFPS csv: {exc}",
            p25=float("nan"), p50=float("nan"), p75=float("nan"), p90=float("nan"), n=0,
        )
        _AGRI_INCOME_SHARE_DISTRIBUTION_CACHE = result
        return dict(result)

    if "inc_total" not in df.columns or "inc_c" not in df.columns:
        result = _degraded_payload(
            "NFPS columns missing",
            "expected columns inc_total / inc_c absent; cannot compute agri_share",
            p25=float("nan"), p50=float("nan"), p75=float("nan"), p90=float("nan"), n=0,
        )
        _AGRI_INCOME_SHARE_DISTRIBUTION_CACHE = result
        return dict(result)

    valid = df[(df["inc_total"] > 0) & df["inc_c"].notna()].copy()
    if valid.empty:
        result = _degraded_payload(
            "NFPS empty valid sample",
            "no rows survive inc_total>0 & inc_c not null filter",
            p25=float("nan"), p50=float("nan"), p75=float("nan"), p90=float("nan"), n=0,
        )
        _AGRI_INCOME_SHARE_DISTRIBUTION_CACHE = result
        return dict(result)

    valid["agri_share"] = (valid["inc_c"].fillna(0.0) / valid["inc_total"]).clip(0.0, 1.0)
    result = {
        "p25": float(valid["agri_share"].quantile(0.25)),
        "p50": float(valid["agri_share"].quantile(0.50)),
        "p75": float(valid["agri_share"].quantile(0.75)),
        "p90": float(valid["agri_share"].quantile(0.90)),
        "mean": float(valid["agri_share"].mean()),
        "n": int(len(valid)),
        "status": "available",
        "source": "NFPS 2009-2015 actual quantile (校准用数据源/全国农村固定观察点/2009-2015household.csv)",
        "reason": None,
    }
    _AGRI_INCOME_SHARE_DISTRIBUTION_CACHE = result
    return dict(result)


def get_validation_target_moments(reference_year: int = 2020) -> Dict[str, Any]:
    livelihood = get_livelihood_reference(reference_year)
    income = get_household_income_reference()
    land = get_land_transfer_reference()
    agri_share = get_agricultural_income_share_reference()

    agri_mean = _safe_float(agri_share.get("agricultural_income_share"))
    mean_income = _safe_float(income.get("mean_household_income"))
    income_gini = _safe_float(income.get("income_gini"))
    land_transfer = _safe_float(land.get("land_transfer_rate"))
    pure_share = _safe_float(livelihood.get("pure_farmer_share"))
    pluri_share = _safe_float(livelihood.get("pluriactive_share"))
    non_share = _safe_float(livelihood.get("non_farmer_share"))

    # @fix 2026-05-02: 改用真实 NFPS 分位数（取代 mean × {0.5, 0.8, 1.4} 伪分位数）
    distribution = get_agricultural_income_share_distribution()
    p25 = _safe_float(distribution.get("p25"))
    p50 = _safe_float(distribution.get("p50"))
    p75 = _safe_float(distribution.get("p75"))
    distribution_status = distribution.get("status", "degraded")
    distribution_source = distribution.get("source", "n/a")

    degraded = any(
        value is None or (isinstance(value, float) and value != value)
        for value in [agri_mean, mean_income, income_gini, land_transfer, pure_share, pluri_share, non_share, p25, p50, p75]
    )

    return {
        "mean_farmer_income": mean_income if mean_income is not None else float("nan"),
        "income_gini": income_gini if income_gini is not None else float("nan"),
        "agri_income_share_mean": agri_mean if agri_mean is not None else float("nan"),
        "agri_income_share_p25": p25 if p25 is not None else float("nan"),
        "agri_income_share_p50": p50 if p50 is not None else float("nan"),
        "agri_income_share_p75": p75 if p75 is not None else float("nan"),
        "agri_income_share_distribution_status": distribution_status,
        "agri_income_share_distribution_source": distribution_source,
        "land_transfer_rate": land_transfer if land_transfer is not None else float("nan"),
        "grain_planting_ratio": float("nan"),
        "pure_farmer_share": pure_share if pure_share is not None else float("nan"),
        "pluriactive_share": pluri_share if pluri_share is not None else float("nan"),
        "non_farmer_share": non_share if non_share is not None else float("nan"),
        "status": "degraded" if degraded else "available",
        "source": "composed validation targets (quantiles from NFPS 2009-2015 raw data)",
        "reason": "one or more empirical reference targets unavailable" if degraded else None,
    }
