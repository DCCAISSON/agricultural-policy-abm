from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS_PATH = PROJECT_ROOT / "config" / "derived_thresholds.json"


class TTIBaselineFallbackWarning(UserWarning):
    """Raised when TTI normalization falls back to JSON-config defaults instead
    of using a runtime-derived reference. Manuscript-pipeline callers must
    supply runtime_reference; fallback values are not empirically anchored.
    See config/derived_thresholds.json -> tti.sources.* for provenance."""


def load_derived_thresholds(config_path: Path | None = None) -> Dict[str, Any]:
    threshold_path = Path(config_path) if config_path else DEFAULT_THRESHOLDS_PATH
    with open(threshold_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_tti_spec(config_path: Path | None = None) -> Dict[str, Any]:
    spec = load_derived_thresholds(config_path).get("tti")
    if not spec:
        raise KeyError("Missing 'tti' section in derived_thresholds.json")
    return spec


def get_default_tti_threshold(config_path: Path | None = None) -> float:
    return float(get_tti_spec(config_path).get("default_threshold", 0.6))


def get_tti_threshold_context(config_path: Path | None = None) -> Dict[str, Any]:
    threshold_path = Path(config_path) if config_path else DEFAULT_THRESHOLDS_PATH
    spec = get_tti_spec(config_path)
    return {
        "threshold": float(spec.get("default_threshold", 0.6)),
        "default_threshold": float(spec.get("default_threshold", 0.6)),
        "threshold_source": "derived_thresholds_default",
        "threshold_kind": "research_default",
        "threshold_spec_path": str(threshold_path.resolve()),
        "reference_point": get_tti_reference_point(config_path),
        "reference_point_source": "derived_thresholds_surface_reference_point",
        "notes": spec.get("notes", {}),
    }


def get_tti_reference_point(config_path: Path | None = None) -> Dict[str, float]:
    reference_point = get_tti_spec(config_path).get("surface_reference_point", {})
    return {
        "aging_rate": float(reference_point.get("aging_rate", 0.25)),
        "land_gini": float(reference_point.get("land_gini", 0.45)),
    }


def build_runtime_tti_reference(
    indicators: Any,
    *,
    source: str = "runtime_model_output",
) -> Dict[str, float]:
    pure_farmer_ratio = float(getattr(indicators, "pure_farmer_ratio", 0.0) or 0.0) / 100.0
    grain_area_ratio = float(getattr(indicators, "grain_area_ratio", 0.0) or 0.0) / 100.0
    return {
        "baseline_pure_farmer_ratio": max(1e-9, pure_farmer_ratio),
        "baseline_grain_area_ratio": max(1e-9, grain_area_ratio),
        "reference_source": source,
    }


def attach_tti_inputs(indicators: Any, model: Any) -> Any:
    if not hasattr(model, "get_subsidy_incidence_ratio"):
        raise AttributeError(
            "Model is missing get_subsidy_incidence_ratio(); cannot compute TTI without a real rent capitalization metric"
        )

    rent_capitalization_rate = model.get_subsidy_incidence_ratio()
    if rent_capitalization_rate is None:
        raise ValueError("Model returned None for get_subsidy_incidence_ratio()")

    indicators.rent_capitalization_rate = float(rent_capitalization_rate)
    return indicators


def compute_tti(
    indicators: Any,
    config_path: Path | None = None,
    runtime_reference: Dict[str, Any] | None = None,
) -> Tuple[float, Dict[str, float]]:
    spec = get_tti_spec(config_path)
    if runtime_reference:
        baseline_pure_farmer = max(
            1e-9,
            float(
                runtime_reference.get(
                    "baseline_pure_farmer_ratio",
                    spec["baseline_pure_farmer_ratio"],
                )
            ),
        )
        baseline_grain_ratio = max(
            1e-9,
            float(
                runtime_reference.get(
                    "baseline_grain_area_ratio",
                    spec["baseline_grain_area_ratio"],
                )
            ),
        )
        reference_source = str(runtime_reference.get("reference_source", "runtime_reference"))
    else:
        warnings.warn(
            "TTI compute_tti() called WITHOUT runtime_reference; falling back to "
            "config/derived_thresholds.json defaults (baseline_pure_farmer_ratio=0.8, "
            "baseline_grain_area_ratio=0.65). These fallback values are NOT empirically "
            "anchored and MUST NOT be used in manuscript-pipeline outputs. "
            "Pass runtime_reference=build_runtime_tti_reference(initial_indicators) instead. "
            "See config/derived_thresholds.json -> tti.sources for provenance details.",
            TTIBaselineFallbackWarning,
            stacklevel=2,
        )
        baseline_pure_farmer = max(1e-9, float(spec["baseline_pure_farmer_ratio"]))
        baseline_grain_ratio = max(1e-9, float(spec["baseline_grain_area_ratio"]))
        reference_source = "derived_thresholds_default_DEPRECATED_FALLBACK"
    weights = spec.get(
        "weights",
        {
            "rent_capitalization": 0.4,
            "scale_shift": 0.3,
            "crop_shift": 0.3,
        },
    )

    rent_cap = getattr(indicators, "rent_capitalization_rate", None)
    if rent_cap is None:
        raise ValueError(
            "UnifiedIndicators.rent_capitalization_rate is required for TTI calculation; fallback constants are disabled"
        )

    rent_cap = float(rent_cap)
    if math.isnan(rent_cap):
        raise ValueError("UnifiedIndicators.rent_capitalization_rate is NaN; cannot compute TTI")

    rent_cap = min(1.0, max(0.0, rent_cap))

    pure_ratio = float(indicators.pure_farmer_ratio) / 100.0
    grain_ratio = float(indicators.grain_area_ratio) / 100.0
    scale_shift = min(1.0, max(0.0, 1.0 - (pure_ratio / baseline_pure_farmer)))
    crop_shift = min(1.0, max(0.0, 1.0 - (grain_ratio / baseline_grain_ratio)))

    tti = (
        float(weights.get("rent_capitalization", 0.4)) * rent_cap
        + float(weights.get("scale_shift", 0.3)) * scale_shift
        + float(weights.get("crop_shift", 0.3)) * crop_shift
    )

    return tti, {
        "rent_capitalization": rent_cap,
        "scale_shift": scale_shift,
        "crop_shift": crop_shift,
        "baseline_pure_farmer_ratio": baseline_pure_farmer,
        "baseline_grain_area_ratio": baseline_grain_ratio,
        "reference_source": reference_source,
    }
