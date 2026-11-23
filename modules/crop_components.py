"""Endogenous grain-component allocation helpers.

This module is intentionally conservative: it does not use official crop
weights to split aggregate grain output. It only writes crop-component state
when the model config explicitly enables a mechanism-level allocation.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


CORE_GRAIN_COMPONENTS = ("rice", "wheat", "corn", "soybean")

DEFAULT_REGION_PRIORS: dict[str, dict[str, float]] = {
    "东部": {"rice": 0.34, "wheat": 0.22, "corn": 0.28, "soybean": 0.16},
    "中部": {"rice": 0.34, "wheat": 0.24, "corn": 0.30, "soybean": 0.12},
    "西部": {"rice": 0.18, "wheat": 0.28, "corn": 0.38, "soybean": 0.16},
    "东北": {"rice": 0.18, "wheat": 0.08, "corn": 0.48, "soybean": 0.26},
    "global": {"rice": 0.28, "wheat": 0.24, "corn": 0.34, "soybean": 0.14},
}


def _get_config(model: Any) -> Mapping[str, Any]:
    config = getattr(model, "config", {}) or {}
    if not isinstance(config, Mapping):
        return {}
    production = config.get("production", {}) or {}
    if not isinstance(production, Mapping):
        return {}
    crop_components = production.get("crop_components", {}) or {}
    return crop_components if isinstance(crop_components, Mapping) else {}


def _clear_component_state(actor: Any, *, status: str) -> None:
    actor.crop_component_area_mu = {}
    actor.crop_component_production_kg = {}
    actor.crop_component_method = "disabled"
    actor.crop_component_status = status
    actor.crop_component_claim_boundary = (
        "No crop-component state was generated. Aggregate grain output must not be read as "
        "rice/wheat/corn/soybean predictions."
    )


def _actor_region(actor: Any, model: Any) -> str:
    if hasattr(model, "get_region_label"):
        try:
            region = model.get_region_label(actor)
            if region:
                return str(region)
        except (AttributeError, TypeError, ValueError):
            actor.crop_component_region_status = "region_lookup_failed"
    return str(getattr(actor, "region", "global") or "global")


def _region_priors(region: str, cfg: Mapping[str, Any]) -> dict[str, float]:
    configured = cfg.get("region_priors", {}) if isinstance(cfg, Mapping) else {}
    if isinstance(configured, Mapping):
        for key in (region, str(region).lower(), "global"):
            raw = configured.get(key)
            if isinstance(raw, Mapping):
                return {
                    crop: max(0.0, float(raw.get(crop, 0.0) or 0.0))
                    for crop in CORE_GRAIN_COMPONENTS
                }
    return dict(DEFAULT_REGION_PRIORS.get(region, DEFAULT_REGION_PRIORS["global"]))


def _softmax_from_scores(scores: dict[str, float], *, temperature: float) -> dict[str, float]:
    temperature = max(1e-6, float(temperature or 1.0))
    max_score = max(scores.values()) if scores else 0.0
    exp_scores = {
        crop: math.exp((score - max_score) / temperature)
        for crop, score in scores.items()
    }
    total = sum(exp_scores.values())
    if total <= 0.0:
        return {crop: 1.0 / len(CORE_GRAIN_COMPONENTS) for crop in CORE_GRAIN_COMPONENTS}
    return {crop: value / total for crop, value in exp_scores.items()}


def _component_shares(
    *,
    region: str,
    grain_area_mu: float,
    actor_type: str,
    cfg: Mapping[str, Any],
) -> dict[str, float]:
    priors = _region_priors(region, cfg)
    min_prior = max(1e-6, float(cfg.get("min_prior", 1e-4) or 1e-4))
    scale_reference_mu = max(1.0, float(cfg.get("scale_reference_mu", 50.0) or 50.0))
    scale_signal = max(0.0, min(1.0, float(grain_area_mu or 0.0) / scale_reference_mu))
    enterprise_signal = 1.0 if str(actor_type).lower() in {"enterprise", "agri_corporation", "cooperative"} else 0.0

    scores: dict[str, float] = {}
    for crop in CORE_GRAIN_COMPONENTS:
        prior = max(min_prior, float(priors.get(crop, 0.0) or 0.0))
        score = math.log(prior)
        if crop in {"corn", "wheat"}:
            score += float(cfg.get("mechanized_crop_scale_bonus", 0.22) or 0.0) * scale_signal
            score += float(cfg.get("enterprise_mechanized_crop_bonus", 0.12) or 0.0) * enterprise_signal
        if crop == "rice":
            score += float(cfg.get("smallholder_rice_bonus", 0.08) or 0.0) * (1.0 - scale_signal)
        if crop == "soybean":
            score += float(cfg.get("rotation_soybean_bonus", 0.05) or 0.0)
        scores[crop] = score
    shares = _softmax_from_scores(scores, temperature=float(cfg.get("temperature", 0.65) or 0.65))

    soybean_floor = max(0.0, min(0.5, float(cfg.get("soybean_min_share", 0.03) or 0.03)))
    if shares.get("soybean", 0.0) < soybean_floor:
        deficit = soybean_floor - shares.get("soybean", 0.0)
        donor_total = sum(value for crop, value in shares.items() if crop != "soybean")
        if donor_total > 0.0:
            for crop in CORE_GRAIN_COMPONENTS:
                if crop == "soybean":
                    continue
                shares[crop] = max(0.0, shares[crop] - deficit * shares[crop] / donor_total)
            shares["soybean"] = soybean_floor
    total = sum(shares.values())
    return {crop: value / total for crop, value in shares.items()} if total > 0.0 else shares


def assign_grain_crop_components(
    actor: Any,
    *,
    model: Any,
    grain_area_mu: float,
    grain_output_kg: float,
    actor_type: str,
) -> dict[str, dict[str, float]]:
    """Attach crop-component state to an actor when explicitly enabled.

    The resulting state is an endogenous proxy generated from model region,
    scale, and actor type. It is not calibrated enough for publication claims
    until crop-specific official backtests pass.
    """
    cfg = _get_config(model)
    if not bool(cfg.get("enabled", False)):
        _clear_component_state(actor, status="disabled")
        return {}

    grain_area_mu = max(0.0, float(grain_area_mu or 0.0))
    grain_output_kg = max(0.0, float(grain_output_kg or 0.0))
    if grain_area_mu <= 0.0 and grain_output_kg <= 0.0:
        _clear_component_state(actor, status="no_grain_output")
        return {}

    region = _actor_region(actor, model)
    shares = _component_shares(
        region=region,
        grain_area_mu=grain_area_mu,
        actor_type=actor_type,
        cfg=cfg,
    )
    area = {crop: grain_area_mu * share for crop, share in shares.items() if share > 0.0}
    output = {crop: grain_output_kg * share for crop, share in shares.items() if share > 0.0}

    actor.crop_component_area_mu = area
    actor.crop_component_production_kg = output
    actor.crop_component_shares = shares
    actor.crop_component_method = "regional_suitability_scale_softmax"
    actor.crop_component_status = "endogenous_proxy_unvalidated"
    actor.crop_component_region = region
    actor.crop_component_claim_boundary = (
        "Crop components are generated by a model-internal regional suitability and scale proxy. "
        "They are not official-weight allocations, and they require crop-specific official backtests "
        "before forecast claims."
    )
    return {
        crop: {"area_mu": area.get(crop, 0.0), "production_kg": output.get(crop, 0.0)}
        for crop in CORE_GRAIN_COMPONENTS
    }
