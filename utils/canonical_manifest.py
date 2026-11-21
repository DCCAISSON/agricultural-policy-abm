from __future__ import annotations

import json
import hashlib
import subprocess
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


SCIENTIFIC_QUESTION_ZH = (
    "公共财政支持一旦进入乡村食物系统，其边际效力为何会被这个系统内生地吸收，"
    "又在什么制度条件下得到修复？"
)

SCIENTIFIC_QUESTION_EN = (
    "Why does the marginal effectiveness of public support, once injected into a rural food system, "
    "become endogenously absorbed by the very system it seeks to transform, and under what "
    "institutional conditions is this absorption repaired?"
)


CANONICAL_SCENARIO_SEMANTICS: Dict[str, Dict[str, str]] = {
    "policy_baseline": {
        "label_zh": "有政策基准世界线",
        "definition": (
            "保持配置文件中已审计的常规公共支持通道，通常为 baseline_cash 及相应规模补贴。"
            "它不是无政策，也不是初始年份快照。"
        ),
        "claim_boundary": (
            "可作为模型内部有政策参照世界线；不能直接说成现实政策的统计平均效果。"
        ),
    },
    "no_policy_counterfactual": {
        "label_zh": "无政策反事实世界线",
        "definition": (
            "关闭粮补、托市/粮储干预、规模补贴和相应财政预算的反事实路径。"
            "用于和 policy_baseline 做模型内部净效应对比。"
        ),
        "claim_boundary": (
            "可讲模型内部政策通道差异；单路径差异不是现实抽样显著性。"
        ),
    },
    "market_only_baseline": {
        "label_zh": "无干预市场基准",
        "definition": (
            "用于实验设计中的纯市场参照组。除非脚本明确说明，否则不要把它与 "
            "policy_baseline 混用。"
        ),
        "claim_boundary": (
            "可作为机制消融参照；不能替代现实历史基准。"
        ),
    },
    "t0_snapshot": {
        "label_zh": "初始状态快照",
        "definition": (
            "模型初始化或完成第一个农业生产年后的状态记录，用于展示从基期到末期的演化。"
            "它不是政策处理组，也不是无政策反事实。"
        ),
        "claim_boundary": (
            "可用于模型体检和时间轴连续性审计；不能作为政策净效应对照组。"
        ),
    },
    "reference_scenario": {
        "label_zh": "梯度/组合实验参照组",
        "definition": (
            "补贴强度、政策组合或帕累托实验中的标准化参照点，例如 "
            "grain_100_scale_100。它用于比较强度梯度，不代表无政策。"
        ),
        "claim_boundary": (
            "可讲相对于参照组的结构差异；不能说成现实最优补贴标准。"
        ),
    },
}


LEGACY_SCENARIO_ALIASES = {
    "baseline": "policy_baseline",
    "Baseline": "policy_baseline",
    "no_policy": "no_policy_counterfactual",
    "no-policy": "no_policy_counterfactual",
    "market_only": "market_only_baseline",
    "market-only": "market_only_baseline",
    "initial_snapshot": "t0_snapshot",
    "t0": "t0_snapshot",
}

STANDARD_CANONICAL_ARTIFACTS = (
    "config_effective.yaml",
    "main_indicators.csv",
    "three_axis_summary.csv",
    "three_axis_summary.json",
)

CANONICAL_RUN_MODES = {"debug", "smoke", "formal"}


def canonicalize_scenario_name(name: str, *, context: Optional[str] = None) -> str:
    """Return a canonical scenario semantic name without mutating runtime keys.

    The optional context resolves legacy words that were historically
    overloaded. Runtime scenario IDs such as ``baseline_cash`` should stay as
    policy-library IDs; this function only standardizes evidence semantics.
    """
    raw = str(name)
    if raw in CANONICAL_SCENARIO_SEMANTICS:
        return raw
    context_key = str(context or "").lower()
    if raw in {"baseline", "Baseline"} and context_key in {
        "subsidy",
        "subsidy_evaluation",
        "reference",
        "policy_bundle",
    }:
        return "reference_scenario"
    return LEGACY_SCENARIO_ALIASES.get(raw, raw)


def scenario_semantics(name: str, *, context: Optional[str] = None) -> Dict[str, Any]:
    canonical = canonicalize_scenario_name(name, context=context)
    payload = deepcopy(CANONICAL_SCENARIO_SEMANTICS.get(canonical, {}))
    payload.setdefault("label_zh", canonical)
    payload.setdefault("definition", "脚本自定义情景；请以 run_manifest.json 中的 policy_runtime 为准。")
    payload.setdefault("claim_boundary", "只能在本次模型实验设计内解释。")
    payload["canonical_semantics"] = canonical
    return payload


def scenario_record(
    name: str,
    *,
    context: Optional[str] = None,
    policy_scenario: Optional[str] = None,
    runtime_name: Optional[str] = None,
    description: Optional[str] = None,
    artifacts: Optional[Iterable[str]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    semantics = scenario_semantics(name, context=context)
    record: Dict[str, Any] = {
        "name": str(name),
        "canonical_semantics": semantics["canonical_semantics"],
        "label_zh": semantics["label_zh"],
        "definition": semantics["definition"],
        "claim_boundary": semantics["claim_boundary"],
    }
    if runtime_name:
        record["runtime_name"] = runtime_name
    if policy_scenario:
        record["policy_scenario"] = policy_scenario
    if description:
        record["description"] = description
    if artifacts:
        record["artifacts"] = list(artifacts)
    record.update({key: value for key, value in extra.items() if value is not None})
    return record


def baseline_semantics_payload() -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rule": "Do not use bare 'baseline' in public-facing results; choose one canonical meaning.",
        "semantics": deepcopy(CANONICAL_SCENARIO_SEMANTICS),
        "legacy_aliases": deepcopy(LEGACY_SCENARIO_ALIASES),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_payload_hash(payload: Any) -> str:
    encoded = json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_fingerprint(output_dir: Path, artifact_name: str) -> Dict[str, Any]:
    rel = Path(str(artifact_name))
    path = output_dir / rel
    record: Dict[str, Any] = {
        "path": str(rel).replace("\\", "/"),
        "exists": path.exists(),
    }
    if not path.exists():
        return record
    if path.is_dir():
        record.update({"type": "directory"})
        return record
    stat = path.stat()
    record.update(
        {
            "type": "file",
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "sha256": sha256_file(path),
        }
    )
    return record


def build_artifact_fingerprints(output_dir: Path, artifact_names: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    return {
        str(name): _artifact_fingerprint(Path(output_dir), str(name))
        for name in artifact_names
    }


def _normalize_run_mode(run_mode: Optional[str]) -> str:
    if run_mode is None:
        return "debug"
    normalized = str(run_mode).strip().lower()
    if normalized not in CANONICAL_RUN_MODES:
        raise ValueError(f"Unknown run_mode '{run_mode}'. Expected one of {sorted(CANONICAL_RUN_MODES)}")
    return normalized


def _as_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def infer_run_mode(run_mode: Optional[str], run_parameters: Optional[Mapping[str, Any]]) -> str:
    """Infer whether a result is smoke/debug/formal from runner metadata."""
    if run_mode is not None:
        return _normalize_run_mode(run_mode)
    params = run_parameters or {}
    steps = _as_int_or_none(params.get("steps"))
    scenario_set = str(params.get("scenario_set", "") or "").strip().lower()
    workflow_lite = _as_bool(params.get("workflow_lite"))
    if scenario_set == "smoke" or workflow_lite or (steps is not None and steps <= 1):
        return "smoke"
    if steps is not None and steps >= 30:
        return "formal"
    return "debug"


def build_citation_readiness(run_mode: str, run_parameters: Mapping[str, Any]) -> Dict[str, Any]:
    if run_mode == "formal":
        status = "formal_candidate"
        can_cite = True
        reason = "Formal run-mode candidate; final citation still requires canonical_output_audit failures=[] and disclosed warnings."
    elif run_mode == "smoke":
        status = "engineering_smoke_only"
        can_cite = False
        reason = "Smoke outputs prove runner connectivity and output contract only; do not cite numeric results."
    else:
        status = "debug_only"
        can_cite = False
        reason = "Debug outputs are not formal scientific evidence."
    return {
        "status": status,
        "can_cite_numeric_results": can_cite,
        "reason": reason,
        "run_parameters": dict(run_parameters),
    }


def _as_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_grain_subsidy_alias_conflicts(value: Any, *, path: str) -> List[Dict[str, Any]]:
    conflicts: List[Dict[str, Any]] = []
    if isinstance(value, Mapping):
        if "grain_subsidies_rate" in value and "subsidy_rate" in value:
            canonical = value.get("grain_subsidies_rate")
            legacy = value.get("subsidy_rate")
            canonical_float = _as_float_or_none(canonical)
            legacy_float = _as_float_or_none(legacy)
            if canonical_float is not None and legacy_float is not None:
                differs = abs(canonical_float - legacy_float) > 1e-12
            else:
                differs = str(canonical) != str(legacy)
            if differs:
                conflicts.append(
                    {
                        "path": path,
                        "grain_subsidies_rate": canonical,
                        "subsidy_rate": legacy,
                        "reason": "canonical grain_subsidies_rate conflicts with legacy subsidy_rate",
                    }
                )
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            conflicts.extend(_find_grain_subsidy_alias_conflicts(child, path=child_path))
    elif isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            conflicts.extend(_find_grain_subsidy_alias_conflicts(child, path=f"{path}[{idx}]"))
    return conflicts


def build_policy_alias_audit(
    *,
    config: Optional[Mapping[str, Any]] = None,
    model: Any = None,
) -> Dict[str, Any]:
    """Record canonical-vs-legacy grain subsidy naming conflicts.

    ``grain_subsidies_rate`` is the canonical key. ``subsidy_rate`` is retained
    as a legacy alias. Formal outputs must not contain conflicting values.
    """
    checked_sources: List[str] = []
    conflicts: List[Dict[str, Any]] = []
    if config is not None:
        checked_sources.append("config")
        conflicts.extend(_find_grain_subsidy_alias_conflicts(config, path="config"))
    if model is not None:
        env = getattr(model, "environment", None)
        active_policy = getattr(env, "active_policy", None) if env is not None else None
        if isinstance(active_policy, Mapping):
            checked_sources.append("model.environment.active_policy")
            conflicts.extend(
                _find_grain_subsidy_alias_conflicts(
                    active_policy,
                    path="model.environment.active_policy",
                )
            )
    return {
        "schema_version": "v1",
        "canonical_key": "grain_subsidies_rate",
        "legacy_alias": "subsidy_rate",
        "checked_sources": checked_sources,
        "has_conflicts": bool(conflicts),
        "conflicts": conflicts,
    }


def _git_command(args: List[str], *, cwd: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_fingerprint() -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    head = _git_command(["rev-parse", "HEAD"], cwd=repo_root)
    branch = _git_command(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    status = _git_command(["status", "--porcelain=v1"], cwd=repo_root)
    if status is None:
        return {"available": False}
    lines = [line for line in status.splitlines() if line.strip()]
    return {
        "available": True,
        "commit": head,
        "branch": branch,
        "dirty": bool(lines),
        "dirty_entry_count": len(lines),
        "dirty_entry_sample": lines[:25],
    }


def _seed_candidates(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not config:
        return {}

    def read_path(root: Mapping[str, Any], dotted: str) -> Any:
        current: Any = root
        for part in dotted.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return None
            current = current[part]
        return current

    paths = [
        "seed",
        "random_seed",
        "experiment.seed",
        "experiment.random_seed",
        "simulation.seed",
        "simulation.random_seed",
        "model.seed",
        "model.random_seed",
    ]
    return {
        path: value
        for path in paths
        if (value := read_path(config, path)) is not None
    }


def _policy_snapshot_from_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if config is None:
        return {"available": False, "reason": "config_not_provided"}
    try:
        from utils.policy_runtime import snapshot_policy_runtime_from_config

        return {
            "available": True,
            "source": "config",
            "snapshot": snapshot_policy_runtime_from_config(deepcopy(dict(config))),
        }
    except Exception as exc:  # pragma: no cover - defensive manifest path
        return {"available": False, "source": "config", "error": str(exc)}


def _policy_snapshot_from_model(model: Any) -> Dict[str, Any]:
    if model is None:
        return {"available": False, "reason": "model_not_provided"}
    try:
        from utils.policy_runtime import snapshot_policy_runtime_from_model

        return {
            "available": True,
            "source": "model",
            "snapshot": snapshot_policy_runtime_from_model(model),
        }
    except Exception as exc:  # pragma: no cover - defensive manifest path
        return {"available": False, "source": "model", "error": str(exc)}


def write_effective_policy_snapshot(
    output_dir: Path,
    *,
    config: Optional[Mapping[str, Any]] = None,
    model: Any = None,
) -> Path:
    payload = {
        "schema_version": "v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "from_config": _policy_snapshot_from_config(config),
        "from_model": _policy_snapshot_from_model(model),
    }
    return write_json(Path(output_dir) / "effective_policy_snapshot.json", payload)


def write_baseline_semantics(output_dir: Path) -> Path:
    return write_json(Path(output_dir) / "baseline_semantics.json", baseline_semantics_payload())


def write_run_manifest(
    output_dir: Path,
    *,
    script_name: str,
    run_type: str,
    scenarios: Iterable[Mapping[str, Any]],
    config: Optional[Mapping[str, Any]] = None,
    model: Any = None,
    artifacts: Optional[Iterable[str]] = None,
    claim_boundaries: Optional[Iterable[str]] = None,
    notes: Optional[Iterable[str]] = None,
    run_mode: Optional[str] = None,
    run_parameters: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    output_dir = Path(output_dir)
    semantics_path = write_baseline_semantics(output_dir)
    policy_snapshot_path = write_effective_policy_snapshot(output_dir, config=config, model=model)
    parameters = dict(run_parameters or {})
    resolved_run_mode = infer_run_mode(run_mode, parameters)
    artifact_names = list(artifacts or [])
    for name in STANDARD_CANONICAL_ARTIFACTS:
        if (output_dir / name).exists() and name not in artifact_names:
            artifact_names.append(name)
    artifact_fingerprints = build_artifact_fingerprints(output_dir, artifact_names)
    payload: Dict[str, Any] = {
        "schema_version": "v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": script_name,
        "run_type": run_type,
        "run_mode": resolved_run_mode,
        "run_parameters": parameters,
        "canonical_runtime": True,
        "scientific_question_zh": SCIENTIFIC_QUESTION_ZH,
        "scientific_question_en": SCIENTIFIC_QUESTION_EN,
        "baseline_semantics_file": semantics_path.name,
        "effective_policy_snapshot_file": policy_snapshot_path.name,
        "scenarios": list(scenarios),
        "artifacts": artifact_names,
        "artifact_fingerprints": artifact_fingerprints,
        "run_fingerprint": {
            "config_sha256": _stable_payload_hash(config) if config is not None else None,
            "seed_candidates": _seed_candidates(config),
            "git": _git_fingerprint(),
        },
        "policy_alias_audit": build_policy_alias_audit(config=config, model=model),
        "citation_readiness": build_citation_readiness(resolved_run_mode, parameters),
        "claim_boundaries": list(claim_boundaries or []),
        "notes": list(notes or []),
    }
    if extra:
        payload["extra"] = dict(extra)
    manifest_path = write_json(output_dir / "run_manifest.json", payload)
    try:
        from utils.canonical_output_validator import write_canonical_output_audit

        write_canonical_output_audit(output_dir)
    except Exception as exc:
        warnings.warn(f"canonical output audit failed for {output_dir}: {exc}", RuntimeWarning)
    return manifest_path


def write_claim_mapping_markdown(
    output_dir: Path,
    *,
    title: str,
    rows: Iterable[Mapping[str, Any]],
    extra_note: Optional[str] = None,
) -> Path:
    lines: List[str] = [
        f"# {title}",
        "",
        f"生成时间: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| 输出/结果 | 可以怎么讲 | 不能怎么讲 | 证据边界 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {artifact} | {can_claim} | {cannot_claim} | {boundary} |".format(
                artifact=str(row.get("artifact", "")),
                can_claim=str(row.get("can_claim", "")),
                cannot_claim=str(row.get("cannot_claim", "")),
                boundary=str(row.get("boundary", "")),
            )
        )
    if extra_note:
        lines.extend(["", extra_note])
    path = Path(output_dir) / "claim_mapping.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
