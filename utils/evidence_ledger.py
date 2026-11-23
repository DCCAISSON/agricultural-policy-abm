from __future__ import annotations

import csv
import json
import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.reproducibility import DataLineage


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _hash_payload(payload: Any) -> str:
    content = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=_json_default)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _normalize_paths(paths: Optional[Iterable[Any]]) -> List[str]:
    normalized: List[str] = []
    for path in paths or []:
        normalized.append(str(Path(path).resolve()))
    return normalized


def _to_flat_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat_rows: List[Dict[str, Any]] = []
    for record in records:
        flat: Dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, (dict, list)):
                flat[key] = json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)
            else:
                flat[key] = value
        flat_rows.append(flat)
    return flat_rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class ExperimentEvidenceRecord:
    experiment_id: str
    label: str
    source_script: str
    output_dir: str
    seed: Optional[int] = None
    config_hash: str = ""
    config_path: str = ""
    worldline_id: Optional[str] = None
    worldline_hash: str = ""
    scenario_name: str = ""
    policy_runtime: Dict[str, Any] = field(default_factory=dict)
    ai_mechanism: Dict[str, Any] = field(default_factory=dict)
    shock_schedule: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    lineage: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FigureEvidenceRecord:
    figure_id: str
    label: str
    source_script: str
    output_paths: List[str]
    result_dirs: List[str] = field(default_factory=list)
    config_path: str = ""
    config_hash: str = ""
    seeds: List[int] = field(default_factory=list)
    worldline_ids: List[str] = field(default_factory=list)
    manifest_paths: List[str] = field(default_factory=list)
    runtime_definition: Dict[str, Any] = field(default_factory=dict)
    policy_runtime: Dict[str, Any] = field(default_factory=dict)
    placeholder_used: bool = False
    audit_flags: List[str] = field(default_factory=list)
    evidence_status: str = "evidence_backed"
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceLedger:
    def __init__(self, output_dir: Path | str, *, stem: str = "evidence_ledger"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stem = stem
        self.experiments: List[ExperimentEvidenceRecord] = []
        self.figures: List[FigureEvidenceRecord] = []
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "ledger_stem": stem,
        }

    def register_experiment(
        self,
        *,
        experiment_id: str,
        label: str,
        source_script: str,
        output_dir: Path | str,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Path | str] = None,
        seed: Optional[int] = None,
        worldline_id: Optional[str | int] = None,
        worldline_hash: str = "",
        scenario_name: str = "",
        policy_runtime: Optional[Dict[str, Any]] = None,
        ai_mechanism: Optional[Dict[str, Any]] = None,
        shock_schedule: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Iterable[Any]] = None,
        notes: str = "",
    ) -> ExperimentEvidenceRecord:
        lineage: Dict[str, Any] = {}
        config_hash = ""
        if config is not None:
            config_hash = _hash_payload(config)
            if seed is not None:
                try:
                    lineage = DataLineage.create(config, experiment_id, seed).to_dict()
                except Exception:
                    lineage = {}

        record = ExperimentEvidenceRecord(
            experiment_id=experiment_id,
            label=label,
            source_script=str(Path(source_script).resolve()),
            output_dir=str(Path(output_dir).resolve()),
            seed=seed,
            config_hash=config_hash,
            config_path=str(Path(config_path).resolve()) if config_path else "",
            worldline_id=str(worldline_id) if worldline_id is not None else None,
            worldline_hash=worldline_hash,
            scenario_name=scenario_name,
            policy_runtime=policy_runtime or {},
            ai_mechanism=ai_mechanism or {},
            shock_schedule=shock_schedule or {},
            artifacts=_normalize_paths(artifacts),
            lineage=lineage,
            notes=notes,
        )
        self.experiments.append(record)
        return record

    def register_figure(
        self,
        *,
        figure_id: str,
        label: str,
        source_script: str,
        output_paths: Iterable[Any],
        result_dirs: Optional[Iterable[Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Path | str] = None,
        seeds: Optional[Iterable[int]] = None,
        worldline_ids: Optional[Iterable[Any]] = None,
        manifest_paths: Optional[Iterable[Any]] = None,
        runtime_definition: Optional[Dict[str, Any]] = None,
        policy_runtime: Optional[Dict[str, Any]] = None,
        placeholder_used: bool = False,
        audit_flags: Optional[Iterable[str]] = None,
        evidence_status: Optional[str] = None,
        notes: str = "",
    ) -> FigureEvidenceRecord:
        config_hash = _hash_payload(config) if config is not None else ""
        normalized_audit_flags = sorted({str(flag) for flag in (audit_flags or []) if str(flag).strip()})
        resolved_status = evidence_status or ("placeholder" if placeholder_used else ("audit_attention" if normalized_audit_flags else "evidence_backed"))
        record = FigureEvidenceRecord(
            figure_id=figure_id,
            label=label,
            source_script=str(Path(source_script).resolve()),
            output_paths=_normalize_paths(output_paths),
            result_dirs=_normalize_paths(result_dirs),
            config_path=str(Path(config_path).resolve()) if config_path else "",
            config_hash=config_hash,
            seeds=list(seeds or []),
            worldline_ids=[str(item) for item in (worldline_ids or [])],
            manifest_paths=_normalize_paths(manifest_paths),
            runtime_definition=runtime_definition or {},
            policy_runtime=policy_runtime or {},
            placeholder_used=bool(placeholder_used),
            audit_flags=normalized_audit_flags,
            evidence_status=resolved_status,
            notes=notes,
        )
        self.figures.append(record)
        return record

    def save(self) -> Dict[str, Path]:
        payload = {
            "metadata": self.metadata,
            "experiments": [record.to_dict() for record in self.experiments],
            "figures": [record.to_dict() for record in self.figures],
        }

        ledger_path = self.output_dir / f"{self.stem}.json"
        with open(ledger_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)

        experiment_rows = _to_flat_rows([record.to_dict() for record in self.experiments])
        figure_rows = _to_flat_rows([record.to_dict() for record in self.figures])

        experiment_csv = self.output_dir / "experiment_registry.csv"
        figure_csv = self.output_dir / "figure_registry.csv"
        _write_csv(experiment_csv, experiment_rows)
        _write_csv(figure_csv, figure_rows)

        return {
            "ledger_json": ledger_path,
            "experiment_registry_csv": experiment_csv,
            "figure_registry_csv": figure_csv,
        }
