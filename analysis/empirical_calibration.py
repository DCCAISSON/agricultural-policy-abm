import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml


@dataclass
class ColumnMapping:
    source: Optional[str]
    description: str


class EmpiricalMomentLoader:
    """
    Load empirical datasets and derive calibration moments for heterogeneity and
    social network parameters.
    """

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Calibration config not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh)

        self.base_dir = Path(self.config.get("data_source_dir", ".")).resolve()
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Data source directory does not exist: {self.base_dir}")

    def _load_dataset(self, entry: Dict) -> pd.DataFrame:
        file_name = entry.get("file")
        if not file_name:
            raise ValueError("Dataset entry must specify 'file'")

        dataset_path = self.base_dir / file_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        sheet = entry.get("sheet", 0)
        usecols = entry.get("usecols")
        df = pd.read_excel(dataset_path, sheet_name=sheet, usecols=usecols)
        return df

    def _extract_series(self, df: pd.DataFrame, column_name: Optional[str]) -> Optional[pd.Series]:
        if not column_name:
            return None
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in dataset. Available columns: {list(df.columns)}")
        series = df[column_name].dropna()
        if series.empty:
            return None
        return series

    @staticmethod
    def _fit_lognormal(series: pd.Series) -> Optional[Dict[str, float]]:
        positive = series[series > 0]
        if positive.empty:
            return None

        log_values = np.log(positive)
        mu = float(log_values.mean())
        sigma = float(log_values.std(ddof=0))
        return {
            "log_mean": mu,
            "log_std": sigma,
            "mean": float(positive.mean()),
            "std": float(positive.std(ddof=0))
        }

    @staticmethod
    def _basic_stats(series: pd.Series) -> Optional[Dict[str, float]]:
        if series.empty:
            return None
        return {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
            "quantiles": {
                "p25": float(series.quantile(0.25)),
                "p50": float(series.quantile(0.5)),
                "p75": float(series.quantile(0.75)),
            }
        }

    def derive_farmer_statistics(self) -> Dict:
        farmer_cfg = self.config.get("farmer_panel")
        if not farmer_cfg:
            return {}

        df = self._load_dataset(farmer_cfg)
        column_map = farmer_cfg.get("columns", {})

        stats = {}

        capital_col = column_map.get("initial_capital")
        if capital_col:
            series = self._extract_series(df, capital_col)
            if series is not None:
                stats["initial_capital"] = {
                    "lognormal": self._fit_lognormal(series),
                    "summary": self._basic_stats(series)
                }

        ability_col = column_map.get("farming_ability")
        if ability_col:
            series = self._extract_series(df, ability_col)
            if series is not None:
                stats["farming_ability"] = {
                    "summary": self._basic_stats(series)
                }

        social_degree_col = column_map.get("social_degree")
        if social_degree_col:
            series = self._extract_series(df, social_degree_col)
            if series is not None:
                stats["social_degree"] = {
                    "summary": self._basic_stats(series)
                }

        farmland_col = column_map.get("farmland_area")
        if farmland_col:
            series = self._extract_series(df, farmland_col)
            if series is not None:
                stats["farmland_area"] = {
                    "summary": self._basic_stats(series)
                }

        risk_col = column_map.get("risk_aversion")
        if risk_col:
            series = self._extract_series(df, risk_col)
            if series is not None:
                stats["risk_aversion"] = {
                    "summary": self._basic_stats(series)
                }

        return stats

    def derive_village_statistics(self) -> Dict:
        village_cfg = self.config.get("village_panel")
        if not village_cfg:
            return {}

        df = self._load_dataset(village_cfg)
        column_map = village_cfg.get("columns", {})

        stats = {}

        mean_degree_col = column_map.get("mean_degree")
        if mean_degree_col:
            series = self._extract_series(df, mean_degree_col)
            if series is not None:
                stats["mean_degree"] = self._basic_stats(series)

        elite_share_col = column_map.get("elite_share")
        if elite_share_col:
            series = self._extract_series(df, elite_share_col)
            if series is not None:
                stats["elite_share"] = self._basic_stats(series)

        clustering_col = column_map.get("clustering")
        if clustering_col:
            series = self._extract_series(df, clustering_col)
            if series is not None:
                stats["clustering"] = self._basic_stats(series)

        return stats

    def derive_empirical_moments(self) -> Dict:
        return {
            "farmer": self.derive_farmer_statistics(),
            "village": self.derive_village_statistics()
        }

    def export_empirical_moments(self, destination: Path) -> None:
        moments = self.derive_empirical_moments()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as fh:
            json.dump(moments, fh, ensure_ascii=False, indent=2)




