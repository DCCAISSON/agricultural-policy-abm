# -*- coding: utf-8 -*-
"""
@theory Spatial Climate Hazard Module
@created 2025-12-28

Responsible for generating spatially heterogeneous climate risk maps.
"""

import numpy as np

from core.reproducibility import np_randint, np_random_value, np_uniform


class ClimateManager:
    def __init__(self, width, height, config=None):
        self.width = width
        self.height = height
        self.config = config or {}
        self.current_hazard_map = None
        self.current_favorable_map = None
        self.severity_history = []
        self.favorability_history = []
        self.np_random = None
        self.forced_events = []
        self._forced_event_index = -1

        spatial_hazard_config = {}
        if isinstance(self.config, dict):
            spatial_hazard_config = self.config.get("spatial_hazards", {}) or {}
            forced_events = self.config.get("forced_events", []) or []
            if isinstance(forced_events, list):
                self.forced_events = forced_events

        self.base_risk_prob = self._resolve_float_config(
            spatial_hazard_config,
            self.config,
            keys=("base_risk_prob", "spatial_hazard_prob"),
            default=0.15,
        )
        self.cluster_size = self._resolve_float_config(
            spatial_hazard_config,
            self.config,
            keys=("cluster_size", "hazard_cluster_size"),
            default=3.0,
        )
        self.max_severity = self._resolve_float_config(
            spatial_hazard_config,
            self.config,
            keys=("max_severity", "spatial_max_loss"),
            default=0.6,
        )
        self.favorable_prob = self._resolve_float_config(
            spatial_hazard_config,
            self.config,
            keys=("favorable_prob", "spatial_favorable_prob"),
            default=0.08,
        )
        self.max_gain = self._resolve_float_config(
            spatial_hazard_config,
            self.config,
            keys=("max_gain", "spatial_max_gain"),
            default=0.06,
        )
        self.minimum_yield_modifier = self._resolve_float_config(
            spatial_hazard_config,
            self.config,
            keys=("minimum_yield_modifier",),
            default=0.82,
        )
        self.maximum_yield_modifier = self._resolve_float_config(
            spatial_hazard_config,
            self.config,
            keys=("maximum_yield_modifier",),
            default=1.10,
        )

    @staticmethod
    def _resolve_float_config(primary_config, fallback_config, *, keys, default):
        for config in (primary_config, fallback_config):
            if not isinstance(config, dict):
                continue
            for key in keys:
                if key not in config:
                    continue
                try:
                    return float(config.get(key))
                except (TypeError, ValueError):
                    continue
        return float(default)

    def step(self):
        """Annual climate-state refresh."""
        self._forced_event_index += 1
        if self._forced_event_index < len(self.forced_events):
            self._generate_forced_hazard_map(self.forced_events[self._forced_event_index])
            return
        self._generate_hazard_map()

    def _generate_forced_hazard_map(self, event):
        """Generate a pre-committed climate event for paired policy worldlines."""
        self.current_hazard_map = np.zeros((self.width, self.height))
        self.current_favorable_map = np.zeros((self.width, self.height))
        if not isinstance(event, dict):
            return

        kind = str(event.get("kind", "none") or "none").strip().lower()
        if kind not in {"adverse", "favorable"}:
            return

        target_map = self.current_hazard_map if kind == "adverse" else self.current_favorable_map
        default_upper = self.max_severity if kind == "adverse" else self.max_gain
        for center in event.get("centers", []) or []:
            if not isinstance(center, dict):
                continue
            try:
                cx = int(center.get("x", 0))
                cy = int(center.get("y", 0))
                intensity = float(center.get("intensity", default_upper))
            except (TypeError, ValueError):
                continue
            cx = max(0, min(self.width - 1, cx))
            cy = max(0, min(self.height - 1, cy))
            self._apply_gaussian_blob(target_map, cx, cy, intensity)

    def _generate_hazard_map(self):
        """Generate either an adverse spatial shock map or a favorable map."""
        self.current_hazard_map = np.zeros((self.width, self.height))
        self.current_favorable_map = np.zeros((self.width, self.height))

        climate_draw = np_random_value(self)
        adverse_event = climate_draw < self.base_risk_prob
        favorable_event = (
            not adverse_event
            and climate_draw < (self.base_risk_prob + self.favorable_prob)
        )
        if not adverse_event and not favorable_event:
            return

        num_centers = int(np_randint(self, 1, 4 if adverse_event else 3))
        target_map = self.current_hazard_map if adverse_event else self.current_favorable_map
        intensity_upper = self.max_severity if adverse_event else self.max_gain
        intensity_lower = 0.3 if adverse_event else 0.2

        for _ in range(num_centers):
            cx = int(np_randint(self, 0, self.width))
            cy = int(np_randint(self, 0, self.height))
            intensity = float(np_uniform(self, intensity_lower, intensity_upper))
            self._apply_gaussian_blob(target_map, cx, cy, intensity)

    def _apply_gaussian_blob(self, target_map, cx, cy, peak_intensity):
        """Add a Gaussian blob to the provided climate-impact map."""
        radius = int(self.cluster_size * 2)
        x_min = max(0, cx - radius)
        x_max = min(self.width, cx + radius)
        y_min = max(0, cy - radius)
        y_max = min(self.height, cy + radius)

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist >= self.cluster_size * 2:
                    continue
                factor = np.exp(-(dist ** 2) / (2 * self.cluster_size ** 2))
                impact = peak_intensity * factor
                target_map[x, y] = max(target_map[x, y], impact)

    def get_yield_modifier(self, pos):
        """
        Get yield modifier at a given position.
        Returns:
            float: 1.0 = neutral, <1 adverse, >1 favorable
        """
        if self.current_hazard_map is None and self.current_favorable_map is None:
            return 1.0

        x, y = pos
        if 0 <= x < self.width and 0 <= y < self.height:
            severity = 0.0
            gain = 0.0
            if self.current_hazard_map is not None:
                severity = float(self.current_hazard_map[int(x), int(y)] or 0.0)
            if self.current_favorable_map is not None:
                gain = float(self.current_favorable_map[int(x), int(y)] or 0.0)
            modifier = 1.0 - severity + gain
            return max(self.minimum_yield_modifier, min(self.maximum_yield_modifier, modifier))
        return 1.0
