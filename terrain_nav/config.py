"""Configuration models and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class SearchConfig:
    """Runtime settings chosen to keep the search suitable for onboard use."""

    resolution_m: float = 30.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    interpolation: str = "bilinear"

    initial_azimuth_step_deg: float = 1.0
    initial_azimuth_limit_deg: float = 45.0
    local_azimuth_step_deg: float = 2.0
    local_turn_limit_deg: float = 30.0
    speed_step_mps: float = 2.0
    max_delta_v_mps_per_window: float = 8.0
    recovery_speed_multiplier: float = 1.5

    top_k_hypotheses: int = 7 #7
    recovery_top_k: int = 15 #15
    beam_width: int = 7 #7
    tree_depth: int = 3 #3
    window_seconds: float = 10  #10 sec #TODO: сделать динамически
    window_overlap: float = 0.5
    min_window_samples: int = 6
    min_window_time_coverage: float = 0.8
    low_confidence_expand_search: bool = True

    rerank_candidates: int = 40
    dtw_window: int = 10
    history_rerank_interval_windows: int = 2
    history_rerank_pool_multiplier: int = 10
    history_rerank_expensive_top: int = 20
    history_score_weight: float = 0.90
    turn_penalty_weight: float = 0.08
    turn_penalty_scale_deg: float = 10.0
    speed_change_penalty_weight: float = 0.05
    out_of_bounds_penalty_weight: float = 0.75
    low_terrain_variance_penalty_weight: float = 0.15
    random_seed: int = 42

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SearchConfig":
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SearchConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream) or {}
        search_values = values.get("search", values)
        dem_values = values.get("dem", {})
        return cls.from_dict({**search_values, **dem_values})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None) -> SearchConfig:
    return SearchConfig() if path is None else SearchConfig.from_yaml(path)
