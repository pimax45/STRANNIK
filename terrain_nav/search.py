"""Vectorized terrain-profile search with a top-K hypothesis beam."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Any

import numpy as np

from .config import SearchConfig
from .dem import DEMInterpolator
from .profile import (
    GroundProfile,
    remove_outliers_hampel,
    smooth_moving_average,
)
from .scoring import ScoreComponents, fast_scores_batch, score_profiles


@dataclass(slots=True)
class CandidateProfile:
    x: np.ndarray
    y: np.ndarray
    heights_m: np.ndarray
    azimuth_deg: float
    ground_speed_mps: float


@dataclass(slots=True)
class MotionHypothesis:
    x: float
    y: float
    azimuth_deg: float
    ground_speed_mps: float
    score: float
    confidence: float


@dataclass(slots=True)
class TrajectoryHypothesis:
    x: float
    y: float
    azimuth_deg: float
    ground_speed_mps: float
    score: float
    cumulative_score: float
    confidence: float
    path_x: np.ndarray
    path_y: np.ndarray
    path_heights_dem: np.ndarray
    parent_id: int | None = None
    hypothesis_id: int | None = None
    depth: int = 0
    number_of_windows: int = 1
    score_components: ScoreComponents | None = None
    history_score: float | None = None
    ranking_score: float | None = None
    score_history: tuple[float, ...] = field(default_factory=tuple, repr=False)
    speed_history: tuple[float, ...] = field(default_factory=tuple, repr=False)
    heading_history: tuple[float, ...] = field(default_factory=tuple, repr=False)

    @property
    def mean_score(self) -> float:
        return self.cumulative_score / max(self.number_of_windows, 1)

    @property
    def selection_score(self) -> float:
        return self.ranking_score if self.ranking_score is not None else self.mean_score


@dataclass(frozen=True, slots=True)
class ScoreGrid:
    azimuths_deg: np.ndarray
    speeds_mps: np.ndarray
    scores: np.ndarray


@dataclass(slots=True)
class SearchResult:
    best: TrajectoryHypothesis
    alternatives: list[TrajectoryHypothesis]
    confidence: str
    terrain_informativeness: float
    best_score: float
    second_best_score: float
    score_gap: float
    tree_depth: int
    top_k_hypotheses: int
    window_seconds: float
    score_grid: ScoreGrid | None = None
    measured_timestamps: np.ndarray | None = None
    measured_heights_m: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        components = self.best.score_components
        return {
            "current_x": self.best.x,
            "current_y": self.best.y,
            "azimuth_deg": self.best.azimuth_deg,
            "ground_speed_mps": self.best.ground_speed_mps,
            "best_score": self.best_score,
            "history_score": self.best.history_score,
            "mean_window_score": self.best.mean_score,
            "second_best_score": self.second_best_score,
            "score_gap": self.score_gap,
            "confidence": self.confidence,
            "confidence_value": self.best.confidence,
            "terrain_informativeness": self.terrain_informativeness,
            "tree_depth": self.tree_depth,
            "top_k_hypotheses": self.top_k_hypotheses,
            "window_seconds": self.window_seconds,
            "best_path": [
                {"x": float(x), "y": float(y)}
                for x, y in zip(self.best.path_x, self.best.path_y, strict=True)
            ],
            "alternative_hypotheses": [
                {
                    "x": item.x,
                    "y": item.y,
                    "azimuth_deg": item.azimuth_deg,
                    "ground_speed_mps": item.ground_speed_mps,
                    "score": item.score,
                    "mean_score": item.mean_score,
                    "history_score": item.history_score,
                }
                for item in self.alternatives
            ],
            "scores": components.as_dict() if components else {},
        }


def build_dem_profile(
    dem_interpolator: DEMInterpolator,
    start_x: float,
    start_y: float,
    azimuth_deg: float,
    ground_speed_mps: float,
    timestamps: np.ndarray,
) -> CandidateProfile:
    times = np.asarray(timestamps, dtype=np.float64)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("timestamps must contain at least two values")
    elapsed = times - times[0]
    theta = np.deg2rad(azimuth_deg)
    distance = ground_speed_mps * elapsed
    x = start_x + distance * np.sin(theta)
    y = start_y + distance * np.cos(theta)
    heights = dem_interpolator.sample_batch(x, y)
    return CandidateProfile(x, y, heights, azimuth_deg % 360.0, ground_speed_mps)


def terrain_informativeness(profile: np.ndarray, score_gap: float = 0.0) -> float:
    values = np.asarray(profile, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return 0.0
    slopes = np.diff(values)
    extrema = np.count_nonzero(np.diff(np.sign(slopes)) != 0)
    height_term = np.clip(np.std(values) / 20.0, 0.0, 1.0)
    slope_term = np.clip(np.std(slopes) / 3.0, 0.0, 1.0)
    extrema_term = np.clip(extrema / max(values.size / 12.0, 1.0), 0.0, 1.0)
    gap_term = np.clip(score_gap / 0.08, 0.0, 1.0)
    return float(
        0.35 * height_term
        + 0.30 * slope_term
        + 0.20 * extrema_term
        + 0.15 * gap_term
    )


def confidence_level(
    profile: np.ndarray,
    best_score: float,
    score_gap: float,
    *,
    speed_consistency: float = 1.0,
    heading_consistency: float = 1.0,
) -> tuple[str, float]:
    terrain_std = float(np.nanstd(profile))
    informative = terrain_informativeness(profile, score_gap)
    stability = np.clip(0.5 * speed_consistency + 0.5 * heading_consistency, 0.0, 1.0)
    numeric = float(
        np.clip(
            0.50 * ((best_score + 1.0) / 2.0)
            + 0.25 * informative
            + 0.15 * np.clip(score_gap / 0.08, 0.0, 1.0)
            + 0.10 * stability,
            0.0,
            1.0,
        )
    )
    if terrain_std < 5.0:
        return "low", min(numeric, 0.35)
    if best_score >= 0.85 and score_gap >= 0.08 and numeric >= 0.72:
        return "high", numeric
    if best_score >= 0.65 and numeric >= 0.52:
        return "medium", numeric
    return "low", min(numeric, 0.49)


def _angular_difference(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _inclusive_range(low: float, high: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("Search step must be positive")
    count_values = max(1, int(np.floor((high - low) / step + 1e-9)) + 1)
    values = low + np.arange(count_values, dtype=np.float64) * step
    if values[-1] < high - step * 0.25:
        values = np.append(values, high)
    return values


def _speed_consistency(history: tuple[float, ...], max_delta: float) -> float:
    if len(history) < 2:
        return 1.0
    changes = np.abs(np.diff(np.asarray(history[-4:], dtype=np.float64)))
    return float(np.exp(-np.mean(changes) / max(max_delta, 1e-6)))


def _heading_consistency(history: tuple[float, ...], turn_limit: float) -> float:
    if len(history) < 2:
        return 1.0
    recent = history[-4:]
    changes = [
        _angular_difference(a, b)
        for a, b in zip(recent[:-1], recent[1:], strict=True)
    ]
    return float(np.exp(-np.mean(changes) / max(turn_limit, 1e-6)))


class TerrainNavigator:
    """Estimate motion by matching measured terrain profiles against a DEM."""

    def __init__(
        self,
        dem_interpolator: DEMInterpolator,
        config: SearchConfig | None = None,
    ) -> None:
        self.dem = dem_interpolator
        self.config = config or SearchConfig(
            resolution_m=dem_interpolator.resolution_m,
            origin_x=dem_interpolator.origin_x,
            origin_y=dem_interpolator.origin_y,
        )
        self._ids = count(1)

    def localize(
        self,
        profile: GroundProfile,
        *,
        start_x: float,
        start_y: float,
        initial_speed_mps: float,
        initial_azimuth_deg: float | None = None,
    ) -> SearchResult:
        if initial_speed_mps <= 0:
            raise ValueError("initial_speed_mps must be positive")
        if len(profile.timestamps) < 4:
            raise ValueError("At least four profile samples are required")

        processed = smooth_moving_average(
            remove_outliers_hampel(profile.heights_m, window_size=7), window_size=3
        )
        windows = self._segments(profile.timestamps, profile.sample_rate_hz)
        if not windows:
            raise ValueError(
                "Profile does not contain a sufficiently complete time window"
            )

        start_height = self.dem.sample(start_x, start_y)
        root = TrajectoryHypothesis(
            x=float(start_x),
            y=float(start_y),
            azimuth_deg=float(initial_azimuth_deg or 0.0) % 360.0,
            ground_speed_mps=float(initial_speed_mps),
            score=0.0,
            cumulative_score=0.0,
            confidence=0.0,
            path_x=np.asarray([start_x], dtype=np.float64),
            path_y=np.asarray([start_y], dtype=np.float64),
            path_heights_dem=np.asarray([start_height], dtype=np.float64),
            hypothesis_id=0,
            depth=0,
            number_of_windows=0,
        )

        active = [root]
        initial_grid: ScoreGrid | None = None
        previous_confidence = "medium"
        final_measured = processed[windows[-1][0] : windows[-1][1]]
        previous_end = 1

        for window_index, (begin, end) in enumerate(windows):
            low_recovery = (
                previous_confidence == "low"
                and self.config.low_confidence_expand_search
                and window_index > 0
            )
            extension_start = max(0, previous_end - 1)
            extension_timestamps = profile.timestamps[extension_start:end]
            measured = processed[begin:end]
            candidates: list[TrajectoryHypothesis] = []
            desired_k = (
                self.config.recovery_top_k if low_recovery else self.config.top_k_hypotheses
            )

            for parent in active:
                generated, grid = self._expand_parent(
                    parent,
                    measured,
                    extension_timestamps,
                    window_start=begin,
                    previous_end=previous_end,
                    initial=window_index == 0,
                    initial_speed_mps=initial_speed_mps,
                    initial_azimuth_deg=initial_azimuth_deg,
                    recovery=low_recovery,
                    keep=max(self.config.rerank_candidates, desired_k * 5),
                )
                candidates.extend(generated)
                if initial_grid is None and grid is not None:
                    initial_grid = grid

            if not candidates:
                raise RuntimeError("No in-bounds trajectory hypotheses remain")
            rerank_interval = max(1, self.config.history_rerank_interval_windows)
            history_rerank_due = (
                (window_index + 1) % rerank_interval == 0
                or window_index == len(windows) - 1
            )
            if history_rerank_due:
                pool_size = max(
                    desired_k,
                    self.config.rerank_candidates,
                    desired_k * self.config.history_rerank_pool_multiplier,
                )
                pool = self._select_diverse(candidates, min(pool_size, len(candidates)))
                self._rerank_by_history(pool, processed[:end])
                active = self._select_diverse(pool, desired_k, use_ranking_score=True)
            else:
                active = self._select_diverse(candidates, desired_k)
            best_current = active[0].selection_score
            second_current = active[1].selection_score if len(active) > 1 else -1.0
            gap = best_current - second_current
            best = active[0]
            previous_confidence, numeric = confidence_level(
                measured,
                best.score,
                gap,
                speed_consistency=_speed_consistency(
                    best.speed_history, self.config.max_delta_v_mps_per_window
                ),
                heading_consistency=_heading_consistency(
                    best.heading_history, self.config.local_turn_limit_deg
                ),
            )
            for item in active:
                item.confidence = numeric
            final_measured = measured
            previous_end = end

        active.sort(key=lambda item: item.selection_score, reverse=True)
        best = active[0]
        best_score = best.selection_score
        second_score = active[1].selection_score if len(active) > 1 else -1.0
        gap = best_score - second_score
        confidence_profile = processed[:previous_end]
        label, numeric = confidence_level(
            confidence_profile,
            best.history_score if best.history_score is not None else best.score,
            gap,
            speed_consistency=_speed_consistency(
                best.speed_history, self.config.max_delta_v_mps_per_window
            ),
            heading_consistency=_heading_consistency(
                best.heading_history, self.config.local_turn_limit_deg
            ),
        )
        best.confidence = numeric
        return SearchResult(
            best=best,
            alternatives=active[1:],
            confidence=label,
            terrain_informativeness=terrain_informativeness(confidence_profile, gap),
            best_score=best_score,
            second_best_score=second_score,
            score_gap=gap,
            tree_depth=self.config.tree_depth,
            top_k_hypotheses=len(active),
            window_seconds=self.config.window_seconds,
            score_grid=initial_grid,
            measured_timestamps=profile.timestamps.copy(),
            measured_heights_m=processed.copy(),
        )

    def _segments(
        self,
        timestamps: np.ndarray,
        sample_rate_hz: float | None = None,
    ) -> list[tuple[int, int]]:
        """Create complete windows from timestamps, independent of sample rate."""

        del sample_rate_hz  # Kept for API compatibility; timestamps are authoritative.
        times = np.asarray(timestamps, dtype=np.float64)
        if times.ndim != 1 or times.size < self.config.min_window_samples:
            return []
        if np.any(np.diff(times) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        window_seconds = float(self.config.window_seconds)
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        overlap = float(self.config.window_overlap)
        if not 0.0 <= overlap < 1.0:
            raise ValueError("window_overlap must be in the range [0, 1)")
        if times[-1] - times[0] < window_seconds:
            return []

        hop_seconds = window_seconds * (1.0 - overlap)
        tolerance = max(1e-9, window_seconds * 1e-9)
        windows: list[tuple[int, int]] = []
        target_start = float(times[0])
        last_time = float(times[-1])
        while target_start + window_seconds <= last_time + tolerance:
            begin = int(np.searchsorted(times, target_start - tolerance, side="left"))
            end = int(
                np.searchsorted(
                    times, target_start + window_seconds + tolerance, side="right"
                )
            )
            if self._window_is_complete(times, begin, end, window_seconds):
                windows.append((begin, end))
            target_start += hop_seconds

        if windows and windows[-1][1] < times.size:
            begin = int(
                np.searchsorted(times, last_time - window_seconds - tolerance, side="left")
            )
            final_window = (begin, times.size)
            if (
                final_window != windows[-1]
                and self._window_is_complete(times, *final_window, window_seconds)
            ):
                windows.append(final_window)
        return windows

    def _window_is_complete(
        self,
        timestamps: np.ndarray,
        begin: int,
        end: int,
        requested_seconds: float,
    ) -> bool:
        if end - begin < self.config.min_window_samples:
            return False
        covered_seconds = float(timestamps[end - 1] - timestamps[begin])
        return covered_seconds >= requested_seconds * self.config.min_window_time_coverage

    def _expand_parent(
        self,
        parent: TrajectoryHypothesis,
        measured: np.ndarray,
        extension_timestamps: np.ndarray,
        *,
        window_start: int,
        previous_end: int,
        initial: bool,
        initial_speed_mps: float,
        initial_azimuth_deg: float | None,
        recovery: bool,
        keep: int,
    ) -> tuple[list[TrajectoryHypothesis], ScoreGrid | None]:
        if initial:
            if initial_azimuth_deg is None:
                headings = np.arange(
                    0.0, 360.0, self.config.initial_azimuth_step_deg, dtype=np.float64
                )
            else:
                headings = _inclusive_range(
                    initial_azimuth_deg - self.config.initial_azimuth_limit_deg,
                    initial_azimuth_deg + self.config.initial_azimuth_limit_deg,
                    self.config.initial_azimuth_step_deg,
                )
            speeds = _inclusive_range(
                max(0.5, initial_speed_mps * 0.7),
                initial_speed_mps * 1.3,
                self.config.speed_step_mps,
            )
            speeds = np.unique(np.append(speeds, initial_speed_mps))
        else:
            turn_limit = self.config.local_turn_limit_deg * (2.0 if recovery else 1.0)
            speed_delta = self.config.max_delta_v_mps_per_window * (
                self.config.recovery_speed_multiplier if recovery else 1.0
            )
            headings = _inclusive_range(
                parent.azimuth_deg - turn_limit,
                parent.azimuth_deg + turn_limit,
                self.config.local_azimuth_step_deg,
            )
            speeds = _inclusive_range(
                max(0.5, parent.ground_speed_mps - speed_delta),
                parent.ground_speed_mps + speed_delta,
                self.config.speed_step_mps,
            )

        speed_mesh, heading_mesh = np.meshgrid(speeds, headings, indexing="ij")
        flat_speeds = speed_mesh.ravel()
        flat_headings = np.mod(heading_mesh.ravel(), 360.0)
        elapsed = (
            np.asarray(extension_timestamps, dtype=np.float64)
            - extension_timestamps[0]
        )
        distance = flat_speeds[:, None] * elapsed[None, :]
        theta = np.deg2rad(flat_headings)
        x = parent.x + distance * np.sin(theta)[:, None]
        y = parent.y + distance * np.cos(theta)[:, None]
        extension_heights = self.dem.sample_batch(x.ravel(), y.ravel()).reshape(x.shape)
        if window_start < previous_end:
            history_prefix = parent.path_heights_dem[window_start:previous_end]
            extension_column = 1
        else:
            history_prefix = np.empty(0, dtype=np.float64)
            extension_column = window_start - (previous_end - 1)
        prefix_matrix = np.broadcast_to(
            history_prefix[None, :], (flat_speeds.size, history_prefix.size)
        )
        heights = np.concatenate(
            (prefix_matrix, extension_heights[:, extension_column:]), axis=1
        )
        if heights.shape[1] != measured.size:
            raise RuntimeError("Candidate profile is not aligned with the time window")
        raw_scores, valid_fraction = fast_scores_batch(measured, heights)

        if initial:
            if initial_azimuth_deg is None:
                turn_penalty = np.zeros_like(raw_scores)
            else:
                turn_penalty = 0.5 * self.config.turn_penalty_weight * np.asarray(
                    [
                        _angular_difference(value, initial_azimuth_deg)
                        for value in flat_headings
                    ]
                ) / max(self.config.turn_penalty_scale_deg, 1.0)
            speed_penalty = self.config.speed_change_penalty_weight * np.abs(
                flat_speeds - initial_speed_mps
            ) / max(initial_speed_mps * 0.3, 1.0)
        else:
            turn_limit = self.config.local_turn_limit_deg * (2.0 if recovery else 1.0)
            turn_penalty = self.config.turn_penalty_weight * np.asarray(
                [_angular_difference(value, parent.azimuth_deg) for value in flat_headings]
            ) / max(self.config.turn_penalty_scale_deg, 1.0)
            speed_delta = self.config.max_delta_v_mps_per_window * (
                self.config.recovery_speed_multiplier if recovery else 1.0
            )
            speed_penalty = self.config.speed_change_penalty_weight * np.abs(
                flat_speeds - parent.ground_speed_mps
            ) / max(speed_delta, 1.0)

        candidate_std = np.nanstd(heights, axis=1)
        variance_penalty = self.config.low_terrain_variance_penalty_weight * np.clip(
            (5.0 - candidate_std) / 5.0, 0.0, 1.0
        )
        bounds_penalty = self.config.out_of_bounds_penalty_weight * (1.0 - valid_fraction)
        adjusted = raw_scores - turn_penalty - speed_penalty - variance_penalty - bounds_penalty
        adjusted[valid_fraction < 0.80] = -np.inf

        finite = np.flatnonzero(np.isfinite(adjusted))
        if finite.size == 0:
            return [], None
        retain = finite[np.argsort(adjusted[finite])[::-1][:keep]]
        expensive_count = min(self.config.rerank_candidates, retain.size)
        expensive_indices = set(int(index) for index in retain[:expensive_count])
        for index in expensive_indices:
            reranked, _ = score_profiles(
                measured,
                heights[index],
                expensive=True,
                dtw_window=self.config.dtw_window,
            )
            adjusted[index] = (
                reranked
                - turn_penalty[index]
                - speed_penalty[index]
                - variance_penalty[index]
                - bounds_penalty[index]
            )
        retain = retain[np.argsort(adjusted[retain])[::-1]]

        generated: list[TrajectoryHypothesis] = []
        for index in retain:
            final_score, components = score_profiles(
                measured,
                heights[index],
                expensive=int(index) in expensive_indices,
                dtw_window=self.config.dtw_window,
            )
            penalty = (
                turn_penalty[index]
                + speed_penalty[index]
                + variance_penalty[index]
                + bounds_penalty[index]
            )
            current_score = float(final_score - penalty)
            candidate_x = x[index]
            candidate_y = y[index]
            candidate_heights = extension_heights[index]
            generated.append(
                TrajectoryHypothesis(
                    x=float(candidate_x[-1]),
                    y=float(candidate_y[-1]),
                    azimuth_deg=float(flat_headings[index]),
                    ground_speed_mps=float(flat_speeds[index]),
                    score=current_score,
                    cumulative_score=parent.cumulative_score + current_score,
                    confidence=parent.confidence,
                    path_x=np.concatenate((parent.path_x, candidate_x[1:])),
                    path_y=np.concatenate((parent.path_y, candidate_y[1:])),
                    path_heights_dem=np.concatenate(
                        (parent.path_heights_dem, candidate_heights[1:])
                    ),
                    parent_id=parent.hypothesis_id,
                    hypothesis_id=next(self._ids),
                    depth=min(parent.depth + 1, self.config.tree_depth),
                    number_of_windows=parent.number_of_windows + 1,
                    score_components=components,
                    history_score=None,
                    ranking_score=None,
                    score_history=parent.score_history + (current_score,),
                    speed_history=parent.speed_history + (float(flat_speeds[index]),),
                    heading_history=parent.heading_history + (float(flat_headings[index]),),
                )
            )

        grid = None
        if initial:
            grid = ScoreGrid(
                azimuths_deg=np.mod(headings, 360.0),
                speeds_mps=speeds,
                scores=adjusted.reshape(speeds.size, headings.size),
            )
        return generated, grid

    def _select_diverse(
        self,
        candidates: list[TrajectoryHypothesis],
        k: int,
        *,
        use_ranking_score: bool = False,
    ) -> list[TrajectoryHypothesis]:
        score_key = (
            (lambda item: item.selection_score)
            if use_ranking_score
            else (lambda item: item.mean_score)
        )
        ordered = sorted(candidates, key=score_key, reverse=True)
        selected: list[TrajectoryHypothesis] = []
        skipped: list[TrajectoryHypothesis] = []
        for item in ordered:
            duplicate = any(
                np.hypot(item.x - kept.x, item.y - kept.y) < self.dem.resolution_m
                and _angular_difference(item.azimuth_deg, kept.azimuth_deg)
                <= self.config.local_azimuth_step_deg
                and abs(item.ground_speed_mps - kept.ground_speed_mps)
                <= self.config.speed_step_mps
                for kept in selected
            )
            if duplicate:
                skipped.append(item)
                continue
            selected.append(item)
            if len(selected) == k:
                break
        if len(selected) < k:
            selected.extend(skipped[: k - len(selected)])
        selected.sort(key=score_key, reverse=True)
        return selected

    def _rerank_by_history(
        self,
        hypotheses: list[TrajectoryHypothesis],
        measured_history: np.ndarray,
    ) -> None:
        """Re-score a candidate pool against all measurements accumulated so far."""

        weight = float(np.clip(self.config.history_score_weight, 0.0, 1.0))
        for item in hypotheses:
            history_score, _ = score_profiles(
                measured_history,
                item.path_heights_dem,
                expensive=False,
                dtw_window=self.config.dtw_window,
            )
            item.history_score = history_score
            item.ranking_score = (1.0 - weight) * item.mean_score + weight * history_score

        hypotheses.sort(key=lambda item: item.selection_score, reverse=True)
        expensive_count = min(
            self.config.history_rerank_expensive_top, len(hypotheses)
        )
        for item in hypotheses[:expensive_count]:
            history_score, components = score_profiles(
                measured_history,
                item.path_heights_dem,
                expensive=True,
                dtw_window=self.config.dtw_window,
            )
            item.history_score = history_score
            item.ranking_score = (1.0 - weight) * item.mean_score + weight * history_score
            item.score_components = components
