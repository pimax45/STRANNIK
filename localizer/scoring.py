"""Profile similarity metrics and fast vectorized candidate scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .profile import compute_slope, normalize_profile, remove_mean_bias


@dataclass(frozen=True, slots=True)
class ScoreComponents:
    pearson_height: float
    pearson_slope: float
    spearman: float
    bias_corrected_rmse_score: float
    absolute_height_likelihood: float = 0.0
    nmi: float = 0.0
    dtw: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "pearson_height": self.pearson_height,
            "pearson_slope": self.pearson_slope,
            "spearman": self.spearman,
            "bias_corrected_rmse_score": self.bias_corrected_rmse_score,
            "absolute_height_likelihood": self.absolute_height_likelihood,
            "nmi": self.nmi,
            "dtw": self.dtw,
        }


def _paired(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(a, dtype=np.float64).ravel()
    second = np.asarray(b, dtype=np.float64).ravel()
    if first.shape != second.shape:
        raise ValueError("Profiles must have equal lengths")
    valid = np.isfinite(first) & np.isfinite(second)
    return first[valid], second[valid]


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    first, second = _paired(a, b)
    if first.size < 2 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return 0.0
    return float(np.clip(np.corrcoef(first, second)[0, 1], -1.0, 1.0))


def slope_corr(a: np.ndarray, b: np.ndarray) -> float:
    first, second = _paired(a, b)
    if first.size < 3:
        return 0.0
    return pearson_corr(compute_slope(first), compute_slope(second))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    first, second = _paired(a, b)
    if first.size < 2:
        return 0.0
    return pearson_corr(_rankdata(first), _rankdata(second))


def bias_corrected_rmse_score(a: np.ndarray, b: np.ndarray) -> float:
    first, second = _paired(a, b)
    if first.size < 2:
        return 0.0
    corrected_first, corrected_second = remove_mean_bias(first, second)
    rmse = float(np.sqrt(np.mean((corrected_first - corrected_second) ** 2)))
    scale = max(float(np.std(first)), float(np.std(second)), 1.0)
    return float(1.0 / (1.0 + rmse / scale))


def absolute_height_likelihood(
    measured: np.ndarray,
    candidate: np.ndarray,
    sigma_m: float = 30.0,
) -> float:
    """Gaussian likelihood-like score for absolute vertical agreement.

    Unlike normalized correlations and bias-corrected RMSE, this component
    retains the mean vertical difference between the measured ground profile
    and DEM. ``sigma_m`` represents the expected combined vertical error of
    the barometer, radio altimeter and DEM.
    """

    if not np.isfinite(sigma_m) or sigma_m <= 0:
        raise ValueError("sigma_m must be positive")
    first, second = _paired(measured, candidate)
    if first.size < 2:
        return 0.0
    rmse = float(np.sqrt(np.mean((first - second) ** 2)))
    return float(np.exp(-0.5 * (rmse / sigma_m) ** 2))


def normalized_mutual_information(
    a: np.ndarray,
    b: np.ndarray,
    bins: int = 32,
) -> float:
    first, second = _paired(a, b)
    if first.size < 3 or np.ptp(first) < 1e-12 or np.ptp(second) < 1e-12:
        return 0.0
    effective_bins = max(2, min(int(bins), int(np.sqrt(first.size)) + 1))
    histogram, _, _ = np.histogram2d(first, second, bins=effective_bins)
    probability = histogram / np.sum(histogram)
    px = probability.sum(axis=1)
    py = probability.sum(axis=0)
    expected = px[:, None] * py[None, :]
    nonzero = probability > 0
    mutual = float(np.sum(probability[nonzero] * np.log(probability[nonzero] / expected[nonzero])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    denominator = max(hx + hy, 1e-12)
    return float(np.clip(2.0 * mutual / denominator, 0.0, 1.0))


def dtw_similarity(a: np.ndarray, b: np.ndarray, window: int = 10) -> float:
    first, second = _paired(a, b)
    if first.size < 2:
        return 0.0
    first = normalize_profile(first)
    second = normalize_profile(second)
    n, m = first.size, second.size
    radius = max(abs(n - m), int(window))
    previous = np.full(m + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = np.full(m + 1, np.inf, dtype=np.float64)
        for j in range(max(1, i - radius), min(m, i + radius) + 1):
            cost = abs(first[i - 1] - second[j - 1])
            current[j] = cost + min(current[j - 1], previous[j], previous[j - 1])
        previous = current
    distance = previous[m] / max(n, m)
    return float(np.exp(-distance))


def score_profiles(
    measured: np.ndarray,
    candidate: np.ndarray,
    *,
    expensive: bool = False,
    dtw_window: int = 10,
    absolute_height_sigma_m: float = 30.0,
) -> tuple[float, ScoreComponents]:
    pearson_height = pearson_corr(normalize_profile(measured), normalize_profile(candidate))
    pearson_slopes = slope_corr(measured, candidate)
    spearman = spearman_corr(measured, candidate)
    rmse_score = bias_corrected_rmse_score(measured, candidate)
    absolute_likelihood = absolute_height_likelihood(
        measured,
        candidate,
        absolute_height_sigma_m,
    )
    if not expensive:
        components = ScoreComponents(
            pearson_height,
            pearson_slopes,
            spearman,
            rmse_score,
            absolute_likelihood,
        )
        score = (
            0.23 * pearson_height
            + 0.24 * pearson_slopes
            + 0.15 * spearman
            + 0.10 * rmse_score
            + 0.28 * absolute_likelihood
        )
        return float(score), components
    nmi = normalized_mutual_information(measured, candidate)
    dtw = dtw_similarity(measured, candidate, dtw_window)
    components = ScoreComponents(
        pearson_height,
        pearson_slopes,
        spearman,
        rmse_score,
        absolute_likelihood,
        nmi,
        dtw,
    )
    score = (
        0.20 * pearson_height
        + 0.15 * pearson_slopes
        + 0.12 * spearman
        + 0.08 * rmse_score
        + 0.3 * absolute_likelihood
        + 0.10 * nmi
        + 0.05 * dtw
    )
    return float(score), components


def fast_scores_batch(
    measured: np.ndarray,
    candidates: np.ndarray,
    *,
    absolute_height_sigma_m: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cheap scores for an MxN matrix of candidate profiles."""

    if not np.isfinite(absolute_height_sigma_m) or absolute_height_sigma_m <= 0:
        raise ValueError("absolute_height_sigma_m must be positive")
    target = np.asarray(measured, dtype=np.float64).ravel()
    matrix = np.asarray(candidates, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != target.size:
        raise ValueError("candidates must have shape (count, len(measured))")
    valid = np.isfinite(matrix) & np.isfinite(target)[None, :]
    valid_fraction = valid.mean(axis=1)
    safe = matrix.copy()
    row_counts = np.maximum(valid.sum(axis=1), 1)
    row_means = np.sum(np.where(valid, safe, 0.0), axis=1) / row_counts
    safe[~valid] = np.broadcast_to(row_means[:, None], safe.shape)[~valid]

    target_safe = np.where(np.isfinite(target), target, np.nanmean(target))
    target_norm = normalize_profile(target_safe)
    centered = safe - safe.mean(axis=1, keepdims=True)
    std = safe.std(axis=1, keepdims=True)
    normalized = np.divide(centered, std, out=np.zeros_like(centered), where=std > 1e-12)
    pearson_height = np.mean(normalized * target_norm[None, :], axis=1)

    target_slope = np.diff(target_safe)
    target_slope_norm = normalize_profile(target_slope)
    slopes = np.diff(safe, axis=1)
    slope_centered = slopes - slopes.mean(axis=1, keepdims=True)
    slope_std = slopes.std(axis=1, keepdims=True)
    slope_norm = np.divide(
        slope_centered,
        slope_std,
        out=np.zeros_like(slope_centered),
        where=slope_std > 1e-12,
    )
    pearson_slope = np.mean(slope_norm * target_slope_norm[None, :], axis=1)

    target_rank = np.argsort(np.argsort(target_safe, kind="mergesort"), kind="mergesort")
    target_rank_norm = normalize_profile(target_rank.astype(np.float64))
    candidate_rank = np.argsort(
        np.argsort(safe, axis=1, kind="mergesort"), axis=1, kind="mergesort"
    ).astype(np.float64)
    candidate_rank -= candidate_rank.mean(axis=1, keepdims=True)
    rank_std = candidate_rank.std(axis=1, keepdims=True)
    candidate_rank = np.divide(
        candidate_rank,
        rank_std,
        out=np.zeros_like(candidate_rank),
        where=rank_std > 1e-12,
    )
    spearman = np.mean(candidate_rank * target_rank_norm[None, :], axis=1)

    bias = (safe - target_safe[None, :]).mean(axis=1, keepdims=True)
    rmse = np.sqrt(np.mean((safe - bias - target_safe[None, :]) ** 2, axis=1))
    scale = np.maximum(np.maximum(safe.std(axis=1), np.std(target_safe)), 1.0)
    rmse_score = 1.0 / (1.0 + rmse / scale)

    absolute_residual = np.where(valid, matrix - target[None, :], 0.0)
    absolute_rmse = np.sqrt(
        np.sum(absolute_residual**2, axis=1) / row_counts
    )
    absolute_likelihood = np.exp(
        -0.5 * (absolute_rmse / absolute_height_sigma_m) ** 2
    )

    score = (
        0.27 * pearson_height
        + 0.28 * pearson_slope
        + 0.15 * spearman
        + 0.10 * rmse_score
        + 0.20 * absolute_likelihood
    )
    score = np.where(valid_fraction >= 0.90, score, -1.0)
    return score.astype(np.float64), valid_fraction.astype(np.float64)
