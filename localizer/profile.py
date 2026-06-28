"""Radio-altimeter profile conversion and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GroundProfile:
    timestamps: np.ndarray
    heights_m: np.ndarray
    sample_rate_hz: float

    def __post_init__(self) -> None:
        if self.timestamps.ndim != 1 or self.heights_m.ndim != 1:
            raise ValueError("Profile arrays must be one-dimensional")
        if len(self.timestamps) != len(self.heights_m):
            raise ValueError("Profile timestamps and heights must have equal length")


def radio_to_ground_profile(
    timestamps: np.ndarray,
    radio_altitude_m: np.ndarray,
    baro_altitude_m: float | np.ndarray,
) -> GroundProfile:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    radio = np.asarray(radio_altitude_m, dtype=np.float64)
    if timestamps.ndim != 1 or radio.ndim != 1 or timestamps.size != radio.size:
        raise ValueError("Timestamp and radio-altitude arrays must have equal lengths")
    if timestamps.size < 2:
        raise ValueError("At least two valid radio-altimeter samples are required")
    baro = np.asarray(baro_altitude_m, dtype=np.float64)
    if baro.ndim > 1 or (baro.ndim == 1 and baro.size != radio.size):
        raise ValueError("Barometric altitude must be scalar or match sample count")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Sample timestamps must be strictly increasing")
    ground = baro - radio
    median_dt = float(np.median(np.diff(timestamps)))
    rate = 1.0 / median_dt if median_dt > 0 else 0.0
    return GroundProfile(timestamps, ground.astype(np.float64), rate)


def remove_outliers_hampel(
    profile: np.ndarray,
    window_size: int = 9,
    n_sigma: float = 2.0,
    passes: int = 2,
) -> np.ndarray:
    """Replace impulsive outliers using repeated local median/MAD checks."""

    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("profile must be one-dimensional")
    if window_size < 3:
        raise ValueError("window_size must be at least 3")
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive")
    if passes < 1:
        raise ValueError("passes must be at least 1")
    radius = int(window_size) // 2
    result = values.copy()
    for _ in range(int(passes)):
        source = result.copy()
        for index in range(source.size):
            window = source[
                max(0, index - radius) : min(source.size, index + radius + 1)
            ]
            median = float(np.nanmedian(window))
            mad = float(1.4826 * np.nanmedian(np.abs(window - median)))
            if np.isfinite(source[index]) and mad > 1e-9:
                if abs(source[index] - median) > n_sigma * mad:
                    result[index] = median
    return result


def smooth_moving_average(profile: np.ndarray, window_size: int = 3) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float64)
    if window_size <= 1 or values.size < 2:
        return values.copy()
    window_size = min(int(window_size), values.size)
    left = window_size // 2
    right = window_size - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window_size) / window_size, mode="valid")


def normalize_profile(profile: np.ndarray) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float64)
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(values)
    return (values - mean) / std


def z_normalize(profile: np.ndarray) -> np.ndarray:
    return normalize_profile(profile)


def compute_slope(profile: np.ndarray) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float64)
    return np.diff(values)


def compute_slope_profile(profile: np.ndarray) -> np.ndarray:
    return compute_slope(profile)


def remove_mean_bias(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("Profiles must have identical shapes")
    valid = np.isfinite(first) & np.isfinite(second)
    if not np.any(valid):
        return first.copy(), second.copy()
    bias = float(np.mean(second[valid] - first[valid]))
    return first.copy(), second - bias
