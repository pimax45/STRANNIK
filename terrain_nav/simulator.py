"""Synthetic DEM, flight path and NMEA generation for repeatable tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .config import SearchConfig
from .dem import DEMInterpolator
from .nmea import nmea_checksum


@dataclass(frozen=True, slots=True)
class TruePath:
    timestamps: np.ndarray
    x: np.ndarray
    y: np.ndarray
    azimuth_deg: np.ndarray
    ground_speed_mps: np.ndarray


def _gaussian(
    x: np.ndarray,
    y: np.ndarray,
    cx: float,
    cy: float,
    width: float,
    amplitude: float,
) -> np.ndarray:
    return amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * width**2))


def generate_synthetic_dem(
    terrain: str = "mixed",
    *,
    shape: tuple[int, int] = (256, 256),
    resolution_m: float = 30.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate deterministic terrain in metres above sea level."""

    del resolution_m  # Shape functions operate in normalized map coordinates.
    rows, cols = shape
    yy, xx = np.mgrid[0:1:complex(rows), 0:1:complex(cols)]
    rng = np.random.default_rng(seed)
    terrain = terrain.lower()
    if terrain == "flat":
        dem = 520.0 + 1.2 * np.sin(2 * np.pi * xx) + 0.8 * np.sin(2 * np.pi * yy)
        dem += rng.normal(0.0, 0.25, shape)
    elif terrain == "hilly":
        dem = (
            560.0
            + 65.0 * np.sin(4 * np.pi * xx + 0.4) * np.cos(3 * np.pi * yy)
            + 35.0 * np.sin(7 * np.pi * (xx + 0.35 * yy))
            + _gaussian(xx, yy, 0.68, 0.36, 0.12, 90.0)
        )
    elif terrain == "mountainous":
        dem = (
            620.0
            + 130.0 * np.sin(5 * np.pi * xx) * np.cos(4 * np.pi * yy)
            + 85.0 * np.sin(11 * np.pi * (xx + 0.2 * yy))
            + _gaussian(xx, yy, 0.28, 0.66, 0.08, 260.0)
            + _gaussian(xx, yy, 0.75, 0.25, 0.11, 190.0)
        )
    elif terrain == "valley":
        centerline = 0.50 + 0.10 * np.sin(2.5 * np.pi * yy)
        dem = (
            490.0
            + 500.0 * (xx - centerline) ** 2
            + 45.0 * np.sin(6 * np.pi * yy + 2 * xx)
            + _gaussian(xx, yy, 0.72, 0.72, 0.10, 110.0)
        )
    elif terrain == "mixed":
        dem = (
            570.0
            + 75.0 * np.sin(4.3 * np.pi * xx + 0.7) * np.cos(3.7 * np.pi * yy)
            + 48.0 * np.sin(8.1 * np.pi * (xx + 0.31 * yy))
            + 27.0 * np.cos(13.0 * np.pi * (yy - 0.17 * xx))
            + _gaussian(xx, yy, 0.24, 0.72, 0.075, 165.0)
            - _gaussian(xx, yy, 0.62, 0.58, 0.10, 115.0)
            + _gaussian(xx, yy, 0.79, 0.27, 0.12, 135.0)
        )
    else:
        raise ValueError(
            "terrain must be one of: flat, hilly, mountainous, valley, mixed"
        )
    return np.asarray(dem, dtype=np.float32)


def generate_true_path(
    *,
    duration_s: float = 60.0,
    sample_rate_hz: float = 10.0,
    start_x: float = 1000.0,
    start_y: float = 1000.0,
    initial_speed_mps: float = 50.0,
    azimuth_deg: float = 73.0,
    motion: str = "straight",
) -> TruePath:
    if duration_s <= 0 or sample_rate_hz <= 0 or initial_speed_mps <= 0:
        raise ValueError("duration, sample rate and initial speed must be positive")
    count = int(round(duration_s * sample_rate_hz)) + 1
    timestamps = np.arange(count, dtype=np.float64) / sample_rate_hz
    progress = timestamps / max(duration_s, 1e-9)
    motion = motion.lower()
    sharp_headings = np.where(
        progress < 1.0 / 3.0,
        azimuth_deg,
        np.where(progress < 2.0 / 3.0, azimuth_deg + 15.0, azimuth_deg + 30.0),
    )
    sharp_speeds = np.where(
        progress < 0.35,
        initial_speed_mps,
        np.where(progress < 0.70, initial_speed_mps * 1.30, initial_speed_mps * 0.80),
    )
    if motion == "straight":
        headings = np.full(count, azimuth_deg, dtype=np.float64)
        speeds = np.full(count, initial_speed_mps, dtype=np.float64)
    elif motion == "turn":
        headings = azimuth_deg + 24.0 * progress
        speeds = np.full(count, initial_speed_mps, dtype=np.float64)
    elif motion == "yaw":
        headings = azimuth_deg + 4.0 * np.sin(2.0 * np.pi * timestamps / 12.0)
        speeds = np.full(count, initial_speed_mps, dtype=np.float64)
    elif motion == "speed-change":
        headings = np.full(count, azimuth_deg, dtype=np.float64)
        speeds = initial_speed_mps * (1.0 + 0.18 * progress)
    elif motion == "mixed":
        headings = azimuth_deg + 14.0 * progress + 2.0 * np.sin(2 * np.pi * progress)
        speeds = initial_speed_mps * (1.0 + 0.12 * progress)
    elif motion == "sharp-turn":
        headings = sharp_headings
        speeds = np.full(count, initial_speed_mps, dtype=np.float64)
    elif motion == "sharp-speed-change":
        headings = np.full(count, azimuth_deg, dtype=np.float64)
        speeds = sharp_speeds
    elif motion == "sharp-mixed":
        headings = sharp_headings
        speeds = sharp_speeds
    else:
        raise ValueError(
            "motion must be one of: straight, turn, yaw, speed-change, mixed, "
            "sharp-turn, sharp-speed-change, sharp-mixed"
        )

    dt = np.diff(timestamps, prepend=timestamps[0])
    theta = np.deg2rad(headings)
    dx = speeds * dt * np.sin(theta)
    dy = speeds * dt * np.cos(theta)
    x = start_x + np.cumsum(dx)
    y = start_y + np.cumsum(dy)
    x[0], y[0] = start_x, start_y
    return TruePath(timestamps, x, y, np.mod(headings, 360.0), speeds)


def _format_nmea_time(seconds: float) -> str:
    seconds %= 24 * 3600
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - hours * 3600 - minutes * 60
    return f"{hours:02d}{minutes:02d}{secs:06.3f}"


def generate_nmea_from_path(
    path: TruePath,
    dem_interpolator: DEMInterpolator,
    *,
    baro_altitude_m: float = 1500.0,
    noise_std_m: float = 0.8,
    outlier_probability: float = 0.002,
    seed: int = 42,
) -> list[str]:
    if noise_std_m < 0:
        raise ValueError("noise_std_m must be non-negative")
    if not 0.0 <= outlier_probability <= 1.0:
        raise ValueError("outlier_probability must be between 0 and 1")
    ground = dem_interpolator.sample_batch(path.x, path.y).astype(np.float64)
    if np.any(~np.isfinite(ground)):
        raise ValueError("The generated path leaves the DEM bounds")
    rng = np.random.default_rng(seed)
    radio = baro_altitude_m - ground + rng.normal(0.0, noise_std_m, ground.size)
    outliers = rng.random(ground.size) < outlier_probability
    radio[outliers] += rng.normal(0.0, 25.0, np.count_nonzero(outliers))
    lines: list[str] = []
    for timestamp, altitude in zip(path.timestamps, radio, strict=True):
        fields = [
            "GPGGA",
            _format_nmea_time(timestamp),
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            f"{altitude:.3f}",
            "M",
            "",
            "M",
            "",
            "",
        ]
        body = ",".join(fields)
        lines.append(f"${body}*{nmea_checksum(body):02X}")
    return lines


def create_test_dataset(
    output_dir: str | Path,
    *,
    terrain: str = "mixed",
    baro_altitude_m: float = 1500.0,
    sample_rate_hz: float = 10.0,
    duration_s: float = 60.0,
    start_x: float = 1000.0,
    start_y: float = 1000.0,
    initial_speed_mps: float = 50.0,
    azimuth_deg: float = 73.0,
    motion: str = "straight",
    resolution_m: float = 30.0,
    radio_noise_std_m: float = 0.8,
    radio_outlier_probability: float = 0.002,
    seed: int = 42,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dem = generate_synthetic_dem(terrain, resolution_m=resolution_m, seed=seed)
    interpolator = DEMInterpolator(dem, resolution_m)
    path = generate_true_path(
        duration_s=duration_s,
        sample_rate_hz=sample_rate_hz,
        start_x=start_x,
        start_y=start_y,
        initial_speed_mps=initial_speed_mps,
        azimuth_deg=azimuth_deg,
        motion=motion,
    )
    lines = generate_nmea_from_path(
        path,
        interpolator,
        baro_altitude_m=baro_altitude_m,
        noise_std_m=radio_noise_std_m,
        outlier_probability=radio_outlier_probability,
        seed=seed,
    )

    dem_path = destination / "dem.npy"
    nmea_path = destination / "radio_altimeter.nmea"
    truth_path = destination / "true_path.npz"
    config_path = destination / "config.yaml"
    np.save(dem_path, dem)
    nmea_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    np.savez(
        truth_path,
        timestamps=path.timestamps,
        x=path.x,
        y=path.y,
        azimuth_deg=path.azimuth_deg,
        ground_speed_mps=path.ground_speed_mps,
    )
    config = SearchConfig(resolution_m=resolution_m)
    payload = {
        "dem": {
            "resolution_m": resolution_m,
            "origin_x": 0.0,
            "origin_y": 0.0,
            "interpolation": "bilinear",
        },
        "search": {
            key: value
            for key, value in config.to_dict().items()
            if key not in {"resolution_m", "origin_x", "origin_y", "interpolation"}
        },
        "simulation": {
            "terrain": terrain,
            "motion": motion,
            "baro_altitude_m": baro_altitude_m,
            "sample_rate_hz": sample_rate_hz,
            "duration_s": duration_s,
            "start_x": float(path.x[0]),
            "start_y": float(path.y[0]),
            "initial_speed_mps": initial_speed_mps,
            "initial_azimuth_deg": azimuth_deg,
            "radio_noise_std_m": radio_noise_std_m,
            "radio_outlier_probability": radio_outlier_probability,
            "seed": seed,
        },
    }
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return {
        "dem": dem_path,
        "nmea": nmea_path,
        "truth": truth_path,
        "config": config_path,
    }
