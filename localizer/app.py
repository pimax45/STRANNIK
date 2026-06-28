"""Standalone localization service with no dependency on test generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .config import SearchConfig, load_config
from .contracts import InitialState, RadioMeasurementSeries
from .dem import PreparedGeoTIFF, load_dem_grid, prepare_geotiff
from .profile import (
    GroundProfile,
    radio_to_ground_profile,
    remove_outliers_hampel,
    smooth_moving_average,
)
from .scoring import score_profiles
from .search import SearchResult, TerrainNavigator, build_dem_profile
from .visualization import save_result_plots


def _load_truth(path: str | Path | None) -> tuple[np.ndarray, np.ndarray] | None:
    if path is None:
        return None
    source = Path(path)
    if not source.exists():
        return None
    with np.load(source, allow_pickle=False) as values:
        return np.asarray(values["x"]), np.asarray(values["y"])


def _run_localization(
    grid,
    profile: GroundProfile,
    initial_state: InitialState,
    output_dir: str | Path,
    *,
    config: SearchConfig,
    truth_path: str | Path | None = None,
    preview_trajectory: bool = False,
    save_plots: bool = True,
) -> SearchResult:
    initial_state.validate()
    interpolator = grid.interpolator(config.interpolation)
    if not bool(interpolator.contains(initial_state.x, initial_state.y)):
        raise ValueError("Initial position is outside the DEM")
    result = TerrainNavigator(interpolator, config).localize(
        profile,
        start_x=initial_state.x,
        start_y=initial_state.y,
        initial_speed_mps=initial_state.ground_speed_mps,
        initial_azimuth_deg=initial_state.azimuth_deg,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "result.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if save_plots:
        save_result_plots(
            result,
            interpolator,
            destination,
            true_path=_load_truth(truth_path),
            preview_trajectory=preview_trajectory,
        )
    return result


def localize_from_inputs(
    dem_path: str | Path,
    measurements_path: str | Path,
    initial_state: InitialState,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    truth_path: str | Path | None = None,
    preview_trajectory: bool = False,
    save_plots: bool = True,
) -> SearchResult:
    """Run localization using the timestamped NPZ operational contract."""

    config = load_config(config_path)
    grid = load_dem_grid(
        dem_path,
        resolution_m=config.resolution_m,
        origin_x=config.origin_x,
        origin_y=config.origin_y,
    )
    measurements = RadioMeasurementSeries.load(measurements_path)
    profile = radio_to_ground_profile(
        measurements.timestamps,
        measurements.radio_altitude_m,
        measurements.baro_altitude_m,
    )
    return _run_localization(
        grid,
        profile,
        initial_state,
        output_dir,
        config=config,
        truth_path=truth_path,
        preview_trajectory=preview_trajectory,
        save_plots=save_plots,
    )


def read_heights_text(path: str | Path) -> np.ndarray:
    """Read one finite height in metres from each non-empty text line."""

    values: list[float] = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.partition("#")[0].strip()
        if not line:
            continue
        if "," in line and "." not in line:
            line = line.replace(",", ".")
        try:
            value = float(line)
        except ValueError as exc:
            raise ValueError(
                f"Invalid height at line {line_number}: {raw_line!r}"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(f"Non-finite height at line {line_number}")
        values.append(value)
    if len(values) < 4:
        raise ValueError("The height file must contain at least four values")
    return np.asarray(values, dtype=np.float64)


def _ground_heights_from_text(
    heights_m: np.ndarray,
    *,
    height_type: str,
    start_ground_height_m: float,
    baro_altitude_m: float | None,
) -> tuple[np.ndarray, float | None]:
    if height_type == "ground":
        return heights_m.copy(), baro_altitude_m
    if height_type != "radio":
        raise ValueError("height_type must be 'radio' or 'ground'")
    if np.any(heights_m < 0):
        raise ValueError("Radio-altimeter heights cannot be negative")
    effective_baro = (
        float(baro_altitude_m)
        if baro_altitude_m is not None
        else float(start_ground_height_m + heights_m[0])
    )
    return effective_baro - heights_m, effective_baro


def infer_constant_sample_rate(
    ground_heights_m: np.ndarray,
    interpolator,
    *,
    start_x: float,
    start_y: float,
    initial_azimuth_deg: float,
    initial_speed_mps: float,
    config: SearchConfig,
    minimum_hz: float = 1.0,
    maximum_hz: float = 10.0,
    step_hz: float = 0.25,
    prefix_samples: int = 31,
) -> tuple[float, float, list[dict[str, float]]]:
    """Infer one effective constant frequency from the initial terrain profile."""

    if not (0 < minimum_hz <= maximum_hz) or step_hz <= 0:
        raise ValueError("Invalid frequency-search range")
    count = min(max(4, int(prefix_samples)), len(ground_heights_m))
    measured = remove_outliers_hampel(
        np.asarray(ground_heights_m[:count], dtype=np.float64),
        window_size=config.hampel_window_size,
        n_sigma=config.hampel_n_sigma,
        passes=config.hampel_passes,
    )
    measured = smooth_moving_average(measured, window_size=3)
    rates = np.arange(
        minimum_hz,
        maximum_hz + 0.5 * step_hz,
        step_hz,
        dtype=np.float64,
    )
    rates = rates[rates <= maximum_hz + 1e-9]
    if rates.size == 0 or rates[-1] < maximum_hz - 1e-9:
        rates = np.append(rates, maximum_hz)

    scored: list[tuple[float, float]] = []
    for rate in rates:
        timestamps = np.arange(count, dtype=np.float64) / float(rate)
        candidate = build_dem_profile(
            interpolator,
            start_x,
            start_y,
            initial_azimuth_deg,
            initial_speed_mps,
            timestamps,
        )
        valid_fraction = float(np.mean(np.isfinite(candidate.heights_m)))
        if valid_fraction < 0.9:
            score = -1.0
        else:
            score, _ = score_profiles(
                measured,
                candidate.heights_m,
                expensive=False,
                absolute_height_sigma_m=config.absolute_height_sigma_m,
            )
        scored.append((float(rate), float(score)))
    best_rate, best_score = max(scored, key=lambda item: item[1])
    diagnostics = [
        {"sample_rate_hz": rate, "score": score} for rate, score in scored
    ]
    return best_rate, best_score, diagnostics


def _save_coordinate_outputs(
    prepared: PreparedGeoTIFF,
    result: SearchResult,
    profile: GroundProfile,
    output_dir: Path,
) -> dict[str, object]:
    count = min(
        len(result.best.path_x),
        len(result.best.path_y),
        len(profile.timestamps),
    )
    working_x = np.asarray(result.best.path_x[:count], dtype=np.float64)
    working_y = np.asarray(result.best.path_y[:count], dtype=np.float64)
    timestamps = np.asarray(profile.timestamps[:count], dtype=np.float64)
    source_x, source_y = prepared.to_source(working_x, working_y)
    longitude, latitude = prepared.to_wgs84(working_x, working_y)
    local_x = working_x - prepared.start_working_x
    local_y = working_y - prepared.start_working_y

    fieldnames = [
        "index",
        "timestamp_s",
        "local_x_m",
        "local_y_m",
        "working_x_m",
        "working_y_m",
        "source_x",
        "source_y",
        "longitude_deg",
        "latitude_deg",
    ]
    rows = [
        {
            "index": index,
            "timestamp_s": float(timestamps[index]),
            "local_x_m": float(local_x[index]),
            "local_y_m": float(local_y[index]),
            "working_x_m": float(working_x[index]),
            "working_y_m": float(working_y[index]),
            "source_x": float(source_x[index]),
            "source_y": float(source_y[index]),
            "longitude_deg": float(longitude[index]),
            "latitude_deg": float(latitude[index]),
        }
        for index in range(count)
    ]
    csv_path = output_dir / "trajectory_coordinates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "trajectory_coordinates.json"
    json_path.write_text(
        json.dumps(
            {
                "schema": "terrain-nav-trajectory/v1",
                "coordinate_reference_systems": {
                    "local": "metres east/north from initial point",
                    "working": prepared.working_crs,
                    "source": prepared.source_crs,
                    "geographic": "EPSG:4326",
                },
                "points": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "point_count": count,
        "source_crs": prepared.source_crs,
        "working_crs": prepared.working_crs,
        "current": rows[-1],
    }


def localize_text_heights(
    dem_path: str | Path,
    heights_path: str | Path,
    output_dir: str | Path,
    *,
    start_x: float,
    start_y: float,
    initial_azimuth_deg: float,
    initial_speed_mps: float,
    sample_rate_hz: float | None = None,
    height_type: str = "radio",
    baro_altitude_m: float | None = None,
    config_path: str | Path | None = None,
    map_radius_m: float | None = None,
    working_resolution_m: float | None = None,
    frequency_min_hz: float = 1.0,
    frequency_max_hz: float = 10.0,
    frequency_step_hz: float = 0.25,
    frequency_prefix_samples: int = 31,
    truth_path: str | Path | None = None,
    preview_trajectory: bool = False,
    save_plots: bool = True,
) -> tuple[SearchResult, dict[str, object]]:
    """Localize newline-separated heights directly against a GeoTIFF map."""

    if not np.all(
        np.isfinite([start_x, start_y, initial_azimuth_deg, initial_speed_mps])
    ):
        raise ValueError("Initial position, heading and speed must be finite")
    if not np.isfinite(initial_speed_mps) or initial_speed_mps <= 0:
        raise ValueError("initial_speed_mps must be positive")
    if baro_altitude_m is not None and not np.isfinite(baro_altitude_m):
        raise ValueError("baro_altitude_m must be finite")
    if sample_rate_hz is not None and (
        not np.isfinite(sample_rate_hz) or not 1.0 <= sample_rate_hz <= 10.0
    ):
        raise ValueError("sample_rate_hz must be in the range 1..10 Hz")
    config = load_config(config_path)
    resolution = (
        config.resolution_m
        if working_resolution_m is None
        else float(working_resolution_m)
    )
    prepared = prepare_geotiff(
        dem_path,
        start_x=float(start_x),
        start_y=float(start_y),
        resolution_m=resolution,
        map_radius_m=(None if map_radius_m is None else float(map_radius_m)),
    )
    interpolator = prepared.grid.interpolator(config.interpolation)
    heights = read_heights_text(heights_path)
    ground, effective_baro = _ground_heights_from_text(
        heights,
        height_type=height_type,
        start_ground_height_m=interpolator.sample(
            prepared.start_working_x, prepared.start_working_y
        ),
        baro_altitude_m=baro_altitude_m,
    )

    frequency_score: float | None = None
    frequency_diagnostics: list[dict[str, float]] = []
    frequency_source = "provided"
    if sample_rate_hz is None:
        sample_rate_hz, frequency_score, frequency_diagnostics = infer_constant_sample_rate(
            ground,
            interpolator,
            start_x=prepared.start_working_x,
            start_y=prepared.start_working_y,
            initial_azimuth_deg=float(initial_azimuth_deg),
            initial_speed_mps=float(initial_speed_mps),
            config=config,
            minimum_hz=frequency_min_hz,
            maximum_hz=frequency_max_hz,
            step_hz=frequency_step_hz,
            prefix_samples=frequency_prefix_samples,
        )
        frequency_source = "inferred_effective_constant"

    timestamps = np.arange(len(ground), dtype=np.float64) / float(sample_rate_hz)
    profile = GroundProfile(timestamps, ground, float(sample_rate_hz))
    theta = np.deg2rad(float(initial_azimuth_deg))
    state = InitialState(
        prepared.start_working_x,
        prepared.start_working_y,
        float(initial_speed_mps * np.sin(theta)),
        float(initial_speed_mps * np.cos(theta)),
    )
    destination = Path(output_dir)
    result = _run_localization(
        prepared.grid,
        profile,
        state,
        destination,
        config=config,
        truth_path=truth_path,
        preview_trajectory=preview_trajectory,
        save_plots=save_plots,
    )
    coordinate_outputs = _save_coordinate_outputs(
        prepared, result, profile, destination
    )
    metadata: dict[str, object] = {
        "height_type": height_type,
        "height_count": int(len(heights)),
        "sample_rate_hz": float(sample_rate_hz),
        "sample_rate_source": frequency_source,
        "frequency_inference_score": frequency_score,
        "frequency_candidates": frequency_diagnostics,
        "baro_altitude_m": effective_baro,
        "baro_altitude_source": (
            "not_required"
            if height_type == "ground"
            else "provided"
            if baro_altitude_m is not None
            else "inferred_from_dem_at_start"
        ),
        "input_start": {"x": float(start_x), "y": float(start_y)},
        "input_start_crs": prepared.source_crs,
        "coordinates": coordinate_outputs,
    }
    result_payload = result.to_dict()
    result_payload["input"] = metadata
    (destination / "result.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result, metadata


def result_summary(result: SearchResult, result_path: str | Path) -> dict[str, object]:
    return {
        "result": str(result_path),
        "current_x": result.best.x,
        "current_y": result.best.y,
        "azimuth_deg": result.best.azimuth_deg,
        "ground_speed_mps": result.best.ground_speed_mps,
        "confidence": result.confidence,
    }
