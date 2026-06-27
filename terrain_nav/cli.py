"""Command-line interface for dataset generation and localization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .dem import DEMInterpolator, load_dem
from .nmea import read_nmea
from .profile import radio_to_ground_profile
from .search import TerrainNavigator
from .simulator import create_test_dataset
from .visualization import save_result_plots


def _non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("значение должно быть неотрицательным")
    return number


def _probability(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("вероятность должна находиться в диапазоне 0..1")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="terrain-nav",
        description="Навигационная привязка БПЛА по радиовысотомеру и DEM",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-test", help="Создать синтетический набор")
    generate.add_argument(
        "--terrain",
        choices=["mixed", "mountainous", "hilly", "flat", "valley"],
        default="mixed",
    )
    generate.add_argument(
        "--motion",
        choices=[
            "straight",
            "turn",
            "yaw",
            "speed-change",
            "mixed",
            "sharp-turn",
            "sharp-speed-change",
            "sharp-mixed",
        ],
        default="straight",
    )
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--baro-altitude", type=float, default=1500.0)
    generate.add_argument("--sample-rate", type=float, default=10.0)
    generate.add_argument("--duration", type=float, default=60.0)
    generate.add_argument("--start-x", type=float, default=1000.0)
    generate.add_argument("--start-y", type=float, default=1000.0)
    generate.add_argument("--initial-speed", type=float, default=50.0)
    generate.add_argument("--azimuth", type=float, default=73.0)
    generate.add_argument("--resolution", type=float, default=30.0)
    generate.add_argument(
        "--radio-noise-std",
        type=_non_negative_float,
        default=0.8,
        metavar="METERS",
        help="Стандартное отклонение гауссова шума радиовысотомера, м",
    )
    generate.add_argument(
        "--radio-outlier-probability",
        type=_probability,
        default=0.002,
        metavar="PROBABILITY",
        help="Вероятность импульсного выброса в одном измерении, 0..1",
    )
    generate.add_argument("--seed", type=int, default=42)

    localize = subparsers.add_parser("localize", help="Оценить положение и движение")
    localize.add_argument("--dem", type=Path, required=True)
    localize.add_argument("--config", type=Path)
    localize.add_argument("--nmea", type=Path, required=True)
    localize.add_argument("--start-x", type=float, required=True)
    localize.add_argument("--start-y", type=float, required=True)
    localize.add_argument("--initial-speed", type=float, required=True)
    localize.add_argument("--initial-azimuth", type=float)
    localize.add_argument("--baro-altitude", type=float, required=True)
    localize.add_argument("--output-dir", type=Path, required=True)
    localize.add_argument("--verify-checksum", action="store_true")
    localize.add_argument(
        "--preview-trajectory",
        action="store_true",
        help="Показать trajectory.png в окне перед сохранением",
    )
    return parser


def _generate(args: argparse.Namespace) -> int:
    paths = create_test_dataset(
        args.output_dir,
        terrain=args.terrain,
        motion=args.motion,
        baro_altitude_m=args.baro_altitude,
        sample_rate_hz=args.sample_rate,
        duration_s=args.duration,
        start_x=args.start_x,
        start_y=args.start_y,
        initial_speed_mps=args.initial_speed,
        azimuth_deg=args.azimuth,
        resolution_m=args.resolution,
        radio_noise_std_m=args.radio_noise_std,
        radio_outlier_probability=args.radio_outlier_probability,
        seed=args.seed,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))
    return 0


def _load_true_path(nmea_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    truth = nmea_path.with_name("true_path.npz")
    if not truth.exists():
        return None
    values = np.load(truth)
    return np.asarray(values["x"]), np.asarray(values["y"])


def _localize(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    dem = load_dem(args.dem)
    interpolator = DEMInterpolator(
        dem,
        config.resolution_m,
        config.origin_x,
        config.origin_y,
        config.interpolation,
    )
    samples = read_nmea(args.nmea, verify_checksum=args.verify_checksum)
    profile = radio_to_ground_profile(samples, args.baro_altitude)
    result = TerrainNavigator(interpolator, config).localize(
        profile,
        start_x=args.start_x,
        start_y=args.start_y,
        initial_speed_mps=args.initial_speed,
        initial_azimuth_deg=args.initial_azimuth,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_result_plots(
        result,
        interpolator,
        args.output_dir,
        true_path=_load_true_path(args.nmea),
        preview_trajectory=args.preview_trajectory,
    )
    summary = {
        "result": str(result_path),
        "current_x": result.best.x,
        "current_y": result.best.y,
        "azimuth_deg": result.best.azimuth_deg,
        "ground_speed_mps": result.best.ground_speed_mps,
        "confidence": result.confidence,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate-test":
        return _generate(args)
    if args.command == "localize":
        return _localize(args)
    raise AssertionError("Unknown command")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
