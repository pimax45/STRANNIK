"""CLI for the standalone localization program."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from localizer.app import (
        localize_from_inputs,
        localize_text_heights,
        result_summary,
    )
    from localizer.contracts import InitialState, ScenarioManifest
else:
    from .app import localize_from_inputs, localize_text_heights, result_summary
    from .contracts import InitialState, ScenarioManifest


def build_parser(prog: str = "terrain-nav-localizer") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Локализация по DEM, массиву измерений и начальному вектору скорости",
    )
    parser.add_argument("--scenario", type=Path, help="scenario.json от генератора")
    parser.add_argument(
        "--input-contract",
        type=Path,
        help="localizer_input.json: GeoTIFF, heights.txt и начальные параметры",
    )
    parser.add_argument("--dem", type=Path)
    parser.add_argument("--measurements", type=Path, help="radio_samples.npz")
    parser.add_argument(
        "--heights-text",
        type=Path,
        help="Текстовый файл: одна высота в метрах на строку",
    )
    parser.add_argument(
        "--height-type",
        choices=("radio", "ground"),
        default="radio",
        help="radio — дальность до земли; ground — абсолютная высота рельефа",
    )
    parser.add_argument("--initial-state", type=Path, help="initial_state.json")
    parser.add_argument("--start-x", type=float)
    parser.add_argument("--start-y", type=float)
    parser.add_argument("--initial-vx", type=float)
    parser.add_argument("--initial-vy", type=float)
    parser.add_argument("--heading-deg", type=float, help="Азимут от севера по часовой стрелке")
    parser.add_argument("--initial-speed", type=float, help="Начальная скорость, м/с")
    parser.add_argument(
        "--sample-rate",
        type=float,
        help="Постоянная частота, Гц; если не задана, оценивается в диапазоне 1–10 Гц",
    )
    parser.add_argument("--baro-altitude", type=float, help="Высота БПЛА над уровнем моря, м")
    parser.add_argument("--frequency-min", type=float, default=1.0)
    parser.add_argument("--frequency-max", type=float, default=10.0)
    parser.add_argument("--frequency-step", type=float, default=0.25)
    parser.add_argument("--frequency-prefix-samples", type=int, default=31)
    parser.add_argument(
        "--map-radius",
        type=float,
        help="Радиус вырезки, м; без флага загружается весь GeoTIFF",
    )
    parser.add_argument("--working-resolution", type=float)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--truth", type=Path, help="Необязательная истинная траектория для графика")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview-trajectory", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _resolve_inputs(args: argparse.Namespace):
    if args.scenario is not None:
        explicit = [
            args.dem,
            args.measurements,
            args.initial_state,
            args.start_x,
            args.start_y,
            args.initial_vx,
            args.initial_vy,
        ]
        if any(value is not None for value in explicit):
            raise ValueError("--scenario cannot be combined with explicit input-state flags")
        manifest = ScenarioManifest.load(args.scenario)
        return (
            manifest.dem_path,
            manifest.measurements_path,
            InitialState.load(manifest.initial_state_path),
            args.config or manifest.config_path,
            args.truth or manifest.truth_path,
        )

    if args.dem is None or args.measurements is None:
        raise ValueError("--dem and --measurements are required without --scenario")
    if args.initial_state is not None:
        if any(
            value is not None
            for value in (args.start_x, args.start_y, args.initial_vx, args.initial_vy)
        ):
            raise ValueError("--initial-state cannot be combined with individual state flags")
        state = InitialState.load(args.initial_state)
    else:
        values = (args.start_x, args.start_y, args.initial_vx, args.initial_vy)
        if any(value is None for value in values):
            raise ValueError(
                "Supply --initial-state or all of --start-x, --start-y, "
                "--initial-vx and --initial-vy"
            )
        state = InitialState(*values)
    return args.dem, args.measurements, state, args.config, args.truth


def _contract_path(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else base / candidate


def _load_text_contract(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "terrain-nav-text-input/v1":
        raise ValueError("Unsupported localizer input contract schema")
    required = (
        "dem",
        "heights_text",
        "height_type",
        "start_x",
        "start_y",
        "initial_heading_deg",
        "initial_speed_mps",
    )
    missing = [key for key in required if payload.get(key) is None]
    if missing:
        raise ValueError(
            "localizer_input.json is missing: " + ", ".join(missing)
        )
    base = path.parent
    payload["dem"] = _contract_path(base, str(payload["dem"]))
    payload["heights_text"] = _contract_path(base, str(payload["heights_text"]))
    payload["truth"] = _contract_path(
        base,
        None if payload.get("truth") is None else str(payload["truth"]),
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        metadata = None
        if args.input_contract is not None:
            conflicting = (
                args.scenario,
                args.dem,
                args.measurements,
                args.heights_text,
                args.initial_state,
                args.start_x,
                args.start_y,
                args.initial_vx,
                args.initial_vy,
                args.heading_deg,
                args.initial_speed,
                args.sample_rate,
                args.baro_altitude,
                args.truth,
                args.working_resolution,
            )
            if any(value is not None for value in conflicting):
                raise ValueError(
                    "--input-contract cannot be combined with explicit input flags"
                )
            contract = _load_text_contract(args.input_contract)
            result, metadata = localize_text_heights(
                contract["dem"],
                contract["heights_text"],
                args.output_dir,
                start_x=float(contract["start_x"]),
                start_y=float(contract["start_y"]),
                initial_azimuth_deg=float(contract["initial_heading_deg"]),
                initial_speed_mps=float(contract["initial_speed_mps"]),
                sample_rate_hz=(
                    None
                    if contract.get("sample_rate_hz") is None
                    else float(contract["sample_rate_hz"])
                ),
                height_type=str(contract["height_type"]),
                baro_altitude_m=(
                    None
                    if contract.get("baro_altitude_m") is None
                    else float(contract["baro_altitude_m"])
                ),
                config_path=args.config,
                map_radius_m=(
                    None
                    if contract.get("map_radius_m") is None
                    else float(contract["map_radius_m"])
                ),
                working_resolution_m=(
                    None
                    if contract.get("working_resolution_m") is None
                    else float(contract["working_resolution_m"])
                ),
                frequency_min_hz=float(contract.get("frequency_min_hz", 1.0)),
                frequency_max_hz=float(contract.get("frequency_max_hz", 10.0)),
                frequency_step_hz=args.frequency_step,
                frequency_prefix_samples=args.frequency_prefix_samples,
                truth_path=contract.get("truth"),
                preview_trajectory=args.preview_trajectory,
                save_plots=not args.no_plots,
            )
        elif args.heights_text is not None:
            conflicting = (
                args.scenario,
                args.measurements,
                args.initial_state,
                args.initial_vx,
                args.initial_vy,
            )
            if any(value is not None for value in conflicting):
                raise ValueError(
                    "--heights-text cannot be combined with scenario/NPZ input flags"
                )
            required = {
                "--dem": args.dem,
                "--start-x": args.start_x,
                "--start-y": args.start_y,
                "--heading-deg": args.heading_deg,
                "--initial-speed": args.initial_speed,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    "Text-height mode requires " + ", ".join(missing)
                )
            result, metadata = localize_text_heights(
                args.dem,
                args.heights_text,
                args.output_dir,
                start_x=args.start_x,
                start_y=args.start_y,
                initial_azimuth_deg=args.heading_deg,
                initial_speed_mps=args.initial_speed,
                sample_rate_hz=args.sample_rate,
                height_type=args.height_type,
                baro_altitude_m=args.baro_altitude,
                config_path=args.config,
                map_radius_m=args.map_radius,
                working_resolution_m=args.working_resolution,
                frequency_min_hz=args.frequency_min,
                frequency_max_hz=args.frequency_max,
                frequency_step_hz=args.frequency_step,
                frequency_prefix_samples=args.frequency_prefix_samples,
                truth_path=args.truth,
                preview_trajectory=args.preview_trajectory,
                save_plots=not args.no_plots,
            )
        else:
            text_only = (
                args.heading_deg,
                args.initial_speed,
                args.sample_rate,
                args.baro_altitude,
                args.working_resolution,
            )
            if any(value is not None for value in text_only):
                raise ValueError(
                    "Text-height flags require --heights-text"
                )
            dem, measurements, state, config, truth = _resolve_inputs(args)
            result = localize_from_inputs(
                dem,
                measurements,
                state,
                args.output_dir,
                config_path=config,
                truth_path=truth,
                preview_trajectory=args.preview_trajectory,
                save_plots=not args.no_plots,
            )
    except ValueError as exc:
        parser.error(str(exc))
    summary = result_summary(result, args.output_dir / "result.json")
    if metadata is not None:
        summary["sample_rate_hz"] = metadata["sample_rate_hz"]
        summary["sample_rate_source"] = metadata["sample_rate_source"]
        summary["coordinates"] = metadata["coordinates"]
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
