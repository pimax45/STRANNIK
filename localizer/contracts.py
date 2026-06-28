"""Independent readers for the localization input contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class InitialState:
    """Last trusted position and horizontal ground-velocity vector."""

    x: float
    y: float
    velocity_x_mps: float
    velocity_y_mps: float

    @property
    def ground_speed_mps(self) -> float:
        return float(np.hypot(self.velocity_x_mps, self.velocity_y_mps))

    @property
    def azimuth_deg(self) -> float:
        if self.ground_speed_mps <= 0:
            raise ValueError("Initial velocity vector must be non-zero")
        return float(
            np.mod(
                np.degrees(np.arctan2(self.velocity_x_mps, self.velocity_y_mps)),
                360.0,
            )
        )

    def validate(self) -> None:
        values = np.asarray(
            [self.x, self.y, self.velocity_x_mps, self.velocity_y_mps],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Initial state values must be finite")
        if self.ground_speed_mps <= 0:
            raise ValueError("Initial velocity vector must be non-zero")

    def to_dict(self) -> dict[str, float]:
        self.validate()
        return {
            "x": float(self.x),
            "y": float(self.y),
            "velocity_x_mps": float(self.velocity_x_mps),
            "velocity_y_mps": float(self.velocity_y_mps),
            "ground_speed_mps": self.ground_speed_mps,
            "azimuth_deg": self.azimuth_deg,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "InitialState":
        state = cls(
            x=float(values["x"]),
            y=float(values["y"]),
            velocity_x_mps=float(values["velocity_x_mps"]),
            velocity_y_mps=float(values["velocity_y_mps"]),
        )
        state.validate()
        return state

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "InitialState":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(values)


@dataclass(frozen=True, slots=True)
class RadioMeasurementSeries:
    """Timestamped radio and barometric altitudes passed to localization."""

    timestamps: np.ndarray
    radio_altitude_m: np.ndarray
    baro_altitude_m: np.ndarray

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        radio = np.asarray(self.radio_altitude_m, dtype=np.float64)
        baro = np.asarray(self.baro_altitude_m, dtype=np.float64)
        if baro.ndim == 0:
            baro = np.full(timestamps.shape, float(baro), dtype=np.float64)
        if timestamps.ndim != 1 or radio.ndim != 1 or baro.ndim != 1:
            raise ValueError("Measurement arrays must be one-dimensional")
        if not (timestamps.size == radio.size == baro.size):
            raise ValueError("Measurement arrays must have equal lengths")
        if timestamps.size < 2:
            raise ValueError("At least two radio-altimeter measurements are required")
        if np.any(~np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
            raise ValueError("Measurement timestamps must be finite and strictly increasing")
        if np.any(~np.isfinite(radio)) or np.any(radio < 0):
            raise ValueError("Radio altitudes must be finite and non-negative")
        if np.any(~np.isfinite(baro)):
            raise ValueError("Barometric altitudes must be finite")
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "radio_altitude_m", radio)
        object.__setattr__(self, "baro_altitude_m", baro)

    @property
    def count(self) -> int:
        return int(self.timestamps.size)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            destination,
            timestamps=self.timestamps,
            radio_altitude_m=self.radio_altitude_m.astype(np.float32),
            baro_altitude_m=self.baro_altitude_m.astype(np.float32),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "RadioMeasurementSeries":
        with np.load(Path(path), allow_pickle=False) as values:
            return cls(
                values["timestamps"],
                values["radio_altitude_m"],
                values["baro_altitude_m"],
            )


@dataclass(frozen=True, slots=True)
class ScenarioManifest:
    """Portable description of generator output consumed by localization."""

    path: Path
    dem: str
    measurements: str
    initial_state: str
    config: str | None = None
    truth: str | None = None
    source_dem: str | None = None

    def _resolve(self, value: str | None) -> Path | None:
        if value is None:
            return None
        candidate = Path(value)
        return candidate if candidate.is_absolute() else self.path.parent / candidate

    @property
    def dem_path(self) -> Path:
        return self._resolve(self.dem)  # type: ignore[return-value]

    @property
    def measurements_path(self) -> Path:
        return self._resolve(self.measurements)  # type: ignore[return-value]

    @property
    def initial_state_path(self) -> Path:
        return self._resolve(self.initial_state)  # type: ignore[return-value]

    @property
    def config_path(self) -> Path | None:
        return self._resolve(self.config)

    @property
    def truth_path(self) -> Path | None:
        return self._resolve(self.truth)

    def save(self) -> Path:
        payload = {
            "schema": "terrain-nav-scenario/v1",
            "dem": self.dem,
            "measurements": self.measurements,
            "initial_state": self.initial_state,
            "config": self.config,
            "truth": self.truth,
            "source_dem": self.source_dem,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.path

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioManifest":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema") != "terrain-nav-scenario/v1":
            raise ValueError("Unsupported scenario manifest schema")
        return cls(
            path=source,
            dem=str(payload["dem"]),
            measurements=str(payload["measurements"]),
            initial_state=str(payload["initial_state"]),
            config=payload.get("config"),
            truth=payload.get("truth"),
            source_dem=payload.get("source_dem"),
        )
