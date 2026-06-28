"""Metric DEM loader and interpolator owned by the localizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class DEMGrid:
    dem: np.ndarray
    resolution_m: float
    origin_x: float
    origin_y: float
    crs: str | None = None

    def __post_init__(self) -> None:
        data = np.asarray(self.dem, dtype=np.float32)
        if data.ndim != 2 or min(data.shape) < 2:
            raise ValueError("DEM must be a two-dimensional grid of at least 2x2")
        if not np.isfinite(self.resolution_m) or self.resolution_m <= 0:
            raise ValueError("DEM resolution must be positive")
        object.__setattr__(self, "dem", data)

    def interpolator(self, method: str = "bilinear") -> "DEMInterpolator":
        return DEMInterpolator(
            self.dem,
            self.resolution_m,
            self.origin_x,
            self.origin_y,
            method,
        )


@dataclass(frozen=True, slots=True)
class PreparedGeoTIFF:
    """Metric working grid plus transformations back to global coordinates."""

    grid: DEMGrid
    start_working_x: float
    start_working_y: float
    source_crs: str
    working_crs: str

    def transform_from_working(
        self,
        x: np.ndarray,
        y: np.ndarray,
        destination_crs: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            from rasterio.warp import transform
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Coordinate transformation requires rasterio") from exc
        transformed_x, transformed_y = transform(
            self.working_crs,
            destination_crs,
            np.asarray(x, dtype=np.float64).tolist(),
            np.asarray(y, dtype=np.float64).tolist(),
        )
        return np.asarray(transformed_x), np.asarray(transformed_y)

    def to_source(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.transform_from_working(x, y, self.source_crs)

    def to_wgs84(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.transform_from_working(x, y, "EPSG:4326")


class DEMInterpolator:
    """Continuous local interpolation over a metric ``dem[y, x]`` grid."""

    def __init__(
        self,
        dem: np.ndarray,
        resolution_m: float,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        method: str = "bilinear",
    ) -> None:
        data = np.asarray(dem, dtype=np.float32)
        if data.ndim != 2 or min(data.shape) < 2:
            raise ValueError("DEM must be a two-dimensional grid of at least 2x2")
        if not np.isfinite(resolution_m) or resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        if method not in {"bilinear", "bicubic"}:
            raise ValueError("method must be 'bilinear' or 'bicubic'")
        self.dem = data
        self.resolution_m = float(resolution_m)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.method = method
        self._map_coordinates = None
        if method == "bicubic":
            try:
                from scipy.ndimage import map_coordinates
            except ImportError as exc:  # pragma: no cover
                raise ImportError("bicubic interpolation requires scipy") from exc
            self._map_coordinates = map_coordinates

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        rows, cols = self.dem.shape
        return (
            self.origin_x,
            self.origin_y,
            self.origin_x + (cols - 1) * self.resolution_m,
            self.origin_y + (rows - 1) * self.resolution_m,
        )

    def contains(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        gx = (np.asarray(x) - self.origin_x) / self.resolution_m
        gy = (np.asarray(y) - self.origin_y) / self.resolution_m
        rows, cols = self.dem.shape
        return (gx >= 0.0) & (gy >= 0.0) & (gx <= cols - 1) & (gy <= rows - 1)

    def sample(self, x: float, y: float) -> float:
        return float(self.sample_batch(np.asarray([x]), np.asarray([y]))[0])

    def sample_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        bx, by = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
        )
        shape = bx.shape
        gx = ((bx.ravel() - self.origin_x) / self.resolution_m).astype(np.float64)
        gy = ((by.ravel() - self.origin_y) / self.resolution_m).astype(np.float64)
        rows, cols = self.dem.shape
        valid = (gx >= 0.0) & (gy >= 0.0) & (gx <= cols - 1) & (gy <= rows - 1)
        result = np.full(gx.shape, np.nan, dtype=np.float32)
        if not np.any(valid):
            return result.reshape(shape)
        if self.method == "bicubic":
            coordinates = np.vstack((gy[valid], gx[valid]))
            result[valid] = self._map_coordinates(  # type: ignore[misc]
                self.dem,
                coordinates,
                order=3,
                mode="nearest",
                prefilter=True,
            ).astype(np.float32)
            return result.reshape(shape)

        vx, vy = gx[valid], gy[valid]
        x0 = np.floor(vx).astype(np.intp)
        y0 = np.floor(vy).astype(np.intp)
        x1 = np.minimum(x0 + 1, cols - 1)
        y1 = np.minimum(y0 + 1, rows - 1)
        tx, ty = vx - x0, vy - y0
        result[valid] = (
            self.dem[y0, x0] * (1.0 - tx) * (1.0 - ty)
            + self.dem[y0, x1] * tx * (1.0 - ty)
            + self.dem[y1, x0] * (1.0 - tx) * ty
            + self.dem[y1, x1] * tx * ty
        ).astype(np.float32)
        return result.reshape(shape)


def load_dem_grid(
    path: str | Path,
    *,
    resolution_m: float = 30.0,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> DEMGrid:
    """Load only navigation-ready metric NPY/NPZ/GeoTIFF data."""

    source = Path(path)
    if source.suffix.lower() == ".npy":
        return DEMGrid(np.load(source), resolution_m, origin_x, origin_y)
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as values:
            key = "dem" if "dem" in values.files else values.files[0]
            return DEMGrid(
                values[key],
                float(values["resolution_m"].item()) if "resolution_m" in values else resolution_m,
                float(values["origin_x"].item()) if "origin_x" in values else origin_x,
                float(values["origin_y"].item()) if "origin_y" in values else origin_y,
                (str(values["crs"].item()) or None) if "crs" in values else None,
            )
    if source.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"Unsupported DEM format: {source.suffix}")
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("GeoTIFF support requires rasterio") from exc
    with rasterio.open(source) as dataset:
        if dataset.crs is None or not dataset.crs.is_projected:
            raise ValueError("Localizer requires a projected metric GeoTIFF")
        x_resolution, y_resolution = map(abs, dataset.res)
        if not np.isclose(x_resolution, y_resolution, rtol=1e-3):
            raise ValueError("Navigation DEM must have square metric pixels")
        data = np.flipud(
            dataset.read(1, masked=True).filled(np.nan).astype(np.float32)
        )
        first_x, first_y = dataset.xy(dataset.height - 1, 0, offset="center")
        return DEMGrid(
            data,
            x_resolution,
            float(first_x),
            float(first_y),
            dataset.crs.to_string(),
        )


def _utm_crs(longitude: float, latitude: float) -> str:
    zone = min(60, max(1, int(np.floor((longitude + 180.0) / 6.0)) + 1))
    return f"EPSG:{(32600 if latitude >= 0 else 32700) + zone}"


def prepare_geotiff(
    path: str | Path,
    *,
    start_x: float,
    start_y: float,
    resolution_m: float = 30.0,
    map_radius_m: float | None = None,
) -> PreparedGeoTIFF:
    """Prepare a metric grid; ``None`` loads the complete GeoTIFF."""

    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")
    if map_radius_m is not None and map_radius_m <= 0:
        raise ValueError("map_radius_m must be positive when supplied")
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.vrt import WarpedVRT
        from rasterio.windows import Window, from_bounds
        from rasterio.warp import calculate_default_transform, transform
    except ImportError as exc:  # pragma: no cover
        raise ImportError("GeoTIFF preparation requires rasterio") from exc

    source_path = Path(path)
    with rasterio.open(source_path) as source:
        if source.crs is None:
            raise ValueError("GeoTIFF has no coordinate reference system")
        source_crs = source.crs.to_string()
        if source.crs.is_geographic:
            if not (
                source.bounds.left <= start_x <= source.bounds.right
                and source.bounds.bottom <= start_y <= source.bounds.top
            ):
                raise ValueError("Initial point is outside the GeoTIFF")
            working_crs = _utm_crs(start_x, start_y)
        elif source.crs.is_projected:
            try:
                unit_factor = float(source.crs.linear_units_factor[1])
            except (AttributeError, TypeError, ValueError):
                unit_factor = np.nan
            if np.isclose(unit_factor, 1.0):
                working_crs = source_crs
            else:
                longitude, latitude = transform(
                    source_crs,
                    "EPSG:4326",
                    [float(start_x)],
                    [float(start_y)],
                )
                working_crs = _utm_crs(longitude[0], latitude[0])
        else:
            raise ValueError("Unsupported GeoTIFF coordinate reference system")

        working_x, working_y = transform(
            source_crs,
            working_crs,
            [float(start_x)],
            [float(start_y)],
        )
        start_working_x = float(working_x[0])
        start_working_y = float(working_y[0])
        destination_transform, destination_width, destination_height = (
            calculate_default_transform(
                source.crs,
                working_crs,
                source.width,
                source.height,
                *source.bounds,
                resolution=resolution_m,
            )
        )
        with WarpedVRT(
            source,
            crs=working_crs,
            transform=destination_transform,
            width=destination_width,
            height=destination_height,
            resampling=Resampling.bilinear,
            nodata=np.nan,
            dtype="float32",
        ) as warped:
            full = Window(0, 0, warped.width, warped.height)
            if map_radius_m is None:
                window = full
            else:
                requested = from_bounds(
                    start_working_x - map_radius_m,
                    start_working_y - map_radius_m,
                    start_working_x + map_radius_m,
                    start_working_y + map_radius_m,
                    warped.transform,
                )
                try:
                    window = requested.intersection(full).round_offsets().round_lengths()
                except rasterio.errors.WindowError as exc:
                    raise ValueError(
                        "Requested map region does not overlap the GeoTIFF"
                    ) from exc
            values = warped.read(1, window=window, masked=True)
            data = np.flipud(values.filled(np.nan).astype(np.float32))
            window_transform = warped.window_transform(window)
            origin_x, origin_y = rasterio.transform.xy(
                window_transform,
                data.shape[0] - 1,
                0,
                offset="center",
            )
            grid = DEMGrid(
                data,
                resolution_m,
                float(origin_x),
                float(origin_y),
                working_crs,
            )

    if not bool(grid.interpolator().contains(start_working_x, start_working_y)):
        raise ValueError("Initial point is outside the prepared DEM grid")
    return PreparedGeoTIFF(
        grid,
        start_working_x,
        start_working_y,
        source_crs,
        working_crs,
    )
