"""DEM loading and continuous local interpolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class DEMInterpolator:
    """Continuous sampler over a discrete ``dem[y, x]`` height grid.

    Bilinear interpolation is the default and operational mode. Bicubic
    interpolation is offered for offline analysis when SciPy is installed.
    Coordinates use metres, with x pointing east and y north.
    """

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
        self._spline = None
        if method == "bicubic":
            try:
                from scipy.ndimage import map_coordinates
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("bicubic interpolation requires scipy") from exc
            self._spline = map_coordinates

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        rows, cols = self.dem.shape
        return (
            self.origin_x,
            self.origin_y,
            self.origin_x + (cols - 1) * self.resolution_m,
            self.origin_y + (rows - 1) * self.resolution_m,
        )

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gx = (np.asarray(x) - self.origin_x) / self.resolution_m
        gy = (np.asarray(y) - self.origin_y) / self.resolution_m
        rows, cols = self.dem.shape
        return (gx >= 0.0) & (gy >= 0.0) & (gx <= cols - 1) & (gy <= rows - 1)

    def sample(self, x: float, y: float) -> float:
        return float(self.sample_batch(np.asarray([x]), np.asarray([y]))[0])

    def sample_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Sample arbitrary, broadcast-compatible coordinate arrays.

        Points outside the DEM are returned as NaN so that the search can apply
        an explicit out-of-bounds penalty rather than silently clipping paths.
        """

        bx, by = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
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
            coords = np.vstack((gy[valid], gx[valid]))
            result[valid] = self._spline(  # type: ignore[misc]
                self.dem, coords, order=3, mode="nearest", prefilter=True
            ).astype(np.float32)
            return result.reshape(shape)

        vx = gx[valid]
        vy = gy[valid]
        x0 = np.floor(vx).astype(np.intp)
        y0 = np.floor(vy).astype(np.intp)
        x1 = np.minimum(x0 + 1, cols - 1)
        y1 = np.minimum(y0 + 1, rows - 1)
        tx = vx - x0
        ty = vy - y0

        z00 = self.dem[y0, x0]
        z10 = self.dem[y0, x1]
        z01 = self.dem[y1, x0]
        z11 = self.dem[y1, x1]
        interpolated = (
            z00 * (1.0 - tx) * (1.0 - ty)
            + z10 * tx * (1.0 - ty)
            + z01 * (1.0 - tx) * ty
            + z11 * tx * ty
        )
        result[valid] = interpolated.astype(np.float32)
        return result.reshape(shape)


def load_dem(path: str | Path) -> np.ndarray:
    """Load a DEM from NPY/NPZ or, when rasterio is available, GeoTIFF."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(source), dtype=np.float32)
    if suffix == ".npz":
        archive = np.load(source)
        key = "dem" if "dem" in archive.files else archive.files[0]
        return np.asarray(archive[key], dtype=np.float32)
    if suffix in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("GeoTIFF support requires rasterio") from exc
        with rasterio.open(source) as dataset:
            return np.asarray(dataset.read(1), dtype=np.float32)
    raise ValueError(f"Unsupported DEM format: {source.suffix}")

