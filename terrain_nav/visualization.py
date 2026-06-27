"""Diagnostic plots for terrain navigation results."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from .dem import DEMInterpolator
from .profile import compute_slope, normalize_profile
from .search import ScoreGrid, SearchResult, TrajectoryHypothesis


def _pyplot(*, interactive: bool = False):
    cache_dir = Path(tempfile.gettempdir()) / "terrain_nav_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    if not interactive and "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if interactive and "agg" in matplotlib.get_backend().lower():
        raise RuntimeError(
            "Предпросмотр требует интерактивный backend Matplotlib. "
            "Запустите команду в графической сессии или задайте MPLBACKEND "
            "(например, MacOSX, TkAgg или QtAgg)."
        )
    return plt


def plot_trajectory_on_dem(
    dem_interpolator: DEMInterpolator,
    best: TrajectoryHypothesis,
    alternatives: list[TrajectoryHypothesis] | None = None,
    *,
    true_path: tuple[np.ndarray, np.ndarray] | None = None,
    output_path: str | Path | None = None,
    preview: bool = False,
):
    """Build a trajectory plot, optionally showing it before PNG saving.

    With ``preview=True`` the call blocks until the user closes the Matplotlib
    window. Only then is ``output_path`` written.
    """

    plt = _pyplot(interactive=preview)
    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    xmin, ymin, xmax, ymax = dem_interpolator.bounds
    image = axis.imshow(
        dem_interpolator.dem,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        cmap="terrain",
        aspect="equal",
    )
    figure.colorbar(image, ax=axis, label="Высота DEM, м")
    for index, hypothesis in enumerate(alternatives or []):
        axis.plot(
            hypothesis.path_x,
            hypothesis.path_y,
            color="white",
            alpha=0.28,
            linewidth=1.0,
            zorder=2,
            label="Альтернативы" if index == 0 else None,
        )
    axis.plot(
        best.path_x,
        best.path_y,
        color="#e31a1c",
        linewidth=2.4,
        zorder=4,
        label="Оценка",
    )
    axis.scatter(
        best.path_x[0],
        best.path_y[0],
        marker="o",
        s=70,
        color="cyan",
        zorder=5,
        label="Старт",
    )
    axis.scatter(
        best.x,
        best.y,
        marker="X",
        s=100,
        color="yellow",
        edgecolor="black",
        zorder=5,
        label="Текущая точка",
    )
    if true_path is not None:
        axis.plot(
            true_path[0],
            true_path[1],
            "--",
            color="#1f78b4",
            linewidth=2.0,
            zorder=4,
            label="Истинная траектория",
        )
    axis.set(title="Траектория на цифровой модели рельефа", xlabel="X, восток, м", ylabel="Y, север, м")
    axis.legend(loc="best")
    if preview:
        manager = getattr(figure.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("Предпросмотр trajectory.png")
        plt.show(block=True)
    if output_path is not None:
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
    return figure


def plot_score_heatmap(
    score_grid: ScoreGrid,
    *,
    output_path: str | Path | None = None,
):
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    finite_scores = np.where(np.isfinite(score_grid.scores), score_grid.scores, np.nan)
    image = axis.imshow(finite_scores, origin="lower", aspect="auto", cmap="viridis")
    x_ticks = np.linspace(0, len(score_grid.azimuths_deg) - 1, min(9, len(score_grid.azimuths_deg))).astype(int)
    y_ticks = np.linspace(0, len(score_grid.speeds_mps) - 1, min(8, len(score_grid.speeds_mps))).astype(int)
    axis.set_xticks(x_ticks, [f"{score_grid.azimuths_deg[i]:.0f}" for i in x_ticks])
    axis.set_yticks(y_ticks, [f"{score_grid.speeds_mps[i]:.1f}" for i in y_ticks])
    axis.set(xlabel="Азимут, градусы", ylabel="Путевая скорость, м/с", title="Тепловая карта score первого окна")
    figure.colorbar(image, ax=axis, label="Score")
    if output_path is not None:
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
    return figure


def plot_profile_comparison(
    measured_profile: np.ndarray,
    dem_profile: np.ndarray,
    *,
    timestamps: np.ndarray | None = None,
    score: float | None = None,
    output_path: str | Path | None = None,
):
    plt = _pyplot()
    measured = np.asarray(measured_profile, dtype=np.float64)
    candidate = np.asarray(dem_profile, dtype=np.float64)
    count = min(measured.size, candidate.size)
    measured, candidate = measured[:count], candidate[:count]
    horizontal = np.arange(count) if timestamps is None else np.asarray(timestamps)[:count] - np.asarray(timestamps)[0]
    x_label = "Номер измерения" if timestamps is None else "Время, с"
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False, constrained_layout=True)
    axes[0].plot(horizontal, measured, label="Измеренный H_baro − H_radio")
    axes[0].plot(horizontal, candidate, label="DEM вдоль гипотезы", alpha=0.85)
    axes[0].set(ylabel="Высота, м", title=f"Сравнение профилей, score={score:.3f}" if score is not None else "Сравнение профилей")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(horizontal, normalize_profile(measured), label="Измеренный")
    axes[1].plot(horizontal, normalize_profile(candidate), label="DEM")
    axes[1].set(ylabel="Z-score", title="Нормализованные профили")
    axes[1].grid(alpha=0.25)
    slope_x = horizontal[1:]
    axes[2].plot(slope_x, compute_slope(measured), label="Измеренный")
    axes[2].plot(slope_x, compute_slope(candidate), label="DEM")
    axes[2].set(xlabel=x_label, ylabel="Первая разность, м", title="Уклоны профилей")
    axes[2].grid(alpha=0.25)
    if output_path is not None:
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
    return figure


def save_result_plots(
    result: SearchResult,
    dem_interpolator: DEMInterpolator,
    output_dir: str | Path,
    *,
    true_path: tuple[np.ndarray, np.ndarray] | None = None,
    preview_trajectory: bool = False,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plot_trajectory_on_dem(
        dem_interpolator,
        result.best,
        result.alternatives,
        true_path=true_path,
        output_path=destination / "trajectory.png",
        preview=preview_trajectory,
    )
    if result.score_grid is not None:
        plot_score_heatmap(result.score_grid, output_path=destination / "score_heatmap.png")
    if result.measured_heights_m is not None:
        plot_profile_comparison(
            result.measured_heights_m,
            result.best.path_heights_dem,
            timestamps=result.measured_timestamps,
            score=result.best_score,
            output_path=destination / "profile_comparison.png",
        )
