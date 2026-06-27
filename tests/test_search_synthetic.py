import numpy as np

from terrain_nav.config import SearchConfig
from terrain_nav.dem import DEMInterpolator
from terrain_nav.nmea import parse_nmea_lines
from terrain_nav.profile import radio_to_ground_profile
from terrain_nav.search import TerrainNavigator, terrain_informativeness
from terrain_nav.simulator import (
    TruePath,
    generate_nmea_from_path,
    generate_synthetic_dem,
    generate_true_path,
)


def _angle_error(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def test_search_recovers_straight_flight_on_informative_terrain():
    dem = generate_synthetic_dem("mixed", shape=(192, 192), seed=7)
    interpolator = DEMInterpolator(dem, 30.0)
    truth = generate_true_path(
        duration_s=20.0,
        sample_rate_hz=5.0,
        initial_speed_mps=48.0,
        azimuth_deg=72.0,
    )
    lines = generate_nmea_from_path(
        truth, interpolator, noise_std_m=0.25, outlier_probability=0.0, seed=7
    )
    profile = radio_to_ground_profile(parse_nmea_lines(lines), 1500.0)
    config = SearchConfig(
        top_k_hypotheses=7,
        recovery_top_k=15,
        window_seconds=10.0,
        window_overlap=0.5,
        rerank_candidates=25,
    )
    result = TerrainNavigator(interpolator, config).localize(
        profile,
        start_x=truth.x[0],
        start_y=truth.y[0],
        initial_speed_mps=48.0,
        initial_azimuth_deg=72.0,
    )

    position_error = np.hypot(result.best.x - truth.x[-1], result.best.y - truth.y[-1])
    assert _angle_error(result.best.azimuth_deg, truth.azimuth_deg[-1]) <= 5.0
    assert abs(result.best.ground_speed_mps - truth.ground_speed_mps[-1]) / 48.0 <= 0.15
    assert position_error <= 90.0
    assert len(result.alternatives) >= 1
    assert len(result.best.path_x) == len(profile.timestamps)
    assert result.best.history_score is not None


def test_flat_profile_is_not_informative():
    assert terrain_informativeness(np.full(100, 500.0)) < 0.1


def test_search_never_reports_high_confidence_on_flat_terrain():
    dem = generate_synthetic_dem("flat", shape=(128, 128), seed=11)
    interpolator = DEMInterpolator(dem, 30.0)
    truth = generate_true_path(
        duration_s=10.0,
        sample_rate_hz=5.0,
        initial_speed_mps=45.0,
        azimuth_deg=120.0,
    )
    profile = radio_to_ground_profile(
        parse_nmea_lines(
            generate_nmea_from_path(
                truth,
                interpolator,
                noise_std_m=0.2,
                outlier_probability=0.0,
                seed=11,
            )
        ),
        1500.0,
    )
    result = TerrainNavigator(
        interpolator,
        SearchConfig(
            top_k_hypotheses=7,
            recovery_top_k=15,
            window_seconds=10.0,
        ),
    ).localize(
        profile,
        start_x=truth.x[0],
        start_y=truth.y[0],
        initial_speed_mps=45.0,
        initial_azimuth_deg=None,
    )

    assert result.confidence == "low"


def test_time_windows_use_full_duration_at_three_hz():
    dem = generate_synthetic_dem("mixed", shape=(64, 64), seed=5)
    navigator = TerrainNavigator(
        DEMInterpolator(dem, 30.0),
        SearchConfig(window_seconds=10.0, window_overlap=0.5),
    )
    timestamps = np.arange(61, dtype=np.float64) / 3.0

    windows = navigator._segments(timestamps, sample_rate_hz=99.0)

    assert windows == [(0, 31), (15, 46), (30, 61)]
    assert [end - begin for begin, end in windows] == [31, 31, 31]


def test_time_windows_support_irregular_measurement_rate():
    dem = generate_synthetic_dem("mixed", shape=(64, 64), seed=6)
    config = SearchConfig(
        window_seconds=10.0,
        window_overlap=0.5,
        min_window_samples=6,
        min_window_time_coverage=0.8,
    )
    navigator = TerrainNavigator(DEMInterpolator(dem, 30.0), config)
    intervals = np.random.default_rng(6).uniform(0.12, 0.62, size=100)
    timestamps = np.concatenate(([0.0], np.cumsum(intervals)))
    timestamps = timestamps[timestamps <= 25.0]

    windows = navigator._segments(timestamps, sample_rate_hz=1.0)

    assert len(windows) >= 4
    assert windows[-1][1] == len(timestamps)
    assert len({end - begin for begin, end in windows}) > 1
    for begin, end in windows:
        assert timestamps[end - 1] - timestamps[begin] >= 8.0
        assert end - begin >= 6


def test_history_reranking_receives_all_accumulated_samples():
    class RecordingNavigator(TerrainNavigator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.history_lengths: list[int] = []

        def _rerank_by_history(self, hypotheses, measured_history):
            self.history_lengths.append(len(measured_history))
            super()._rerank_by_history(hypotheses, measured_history)

    dem = generate_synthetic_dem("mixed", shape=(128, 128), seed=13)
    interpolator = DEMInterpolator(dem, 30.0)
    truth = generate_true_path(
        duration_s=20.0,
        sample_rate_hz=5.0,
        initial_speed_mps=45.0,
        azimuth_deg=70.0,
    )
    profile = radio_to_ground_profile(
        parse_nmea_lines(
            generate_nmea_from_path(
                truth,
                interpolator,
                noise_std_m=0.2,
                outlier_probability=0.0,
                seed=13,
            )
        ),
        1500.0,
    )
    navigator = RecordingNavigator(
        interpolator,
        SearchConfig(
            top_k_hypotheses=7,
            recovery_top_k=15,
            window_seconds=10.0,
            history_rerank_interval_windows=2,
        ),
    )

    result = navigator.localize(
        profile,
        start_x=truth.x[0],
        start_y=truth.y[0],
        initial_speed_mps=45.0,
        initial_azimuth_deg=70.0,
    )

    assert navigator.history_lengths == [76, 101]
    assert len(result.best.path_x) == 101


def test_localization_accepts_irregular_radio_timestamps():
    dem = generate_synthetic_dem("mixed", shape=(160, 160), seed=23)
    interpolator = DEMInterpolator(dem, 30.0)
    intervals = np.random.default_rng(23).uniform(0.12, 0.95, size=80)
    timestamps = np.concatenate(([0.0], np.cumsum(intervals)))
    timestamps = timestamps[timestamps < 20.0]
    timestamps = np.append(timestamps, 20.0)
    speed = 42.0
    heading = 68.0
    dt = np.diff(timestamps, prepend=timestamps[0])
    theta = np.deg2rad(heading)
    truth = TruePath(
        timestamps=timestamps,
        x=1000.0 + np.cumsum(speed * dt * np.sin(theta)),
        y=1000.0 + np.cumsum(speed * dt * np.cos(theta)),
        azimuth_deg=np.full(timestamps.size, heading),
        ground_speed_mps=np.full(timestamps.size, speed),
    )
    profile = radio_to_ground_profile(
        parse_nmea_lines(
            generate_nmea_from_path(
                truth,
                interpolator,
                noise_std_m=0.15,
                outlier_probability=0.0,
                seed=23,
            )
        ),
        1500.0,
    )
    result = TerrainNavigator(
        interpolator,
        SearchConfig(
            top_k_hypotheses=7,
            recovery_top_k=15,
            window_seconds=10.0,
        ),
    ).localize(
        profile,
        start_x=1000.0,
        start_y=1000.0,
        initial_speed_mps=speed,
        initial_azimuth_deg=heading,
    )

    position_error = np.hypot(result.best.x - truth.x[-1], result.best.y - truth.y[-1])
    assert len(result.best.path_x) == len(profile.timestamps)
    assert position_error <= 90.0
