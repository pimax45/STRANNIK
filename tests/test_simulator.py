from pathlib import Path

import numpy as np
import pytest
import yaml

from terrain_nav.dem import DEMInterpolator
from terrain_nav.nmea import parse_nmea_lines
from terrain_nav.simulator import (
    create_test_dataset,
    generate_nmea_from_path,
    generate_synthetic_dem,
    generate_true_path,
)


def test_true_path_uses_custom_start_point():
    path = generate_true_path(
        duration_s=5.0,
        sample_rate_hz=2.0,
        start_x=1234.5,
        start_y=678.9,
    )

    assert path.x[0] == 1234.5
    assert path.y[0] == 678.9


@pytest.mark.parametrize("motion", ["sharp-turn", "sharp-mixed"])
def test_sharp_turn_modes_contain_heading_steps(motion: str):
    path = generate_true_path(
        duration_s=30.0,
        sample_rate_hz=10.0,
        initial_speed_mps=50.0,
        azimuth_deg=70.0,
        motion=motion,
    )

    assert np.max(np.abs(np.diff(path.azimuth_deg))) == pytest.approx(15.0)


@pytest.mark.parametrize("motion", ["sharp-speed-change", "sharp-mixed"])
def test_sharp_speed_modes_contain_speed_steps(motion: str):
    path = generate_true_path(
        duration_s=30.0,
        sample_rate_hz=10.0,
        initial_speed_mps=50.0,
        azimuth_deg=70.0,
        motion=motion,
    )

    assert set(np.unique(path.ground_speed_mps)) == {40.0, 50.0, 65.0}
    assert np.max(np.abs(np.diff(path.ground_speed_mps))) == pytest.approx(25.0)


def test_dataset_records_custom_start_and_motion(tmp_path: Path):
    paths = create_test_dataset(
        tmp_path,
        duration_s=5.0,
        sample_rate_hz=2.0,
        start_x=1400.0,
        start_y=1200.0,
        initial_speed_mps=20.0,
        motion="sharp-mixed",
        radio_noise_std_m=2.5,
        radio_outlier_probability=0.01,
    )
    truth = np.load(paths["truth"])
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))

    assert truth["x"][0] == 1400.0
    assert truth["y"][0] == 1200.0
    assert config["simulation"]["start_x"] == 1400.0
    assert config["simulation"]["start_y"] == 1200.0
    assert config["simulation"]["motion"] == "sharp-mixed"
    assert config["simulation"]["radio_noise_std_m"] == 2.5
    assert config["simulation"]["radio_outlier_probability"] == 0.01
    assert config["search"]["min_window_samples"] == 6
    assert config["search"]["history_rerank_interval_windows"] == 2
    assert config["search"]["history_score_weight"] == 0.9


def test_radio_noise_can_be_disabled_or_increased():
    dem = generate_synthetic_dem("mixed", shape=(96, 96), seed=3)
    interpolator = DEMInterpolator(dem, 30.0)
    path = generate_true_path(
        duration_s=5.0,
        sample_rate_hz=10.0,
        start_x=1000.0,
        start_y=1000.0,
        initial_speed_mps=20.0,
    )
    exact = parse_nmea_lines(
        generate_nmea_from_path(
            path,
            interpolator,
            noise_std_m=0.0,
            outlier_probability=0.0,
            seed=3,
        )
    )
    noisy = parse_nmea_lines(
        generate_nmea_from_path(
            path,
            interpolator,
            noise_std_m=5.0,
            outlier_probability=0.0,
            seed=3,
        )
    )
    expected_radio = 1500.0 - interpolator.sample_batch(path.x, path.y)
    exact_radio = np.asarray([sample.radio_altitude_m for sample in exact])
    noisy_radio = np.asarray([sample.radio_altitude_m for sample in noisy])

    np.testing.assert_allclose(exact_radio, expected_radio, atol=0.001)
    assert np.std(noisy_radio - expected_radio) > 3.0


def test_negative_radio_noise_is_rejected():
    dem = generate_synthetic_dem("mixed", shape=(96, 96), seed=4)
    interpolator = DEMInterpolator(dem, 30.0)
    path = generate_true_path(duration_s=1.0, initial_speed_mps=10.0)

    with pytest.raises(ValueError, match="non-negative"):
        generate_nmea_from_path(path, interpolator, noise_std_m=-1.0)
