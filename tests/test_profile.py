import numpy as np

from terrain_nav.nmea import RadioAltimeterSample
from terrain_nav.profile import (
    compute_slope,
    normalize_profile,
    radio_to_ground_profile,
    remove_mean_bias,
    remove_outliers_hampel,
    smooth_moving_average,
)


def test_radio_to_ground_profile_scalar_and_series_baro():
    samples = [RadioAltimeterSample(float(i), 500.0 + i) for i in range(4)]
    scalar = radio_to_ground_profile(samples, 1500.0)
    series = radio_to_ground_profile(samples, np.full(4, 1500.0))

    np.testing.assert_allclose(scalar.heights_m, [1000, 999, 998, 997])
    np.testing.assert_allclose(series.heights_m, scalar.heights_m)
    assert scalar.sample_rate_hz == 1.0


def test_preprocessing_helpers():
    profile = np.array([1.0, 2.0, 100.0, 4.0, 5.0])
    filtered = remove_outliers_hampel(profile, window_size=5, n_sigma=2.0)
    assert filtered[2] != 100.0
    assert smooth_moving_average(profile, 3).shape == profile.shape
    assert abs(np.mean(normalize_profile(profile))) < 1e-12
    np.testing.assert_allclose(compute_slope([1, 3, 6]), [2, 3])
    first, second = remove_mean_bias(np.array([1, 2, 3]), np.array([11, 12, 13]))
    np.testing.assert_allclose(first, second)

