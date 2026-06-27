import numpy as np

from terrain_nav.dem import DEMInterpolator


def test_bilinear_interpolation_at_center_and_nodes():
    dem = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)
    interpolator = DEMInterpolator(dem, resolution_m=10.0, origin_x=100.0, origin_y=200.0)

    assert interpolator.sample(100.0, 200.0) == 0.0
    assert interpolator.sample(110.0, 210.0) == 30.0
    assert interpolator.sample(105.0, 205.0) == 15.0


def test_batch_shape_and_out_of_bounds_nan():
    dem = np.arange(9, dtype=np.float32).reshape(3, 3)
    interpolator = DEMInterpolator(dem, resolution_m=1.0)
    values = interpolator.sample_batch(
        np.array([[0.0, 0.5], [1.0, 3.0]]),
        np.array([[0.0, 0.5], [1.0, 1.0]]),
    )

    assert values.shape == (2, 2)
    assert np.isclose(values[0, 1], 2.0)
    assert np.isnan(values[1, 1])

