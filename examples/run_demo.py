"""Create a synthetic dataset and run the same pipeline as the CLI."""

from pathlib import Path

from terrain_nav.config import SearchConfig
from terrain_nav.dem import DEMInterpolator, load_dem
from terrain_nav.nmea import read_nmea
from terrain_nav.profile import radio_to_ground_profile
from terrain_nav.search import TerrainNavigator
from terrain_nav.simulator import create_test_dataset


workspace = Path("data/demo")
paths = create_test_dataset(workspace, terrain="mixed", duration_s=30.0)
config = SearchConfig()
interpolator = DEMInterpolator(load_dem(paths["dem"]), config.resolution_m)
profile = radio_to_ground_profile(read_nmea(paths["nmea"]), 1500.0)
result = TerrainNavigator(interpolator, config).localize(
    profile,
    start_x=1000.0,
    start_y=1000.0,
    initial_speed_mps=50.0,
    initial_azimuth_deg=73.0,
)
print(result.to_dict())

