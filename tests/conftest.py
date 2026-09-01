import warnings

import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def small_terrain():
    from setu.bench.terrain import synthetic_terrain
    return synthetic_terrain(384, 5.0, "highland", seed=26166)


@pytest.fixture(scope="session")
def tiny_pair(small_terrain):
    from setu.bench.generate import make_pair
    from setu.types import IlluminationState

    return make_pair(
        small_terrain,
        illum_src=IlluminationState(sun_az_deg=135.0, sun_elev_deg=25.0, source="synthetic"),
        illum_ref=IlluminationState(sun_az_deg=135.0, sun_elev_deg=55.0, source="synthetic"),
        scale_ratio=1.0, tile_px=256, warp_kind="similarity", pair_id="tiny", seed=3,
    )
