import pytest
import rasterio
from shapely.geometry import box

from src._types import HeightWidth
from src.models import Raster


@pytest.fixture
def scl_raster():
    with open("tests/assets/scl_image.tif", "rb") as f:
        content = f.read()
    with rasterio.open("tests/assets/scl_image.tif") as src:
        meta = src.meta.copy()

    return Raster(
        content=content,
        size=HeightWidth(meta["width"], meta["height"]),
        dtype=meta["dtype"],
        crs=meta["crs"].to_epsg(),
        bands=[i for i in range(1, meta["count"] + 1)],
        resolution=src.res[0],
        geometry=box(*src.bounds),
    )
