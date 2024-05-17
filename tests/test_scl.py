import pytest
import rasterio
from shapely.geometry import box

from src._types import HeightWidth
from src.models import Raster
from src.scl import SCL, get_scl_vectors


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


def test_get_scl_vectors(scl_raster):
    vectors = list(get_scl_vectors(scl_raster, band=13))
    assert len(vectors) > 100
    for vector in vectors:
        assert vector.pixel_value >= SCL.min() and vector.pixel_value <= SCL.max()
        assert vector.geometry.geom_type == "Polygon"
        assert vector.crs == 4326
