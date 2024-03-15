import pytest
import rasterio
from shapely.geometry import Polygon, box

from plastic_detection_service.models import Raster, Vector


@pytest.fixture
def s2_l2a_response():
    with open("tests/assets/test_response.tiff", "rb") as f:
        return f.read()


@pytest.fixture
def s2_l2a_rasterio():
    with rasterio.open("tests/assets/test_response.tiff") as src:
        image = src.read()
        meta = src.meta.copy()
        return src, image, meta


@pytest.fixture
def s2_l2a_raster(s2_l2a_rasterio, s2_l2a_response):
    src, image, meta = s2_l2a_rasterio

    return Raster(
        content=s2_l2a_response,
        size=(meta["width"], meta["height"]),
        dtype=meta["dtype"],
        crs=meta["crs"].to_epsg(),
        bands=[i for i in range(1, meta["count"] + 1)],
        geometry=box(*src.bounds),
    )


@pytest.fixture
def content():
    with open("tests/assets/test_exp_pred.tif", "rb") as f:
        return f.read()


@pytest.fixture
def rasterio_ds():
    src = rasterio.open("tests/assets/test_exp_pred.tif")
    yield src
    src.close()


@pytest.fixture
def crs(rasterio_ds) -> int:
    crs = rasterio_ds.crs
    return crs.to_epsg()


@pytest.fixture
def rast_geometry(rasterio_ds):
    return box(*rasterio_ds.bounds)


@pytest.fixture
def raster(content, rasterio_ds, crs, rast_geometry):
    return Raster(
        content=content,
        size=(rasterio_ds.width, rasterio_ds.height),
        dtype=rasterio_ds.meta["dtype"],
        crs=crs,
        bands=[i for i in range(1, rasterio_ds.count + 1)],
        geometry=rast_geometry,
    )


@pytest.fixture
def vector():
    return Vector(
        geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), pixel_value=5, crs=4326
    )
