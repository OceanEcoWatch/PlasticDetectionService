import io

import pytest
import rasterio
import requests
from shapely.geometry import Polygon, box

from src.models import Raster, Vector

FULL_DURBAN_SCENE = "https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/data/durban_20190424.tif"


@pytest.fixture
def s2_l2a_response():
    with open("tests/assets/test_exp_response_durban20190424.tiff", "rb") as f:
        return f.read()


@pytest.fixture
def s2_l2a_rasterio():
    with rasterio.open("tests/assets/test_exp_response_durban20190424.tiff") as src:
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
        resolution=src.res[0],
        geometry=box(*src.bounds),
    )


@pytest.fixture
def pred_durban_first_split():
    with open(
        "tests/assets/test_exp_pred_durban_first_split.tif",
        "rb",
    ) as f:
        return f.read()


@pytest.fixture
def pred_durban_first_split_rasterio():
    with rasterio.open("tests/assets/test_exp_pred_durban_first_split.tif") as src:
        image = src.read()
        meta = src.meta.copy()
        return src, image, meta


@pytest.fixture
def pred_durban_first_split_raster(
    pred_durban_first_split_rasterio, pred_durban_first_split
):
    src, image, meta = pred_durban_first_split_rasterio

    return Raster(
        content=pred_durban_first_split,
        size=(meta["width"], meta["height"]),
        dtype=meta["dtype"],
        crs=meta["crs"].to_epsg(),
        bands=[i for i in range(1, meta["count"] + 1)],
        resolution=src.res[0],
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
        size=(rasterio_ds.meta["width"], rasterio_ds.meta["height"]),
        dtype=rasterio_ds.meta["dtype"],
        crs=crs,
        bands=[i for i in range(1, rasterio_ds.count + 1)],
        resolution=rasterio_ds.res[0],
        geometry=rast_geometry,
    )


@pytest.fixture
def vector():
    return Vector(
        geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), pixel_value=5, crs=4326
    )


@pytest.fixture(scope="session")
def durban_content():
    return requests.get(FULL_DURBAN_SCENE).content


@pytest.fixture
def durban_rasterio_ds(durban_content):
    with rasterio.open(io.BytesIO(durban_content)) as src:
        image = src.read()
        meta = src.meta.copy()
        return src, image, meta


@pytest.fixture
def durban_full_raster(durban_rasterio_ds, durban_content):
    src, image, meta = durban_rasterio_ds

    return Raster(
        content=durban_content,
        size=(meta["width"], meta["height"]),
        dtype=meta["dtype"],
        crs=meta["crs"].to_epsg(),
        bands=[i for i in range(1, meta["count"] + 1)],
        resolution=src.res[0],
        geometry=box(*src.bounds),
    )
