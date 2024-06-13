import io
import json

import numpy as np
import pytest
import rasterio
import requests
import torch
from shapely.geometry import Polygon, box, shape

from src._types import HeightWidth
from src.dt_util import get_past_date, get_today_str
from src.inference.inference_callback import BaseInferenceCallback
from src.models import Raster, Vector
from tests.marinedebrisdetector_mod.checkpoints import CHECKPOINTS
from tests.marinedebrisdetector_mod.model.segmentation_model import (
    SegmentationModel,
)
from tests.marinedebrisdetector_mod.predictor import predict

FULL_DURBAN_SCENE = "https://marinedebrisdetector.s3.eu-central-1.amazonaws.com/data/durban_20190424.tif"

TIME_INTERVAL = (get_past_date(7), get_today_str())
TEST_AOI = (
    120.82481750015815,
    14.619576605802296,
    120.82562856620629,
    14.66462165084734,
)  # manilla bay

TEST_AOI_POLYGON = Polygon(
    [
        (120.82481750015815, 14.619576605802296),
        (120.82562856620629, 14.619576605802296),
        (120.82562856620629, 14.66462165084734),
        (120.82481750015815, 14.66462165084734),
        (120.82481750015815, 14.619576605802296),
    ]
)
MAX_CC = 0.1

RUNPOD_ENDPOINT_ID = "2qzxx0ljdkepts"


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
    src, _, meta = s2_l2a_rasterio

    return Raster(
        content=s2_l2a_response,
        size=HeightWidth(meta["width"], meta["height"]),
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
    src, _, meta = pred_durban_first_split_rasterio

    return Raster(
        content=pred_durban_first_split,
        size=HeightWidth(meta["width"], meta["height"]),
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
        size=HeightWidth(rasterio_ds.meta["width"], rasterio_ds.meta["height"]),
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


@pytest.fixture
def expected_vectors():
    with open("tests/assets/test_exp_vectors.geojson", "r") as f:
        geojson = json.load(f)
    vectors = []
    for feature in geojson["features"]:
        vectors.append(
            Vector(
                geometry=shape(feature["geometry"]),
                pixel_value=int(feature["properties"]["pixel_value"]),
                crs=4326,
            )
        )
    return vectors


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
    src, _, meta = durban_rasterio_ds

    return Raster(
        content=durban_content,
        size=HeightWidth(meta["width"], meta["height"]),
        dtype=meta["dtype"],
        crs=meta["crs"].to_epsg(),
        bands=[i for i in range(1, meta["count"] + 1)],
        resolution=src.res[0],
        geometry=box(*src.bounds),
    )


class MockInferenceCallback(BaseInferenceCallback):
    def __call__(self, payload: bytes) -> bytes:
        with rasterio.open(io.BytesIO(payload)) as src:
            image = src.read()
            band1 = image[0, :, :].astype(np.float32)
            return band1.tobytes()


class LocalInferenceCallback(BaseInferenceCallback):
    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def model(self):
        model = SegmentationModel.load_from_checkpoint(
            checkpoint_path=CHECKPOINTS["unet++1"],
            strict=False,
            map_location=self.device,
        )
        return model.to(self.device).eval()

    def __call__(self, payload: bytes) -> bytes:
        with rasterio.open(io.BytesIO(payload)) as src:
            image = src.read()
        pred_array = predict(self.model, image, device=self.device)
        return pred_array.tobytes()
