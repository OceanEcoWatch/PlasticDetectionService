import numpy as np
import pytest

from src.inference.inference_callback import (
    LocalInferenceCallback,
    RunpodInferenceCallback,
)
from src.models import Raster
from src.raster_op.inference import RasterInference
from src.raster_op.padding import RasterioRasterPad, RasterioRasterUnpad
from src.raster_op.shape import RasterioRemoveBand
from src.raster_op.split import RasterioRasterSplit
from src.types import HeightWidth
from tests.conftest import MockInferenceCallback


def test_inference_raster_mock(s2_l2a_raster):
    operation = RasterInference(inference_func=MockInferenceCallback())

    result = operation.execute(s2_l2a_raster)

    assert isinstance(result, Raster)

    assert result.size == s2_l2a_raster.size
    assert result.dtype == "float32"
    assert result.crs == s2_l2a_raster.crs
    assert result.bands == [1]
    assert result.geometry == s2_l2a_raster.geometry

    assert isinstance(result.content, bytes)


@pytest.mark.slow
def test_inference_raster_real(s2_l2a_raster, pred_durban_first_split_raster):
    raster = next(
        RasterioRasterSplit(image_size=HeightWidth(480, 480), offset=64).execute(
            s2_l2a_raster
        )
    )
    raster = RasterioRasterPad(padding=64).execute(raster)
    raster = RasterioRemoveBand(band=13).execute(raster)
    inference_raster = RasterInference(inference_func=LocalInferenceCallback()).execute(
        raster
    )

    inference_raster = RasterioRasterUnpad().execute(inference_raster)
    inference_raster.to_file("tests/assets/test_out_inference.tif")
    assert isinstance(inference_raster, Raster)

    # assert pixel values are within the expected range
    assert np.all(inference_raster.to_numpy() >= 0)
    assert np.any(inference_raster.to_numpy() > 0)
    assert np.all(inference_raster.to_numpy() <= 1)

    assert inference_raster.size == pred_durban_first_split_raster.size
    assert inference_raster.crs == pred_durban_first_split_raster.crs


@pytest.mark.slow
@pytest.mark.integration
def test_inference_raster_real_runpod(s2_l2a_raster, pred_durban_first_split_raster):
    raster = next(
        RasterioRasterSplit(image_size=HeightWidth(480, 480), offset=64).execute(
            s2_l2a_raster
        )
    )
    raster = RasterioRasterPad(padding=64).execute(raster)
    raster = RasterioRemoveBand(band=13).execute(raster)
    inference_raster = RasterInference(
        inference_func=RunpodInferenceCallback()
    ).execute(raster)

    inference_raster = RasterioRasterUnpad().execute(inference_raster)
    inference_raster.to_file("tests/assets/test_out_inference_runpod.tif")
    assert isinstance(inference_raster, Raster)

    # assert pixel values are within the expected range
    assert np.all(inference_raster.to_numpy() >= 0)
    assert np.any(inference_raster.to_numpy() > 0)
    assert np.all(inference_raster.to_numpy() <= 1)

    assert inference_raster.size == pred_durban_first_split_raster.size
    assert inference_raster.crs == pred_durban_first_split_raster.crs
