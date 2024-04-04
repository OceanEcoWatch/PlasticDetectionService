import numpy as np
import pytest

from src.inference.inference_callback import LocalInferenceCallback
from src.raster_op.abstractions import CompositeRasterOperation
from src.raster_op.convert import RasterioDtypeConversion
from src.raster_op.inference import RasterInference
from src.raster_op.merge import (
    RasterioRasterMerge,
    copy_smooth,
)
from src.raster_op.padding import RasterioRasterPad, RasterioRasterUnpad
from src.raster_op.shape import RasterioRemoveBand
from src.raster_op.split import RasterioRasterSplit
from src.types import HeightWidth
from tests.conftest import MockInferenceCallback


def test_composite_raster_operation(s2_l2a_raster):
    split_op = RasterioRasterSplit(image_size=HeightWidth(480, 480), offset=64)
    comp_op = CompositeRasterOperation(
        [
            RasterioRasterPad(padding=64),
            RasterioRemoveBand(band=13),
            RasterInference(inference_func=MockInferenceCallback()),
            RasterioRasterUnpad(),
        ]
    )
    results = []
    for raster in split_op.execute(s2_l2a_raster):
        result = comp_op.execute(raster)
        results.append(result)

    merge_op = RasterioRasterMerge(offset=64, merge_method=copy_smooth)

    merged = merge_op.execute(results)

    merged.to_file(
        f"tests/assets/test_out_composite_{MockInferenceCallback.__name__}.tif"
    )
    assert merged.size == s2_l2a_raster.size
    assert merged.dtype == "float32"
    assert merged.crs == s2_l2a_raster.crs
    assert merged.bands == [1]
    assert isinstance(merged.content, bytes)
    assert merged.padding_size == (0, 0)
    assert merged.geometry == s2_l2a_raster.geometry


@pytest.mark.slow
def test_composite_raster_real_inference(s2_l2a_raster, raster):
    split_op = RasterioRasterSplit(image_size=HeightWidth(480, 480), offset=64)
    comp_op = CompositeRasterOperation(
        [
            RasterioRasterPad(padding=64),
            RasterioRemoveBand(band=13),
            RasterInference(inference_func=LocalInferenceCallback()),
            RasterioRasterUnpad(),
        ]
    )
    results = []
    for r in split_op.execute(s2_l2a_raster):
        result = comp_op.execute(r)
        results.append(result)

    merge_op = RasterioRasterMerge(offset=64, merge_method=copy_smooth)

    merged = merge_op.execute(results)
    merged = RasterioDtypeConversion(dtype="uint8").execute(merged)
    merged.to_file(
        f"tests/assets/test_out_composite_{LocalInferenceCallback.__name__}.tif"
    )

    assert merged.size == raster.size
    assert merged.dtype == raster.dtype
    assert merged.crs == raster.crs
    assert merged.bands == [1]
    assert isinstance(merged.content, bytes)
    assert merged.padding_size == (0, 0)
    assert merged.geometry == raster.geometry
    assert np.isclose(merged.to_numpy().mean(), raster.to_numpy().mean(), rtol=0.05)
