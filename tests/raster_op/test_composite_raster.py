import numpy as np
import pytest

from src._types import HeightWidth
from src.inference.inference_callback import RunpodInferenceCallback
from src.raster_op.band import RasterioRemoveBand
from src.raster_op.composite import CompositeRasterOperation
from src.raster_op.convert import RasterioDtypeConversion
from src.raster_op.inference import RasterioInference
from src.raster_op.merge import (
    RasterioRasterMerge,
    copy_smooth,
)
from src.raster_op.padding import RasterioRasterPad, RasterioRasterUnpad
from src.raster_op.split import RasterioRasterSplit
from tests import conftest
from tests.conftest import LocalInferenceCallback, MockInferenceCallback


def test_all_raster_op_without_comp(s2_l2a_raster):
    rasters = [s2_l2a_raster]
    splitted_rasters = list(RasterioRasterSplit().execute(rasters))
    padded_rasters = list(RasterioRasterPad().execute(splitted_rasters))
    assert len(padded_rasters) == len(splitted_rasters)
    inferred_rasters = list(
        RasterioInference(
            inference_func=MockInferenceCallback(), output_dtype="float32"
        ).execute(padded_rasters)
    )
    assert len(inferred_rasters) == len(padded_rasters)
    unpadded_rasters = list(RasterioRasterUnpad().execute(inferred_rasters))
    assert len(unpadded_rasters) == len(padded_rasters)
    merged_raster = list(RasterioRasterMerge().execute(unpadded_rasters))
    assert len(merged_raster) == 1


def test_composite_raster_operation(s2_l2a_raster):
    root_op = CompositeRasterOperation()
    root_op.add(RasterioRasterSplit())
    root_op.add(RasterioRasterPad(padding=64))
    root_op.add(RasterioRemoveBand(band=13))
    root_op.add(
        RasterioInference(
            inference_func=MockInferenceCallback(),
            output_dtype="float32",
        )
    )
    root_op.add(RasterioRasterUnpad())
    root_op.add(RasterioRasterMerge(merge_method=copy_smooth))

    merged = list(root_op.execute([s2_l2a_raster]))
    assert len(merged) == 1
    merged = merged[0]
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
@pytest.mark.integration
def test_composite_raster_real_inference(s2_l2a_raster, raster):
    comp_op = CompositeRasterOperation()
    comp_op.add(RasterioRasterSplit(image_size=HeightWidth(480, 480), offset=64))
    comp_op.add(RasterioRasterPad(padding=64))
    comp_op.add(RasterioRemoveBand(band=13))
    comp_op.add(
        RasterioInference(
            inference_func=RunpodInferenceCallback(conftest.RUNPOD_ENDPOINT_ID),
            output_dtype="uint8",
        )
    )
    comp_op.add(RasterioRasterUnpad())
    comp_op.add(RasterioRasterMerge(offset=64, merge_method=copy_smooth))

    _merged = list(comp_op.execute([s2_l2a_raster]))
    assert len(_merged) == 1
    merged = _merged[0]

    merged = next(RasterioDtypeConversion(dtype="uint8").execute([merged]))
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
