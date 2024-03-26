import numpy as np
import pytest

from plastic_detection_service.inference.inference_callback import (
    local_inference_callback,
)
from plastic_detection_service.processing.merge_callable import copy_smooth
from plastic_detection_service.processing.raster_operations import (
    CompositeRasterOperation,
    HeightWidth,
    RasterInference,
    RasterioDtypeConversion,
    RasterioRasterMerge,
    RasterioRasterPad,
    RasterioRasterSplit,
    RasterioRasterUnpad,
    RasterioRemoveBand,
)


@pytest.mark.slow
@pytest.mark.e2e
def test_e2e_full_durban_scene(durban_full_raster):
    split_op = RasterioRasterSplit(image_size=HeightWidth(480, 480), offset=64)
    comp_op = CompositeRasterOperation(
        [
            RasterioRasterPad(padding=64),
            RasterioRemoveBand(band=13),
            RasterInference(inference_func=local_inference_callback),
            RasterioRasterUnpad(),
        ]
    )
    results = []
    for r in split_op.execute(durban_full_raster):
        result = comp_op.execute(r)
        results.append(result)

    merge_op = RasterioRasterMerge(offset=64, merge_method=copy_smooth)

    merged = merge_op.execute(results)
    merged = RasterioDtypeConversion(dtype="uint8").execute(merged)
    merged.to_file(
        "tests/assets/test_out_pred_durban_full.tif",
    )
    assert merged.size == durban_full_raster.size
    assert merged.crs == durban_full_raster.crs
    assert np.allclose(merged.to_numpy(), durban_full_raster.to_numpy())
