import json

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
    RasterioRasterReproject,
    RasterioRasterSplit,
    RasterioRasterToVector,
    RasterioRasterUnpad,
    RasterioRemoveBand,
)


@pytest.mark.slow
@pytest.mark.e2e
def test_e2e(s2_l2a_raster, raster):
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
    for r in split_op.execute(s2_l2a_raster):
        result = comp_op.execute(r)
        results.append(result)

    merge_op = RasterioRasterMerge(offset=64, merge_method=copy_smooth)

    merged = merge_op.execute(results)
    merged = RasterioDtypeConversion(dtype="uint8").execute(merged)

    reprojected = RasterioRasterReproject(target_crs=4326, target_bands=[1]).execute(
        merged
    )

    reprojected.to_file("tests/assets/test_out_e2e.tif")
    vectors = list(RasterioRasterToVector(band=1).execute(reprojected))

    features = [vec.geojson for vec in vectors]
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }
    with open("tests/assets/test_out_e2e.geojson", "w") as f:
        json.dump(geojson, f, indent=2)

    assert all(isinstance(vec.pixel_value, int) for vec in vectors)

    assert all(vec.crs == reprojected.crs for vec in vectors)
    assert all(
        vec.geometry.bounds[0] >= reprojected.geometry.bounds[0] for vec in vectors
    )
    assert all(
        vec.geometry.bounds[1] >= reprojected.geometry.bounds[1] for vec in vectors
    )
    assert all(
        vec.geometry.bounds[2] <= reprojected.geometry.bounds[2] for vec in vectors
    )
    assert all(
        vec.geometry.bounds[3] <= reprojected.geometry.bounds[3] for vec in vectors
    )

    # check pixel values are within the expected range
    assert all(vec.pixel_value >= 0 for vec in vectors)
    assert any(vec.pixel_value > 0 for vec in vectors)
    assert all(vec.pixel_value <= 255 for vec in vectors)

    # assert mean value is close to the original raster
    assert np.isclose(merged.to_numpy().mean(), raster.to_numpy().mean(), rtol=0.05)


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
