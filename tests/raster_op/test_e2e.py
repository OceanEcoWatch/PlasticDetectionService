import json

import numpy as np
import pytest

from src.inference.inference_callback import (
    RunpodInferenceCallback,
)
from src.raster_op.band import RasterioRemoveBand
from src.raster_op.composite import CompositeRasterOperation
from src.raster_op.convert import RasterioDtypeConversion
from src.raster_op.inference import RasterioInference
from src.raster_op.merge import RasterioRasterMerge, copy_smooth
from src.raster_op.padding import RasterioRasterPad, RasterioRasterUnpad
from src.raster_op.reproject import RasterioRasterReproject
from src.raster_op.split import RasterioRasterSplit
from src.raster_op.vectorize import RasterioRasterToVector


@pytest.mark.slow
@pytest.mark.e2e
@pytest.mark.parametrize("inference_func", [RunpodInferenceCallback()])
def test_e2e(s2_l2a_raster, raster, inference_func, expected_vectors):
    comp_op = CompositeRasterOperation()
    comp_op.add(RasterioRasterSplit())
    comp_op.add(RasterioRasterPad())
    comp_op.add(RasterioRemoveBand(band=13))
    comp_op.add(RasterioInference(inference_func=inference_func))
    comp_op.add(RasterioRasterUnpad())
    comp_op.add(RasterioRasterMerge(merge_method=copy_smooth))
    comp_op.add(RasterioDtypeConversion(dtype="uint8"))

    merged = list(comp_op.execute([s2_l2a_raster]))
    assert len(merged) == 1
    merged = merged[0]

    reprojected = next(
        RasterioRasterReproject(target_crs=4326, target_bands=[1]).execute([merged])
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

    assert merged.dtype == raster.dtype
    # assert mean value is close to the original raster
    assert np.isclose(merged.to_numpy().mean(), raster.to_numpy().mean(), rtol=0.05)

    # assert vectors are almost equal to the expected vectors
    assert len(vectors) == len(expected_vectors)
    for vec, exp_vec in zip(vectors, expected_vectors):
        assert vec.pixel_value == exp_vec.pixel_value
        assert vec.geometry.bounds == exp_vec.geometry.bounds


@pytest.mark.slow
@pytest.mark.e2e
@pytest.mark.skip("This test is too slow for CI")
def test_e2e_full_durban_scene(durban_full_raster):
    comp_op = CompositeRasterOperation()
    comp_op.add(RasterioRasterSplit())
    comp_op.add(RasterioRasterPad())
    comp_op.add(RasterioRemoveBand(band=13))
    comp_op.add(RasterioInference(inference_func=RunpodInferenceCallback()))
    comp_op.add(RasterioRasterUnpad())
    comp_op.add(RasterioRasterMerge(merge_method=copy_smooth))
    comp_op.add(RasterioDtypeConversion(dtype="uint8"))

    merged = list(comp_op.execute([durban_full_raster]))
    assert len(merged) == 1
    merged = merged[0]
    merged.to_file(
        "tests/assets/test_out_pred_durban_full.tif",
    )
    assert merged.size == durban_full_raster.size
    assert merged.crs == durban_full_raster.crs
    assert np.allclose(merged.to_numpy(), durban_full_raster.to_numpy())
