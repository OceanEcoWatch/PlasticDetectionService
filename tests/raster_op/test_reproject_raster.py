import numpy as np
import pytest

from src.models import Raster
from src.raster_op.abstractions import CompositeRasterOperation
from src.raster_op.reproject import RasterioRasterReproject


@pytest.mark.parametrize(
    "strategy",
    [
        RasterioRasterReproject(target_crs=4326, target_bands=[1]),
        CompositeRasterOperation(
            [RasterioRasterReproject(target_crs=4326, target_bands=[1])]
        ),
    ],
)
def test_reproject_raster(raster, rasterio_ds, strategy):
    target_crs = 4326
    target_bands = [1]

    reprojected_raster = strategy.execute(raster)

    assert reprojected_raster.crs == target_crs
    assert reprojected_raster.bands == target_bands
    assert isinstance(reprojected_raster, Raster)

    original_mean = rasterio_ds.read().mean()
    reprojected_mean = np.mean(reprojected_raster.to_numpy())
    assert np.isclose(
        original_mean, reprojected_mean, rtol=0.05
    )  # Allow a relative tolerance of 5%

    # check if the reprojected geometry coordinates are in degrees
    assert reprojected_raster.geometry.bounds[0] > -180
    assert reprojected_raster.geometry.bounds[2] < 180
    assert reprojected_raster.geometry.bounds[1] > -90
    assert reprojected_raster.geometry.bounds[3] < 90
