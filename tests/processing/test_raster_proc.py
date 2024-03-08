import numpy as np
import pytest
from osgeo import gdal
from shapely.geometry import Polygon

from plastic_detection_service.models import Raster
from plastic_detection_service.processing.abstractions import RasterProcessor
from plastic_detection_service.processing.gdal_proc import (
    GdalRasterProcessor,
)
from plastic_detection_service.processing.main import (
    RasterProcessingContext,
)

PROCESSORS = [GdalRasterProcessor(), RasterProcessingContext(GdalRasterProcessor())]


def test_get_epsg_from_ds(ds, crs):
    processor = GdalRasterProcessor()
    assert processor._get_epsg_from_ds(ds) == crs


def test_get_raster_geometry(ds, rast_geometry):
    processor = GdalRasterProcessor()
    assert processor._get_raster_geometry(ds) == rast_geometry


def test_ds_to_raster(ds, content, rast_geometry, crs):
    processor = GdalRasterProcessor()
    raster = processor._ds_to_raster(ds)
    assert raster.crs == crs
    assert raster.bands == [i for i in range(1, ds.RasterCount + 1)]
    assert raster.width == ds.RasterXSize
    assert raster.height == ds.RasterYSize
    assert raster.content == content
    assert raster.geometry == rast_geometry


@pytest.mark.parametrize("processor", PROCESSORS)
def test_reproject_raster(ds, raster: Raster, processor: RasterProcessor):
    reprojected_raster = processor.reproject_raster(raster, 4326, [1], "nearest")
    reprojected_raster.to_file("tests/assets/test_out_reprojected.tif")
    assert reprojected_raster.crs == 4326
    assert reprojected_raster.bands == [1]
    assert isinstance(reprojected_raster, Raster)

    # Compare the means of the rasters
    out_ds = gdal.Open("tests/assets/test_out_reprojected.tif")
    numpy_array = out_ds.GetRasterBand(1).ReadAsArray()
    original_mean = np.mean(numpy_array)
    reprojected_mean = np.mean(ds.GetRasterBand(1).ReadAsArray())
    assert np.isclose(
        original_mean, reprojected_mean, rtol=0.05
    )  # Allow a relative tolerance of 5%

    # check if the reprojected geometry coordinates are in degrees
    assert reprojected_raster.geometry.bounds[0] > -180
    assert reprojected_raster.geometry.bounds[2] < 180
    assert reprojected_raster.geometry.bounds[1] > -90
    assert reprojected_raster.geometry.bounds[3] < 90


@pytest.mark.parametrize("processor", PROCESSORS)
def test_to_vector(raster, processor: RasterProcessor):
    vectors = processor.to_vector(
        raster=raster,
        field="pixel_value",
        band=1,
    )
    vec = next(vectors)
    assert isinstance(vec.pixel_value, int)
    assert isinstance(vec.geometry, Polygon)

    # test if geometry is within the bounds of the raster
    assert vec.geometry.bounds[0] >= raster.geometry.bounds[0]
    assert vec.geometry.bounds[1] >= raster.geometry.bounds[1]
    assert vec.geometry.bounds[2] <= raster.geometry.bounds[2]
    assert vec.geometry.bounds[3] <= raster.geometry.bounds[3]


@pytest.mark.parametrize("processor", PROCESSORS)
def test_round_pixel_values(processor: RasterProcessor, raster: Raster, ds):
    rounded_raster = processor.round_pixel_values(raster, 10)

    assert rounded_raster.width == raster.width
    assert rounded_raster.height == raster.height
    assert rounded_raster.bands == raster.bands
    assert rounded_raster.crs == raster.crs
    assert rounded_raster.geometry == raster.geometry
    assert rounded_raster.content != raster.content

    # check if all values are rounded
    for i in range(len(raster.bands)):
        assert (rounded_raster.to_numpy()[i] % 10 == 0).all()
