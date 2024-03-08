import numpy as np
import pytest
from osgeo import gdal
from shapely.geometry import Polygon, box

from plastic_detection_service.models import Raster
from plastic_detection_service.processing.abstractions import RasterProcessor
from plastic_detection_service.processing.gdal_proc import (
    GdalRasterProcessor,
)
from plastic_detection_service.processing.main import RasterProcessingContext

PROCESSORS = [GdalRasterProcessor(), RasterProcessingContext(GdalRasterProcessor())]


@pytest.fixture
def content():
    with open("tests/assets/test_exp_pred.tif", "rb") as f:
        return f.read()


@pytest.fixture
def ds(content):
    _temp_file = "/vsimem/temp.tif"
    gdal.FileFromMemBuffer(_temp_file, content)
    yield gdal.Open(_temp_file)

    gdal.Unlink(_temp_file)


@pytest.fixture
def crs(ds):
    srs = gdal.osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    return int(srs.GetAttrValue("AUTHORITY", 1))


@pytest.fixture
def rast_geometry(ds):
    gt = ds.GetGeoTransform()

    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + gt[1] * ds.RasterXSize
    ymin = ymax + gt[5] * ds.RasterYSize
    return box(xmin, ymin, xmax, ymax)


@pytest.fixture
def raster(content, ds, crs, rast_geometry):
    return Raster(
        content=content,
        width=ds.RasterXSize,
        height=ds.RasterYSize,
        crs=crs,
        bands=[i for i in range(1, ds.RasterCount + 1)],
        geometry=rast_geometry,
    )


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
    reprojected_raster = processor.reproject_raster(raster, 4326, [1])
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
def test_to_vector(ds, raster, processor: RasterProcessor):
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
