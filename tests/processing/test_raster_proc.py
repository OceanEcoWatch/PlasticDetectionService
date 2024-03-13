import io

import numpy as np
import pytest
import rasterio
from osgeo import gdal
from shapely.geometry import Polygon

from plastic_detection_service.models import Raster
from plastic_detection_service.processing.abstractions import RasterProcessor
from plastic_detection_service.processing.context import (
    RasterProcessingContext,
)
from plastic_detection_service.processing.gdal_proc import (
    GdalRasterProcessor,
)
from plastic_detection_service.processing.rasterio_proc import RasterioRasterProcessor

PROCESSORS = [GdalRasterProcessor(), RasterProcessingContext(GdalRasterProcessor())]


def test_get_gdal_ds_from_memory(content, crs):
    processor = GdalRasterProcessor()

    ds = processor._get_gdal_ds_from_memory(content)
    assert isinstance(ds, gdal.Dataset)
    assert ds.RasterCount == 1
    assert ds.RasterXSize == 500
    assert ds.RasterYSize == 500

    srs = gdal.osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    assert int(srs.GetAttrValue("AUTHORITY", 1)) == crs


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
    assert raster.size == (ds.RasterXSize, ds.RasterYSize)
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
    assert vec.crs == raster.crs

    # test if geometry is within the bounds of the raster
    assert vec.geometry.bounds[0] >= raster.geometry.bounds[0]
    assert vec.geometry.bounds[1] >= raster.geometry.bounds[1]
    assert vec.geometry.bounds[2] <= raster.geometry.bounds[2]
    assert vec.geometry.bounds[3] <= raster.geometry.bounds[3]


@pytest.mark.parametrize("processor", [RasterioRasterProcessor()])
def test_split_pad_raster(s2_l2a_raster, processor: RasterioRasterProcessor):
    exp_np = np.load("tests/assets/test_split_pad_image.npy")
    split_raster = next(
        processor.split_pad_raster(s2_l2a_raster, image_size=(480, 480), padding=64)
    )

    split_raster.to_file("tests/assets/test_out_split_pad.tif")

    assert split_raster.size == (608, 608)
    assert split_raster.crs == s2_l2a_raster.crs
    assert split_raster.bands == s2_l2a_raster.bands[:12]
    assert split_raster.content != s2_l2a_raster.content
    assert isinstance(split_raster.content, bytes)

    assert split_raster.to_numpy().shape == exp_np.shape
    assert split_raster.to_numpy().dtype == exp_np.dtype

    assert np.array_equal(split_raster.to_numpy(), exp_np)

    #
    with rasterio.open("tests/assets/test_exp_split_pad.tif") as exp_src:
        exp_image = exp_src.read()
        with rasterio.open(io.BytesIO(split_raster.content)) as src:
            image = src.read()

            assert image.shape == exp_image.shape
            assert image.dtype == exp_image.dtype
            assert src.meta["height"] == exp_src.meta["height"]
            assert src.meta["width"] == exp_src.meta["width"]
            assert src.meta["crs"] == exp_src.meta["crs"]
            assert src.meta["count"] == exp_src.meta["count"]
            assert src.meta["transform"] == exp_src.meta["transform"]
            assert src.meta["dtype"] == exp_src.meta["dtype"]
            assert src.meta["nodata"] == exp_src.meta["nodata"]

            assert np.array_equal(image, exp_image)


@pytest.mark.parametrize("processor", [RasterioRasterProcessor()])
def test_split_raster(s2_l2a_raster, processor):
    split_raster = next(
        processor.split_raster(s2_l2a_raster, image_size=(480, 480), offset=64)
    )

    split_raster.to_file("tests/assets/test_out_split.tif")

    assert split_raster.size == (480, 480)
    assert split_raster.crs == s2_l2a_raster.crs
    assert split_raster.bands == s2_l2a_raster.bands
    assert split_raster.content != s2_l2a_raster.content
    assert isinstance(split_raster.content, bytes)

    with rasterio.open("tests/assets/test_exp_split.tif") as exp_src:
        exp_image = exp_src.read()
        with rasterio.open(io.BytesIO(split_raster.content)) as src:
            image = src.read()

            assert image.shape == exp_image.shape
            assert image.dtype == exp_image.dtype
            assert src.meta["height"] == exp_src.meta["height"]
            assert src.meta["width"] == exp_src.meta["width"]
            assert src.meta["crs"] == exp_src.meta["crs"]
            assert src.meta["count"] == exp_src.meta["count"]
            assert src.meta["transform"] == exp_src.meta["transform"]
            assert src.meta["dtype"] == exp_src.meta["dtype"]
            assert src.meta["nodata"] == exp_src.meta["nodata"]

            assert np.array_equal(image, exp_image)


@pytest.mark.parametrize("processor", [RasterioRasterProcessor()])
def test_pad_raster(s2_l2a_raster, processor: RasterioRasterProcessor):
    padding = 64
    image_size = (480, 480)
    padded_raster = processor.pad_raster(
        s2_l2a_raster, image_size=image_size, padding=padding
    )

    padded_raster.to_file("tests/assets/test_out_pad.tif")

    assert padded_raster.size == (608, 608)
    assert padded_raster.crs == s2_l2a_raster.crs
    assert padded_raster.bands == s2_l2a_raster.bands
    assert padded_raster.content != s2_l2a_raster.content
    assert isinstance(padded_raster.content, bytes)

    # check padding_size is as expected
    exp_padding_size = processor._calculate_padding_size(
        s2_l2a_raster.to_numpy(), image_size, padding
    )
    assert padded_raster.padding_size == exp_padding_size

    with rasterio.open("tests/assets/test_exp_pad.tif") as exp_src:
        exp_image = exp_src.read()
        with rasterio.open(io.BytesIO(padded_raster.content)) as src:
            image = src.read()

            assert image.shape == exp_image.shape
            assert image.dtype == exp_image.dtype
            assert src.meta["height"] == exp_src.meta["height"]
            assert src.meta["width"] == exp_src.meta["width"]
            assert src.meta["crs"] == exp_src.meta["crs"]
            assert src.meta["count"] == exp_src.meta["count"]
            assert src.meta["transform"] == exp_src.meta["transform"]
            assert src.meta["dtype"] == exp_src.meta["dtype"]
            assert src.meta["nodata"] == exp_src.meta["nodata"]

            assert np.array_equal(image, exp_image)


@pytest.mark.parametrize("processor", [RasterioRasterProcessor()])
def test_unpad_raster(s2_l2a_raster, processor: RasterioRasterProcessor):
    padded_raster = processor.pad_raster(
        s2_l2a_raster, image_size=(480, 480), padding=64
    )
    unpadded_raster = processor.unpad_raster(padded_raster)

    unpadded_raster.to_file("tests/assets/test_out_unpad.tif")

    # check if the unpadded raster is the same as the original raster
    assert unpadded_raster.size == s2_l2a_raster.size
    assert unpadded_raster.crs == s2_l2a_raster.crs
    assert unpadded_raster.bands == s2_l2a_raster.bands

    assert isinstance(unpadded_raster.content, bytes)
    assert unpadded_raster.padding_size == (0, 0)
    assert unpadded_raster.geometry == s2_l2a_raster.geometry
    assert np.array_equal(unpadded_raster.to_numpy(), s2_l2a_raster.to_numpy())
