import numpy as np
import pytest
from osgeo import gdal

from plastic_detection_service.models import Raster
from plastic_detection_service.processing.gdal_proc import (
    GdalRasterProcessor,
)


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


def crs(ds):
    srs = gdal.osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    return int(srs.GetAttrValue("AUTHORITY", 1))


def test_reproject_raster(content, ds):
    raster = Raster(
        content=content,
        width=ds.RasterXSize,
        height=ds.RasterYSize,
        crs=crs(ds),
        bands=[i for i in range(1, ds.RasterCount + 1)],
    )
    processor = GdalRasterProcessor()
    reprojected_raster = processor.reproject_raster(raster, 4326, [1])
    reprojected_raster.to_file("tests/assets/test_out_reprojected.tif")
    assert reprojected_raster.crs == 4326
    assert reprojected_raster.bands == [1]

    # Compare the means of the rasters
    out_ds = gdal.Open("tests/assets/test_out_reprojected.tif")
    numpy_array = out_ds.GetRasterBand(1).ReadAsArray()
    original_mean = np.mean(numpy_array)
    reprojected_mean = np.mean(ds.GetRasterBand(1).ReadAsArray())
    assert np.isclose(
        original_mean, reprojected_mean, rtol=0.05
    )  # Allow a relative tolerance of 5%


def test_ds_to_raster(ds, content):
    processor = GdalRasterProcessor()
    raster = processor._ds_to_raster(ds)
    assert raster.crs == crs(ds)
    assert raster.bands == [i for i in range(1, ds.RasterCount + 1)]
    assert raster.width == ds.RasterXSize
    assert raster.height == ds.RasterYSize
    assert raster.content == content


# def test_to_vector():
#     gdal_raster = GdalRaster.from_memory(
#         open("images/5cb12a6cbd6df0865947f21170bc432a/response.tiff", "rb").read()
#     )
#     processor = GdalRasterProcessor()
#     reprojected_raster = processor.to_vector(gdal_raster, "pixel_value")

#     for feature in reprojected_raster.ds.GetLayer():
#         print(feature.GetField("pixel_value"))

#     # save
#     with open("response_wgs84_test.geojson", "w") as f:
#         schema = {
#             "geometry": "Polygon",
#             "properties": {"pixel_value": "int"},
#         }
#         feature_collection = geojson.FeatureCollection([])
#         for feature in ds.GetLayer():
#             pixel_value = int(feature.GetField("pixel_value"))
#             geometry = json.loads(feature.ExportToJson())["geometry"]
#             feature_collection["features"].append(
#                 geojson.Feature(
#                     geometry=geometry, properties={"pixel_value": pixel_value}
#                 )
#             )
#         f.write(geojson.dumps(feature_collection))
