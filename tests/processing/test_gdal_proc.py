import numpy as np
import pytest
from osgeo import gdal
from shapely.geometry import Polygon, box

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


def test_ds_to_raster(ds, content, rast_geometry, crs):
    processor = GdalRasterProcessor()
    raster = processor._ds_to_raster(ds)
    assert raster.crs == crs
    assert raster.bands == [i for i in range(1, ds.RasterCount + 1)]
    assert raster.width == ds.RasterXSize
    assert raster.height == ds.RasterYSize
    assert raster.content == content
    assert raster.geometry == rast_geometry


def test_reproject_raster(ds, raster: Raster):
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

    # check if the reprojected geometry coordinates are in degrees
    assert reprojected_raster.geometry.bounds[0] > -180
    assert reprojected_raster.geometry.bounds[2] < 180
    assert reprojected_raster.geometry.bounds[1] > -90
    assert reprojected_raster.geometry.bounds[3] < 90


def test_to_vector(ds, raster):
    processor = GdalRasterProcessor()
    vectors = processor.to_vector(
        raster=raster,
        field="pixel_value",
        band=1,
    )
    vec = next(vectors)
    assert isinstance(vec.pixel_value, int)
    assert isinstance(vec.geometry, Polygon)

    # test if geometry is within the bounds of the raster


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
