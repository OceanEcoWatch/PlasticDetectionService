import pytest
import rasterio
from osgeo import gdal
from shapely.geometry import Polygon, box

from plastic_detection_service.models import Raster, Vector


@pytest.fixture
def s2_l2a_response():
    with open("tests/assets/test_response.tiff", "rb") as f:
        return f.read()


@pytest.fixture
def s2_l2a_rasterio():
    with rasterio.open("tests/assets/test_response.tiff") as src:
        image = src.read()
        meta = src.meta.copy()
        return src, image, meta


@pytest.fixture
def s2_l2a_raster(s2_l2a_rasterio, s2_l2a_response):
    src, image, meta = s2_l2a_rasterio

    return Raster(
        content=s2_l2a_response,
        size=(meta["width"], meta["height"]),
        crs=meta["crs"].to_epsg(),
        bands=[i for i in range(1, meta["count"] + 1)],
        geometry=box(*src.bounds),
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
        size=(ds.RasterXSize, ds.RasterYSize),
        crs=crs,
        bands=[i for i in range(1, ds.RasterCount + 1)],
        geometry=rast_geometry,
    )


@pytest.fixture
def vector():
    return Vector(
        geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), pixel_value=5, crs=4326
    )
