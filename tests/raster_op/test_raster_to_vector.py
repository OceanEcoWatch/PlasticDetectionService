from shapely.geometry import Point, Polygon

from src.models import Raster
from src.raster_op.vectorize import RasterioRasterToPoint, RasterioRasterToPolygon


def test_to_point(raster: Raster):
    strategy = RasterioRasterToPoint(band=1)
    vectors = strategy.execute(
        raster=raster,
    )

    for vec in vectors:
        assert isinstance(vec.pixel_value, int), "Pixel value is not an integer"
        assert isinstance(vec.geometry, Point), "Geometry is not a Point"
        assert vec.crs == raster.crs

        # test if geometry is within the bounds of the raster
        assert vec.geometry.bounds[0] >= raster.geometry.bounds[0]
        assert vec.geometry.bounds[1] >= raster.geometry.bounds[1]
        assert vec.geometry.bounds[2] <= raster.geometry.bounds[2]
        assert vec.geometry.bounds[3] <= raster.geometry.bounds[3]


def test_to_polygon(raster: Raster):
    strategy = RasterioRasterToPolygon(band=1)
    vectors = strategy.execute(
        raster=raster,
    )

    for vec in vectors:
        assert isinstance(vec.pixel_value, int), "Pixel value is not an integer"
        assert isinstance(vec.geometry, Polygon), "Geometry is not a Polygon"
        assert vec.crs == raster.crs

        # test if geometry is within the bounds of the raster
        assert vec.geometry.bounds[0] >= raster.geometry.bounds[0]
        assert vec.geometry.bounds[1] >= raster.geometry.bounds[1]
        assert vec.geometry.bounds[2] <= raster.geometry.bounds[2]
        assert vec.geometry.bounds[3] <= raster.geometry.bounds[3]
