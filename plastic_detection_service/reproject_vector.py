from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import Optional, Union

import pyproj
import rasterio
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import BaseGeometry


@cache
def determine_utm_epsg(
    source_epsg: int,
    west_lon: float,
    south_lat: float,
    east_lon: float,
    north_lat: float,
    contains: bool = True,
) -> int:
    """
    Determine the UTM EPSG code for a given epsg code and bounding box

    :param source_epsg: The source EPSG code
    :param west_lon: The western longitude
    :param south_lat: The southern latitude
    :param east_lon: The eastern longitude
    :param north_lat: The northern latitude
    :param contains: If True, the UTM CRS must contain the bounding box,
        if False, the UTM CRS must intersect the bounding box.

    :return: The UTM EPSG code

    :raises ValueError: If no UTM CRS is found for the epsg and bbox
    """
    datum_name = pyproj.CRS.from_epsg(source_epsg).to_dict()["datum"]

    utm_crs_info = query_utm_crs_info(
        datum_name=datum_name,
        area_of_interest=AreaOfInterest(west_lon, south_lat, east_lon, north_lat),
        contains=contains,
    )

    if not utm_crs_info:
        raise ValueError(f"No UTM CRS found for the datum {datum_name} and bbox")

    return int(utm_crs_info[0].code)


def is_utm_epsg(epsg: int) -> bool:
    """Check if an EPSG code is a UTM code."""
    return pyproj.CRS.from_epsg(epsg).to_dict()["proj"] == "utm"


@cache
def create_transformer(
    source_epsg: int, target_epsg: int, centroid: Optional[Point] = None
) -> pyproj.Transformer:
    """
    Create a pyproj transformer for a given source and target EPSG code.

    :param source_epsg: The source EPSG code
    :param target_epsg: The target EPSG code
    :param centroid: The centroid of the geometry to be projected.
        Only required for non-UTM projections.

    :return: A pyproj transformer
    :raises ValueError: If a centroid is not provided for non-UTM projections
    """
    source_crs = pyproj.CRS.from_epsg(source_epsg)

    if is_utm_epsg(target_epsg):
        target_crs = pyproj.CRS.from_epsg(target_epsg)
    else:
        if centroid is None:
            raise ValueError("A centroid must be provided for non-UTM projections")

        lon, lat = centroid.x, centroid.y
        proj_name = pyproj.CRS.from_epsg(target_epsg).to_dict()["proj"]
        target_crs = pyproj.CRS.from_proj4(
            f"+proj={proj_name} +lat_1={lat} +lat_2={lat} +lat_0={lat} +lon_0={lon}"
        )

    return pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)


def reproject_geometry(
    geometry: BaseGeometry, source_epsg: int = 4326, target_epsg: int = 9822
) -> BaseGeometry:
    """
    Reproject a shapely geometry to a EPSG projection centered on the centroid.
    For multi-part geometries, each part is processed separately.

    :param geometry: shapely geometry to project
    :param source_epsg: EPSG code of the source geometry.
        Default is WGS84 (4326)
    :param target_epsg: EPSG code of the target geometry.
        Defaults is Lambert Conformal Conic (9822)

    :return: projected shapely geometry
    :raises TypeError: if the geometry type is not supported
    """
    return _reproject(geometry, source_epsg, target_epsg)


def reproject_geometry_local_utm(
    geometry: BaseGeometry,
    source_epsg: int = 4326,
) -> BaseGeometry:
    """
    Reproject a shapely geometry to a local UTM projection.
    For multi-part geometries, each part is processed separately.

    :param geometry: shapely geometry to project
    :param source_epsg: EPSG code of the source geometry.
        Default is WGS84 (4326)

    :return: projected shapely geometry
    :raises TypeError: if the geometry type is not supported
    """
    return _reproject(geometry, source_epsg)


def _reproject(
    geometry: BaseGeometry,
    source_epsg: int = 4326,
    target_epsg: Optional[int] = None,
) -> BaseGeometry:
    geom_map = {
        Point: _project_point,
        LineString: _project_line,
        LinearRing: _project_line,
        Polygon: _project_polygon,
        MultiPoint: _project_multi_geom,
        MultiLineString: _project_multi_geom,
        MultiPolygon: _project_multi_geom,
        GeometryCollection: _project_multi_geom,
    }

    if type(geometry) not in geom_map:
        raise TypeError(f"Unsupported geometry type: {type(geometry)}")

    if target_epsg is None and isinstance(
        geometry, (Point, LineString, LinearRing, Polygon)
    ):
        x, y = geometry.centroid.x, geometry.centroid.y
        target_epsg = determine_utm_epsg(source_epsg, x, y, x, y)

    return geom_map[type(geometry)](geometry, source_epsg, target_epsg)


def _project_multi_geom(
    geom: BaseGeometry, source_epsg: int, target_epsg: int
) -> BaseGeometry:
    with ThreadPoolExecutor() as executor:
        projected_geoms = executor.map(
            lambda g: _reproject(g, source_epsg, target_epsg), geom.geoms
        )

    return type(geom)(list(projected_geoms))


def _project_point(point: Point, source_epsg: int, target_epsg: int) -> Point:
    transformer = create_transformer(source_epsg, target_epsg, point)
    return Point(transformer.transform(point.x, point.y))


def _project_line(
    linestring: LineString, source_epsg: int, target_epsg: int
) -> Union[LineString, LinearRing]:
    transformer = create_transformer(source_epsg, target_epsg, linestring.centroid)
    return type(linestring)([transformer.transform(*xy) for xy in linestring.coords])


def _project_polygon(polygon: Polygon, source_epsg: int, target_epsg: int) -> Polygon:
    transformer = create_transformer(source_epsg, target_epsg, polygon.centroid)

    exterior_coords = [transformer.transform(*xy) for xy in polygon.exterior.coords]

    interior_coords = []
    for interior in polygon.interiors:
        interior_coords.append([transformer.transform(*xy) for xy in interior.coords])

    return Polygon(exterior_coords, interior_coords)


def reproject_raster(
    raster: str,
    target_epsg: int,
    output_path: str,
    resampling: Resampling = Resampling.nearest,
) -> None:
    """
    Reproject a raster to a target EPSG projection.

    :param raster: The raster to reproject
    :param target_epsg: The target EPSG code
    :param output_path: The output path
    :param resampling: The resampling method to use. Default is nearest neighbor.
    """
    with rasterio.open(raster) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_epsg, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": target_epsg,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_epsg,
                    resampling=resampling,
                )
