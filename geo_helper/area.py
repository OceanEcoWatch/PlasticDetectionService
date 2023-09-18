from shapely.geometry.base import BaseGeometry

from .reproject import reproject_geometry, reproject_geometry_local_utm


def calculate_area(
    geometry: BaseGeometry,
    source_epsg: int = 4326,
    target_epsg: int = 9822,
    use_local_utm: bool = False,
) -> float:
    """
    Calculate area of a geometry in square kilometers.

    :param geometry: Geometry to calculate area for.
    :param source_epsg: EPSG code of the geometry.
    :param target_epsg: EPSG code of the target projection. Defaults is Lambert Conformal Conic (9822).
        Only used if use_local_utm is False.
    :param use_local_utm: If True, use local UTM projection instead of target_epsg.
        Defaults to False.

    """
    if use_local_utm:
        projected_geom = reproject_geometry_local_utm(geometry, source_epsg)
    else:
        projected_geom = reproject_geometry(geometry, source_epsg, target_epsg)
    return projected_geom.area / 1e6
