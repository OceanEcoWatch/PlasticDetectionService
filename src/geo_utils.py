from pyproj import Transformer
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform


def reproject_geometry(geometry, src_crs: int, dst_crs: int) -> BaseGeometry:
    """Reproject a geometry from one CRS to another."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transform(transformer.transform, geometry)
