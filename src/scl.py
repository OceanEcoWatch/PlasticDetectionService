import enum
from typing import Generator

from src.models import (
    Raster,
    Vector,
)
from src.raster_op.band import RasterioRasterBandSelect
from src.raster_op.reproject import RasterioRasterReproject
from src.raster_op.vectorize import RasterioRasterToPolygon


class SCL(enum.Enum):
    NO_DATA = 0
    SATURATED = 1
    SHADOWS = 2
    CLOUD_SHADOWS = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM_PROB = 8
    CLOUD_HIGH_PROB = 9
    THIN_CIRRUS = 10
    SNOW_ICE = 11

    @classmethod
    def max(cls) -> int:
        return 11

    @classmethod
    def min(cls) -> int:
        return 0


def get_scl_vectors(image: Raster, band: int) -> Generator[Vector, None, None]:
    """Get Scene Classification Layer (SCL) from the image."""
    scl_img = next(RasterioRasterBandSelect([band]).execute([image]))
    reproj_scl_img = next(
        RasterioRasterReproject(target_crs=4326, target_bands=[1]).execute([scl_img])
    )
    return RasterioRasterToPolygon().execute(reproj_scl_img)
