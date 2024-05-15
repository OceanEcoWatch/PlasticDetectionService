from typing import Generator

from src.models import (
    Raster,
    Vector,
)
from src.raster_op.reproject import RasterioRasterReproject
from src.raster_op.vectorize import RasterioRasterToVector


def get_scl_vectors(image: Raster, band: int = 13) -> Generator[Vector, None, None]:
    """Get Scene Classification Layer (SCL) from the image."""

    reproj_img = next(RasterioRasterReproject(4326, [band]).execute([image]))

    return RasterioRasterToVector().execute(reproj_img)
