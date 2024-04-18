import io
from typing import Generator

import numpy as np
import rasterio
from shapely.geometry import Point

from src.models import Raster, Vector

from .abstractions import (
    RasterToVectorStrategy,
)


class RasterioRasterToVector(RasterToVectorStrategy):
    def __init__(self, band: int = 1, threshold: int = 0):
        self.band = band
        self.threshold = threshold

    def execute(self, raster: Raster) -> Generator[Vector, None, None]:
        with rasterio.open(io.BytesIO(raster.content)) as src:
            image = src.read(self.band)
            meta = src.meta.copy()

            transform = src.transform

            if not np.issubdtype(image.dtype, np.integer):
                raise NotImplementedError(
                    "Raster to vector conversion only supported for integer data types"
                )

            for (row, col), value in np.ndenumerate(image):
                if value <= self.threshold:
                    continue
                x, y = transform * (col + 0.5, row + 0.5)

                yield Vector(
                    pixel_value=round(value),
                    geometry=Point(x, y),
                    crs=meta["crs"].to_epsg(),
                )