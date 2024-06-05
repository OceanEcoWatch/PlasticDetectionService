import io
from typing import Generator, Iterable

import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon

from src.models import Raster

from .abstractions import (
    RasterOperationStrategy,
)
from .utils import create_raster, write_image


class RasterioClip(RasterOperationStrategy):
    def __init__(self, geometry: Polygon):
        self.geometry = geometry

    def execute(self, rasters: Iterable[Raster]) -> Generator[Raster, None, None]:
        for raster in rasters:
            with rasterio.open(io.BytesIO(raster.content)) as src:
                out_image, out_transform = mask(src, [self.geometry], crop=True)
                out_meta = src.meta.copy()

                out_meta.update(
                    {
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                    }
                )

                yield create_raster(
                    write_image(out_image, out_meta),
                    out_image,
                    src.bounds,
                    out_meta,
                    raster.padding_size,
                )