import io
import logging

import numpy as np
import rasterio

from src.models import Raster
from src.raster_op.utils import create_raster, write_image

from .abstractions import (
    RasterOperationStrategy,
)

LOGGER = logging.getLogger(__name__)


class RasterioRemoveBand(RasterOperationStrategy):
    def __init__(self, band: int):
        self.band = band
        self.band_index = band - 1

    def execute(self, raster: Raster) -> Raster:
        try:
            raster.bands[self.band_index]
        except IndexError:
            LOGGER.warning(f"Band {self.band} does not exist in raster, skipping")
            return raster

        with rasterio.open(io.BytesIO(raster.content)) as src:
            meta = src.meta.copy()
            image = src.read()

            removed_band_image = np.delete(image, self.band_index, axis=0)
            LOGGER.info(f"Removed band {self.band} from raster")
            meta.update(
                {
                    "count": removed_band_image.shape[0],
                    "height": removed_band_image.shape[1],
                    "width": removed_band_image.shape[2],
                }
            )


            return create_raster(
                write_image(removed_band_image, meta),
                removed_band_image,
                src.bounds,
                meta,
                raster.padding_size,
                removed_band=self.band,
            )
