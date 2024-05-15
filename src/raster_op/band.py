import io
import logging
from typing import Generator, Iterable

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

    def execute(self, rasters: Iterable[Raster]) -> Generator[Raster, None, None]:
        for raster in rasters:
            print(raster.bands)
            try:
                raster.bands[self.band_index]
            except IndexError:
                LOGGER.warning(f"Band {self.band} does not exist in raster, skipping")
                yield raster
                continue

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

                yield create_raster(
                    write_image(removed_band_image, meta),
                    removed_band_image,
                    src.bounds,
                    meta,
                    raster.padding_size,
                )


class RasterioRasterBandSelect(RasterOperationStrategy):
    def __init__(self, bands: Iterable[int]):
        self._bands = bands
        self.bands = [band - 1 for band in bands]

    def execute(self, rasters: Iterable[Raster]) -> Generator[Raster, None, None]:
        for raster in rasters:
            with rasterio.open(io.BytesIO(raster.content)) as src:
                meta = src.meta.copy()
                image = src.read()

                selected_bands_image = image[self.bands]
                LOGGER.info(f"Selected bands {self.bands} from raster")
                meta.update(
                    {
                        "count": selected_bands_image.shape[0],
                        "height": selected_bands_image.shape[1],
                        "width": selected_bands_image.shape[2],
                    }
                )

                yield create_raster(
                    write_image(selected_bands_image, meta),
                    selected_bands_image,
                    src.bounds,
                    meta,
                    raster.padding_size,
                )
