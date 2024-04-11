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


class RasterioDtypeConversion(RasterOperationStrategy):
    def __init__(self, dtype: str):
        self.dtype = dtype
        try:
            self.np_dtype = np.dtype(dtype)
        except TypeError:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def execute(self, raster: Raster) -> Raster:
        with rasterio.open(io.BytesIO(raster.content)) as src:
            meta = src.meta.copy()
            image = src.read()

            if image.dtype == self.dtype:
                LOGGER.info(f"Raster already has dtype {self.dtype}, skipping")
                return raster

            image = self._scale(image)

            meta.update(
                {
                    "dtype": self.dtype,
                }
            )
            return create_raster(
                write_image(image, meta),
                image,
                src.bounds,
                meta,
                raster.padding_size,
            )

    def _scale(self, image: np.ndarray) -> np.ndarray:
        image_min = image.min()
        image_max = image.max()

        if np.issubdtype(self.np_dtype, np.integer):
            dtype_min = np.iinfo(self.np_dtype).min
            dtype_max = np.iinfo(self.np_dtype).max
        elif np.issubdtype(self.np_dtype, np.floating):
            dtype_min = 0.0
            dtype_max = 1.0
        else:
            raise ValueError(
                "Unsupported dtype: must be either integer or floating-point."
            )

        scaled_image = (
            (image - image_min) / (image_max - image_min) * (dtype_max - dtype_min)
            + dtype_min
        ).astype(self.np_dtype)

        return scaled_image
