import io
import logging
from typing import Callable, Generator, Iterable

import numpy as np
import rasterio

from src.models import Raster
from src.raster_op.utils import create_raster, write_image

from .abstractions import RasterOperationStrategy

LOGGER = logging.getLogger(__name__)


class RasterioInference(RasterOperationStrategy):
    def __init__(self, inference_func: Callable[[bytes], bytes]):
        self.inference_func = inference_func

    def execute(self, rasters: Iterable[Raster]) -> Generator[Raster, None, None]:
        for raster in rasters:
            with rasterio.open(io.BytesIO(raster.content)) as src:
                meta = src.meta.copy()

                raster_size_mb = len(raster.content) / 1024 / 1024
                LOGGER.info(f"Raster size: {raster_size_mb:.2f} MB")

                np_buffer = np.frombuffer(
                    self.inference_func(raster.content), dtype=np.float32
                )
                prediction = np_buffer.reshape(1, meta["height"], meta["width"])

                meta.update(
                    {
                        "count": prediction.shape[0],
                        "height": prediction.shape[1],
                        "width": prediction.shape[2],
                        "dtype": prediction.dtype,
                    }
                )

                yield create_raster(
                    write_image(prediction, meta),
                    prediction,
                    src.bounds,
                    meta,
                    raster.padding_size,
                )
