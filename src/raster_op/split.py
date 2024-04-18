import io
from itertools import product
from typing import Generator

import numpy as np
import rasterio
from rasterio.windows import Window

from src.models import Raster
from src.raster_op.utils import (
    create_raster,
    update_bounds,
    update_window_meta,
    write_image,
)
from src._types import HeightWidth

from .abstractions import (
    RasterSplitStrategy,
)


class RasterioRasterSplit(RasterSplitStrategy):
    def __init__(
        self,
        image_size: HeightWidth = HeightWidth(480, 480),
        offset: int = 64,
    ):
        self.image_size = image_size
        self.offset = offset

    def execute(self, raster: Raster) -> Generator[Raster, None, None]:
        with rasterio.open(io.BytesIO(raster.content)) as src:
            meta = src.meta.copy()
            for window, src in self._generate_windows(
                raster, self.image_size, self.offset
            ):
                image = src.read(window=window)
                window_meta = update_window_meta(meta, image)
                window_meta = update_bounds(window_meta, src.window_bounds(window))
                window_byte_stream = write_image(image, window_meta)

                yield create_raster(
                    window_byte_stream,
                    image,
                    src.window_bounds(window),
                    window_meta,
                    raster.padding_size,
                )

    def _generate_windows(self, raster: Raster, image_size, offset):
        with rasterio.open(io.BytesIO(raster.content)) as src:
            meta = src.meta.copy()
            rows = np.arange(0, meta["height"], image_size[0])
            cols = np.arange(0, meta["width"], image_size[1])
            image_window = Window(0, 0, meta["width"], meta["height"])  # type: ignore

            for r, c in product(rows, cols):
                window = image_window.intersection(
                    Window(
                        c - offset,  # type: ignore
                        r - offset,
                        image_size[1] + offset,
                        image_size[0] + offset,
                    )
                )
                yield window, src
