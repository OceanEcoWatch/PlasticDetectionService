import io
from typing import Generator, Iterable

import numpy as np
import rasterio

from src._types import BoundingBox, HeightWidth
from src.models import Raster
from src.raster_op.utils import (
    create_raster,
    update_bounds,
    update_window_meta,
    write_image,
)

from .abstractions import (
    RasterOperationStrategy,
)


class RasterioRasterPad(RasterOperationStrategy):
    def __init__(self, padding: int = 64, divisible_by: int = 32):
        self.padding = padding
        self.divisible_by = divisible_by

    def execute(self, rasters: Iterable[Raster]) -> Generator[Raster, None, None]:
        for raster in rasters:
            with rasterio.open(io.BytesIO(raster.content)) as src:
                meta = src.meta.copy()
                padding_size = self._calculate_padding_size(src.read(), self.padding)
                image = self._pad_image(src.read(), padding_size)

                adjusted_bounds = self._adjust_bounds_for_padding(
                    src.bounds, padding_size[0], src.transform
                )
                updated_meta = update_window_meta(meta, image)
                updated_meta = update_bounds(updated_meta, adjusted_bounds)
                byte_stream = write_image(image, updated_meta)

                print("original image shape: ", src.read().shape)
                print("padding size: ", padding_size)
                print("padded image shape: ", image.shape)
                yield create_raster(
                    byte_stream,
                    image,
                    adjusted_bounds,
                    updated_meta,
                    padding_size[0],
                )

    def _ensure_divisible_padding(
        self, original_size: int, padding: int, divisible_by: int
    ) -> float:
        """Ensure that the original size plus padding is divisible by a given number.
        Return the new padding size."""
        target_size = original_size + padding

        while target_size % divisible_by != 0:
            padding += 1
            target_size = original_size + padding

        return padding / 2

    def _calculate_padding_size(
        self, image: np.ndarray, padding: int
    ) -> tuple[HeightWidth, HeightWidth]:
        _, input_image_height, input_image_width = image.shape

        padding_height = self._ensure_divisible_padding(
            input_image_height, padding, self.divisible_by
        )
        padding_width = self._ensure_divisible_padding(
            input_image_width, padding, self.divisible_by
        )

        return HeightWidth(
            int(np.ceil(padding_height)), int(np.ceil(padding_width))
        ), HeightWidth(int(np.floor(padding_height)), int(np.floor(padding_width)))

    def _pad_image(
        self, input_image: np.ndarray, padding_size: tuple[HeightWidth, HeightWidth]
    ) -> np.ndarray:
        padded_image = np.pad(
            input_image,
            (
                (0, 0),
                (padding_size[0].height, padding_size[1].height),
                (padding_size[0].width, padding_size[1].width),
            ),
        )

        return padded_image

    def _adjust_bounds_for_padding(
        self,
        bounds: tuple[float, float, float, float],
        padding_size: tuple[int, int],
        transform: rasterio.Affine,
    ) -> BoundingBox:
        padding_height, padding_width = padding_size
        minx, miny, maxx, maxy = bounds
        x_padding, y_padding = (
            padding_width * transform.a,
            padding_height * transform.e,
        )

        return BoundingBox(minx - x_padding, miny, maxx, maxy - y_padding)


class RasterioRasterUnpad(RasterOperationStrategy):
    def execute(self, rasters: Iterable[Raster]) -> Generator[Raster, None, None]:
        for raster in rasters:
            with rasterio.open(io.BytesIO(raster.content)) as src:
                print("padding size: ", raster.padding_size)
                image = src.read()
                image = self._unpad_image(image, raster.padding_size)

                adjusted_bounds = self._adjust_bounds_for_unpadding(
                    src.bounds, raster.padding_size, src.transform
                )
                updated_meta = update_window_meta(src.meta, image)
                updated_meta = update_bounds(updated_meta, adjusted_bounds)
                byte_stream = write_image(image, updated_meta)

                yield create_raster(
                    byte_stream, image, adjusted_bounds, updated_meta, HeightWidth(0, 0)
                )

    def _unpad_image(
        self,
        input_image: np.ndarray,
        padding_size: tuple[int, int],
    ) -> np.ndarray:
        _, input_image_height, input_image_width = input_image.shape
        padding_height, padding_width = padding_size

        unpadded_image = input_image[
            :,
            int(np.ceil(padding_height)) : input_image_height
            - int(np.floor(padding_height)),
            int(np.ceil(padding_width)) : input_image_width
            - int(np.floor(padding_width)),
        ]

        return unpadded_image

    def _adjust_bounds_for_unpadding(
        self,
        bounds: tuple[float, float, float, float],
        padding_size: tuple[int, int],
        transform: rasterio.Affine,
    ) -> BoundingBox:
        padding_height, padding_width = padding_size
        minx, miny, maxx, maxy = bounds
        x_padding, y_padding = (
            padding_width * transform.a,
            padding_height * transform.e,
        )

        return BoundingBox(
            minx + x_padding, miny - y_padding, maxx - x_padding, maxy + y_padding
        )
