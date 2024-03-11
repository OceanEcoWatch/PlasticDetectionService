import io
from itertools import product
from typing import Generator, Union

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

from plastic_detection_service.models import Raster, Vector

from .abstractions import RasterProcessor


class RasterioRasterProcessor(RasterProcessor):
    def reproject_raster(
        self,
        raster: Raster,
        target_crs: int,
        target_bands: list[int],
        resample_alg: str = "nearest",
    ) -> Raster:
        raise NotImplementedError

    def to_vector(
        self, raster: Raster, field: str, band: int = 1
    ) -> Generator[Vector, None, None]:
        raise NotImplementedError

    def round_pixel_values(self, raster: Raster, round_to: Union[int, float]) -> Raster:
        raise NotImplementedError

    def generate_windows(self, raster: Raster, image_size=(480, 480), offset=64):
        with rasterio.open(io.BytesIO(raster.content)) as src:
            meta = src.meta.copy()
            H, W = image_size
            rows = np.arange(0, meta["height"], H)
            cols = np.arange(0, meta["width"], W)
            image_window = Window(0, 0, meta["width"], meta["height"])

            for r, c in product(rows, cols):
                H, W = image_size
                window = image_window.intersection(
                    Window(c - offset, r - offset, W + offset, H + offset)
                )
                yield window, src

    def pad_image(self, src, window, image_size=(480, 480), offset=64):
        H, W = image_size
        image = src.read(window=window)

        H, W = H + offset * 2, W + offset * 2

        _, h, w = image.shape
        dh = (H - h) / 2
        dw = (W - w) / 2
        image = np.pad(
            image,
            [
                (0, 0),
                (int(np.ceil(dh)), int(np.floor(dh))),
                (int(np.ceil(dw)), int(np.floor(dw))),
            ],
        )
        return image

    def update_window_metaa(self, meta, window):
        window_meta = meta.copy()
        window_meta.update(
            {
                "height": window.height,
                "width": window.width,
            }
        )
        return window_meta

    def write_image(self, image, meta):
        buffer = io.BytesIO()
        with rasterio.open(buffer, "w+", **meta) as mem_dst:
            mem_dst.write(image)

        return buffer.getvalue()

    def create_raster(self, content, src, image, window, window_meta):
        return Raster(
            content=content,
            size=(window_meta["width"], window_meta["height"]),
            crs=window_meta["crs"].to_epsg(),
            bands=[i + 1 for i in range(image.shape[0])],
            geometry=box(*src.window_bounds(window)),
        )

    def split_pad_raster(self, raster: Raster, image_size=(480, 480), offset=64):
        with rasterio.open(io.BytesIO(raster.content)) as src:
            meta = src.meta.copy()
            for window, src in self.generate_windows(raster, image_size, offset):
                image = self.pad_image(src, window, image_size, offset)
                window_meta = self.update_window_metaa(meta, window)
                window_byte_stream = self.write_image(image, window_meta)
                yield self.create_raster(
                    window_byte_stream, src, image, window, window_meta
                )
