import io
from itertools import product
from typing import Generator, Union

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

from plastic_detection_service.config import L1CBANDS, L2ABANDS
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

            rows = np.arange(0, meta["height"], image_size[0])
            cols = np.arange(0, meta["width"], image_size[1])
            image_window = Window(0, 0, meta["width"], meta["height"])

            for r, c in product(rows, cols):
                window = image_window.intersection(
                    Window(
                        c - offset,
                        r - offset,
                        image_size[1] + offset,
                        image_size[0] + offset,
                    )
                )
                yield window, src

    def pad_image(
        self, image: np.ndarray, image_size=(480, 480), offset=64
    ) -> np.ndarray:
        H, W = image_size

        H, W = H + offset * 2, W + offset * 2

        _, h, w = image.shape
        dh = (H - h) / 2
        dw = (W - w) / 2

        padded_image = np.pad(
            image,
            [
                (0, 0),
                (int(np.ceil(dh)), int(np.floor(dh))),
                (int(np.ceil(dw)), int(np.floor(dw))),
            ],
        )
        return padded_image

    def _adjust_bounds_for_padding(self, bounds, padding, transform):
        minx, miny, maxx, maxy = bounds
        x_padding, y_padding = padding * transform.a, padding * transform.e
        return minx - x_padding, miny, maxx, maxy - y_padding

    def _update_bounds(self, meta, new_bounds):
        minx, miny, maxx, maxy = new_bounds
        transform = meta["transform"]
        new_transform = rasterio.Affine(
            transform.a, transform.b, minx, transform.d, transform.e, maxy
        )
        meta["transform"] = new_transform
        return meta

    def _update_window_meta(self, meta, image):
        window_meta = meta.copy()
        window_meta.update(
            {
                "count": image.shape[0],
                "height": image.shape[1],
                "width": image.shape[2],
            }
        )
        return window_meta

    def _write_image(self, image, meta):
        buffer = io.BytesIO()
        with rasterio.open(buffer, "w+", **meta) as mem_dst:
            mem_dst.write(image)

        return buffer.getvalue()

    def _create_raster(self, content, image, bounds, meta):
        return Raster(
            content=content,
            size=(image.shape[1], image.shape[2]),
            crs=meta["crs"].to_epsg(),
            bands=[i + 1 for i in range(image.shape[0])],
            geometry=box(*bounds),
        )

    def remove_bands(self, image):
        if image.shape[0] == 13:
            image = image[[L1CBANDS.index(b) for b in L2ABANDS]]
        return image

    def split_raster(
        self, raster: Raster, image_size=(480, 480), offset=64
    ) -> Generator[Raster, None, None]:
        with rasterio.open(io.BytesIO(raster.content)) as src:
            meta = src.meta.copy()
            for window, src in self.generate_windows(raster, image_size, offset):
                image = src.read(window=window)
                window_meta = self._update_window_meta(meta, image)
                window_byte_stream = self._write_image(image, window_meta)

                yield self._create_raster(
                    window_byte_stream, image, src.window_bounds(window), window_meta
                )

    def pad_raster(self, raster: Raster, image_size=(480, 480), offset=64) -> Raster:
        with rasterio.open(io.BytesIO(raster.content)) as src:
            meta = src.meta.copy()
            image = self.pad_image(src.read(), image_size, offset)
            adjusted_bounds = self._adjust_bounds_for_padding(
                src.bounds, offset, src.transform
            )
            updated_meta = self._update_window_meta(meta, image)
            updated_meta = self._update_bounds(updated_meta, adjusted_bounds)
            byte_stream = self._write_image(image, updated_meta)

            return self._create_raster(
                byte_stream, image, adjusted_bounds, updated_meta
            )

    def split_pad_raster(
        self, raster: Raster, image_size=(480, 480), offset=64
    ) -> Generator[Raster, None, None]:
        with rasterio.open(io.BytesIO(raster.content)) as src:
            meta = src.meta.copy()
            for window, src in self.generate_windows(raster, image_size, offset):
                image = src.read(window=window)
                image = self.pad_image(image, image_size, offset)

                image = self.remove_bands(image)
                adjusted_bounds = self._adjust_bounds_for_padding(
                    src.window_bounds(window), offset, src.transform
                )
                window_meta = self._update_window_meta(meta, image)
                window_meta = self._update_bounds(window_meta, adjusted_bounds)
                window_byte_stream = self._write_image(image, window_meta)

                yield self._create_raster(
                    window_byte_stream, image, adjusted_bounds, window_meta
                )
