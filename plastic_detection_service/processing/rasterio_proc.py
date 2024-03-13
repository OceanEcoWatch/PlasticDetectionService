import io
from itertools import product
from typing import Generator

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window
from shapely.geometry import box, shape

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
        target_crs = CRS.from_epsg(target_crs)
        with rasterio.open(io.BytesIO(raster.content)) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update(
                {
                    "crs": target_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                }
            )
            with rasterio.open(io.BytesIO(raster.content), "w", **kwargs) as dst:
                for band in target_bands:
                    reproject(
                        source=rasterio.band(src, band),
                        destination=rasterio.band(dst, band),
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling[resample_alg],
                    )
                return Raster(
                    content=self._write_image(dst.read(), dst.meta),
                    size=(dst.width, dst.height),
                    crs=dst.crs.to_epsg(),
                    bands=target_bands,
                    geometry=box(*dst.bounds),
                )

    def to_vector(
        self, raster: Raster, field: str, band: int = 1
    ) -> Generator[Vector, None, None]:
        with rasterio.open(io.BytesIO(raster.content)) as src:
            image = src.read(band)
            meta = src.meta.copy()
            for geom, val in shapes(image, transform=src.transform):
                yield Vector(
                    pixel_value=round(val),
                    geometry=shape(geom),
                    crs=meta["crs"].to_epsg(),
                )

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

    def _calculate_padding_size(
        self, image: np.ndarray, target_image_size: tuple[int, int], padding: int
    ) -> tuple[int, int]:
        _, input_image_height, input_image_width = image.shape

        target_height_with_padding = target_image_size[0] + padding * 2
        target_width_with_padding = target_image_size[1] + padding * 2

        padding_height = round(target_height_with_padding - input_image_height) / 2
        padding_width = round(target_width_with_padding - input_image_width) / 2

        return int(padding_height), int(padding_width)

    def _adjust_bounds_for_padding(
        self,
        bounds: tuple[float, float, float, float],
        padding_size: tuple[int, int],
        transform: rasterio.Affine,
    ):
        padding_height, padding_width = padding_size
        minx, miny, maxx, maxy = bounds
        x_padding, y_padding = (
            padding_width * transform.a,
            padding_height * transform.e,
        )

        return minx - x_padding, miny, maxx, maxy - y_padding

    def _adjust_bounds_for_unpadding(
        self,
        bounds: tuple[float, float, float, float],
        padding_size: tuple[int, int],
        transform: rasterio.Affine,
    ):
        padding_height, padding_width = padding_size
        minx, miny, maxx, maxy = bounds
        x_padding, y_padding = (
            padding_width * transform.a,
            padding_height * transform.e,
        )

        return minx + x_padding, miny - y_padding, maxx - x_padding, maxy + y_padding

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

    def _create_raster(
        self,
        content: bytes,
        image: np.ndarray,
        bounds: tuple,
        meta: dict,
        padding_size: tuple[int, int] = (0, 0),
    ):
        return Raster(
            content=content,
            size=(image.shape[1], image.shape[2]),
            crs=meta["crs"].to_epsg(),
            bands=[i + 1 for i in range(image.shape[0])],
            geometry=box(*bounds),
            padding_size=padding_size,
        )

    def _remove_bands(self, image: np.ndarray) -> np.ndarray:
        if image.shape[0] == 13:
            image = image[[L1CBANDS.index(b) for b in L2ABANDS]]
        return image

    def _pad_image(
        self,
        input_image: np.ndarray,
        padding_size: tuple[int, int],
    ) -> np.ndarray:
        padding_height, padding_width = padding_size
        padded_image = np.pad(
            input_image,
            (
                (0, 0),
                (padding_height, padding_height),
                (padding_width, padding_width),
            ),
        )
        return padded_image

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

    def split_raster(
        self,
        raster: Raster,
        image_size: tuple[int, int] = (480, 480),
        offset: int = 64,
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

    def pad_raster(
        self,
        raster: Raster,
        padding: int = 64,
    ) -> Raster:
        with rasterio.open(io.BytesIO(raster.content)) as src:
            meta = src.meta.copy()
            padding_size = self._calculate_padding_size(
                src.read(), raster.size, padding
            )
            image = self._pad_image(src.read(), padding_size)
            adjusted_bounds = self._adjust_bounds_for_padding(
                src.bounds, padding_size, src.transform
            )
            updated_meta = self._update_window_meta(meta, image)
            updated_meta = self._update_bounds(updated_meta, adjusted_bounds)
            byte_stream = self._write_image(image, updated_meta)

            return self._create_raster(
                byte_stream,
                image,
                adjusted_bounds,
                updated_meta,
                padding_size,
            )

    def unpad_raster(
        self,
        raster: Raster,
    ) -> Raster:
        with rasterio.open(io.BytesIO(raster.content)) as src:
            image = src.read()
            image = self._unpad_image(image, raster.padding_size)

            adjusted_bounds = self._adjust_bounds_for_unpadding(
                src.bounds, raster.padding_size, src.transform
            )
            updated_meta = self._update_window_meta(src.meta, image)
            updated_meta = self._update_bounds(updated_meta, adjusted_bounds)
            byte_stream = self._write_image(image, updated_meta)
            print(raster.padding_size)
            print(adjusted_bounds)
            print(updated_meta)
            return self._create_raster(
                byte_stream, image, adjusted_bounds, updated_meta, (0, 0)
            )

    def split_pad_raster(
        self, raster: Raster, image_size=(480, 480), padding=64
    ) -> Generator[Raster, None, None]:
        with rasterio.open(io.BytesIO(raster.content)) as src:
            meta = src.meta.copy()
            for window, src in self.generate_windows(raster, image_size, padding):
                image = src.read(window=window)
                padding_size = self._calculate_padding_size(image, image_size, padding)
                image = self._pad_image(image, padding_size)

                image = self._remove_bands(image)
                adjusted_bounds = self._adjust_bounds_for_padding(
                    src.window_bounds(window), padding_size, src.transform
                )
                window_meta = self._update_window_meta(meta, image)
                window_meta = self._update_bounds(window_meta, adjusted_bounds)
                window_byte_stream = self._write_image(image, window_meta)

                yield self._create_raster(
                    window_byte_stream,
                    image,
                    adjusted_bounds,
                    window_meta,
                    padding_size,
                )
