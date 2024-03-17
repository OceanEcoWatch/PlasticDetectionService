import io
from itertools import product
from typing import Callable, Generator, Iterable, Optional

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.io import DatasetWriter
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter
from shapely.geometry import box, shape

from plastic_detection_service.models import Raster, Vector

from .abstractions import (
    RasterOperationStrategy,
    RasterSplitStrategy,
    RasterToVectorStrategy,
)


def _update_bounds(meta, new_bounds):
    minx, miny, maxx, maxy = new_bounds
    transform = meta["transform"]
    new_transform = rasterio.Affine(
        transform.a, transform.b, minx, transform.d, transform.e, maxy
    )
    meta["transform"] = new_transform
    return meta


def _update_window_meta(meta, image):
    window_meta = meta.copy()
    window_meta.update(
        {
            "count": image.shape[0],
            "height": image.shape[1],
            "width": image.shape[2],
        }
    )
    return window_meta


def _create_raster(
    content: bytes,
    image: np.ndarray,
    bounds: tuple,
    meta: dict,
    padding_size: tuple[int, int] = (0, 0),
):
    return Raster(
        content=content,
        size=(image.shape[1], image.shape[2]),
        dtype=meta["dtype"],
        crs=meta["crs"].to_epsg(),
        bands=[i + 1 for i in range(image.shape[0])],
        geometry=box(*bounds),
        padding_size=padding_size,
    )


def _write_image(image, meta):
    buffer = io.BytesIO()
    with rasterio.open(buffer, "w+", **meta) as mem_dst:
        mem_dst.write(image)

    return buffer.getvalue()


class RasterioRasterReproject(RasterOperationStrategy):
    def __init__(
        self,
        target_crs: int,
        target_bands: Optional[Iterable[int]] = None,
        resample_alg: str = "nearest",
    ):
        self.target_crs = target_crs
        self.target_bands = target_bands
        self.resample_alg = resample_alg

    def execute(self, raster: Raster) -> Raster:
        target_crs = CRS.from_epsg(self.target_crs)
        target_bands = self.target_bands or raster.bands
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
                        resampling=Resampling[self.resample_alg],
                    )
                return _create_raster(
                    _write_image(dst.read(), dst.meta),
                    dst.read(),
                    dst.bounds,
                    dst.meta,
                )


class RasterioRasterToVector(RasterToVectorStrategy):
    def __init__(self, band: int = 1):
        self.band = band

    def execute(self, raster: Raster) -> Generator[Vector, None, None]:
        with rasterio.open(io.BytesIO(raster.content)) as src:
            image = src.read(self.band)
            meta = src.meta.copy()
            for geom, val in shapes(image, transform=src.transform):
                yield Vector(
                    pixel_value=round(val),
                    geometry=shape(geom),
                    crs=meta["crs"].to_epsg(),
                )


class RasterioRasterPad(RasterOperationStrategy):
    def __init__(self, padding: int = 64):
        self.padding = padding

    def execute(self, raster: Raster) -> Raster:
        with rasterio.open(io.BytesIO(raster.content)) as src:
            meta = src.meta.copy()
            padding_size = self._calculate_padding_size(
                src.read(), raster.size, self.padding
            )
            image = self._pad_image(src.read(), padding_size)
            adjusted_bounds = self._adjust_bounds_for_padding(
                src.bounds, padding_size, src.transform
            )
            updated_meta = _update_window_meta(meta, image)
            updated_meta = _update_bounds(updated_meta, adjusted_bounds)
            byte_stream = _write_image(image, updated_meta)

            return _create_raster(
                byte_stream,
                image,
                adjusted_bounds,
                updated_meta,
                padding_size,
            )

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


class RasterioRasterUnpad(RasterOperationStrategy):
    def execute(self, raster: Raster) -> Raster:
        with rasterio.open(io.BytesIO(raster.content)) as src:
            image = src.read()
            image = self._unpad_image(image, raster.padding_size)

            adjusted_bounds = self._adjust_bounds_for_unpadding(
                src.bounds, raster.padding_size, src.transform
            )
            updated_meta = _update_window_meta(src.meta, image)
            updated_meta = _update_bounds(updated_meta, adjusted_bounds)
            byte_stream = _write_image(image, updated_meta)

            return _create_raster(
                byte_stream, image, adjusted_bounds, updated_meta, (0, 0)
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
    ):
        padding_height, padding_width = padding_size
        minx, miny, maxx, maxy = bounds
        x_padding, y_padding = (
            padding_width * transform.a,
            padding_height * transform.e,
        )

        return minx + x_padding, miny - y_padding, maxx - x_padding, maxy + y_padding


class RasterioRasterSplit(RasterSplitStrategy):
    def __init__(
        self,
        image_size: tuple[int, int] = (480, 480),
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
                window_meta = _update_window_meta(meta, image)
                window_byte_stream = _write_image(image, window_meta)

                yield _create_raster(
                    window_byte_stream, image, src.window_bounds(window), window_meta
                )

    def _generate_windows(self, raster: Raster, image_size, offset):
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


class RasterioRasterMerge(RasterOperationStrategy):
    def __init__(
        self, target_raster: Raster, offset: int = 64, smooth_overlap: bool = False
    ):
        self.offset = offset
        self.smooth_overlap = smooth_overlap
        self.target_raster = target_raster
        self.buffer = io.BytesIO()

    def transform_target_raster(self):
        minx, miny, maxx, maxy = self.target_raster.geometry.bounds
        return from_bounds(
            minx,
            miny,
            maxx,
            maxy,
            self.target_raster.size[0],
            self.target_raster.size[1],
        )

    def open_writeable_raster(self):
        return rasterio.open(
            self.buffer,
            "w+",
            driver="GTiff",
            width=self.target_raster.size[0],
            height=self.target_raster.size[1],
            count=len(self.target_raster.bands),
            dtype=self.target_raster.dtype,
            crs=CRS.from_epsg(self.target_raster.crs),
            transform=self.transform_target_raster(),
        )

    def execute(
        self,
        rasters: Iterable[Raster],
    ) -> Raster:
        with self.open_writeable_raster() as dst:
            for raster in rasters:
                raster = self._merge(raster, dst, self.offset, self.smooth_overlap)

            return _create_raster(
                _write_image(dst.read(), dst.meta),
                dst.read(),
                dst.bounds,
                dst.meta,
            )

    def _merge(
        self,
        raster: Raster,
        dst: DatasetWriter,
        offset: int,
        smooth_overlap: bool,
    ):
        with rasterio.open(io.BytesIO(raster.content)) as src:
            for bidx in range(1, src.count + 1):
                y_score = src.read(bidx)
                width, height = src.width, src.height

                window = Window(0, 0, width, height)
                data = dst.read(bidx, window=window) / 255
                overlap = data > 0

                if overlap.any() and smooth_overlap:
                    dx, dy = np.gradient(overlap.astype(float))
                    g = np.abs(dx) + np.abs(dy)
                    transition = gaussian_filter(g, sigma=offset / 2)
                    transition /= transition.max()
                    transition[~overlap] = 1.0

                writedata = (
                    np.expand_dims(y_score, 0).astype(np.float32) * 255
                ).astype(np.uint8)
                dst.write(writedata, window=window, indexes=bidx)


class RasterioRasterCount(RasterOperationStrategy):
    def execute(self, raster: Raster) -> Raster:
        with rasterio.open(io.BytesIO(raster.content)) as src:
            return Raster(
                content=raster.content,
                size=(src.width, src.height),
                dtype=src.meta["dtype"],
                crs=src.meta["crs"].to_epsg(),
                bands=[i for i in range(1, src.meta["count"] + 1)],
                geometry=box(*src.bounds),
            )


class InferenceRasterError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class RasterInferenceOperation(RasterOperationStrategy):
    def __init__(self, inference_func: Callable[[bytes], bytes]):
        self.inference_func = inference_func

    def execute(self, raster: Raster) -> Raster:
        with rasterio.open(io.BytesIO(raster.content)) as src:
            meta = src.meta.copy()

            np_buffer = np.frombuffer(
                self.inference_func(raster.content), dtype=meta["dtype"]
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

            return _create_raster(
                _write_image(prediction, meta),
                prediction,
                src.bounds,
                meta,
            )


class CompositeRasterOperation(RasterOperationStrategy):
    def __init__(self, strategies: Iterable[RasterOperationStrategy]):
        self.strategies = strategies

    def execute(self, raster: Raster) -> Raster:
        for strategy in self.strategies:
            raster = strategy.execute(raster)
        return raster
