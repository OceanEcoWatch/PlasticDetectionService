import io
import logging
from itertools import product
from typing import Callable, Generator, Iterable, Optional, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter
from shapely.geometry import box

from plastic_detection_service.aws.s3 import LOGGER
from plastic_detection_service.models import Raster, Vector

from .abstractions import (
    RasterOperationStrategy,
    RasterSplitStrategy,
    RasterToVectorStrategy,
)

LOGGER = logging.getLogger(__name__)


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
    padding_size: tuple[int, int],
    removed_band: Optional[int] = None,
):
    if removed_band:
        bands = [i + 1 for i in range(image.shape[0])]
        bands.remove(removed_band)
    else:
        bands = [i + 1 for i in range(image.shape[0])]

    return Raster(
        content=content,
        size=(image.shape[1], image.shape[2]),
        dtype=image.dtype,
        crs=meta["crs"].to_epsg(),
        bands=bands,
        geometry=box(*bounds),
        padding_size=padding_size,
    )


def _write_image(image: np.ndarray, meta: dict) -> bytes:
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
                    raster.padding_size,
                )


class RasterioRasterToVector(RasterToVectorStrategy):
    def __init__(self, band: int = 1):
        self.band = band

    def execute(self, raster: Raster) -> Generator[Vector, None, None]:
        with rasterio.open(io.BytesIO(raster.content)) as src:
            image = src.read(self.band)
            meta = src.meta.copy()

            transform = src.transform

            if image.dtype != np.uint8 and image.dtype != np.uint16:
                raise ValueError(
                    "Only uint8 and uint16 data types are supported to convert to Vector, "
                    "please use RasterioRasterDtypeConversion to convert the data type to uint8 or uint16."
                )
            for (row, col), value in np.ndenumerate(image):
                if value <= 0:
                    continue

                left, top = transform * (col, row)
                right, bottom = transform * (col + 1, row + 1)

                polygon = box(left, bottom, right, top)

                yield Vector(
                    pixel_value=round(value),
                    geometry=polygon,
                    crs=meta["crs"].to_epsg(),
                )


class RasterioRasterPad(RasterOperationStrategy):
    def __init__(self, padding: int = 64, divisible_by: int = 32):
        self.padding = padding
        self.divisible_by = divisible_by

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

    def _ensure_image_size_is_divisible_by(self, image_size: tuple[int, int]):
        height, width = image_size
        if height % 32 != 0:
            height = height + (self.divisible_by - (height % self.divisible_by))
        if width % 32 != 0:
            width = width + (self.divisible_by - (width % self.divisible_by))
        return height, width

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

    def _calculate_target_image_size(
        self, image: np.ndarray, padding: int
    ) -> tuple[int, int]:
        _, input_image_height, input_image_width = image.shape
        target_image_size = (
            input_image_height + padding * 2,
            input_image_width + padding * 2,
        )
        return self._ensure_image_size_is_divisible_by(target_image_size)

    def _calculate_padding_size(
        self, image: np.ndarray, target_image_size: tuple[int, int], padding: int
    ) -> tuple[int, int]:
        _, input_image_height, input_image_width = image.shape

        target_height_with_padding, target_width_with_padding = (
            self._calculate_target_image_size(image, padding)
        )

        padding_height = (target_height_with_padding - input_image_height) / 2
        padding_width = (target_width_with_padding - input_image_width) / 2

        padding_height = int(np.ceil(padding_height))
        padding_width = int(np.ceil(padding_width))

        return padding_height, padding_width

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
            print("padding size: ", raster.padding_size)
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
                window_meta = _update_bounds(window_meta, src.window_bounds(window))
                window_byte_stream = _write_image(image, window_meta)

                yield _create_raster(
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
        self,
        offset: int = 64,
        merge_method: Union[str, Callable] = "first",
        bands: Optional[list[int]] = None,
    ):
        self.offset = offset
        self.merge_method = merge_method
        self.bands = bands

        self.buffer = io.BytesIO()

    def execute(
        self,
        rasters: Iterable[Raster],
    ) -> Raster:
        srcs = [rasterio.open(io.BytesIO(r.content)) for r in rasters]

        mosaic, out_trans = merge(srcs, method=self.merge_method, nodata=0)
        out_meta = srcs[0].meta.copy()

        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "dtype": mosaic.dtype,
            }
        )
        if self.bands:
            out_meta["count"] = len(self.bands)
            mosaic = mosaic[self.bands]

        with rasterio.open(self.buffer, "w+", **out_meta) as dst:
            dst.write(mosaic)

        return _create_raster(
            self.buffer.getvalue(),
            mosaic,
            dst.bounds,
            out_meta,
            (0, 0),
        )

    def _apply_smooth_overlap(
        self, src: rasterio.DatasetReader, dst: rasterio.DatasetReader
    ):
        for band in dst.indexes:
            window = src.window(*src.bounds)
            existing_data = dst.read(band, window=window)
            incoming_data = src.read(band, window=window)

            overlap = existing_data > 0
            if overlap.any():
                dx, dy = np.gradient(overlap.astype(float))
                g = np.abs(dx) + np.abs(dy)
                transition = gaussian_filter(g, sigma=self.offset)
                transition = transition / transition.max()
                transition[~overlap] = 1.0

                incoming_data = (
                    transition * incoming_data + (1 - transition) * existing_data
                )
            writedata = incoming_data.astype(np.float32) * 255
            dst.write(writedata, window=window, indexes=band)


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

            print("meta: ", meta)
            return _create_raster(
                _write_image(removed_band_image, meta),
                removed_band_image,
                src.bounds,
                meta,
                raster.padding_size,
                removed_band=self.band,
            )


class RasterioDtypeConversion(RasterOperationStrategy):
    def __init__(self, dtype: str):
        self.dtype = dtype
        try:
            self.np_dtype = np.dtype(dtype)
        except TypeError:
            raise ValueError(f"Unsupported dtype: {dtype}")

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
            return _create_raster(
                _write_image(image, meta),
                image,
                src.bounds,
                meta,
                raster.padding_size,
            )


class RasterInference(RasterOperationStrategy):
    def __init__(self, inference_func: Callable[[bytes], bytes]):
        self.inference_func = inference_func

    def execute(self, raster: Raster) -> Raster:
        with rasterio.open(io.BytesIO(raster.content)) as src:
            meta = src.meta.copy()

            raster_size_mb = len(raster.content) / 1024 / 1024
            LOGGER.info("size of raster content in MB: ", raster_size_mb)

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

            return _create_raster(
                _write_image(prediction, meta),
                prediction,
                src.bounds,
                meta,
                raster.padding_size,
            )


class CompositeRasterOperation(RasterOperationStrategy):
    def __init__(self, strategies: Iterable[RasterOperationStrategy]):
        self.strategies = strategies

    def execute(self, raster: Raster) -> Raster:
        for strategy in self.strategies:
            raster = strategy.execute(raster)
        return raster
