import io
from typing import Callable, Generator, Iterable, Optional, Union

import numpy as np
import rasterio
from rasterio.merge import merge
from scipy.ndimage import gaussian_filter

from src._types import HeightWidth
from src.models import Raster
from src.raster_op.utils import create_raster

from .abstractions import (
    RasterOperationStrategy,
)


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
    ) -> Generator[Raster, None, None]:
        srcs = [rasterio.open(io.BytesIO(r.content)) for r in rasters]

        mosaic, out_trans = merge(srcs, method=self.merge_method, nodata=0)  # type: ignore
        out_meta = srcs[0].meta.copy()

        [src.close() for src in srcs]

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

        yield create_raster(
            self.buffer.getvalue(),
            mosaic,
            dst.bounds,
            out_meta,
            HeightWidth(0, 0),
        )


def smooth_overlap_callable(
    merged_data,
    new_data,
    merged_mask,
    new_mask,
    index=None,
    roff=None,
    coff=None,
    sigma=64,
):
    overlap = merged_mask & new_mask

    if overlap.any():
        # Calculate the gradient of the overlap mask
        dx, dy = np.gradient(overlap.astype(float))
        g = np.abs(dx) + np.abs(dy)

        # Smooth the gradient to create a transition mask
        transition = gaussian_filter(g, sigma=sigma)
        transition /= transition.max()
        transition[overlap] = 1.0

        # Applying the blending logic correctly
        for band in range(merged_data.shape[0]):
            # Only blend where there's overlap
            blend_area = (transition < 1) & overlap
            merged_data[band][blend_area] = (
                transition[blend_area] * new_data[band][blend_area]
                + (1 - transition[blend_area]) * merged_data[band][blend_area]
            )

    else:
        # Simplified approach for cases without overlap
        for band in range(merged_data.shape[0]):
            # Considering new_mask to directly replace data in areas without overlap
            replace_area = ~new_mask
            merged_data[band][replace_area] = new_data[band][replace_area]


def copy_smooth(merged_data, new_data, merged_mask, new_mask, sigma=64, **kwargs):
    """Applies a Gaussian filter to the overlapping pixels."""
    mask = np.empty_like(merged_mask, dtype="bool")
    np.logical_and(merged_mask, new_mask, out=mask)
    np.copyto(
        merged_data,
        gaussian_filter(new_data, sigma=sigma),
        where=mask,
        casting="unsafe",
    )
    np.logical_not(new_mask, out=mask)
    np.logical_and(merged_mask, mask, out=mask)
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")
