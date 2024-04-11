import io
from typing import Iterable, Optional

import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

from src.models import Raster
from src.raster_op.utils import create_raster, write_image

from .abstractions import (
    RasterOperationStrategy,
)


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
                return create_raster(
                    write_image(dst.read(), dst.meta),
                    dst.read(),
                    dst.bounds,
                    dst.meta,
                    raster.padding_size,
                )
