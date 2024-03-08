import logging
from typing import Generator

from .abstractions import Raster, RasterProcessor, Vector

LOGGER = logging.getLogger(__name__)


class RasterProcessingContext:
    def __init__(self, raster_processor: RasterProcessor):
        self.raster_processor = raster_processor

    def set_raster_processor(self, raster_processor: RasterProcessor):
        self.raster_processor = raster_processor

    def reproject_raster(
        self,
        raster: Raster,
        target_crs: int,
        target_bands: list[int],
        resample_alg: str = "nearest",
    ) -> Raster:
        return self.raster_processor.reproject_raster(
            raster, target_crs, target_bands, resample_alg
        )

    def to_vector(
        self, raster: Raster, field: str, band: int = 1
    ) -> Generator[Vector, None, None]:
        return self.raster_processor.to_vector(raster, field, band)


def main():
    pass


if __name__ == "__main__":
    main()
