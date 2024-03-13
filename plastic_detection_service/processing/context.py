import logging
from typing import Generator, Iterable

from .abstractions import Raster, RasterProcessor, Vector, VectorsProcessor

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

    def pad_raster(self, raster: Raster, padding: int) -> Raster:
        return self.raster_processor.pad_raster(raster, padding)

    def unpad_raster(self, raster: Raster) -> Raster:
        return self.raster_processor.unpad_raster(raster)

    def split_raster(
        self, raster: Raster, image_size: tuple[int, int], padding: int
    ) -> Generator[Raster, None, None]:
        return self.raster_processor.split_raster(raster, image_size, padding)

    def split_pad_raster(
        self, raster: Raster, image_size: tuple[int, int], padding: int
    ) -> Generator[Raster, None, None]:
        return self.raster_processor.split_pad_raster(raster, image_size, padding)


class VectorsProcessingContext:
    def __init__(self, vectors_processor: VectorsProcessor):
        self.vectors_processor = vectors_processor

    def set_vectors_processor(self, vectors_processor: VectorsProcessor):
        self.vectors_processor = vectors_processor

    def filter_out_(
        self, vectors: Iterable[Vector], threshold: int
    ) -> Generator[Vector, None, None]:
        return self.vectors_processor.filter_out_(vectors, threshold)
