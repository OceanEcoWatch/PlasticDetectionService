from abc import ABC, abstractmethod
from typing import Generator, Iterable

from plastic_detection_service.models import Raster, Vector


class RasterProcessor(ABC):
    @abstractmethod
    def reproject_raster(
        self,
        raster: Raster,
        target_crs: int,
        target_bands: list[int],
        resample_alg: str = "nearest",
    ) -> Raster:
        pass

    @abstractmethod
    def to_vector(
        self, raster: Raster, field: str, band: int = 1
    ) -> Generator[Vector, None, None]:
        pass

    @abstractmethod
    def pad_raster(self, raster: Raster, padding: int) -> Raster:
        pass

    @abstractmethod
    def unpad_raster(self, raster: Raster) -> Raster:
        pass

    @abstractmethod
    def split_raster(
        self, raster: Raster, image_size: tuple[int, int], padding: int
    ) -> Generator[Raster, None, None]:
        pass


class VectorsProcessor(ABC):
    def filter_out_(
        self, vectors: Iterable[Vector], threshold: int
    ) -> Generator[Vector, None, None]:
        for v in vectors:
            if v.pixel_value > threshold:
                yield v

    @abstractmethod
    def to_raster(self, vectors: Iterable[Vector]) -> Raster:
        pass
