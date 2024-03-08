from abc import ABC, abstractmethod
from typing import Generator, Iterable, Union

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
    def round_pixel_values(self, raster: Raster, round_to: Union[int, float]) -> Raster:
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
