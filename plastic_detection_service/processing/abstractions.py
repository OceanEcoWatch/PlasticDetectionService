from abc import ABC, abstractmethod
from typing import Generator

from plastic_detection_service.models import Raster, Vector


class RasterProcessor(ABC):
    @abstractmethod
    def reproject_raster(
        self,
        raster: Raster,
        target_crs: int,
        target_bands: list[int],
    ) -> Raster:
        pass

    @abstractmethod
    def to_vector(
        self, raster: Raster, field: str, band: int = 1
    ) -> Generator[Vector, None, None]:
        pass
