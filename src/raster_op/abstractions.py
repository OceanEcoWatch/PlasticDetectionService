from abc import ABC, abstractmethod
from typing import Generator, Iterable

from src.models import Raster, Vector


class RasterOperationStrategy(ABC):
    @abstractmethod
    def execute(self, rasters: Iterable[Raster]) -> Generator[Raster, None, None]:
        pass


class RasterToVectorStrategy(RasterOperationStrategy):
    @abstractmethod
    def execute(self, raster: Raster) -> Generator[Vector, None, None]:
        pass
