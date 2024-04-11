from abc import ABC, abstractmethod
from typing import Generator, Iterable, Iterator

from src.models import Raster, Vector


class RasterOperationStrategy(ABC):
    @abstractmethod
    def execute(self, rasters: Raster) -> Raster:
        pass


class RasterToVectorStrategy(ABC):
    @abstractmethod
    def execute(self, raster: Raster) -> Generator[Vector, None, None]:
        pass


class RasterSplitStrategy(ABC):
    @abstractmethod
    def execute(self, raster: Raster) -> Generator[Raster, None, None]:
        pass


class RasterMergeStrategy(ABC):
    @abstractmethod
    def execute(self, rasters: Iterator[Raster]) -> Raster:
        pass


class CompositeRasterOperation(RasterOperationStrategy):
    def __init__(self, strategies: Iterable[RasterOperationStrategy]):
        self.strategies = strategies

    def execute(self, raster: Raster) -> Raster:
        for strategy in self.strategies:
            raster = strategy.execute(raster)
        return raster
