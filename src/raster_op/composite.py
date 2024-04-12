from typing import Generator, Iterable

from src.models import Raster, Vector

from .abstractions import (
    RasterMergeStrategy,
    RasterOperationStrategy,
    RasterSplitStrategy,
    RasterToVectorStrategy,
)


class CompositeRasterOperation(RasterOperationStrategy):
    def __init__(self, strategies: Iterable[RasterOperationStrategy]):
        self.strategies = strategies

    def execute(self, raster: Raster) -> Raster:
        for strategy in self.strategies:
            raster = strategy.execute(raster)
        return raster


class RasterOpHandler(RasterToVectorStrategy):
    def __init__(
        self,
        split: RasterSplitStrategy,
        pad: RasterOperationStrategy,
        inference: RasterOperationStrategy,
        band: RasterOperationStrategy,
        unpad: RasterOperationStrategy,
        merge: RasterMergeStrategy,
        convert: RasterOperationStrategy,
        reproject: RasterOperationStrategy,
        to_vector: RasterToVectorStrategy,
    ):
        self.split = split
        self.pad = pad
        self.inference = inference
        self.band = band
        self.unpad = unpad
        self.merge = merge
        self.convert = convert
        self.reproject = reproject
        self.to_vector = to_vector

    def execute(self, raster: Raster) -> Generator[Vector, None, None]:
        rasters = []
        for window in self.split.execute(raster):
            window = self.pad.execute(window)
            window = self.band.execute(window)
            window = self.inference.execute(window)
            window = self.unpad.execute(window)
            rasters.append(window)

        merged = self.merge.execute(rasters)
        converted = self.convert.execute(merged)
        reprojected = self.reproject.execute(converted)
        return self.to_vector.execute(reprojected)
