from typing import Iterable

from src.models import Raster

from .abstractions import (
    RasterMergeStrategy,
    RasterOperationStrategy,
    RasterSplitStrategy,
)


class CompositeRasterOperation(RasterOperationStrategy):
    def __init__(self, strategies: Iterable[RasterOperationStrategy]):
        self.strategies = strategies

    def execute(self, raster: Raster) -> Raster:
        for strategy in self.strategies:
            raster = strategy.execute(raster)
        return raster


class RasterOpHandler(RasterOperationStrategy):
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
    ):
        self.split = split
        self.pad = pad
        self.inference = inference
        self.band = band
        self.unpad = unpad
        self.merge = merge
        self.convert = convert
        self.reproject = reproject

    def execute(self, raster: Raster) -> Raster:
        rasters = []
        for window in self.split.execute(raster):
            window = self.pad.execute(window)

            window = self.band.execute(window)
            window = self.inference.execute(window)
            window = self.unpad.execute(window)
            rasters.append(window)

        merged = self.merge.execute(rasters)
        converted = self.convert.execute(merged)
        return self.reproject.execute(converted)
