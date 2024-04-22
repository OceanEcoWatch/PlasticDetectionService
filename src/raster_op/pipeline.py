from typing import Generator, Iterable

from src.models import Raster

from .abstractions import (
    RasterOperationStrategy,
)


class CompositeRasterOperation(RasterOperationStrategy):
    def __init__(self):
        self.children = []

    def add(self, component: RasterOperationStrategy):
        self.children.append(component)

    def remove(self, component: RasterOperationStrategy):
        self.children.remove(component)

    def execute(self, rasters: Iterable[Raster]) -> Generator[Raster, None, None]:
        # apply all childern operations in a sequence
        for child in self.children:
            rasters = child.execute(rasters)
        yield from rasters
