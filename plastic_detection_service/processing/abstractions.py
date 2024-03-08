from abc import ABC, abstractmethod

from plastic_detection_service.models import Raster, Vector


class RasterProcessor(ABC):
    @abstractmethod
    def reproject_raster(
        self,
        raster: Raster,
        target_crs: int,
        target_bands: list[int],
        resample_alg: str,
    ) -> Raster:
        pass

    @abstractmethod
    def to_vector(self, raster: Raster, field: str) -> Vector:
        pass
