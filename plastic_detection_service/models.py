import datetime as dt
import io
from dataclasses import dataclass

import numpy as np
import rasterio
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon


@dataclass
class DownloadResponse:
    image_id: str
    timestamp: dt.datetime
    bbox: tuple[float, float, float, float]
    crs: int
    image_size: tuple[int, int]
    maxcc: float
    data_collection: str
    request_timestamp: dt.datetime
    content: bytes
    headers: dict


@dataclass
class Raster:
    content: bytes
    size: tuple[int, int]
    crs: int
    bands: list[int]
    geometry: Polygon

    def to_file(self, path: str):
        with open(path, "wb") as f:
            f.write(self.content)

    def to_numpy(self) -> np.ndarray:
        with rasterio.MemoryFile(io.BytesIO(self.content)) as memfile:
            with memfile.open() as dataset:
                array = dataset.read()
        return array


@dataclass
class Vector:
    geometry: BaseGeometry
    crs: int
    pixel_value: int

    @property
    def geojson(self) -> dict:
        if self.crs != 4326:
            raise ValueError("Only EPSG:4326 is supported")
        return {
            "type": "Feature",
            "geometry": self.geometry.__geo_interface__,
            "properties": {"pixel_value": self.pixel_value},
        }
