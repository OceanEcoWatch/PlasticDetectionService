""" Domain models for the application """
import datetime as dt
import io
from dataclasses import dataclass

import numpy as np
import rasterio
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon

from src._types import IMAGE_DTYPES, HeightWidth


@dataclass(frozen=True)
class DownloadResponse:
    image_id: str
    timestamp: dt.datetime
    bbox: tuple[float, float, float, float]
    crs: int
    image_size: HeightWidth
    maxcc: float
    data_collection: str
    request_timestamp: dt.datetime
    content: bytes
    headers: dict


@dataclass(frozen=True)
class Raster:
    content: bytes
    size: HeightWidth
    dtype: str
    crs: int
    bands: list[int]
    resolution: float
    geometry: Polygon
    padding_size: HeightWidth = HeightWidth(0, 0)

    def __post_init__(self):
        if self.dtype not in IMAGE_DTYPES:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def to_file(self, path: str):
        with open(path, "wb") as f:
            f.write(self.content)

    def to_numpy(self) -> np.ndarray:
        with rasterio.MemoryFile(io.BytesIO(self.content)) as memfile:
            with memfile.open() as dataset:
                array = dataset.read()
        return array


@dataclass(frozen=True)
class Vector:
    geometry: BaseGeometry
    crs: int
    pixel_value: int

    @property
    def geojson(self) -> dict:
        if self.crs != 4326:
            raise ValueError("Only EPSG:4326 format is supported for GeoJSON")
        return {
            "type": "Feature",
            "geometry": self.geometry.__geo_interface__,
            "properties": {"pixel_value": self.pixel_value},
        }
