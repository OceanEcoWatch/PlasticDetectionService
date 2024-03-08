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
    pixel_value: int


@dataclass
class Image:
    image_id: str
    timestamp: dt.datetime
    bbox: tuple[float, float, float, float]
    image_size: tuple[int, int]
    crs: int
    maxcc: float
    data_collection: str
    request_timestamp: dt.datetime
    content: bytes
    headers: dict
    bands: list[int]
    geometry: Polygon

    @classmethod
    def from_download_response_and_raster(
        cls, response: DownloadResponse, raster: Raster
    ):
        return cls(
            image_id=response.image_id,
            timestamp=response.timestamp,
            bbox=response.bbox,
            image_size=raster.size,
            crs=response.crs,
            maxcc=response.maxcc,
            data_collection=response.data_collection,
            request_timestamp=response.request_timestamp,
            content=response.content,
            headers=response.headers,
            bands=raster.bands,
            geometry=raster.geometry,
        )
