import datetime as dt
from dataclasses import dataclass

from shapely.geometry.base import BaseGeometry


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
    width: int
    height: int
    crs: int
    bands: list[int]

    def to_file(self, path: str):
        with open(path, "wb") as f:
            f.write(self.content)


@dataclass
class Vector:
    geometry: BaseGeometry
    pixel_value: int
