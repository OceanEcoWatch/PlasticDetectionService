import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator


@dataclass
class DownloadParams(ABC):
    bbox: tuple[float, float, float, float]
    time_interval: tuple[str, str]
    maxcc: float


class ImageDownload(ABC):
    def __init__(self, params: DownloadParams):
        self.params = params

    @abstractmethod
    def download_images(self) -> Generator:
        yield


@dataclass
class TimestampResponse:
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
