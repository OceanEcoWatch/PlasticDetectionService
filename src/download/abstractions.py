from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator

from src._types import BoundingBox, TimeRange
from src.models import DownloadResponse


@dataclass
class DownloadParams(ABC):
    bbox: BoundingBox
    time_interval: TimeRange
    maxcc: float


class DownloadStrategy(ABC):
    def __init__(self, params: DownloadParams):
        self.params = params

    @abstractmethod
    def download_images(self) -> Generator[DownloadResponse, None, None]:
        pass
