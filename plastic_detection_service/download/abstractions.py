from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator

from plastic_detection_service.models import DownloadResponse


@dataclass
class DownloadParams(ABC):
    bbox: tuple[float, float, float, float]
    time_interval: tuple[str, str]
    maxcc: float


class DownloadStrategy(ABC):
    def __init__(self, params: DownloadParams):
        self.params = params

    @abstractmethod
    def download_images(self) -> Generator[DownloadResponse, None, None]:
        pass
