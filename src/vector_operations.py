from abc import ABC, abstractmethod
from typing import Iterable

from src.models import Vector


class VectorOperationStrategy(ABC):
    @abstractmethod
    def execute(self, vectors: Iterable[Vector]) -> Iterable[Vector]:
        pass


class VectorFilter(VectorOperationStrategy):
    def __init__(self, threshold: int):
        self.threshold = threshold

    def execute(self, vectors: Iterable[Vector]) -> Iterable[Vector]:
        for vector in vectors:
            if vector.pixel_value > self.threshold:
                yield vector
