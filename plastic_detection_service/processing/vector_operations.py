from typing import Iterable

from plastic_detection_service.models import Vector
from plastic_detection_service.processing.abstractions import VectorOperationStrategy


class VectorFilter(VectorOperationStrategy):
    def __init__(self, threshold: int):
        self.threshold = threshold

    def execute(self, vectors: Iterable[Vector]) -> Iterable[Vector]:
        for vector in vectors:
            if vector.pixel_value > self.threshold:
                yield vector
