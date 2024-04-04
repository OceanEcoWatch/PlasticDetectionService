import pytest
from shapely.geometry import Polygon

from src.models import Vector
from src.processing.vector_operations import VectorFilter


@pytest.fixture
def vectors():
    return [
        Vector(
            geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), pixel_value=5, crs=4326
        ),
        Vector(
            geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), pixel_value=25, crs=4326
        ),
        Vector(
            geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), pixel_value=15, crs=4326
        ),
    ]


def test_filter_out_(vectors: list[Vector]):
    threshold = 10
    strategy = VectorFilter(threshold=threshold)

    filtered = list(strategy.execute(vectors))
    assert len(filtered) == 2
    assert filtered[0] == vectors[1]
