import pytest
from shapely.geometry import Polygon

from src.models import Vector
from src.vector_op import (
    VectorFilter,
    pixelvalue_to_probability,
    probability_to_pixelvalue,
)


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


def test_probability_to_pixelvalue():
    assert probability_to_pixelvalue(0.5) == 128
    assert probability_to_pixelvalue(0.1) == 26
    assert probability_to_pixelvalue(0.9) == 230
    assert probability_to_pixelvalue(0.0) == 0
    assert probability_to_pixelvalue(1.0) == 255
    assert probability_to_pixelvalue(0.25) == 64
    assert probability_to_pixelvalue(0.75) == 191


def test_pixelvalue_to_probability():
    assert pixelvalue_to_probability(128) == 0.5
    assert pixelvalue_to_probability(26) == 0.1
    assert pixelvalue_to_probability(230) == 0.9
    assert pixelvalue_to_probability(0) == 0.0
    assert pixelvalue_to_probability(255) == 1.0
    assert pixelvalue_to_probability(64) == 0.25
