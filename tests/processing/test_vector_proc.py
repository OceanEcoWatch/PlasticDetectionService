import pytest
from shapely.geometry import Polygon

from plastic_detection_service.models import Vector
from plastic_detection_service.processing.abstractions import VectorsProcessor
from plastic_detection_service.processing.context import VectorsProcessingContext
from plastic_detection_service.processing.rasterio_proc import RasterioVectorsProcessor

PROCESSORS = [
    RasterioVectorsProcessor(),
    VectorsProcessingContext(RasterioVectorsProcessor()),
]


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


@pytest.mark.parametrize("vectors_processor", PROCESSORS)
def test_filter_out_(vectors_processor: VectorsProcessor, vectors: list[Vector]):
    threshold = 10
    filtered = list(vectors_processor.filter_out_(vectors, threshold))
    assert len(filtered) == 2
    assert filtered[0] == vectors[1]
