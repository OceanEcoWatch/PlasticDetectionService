import pytest
from shapely.geometry import Polygon

from plastic_detection_service.models import Vector
from plastic_detection_service.processing.abstractions import VectorsProcessor
from plastic_detection_service.processing.gdal_proc import GdalVectorsProcessor
from plastic_detection_service.processing.main import VectorsProcessingContext

PROCESSORS = [GdalVectorsProcessor(), VectorsProcessingContext(GdalVectorsProcessor())]


@pytest.fixture
def vectors():
    return [
        Vector(geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), pixel_value=5),
        Vector(geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), pixel_value=25),
        Vector(geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), pixel_value=15),
    ]


@pytest.mark.parametrize("vectors_processor", PROCESSORS)
def test_filter_out_(vectors_processor: VectorsProcessor, vectors: list[Vector]):
    threshold = 10
    filtered = list(vectors_processor.filter_out_(vectors, threshold))
    assert len(filtered) == 2
    assert filtered[0] == vectors[1]
