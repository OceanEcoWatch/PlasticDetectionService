import datetime

import pytest
from shapely.geometry.polygon import Polygon
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, create_session
from sqlalchemy_utils import create_database, database_exists

from plastic_detection_service.database.insert import Insert
from plastic_detection_service.database.models import (
    Base,
    Image,
    Model,
    PredictionVector,
    SceneClassificationVector,
)
from plastic_detection_service.models import DownloadResponse, Raster, Vector
from plastic_detection_service.types import HeightWidth

TEST_DB_URL = "postgresql://postgres:postgres@localhost:5432/oew_dev"


@pytest.fixture
def create_test_db():
    engine = create_engine(TEST_DB_URL)
    if not database_exists(engine.url):
        create_database(engine.url)
    yield engine
    engine.dispose()


@pytest.fixture
def drop_create_tables(create_test_db):
    Base.metadata.drop_all(create_test_db)
    Base.metadata.create_all(create_test_db)


@pytest.fixture
def test_session(create_test_db, drop_create_tables):
    test_session = create_session(create_test_db)
    yield test_session
    test_session.close()


@pytest.fixture
def download_response():
    return DownloadResponse(
        image_id="test_image_id",
        timestamp=datetime.datetime.now(),
        bbox=(0, 0, 0, 0),
        crs=4326,
        image_size=HeightWidth(height=10, width=10),
        maxcc=0.1,
        data_collection="test_data_collection",
        request_timestamp=datetime.datetime.now(),
        content=b"test_content",
        headers={"test_header": "test_value"},
    )


@pytest.fixture
def db_raster():
    return Raster(
        content=b"test_content",
        size=HeightWidth(height=10, width=10),
        dtype="uint8",
        crs=4326,
        bands=[1, 2, 3],
        geometry=Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
    )


@pytest.fixture
def db_vectors():
    return [
        Vector(
            geometry=Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]), crs=4326, pixel_value=1
        )
    ]


@pytest.fixture
def insert(test_session):
    return Insert(test_session)


@pytest.mark.integration
def test_insert_mock(
    insert: Insert,
    db_raster: Raster,
    download_response: DownloadResponse,
    db_vectors: list[Vector],
    test_session: Session,
):
    image = insert.insert_image(download_response, db_raster, "test_image_url")
    model = insert.insert_model("test_model_id", "test_model_url")
    prediction_vectors = insert.insert_prediction_vectors(
        db_vectors, image.id, model.id
    )
    scls_vectors = insert.insert_scls_vectors(db_vectors, image.id)

    assert image.id is not None
    assert model.id is not None
    assert len(prediction_vectors) == 1
    assert len(scls_vectors) == 1

    assert all([pv.model_id == model.id for pv in prediction_vectors])
    assert all([pv.image_id == image.id for pv in prediction_vectors])
    assert all([scls.image_id == image.id for scls in scls_vectors])

    assert len(test_session.query(Image).all()) == 1
    assert len(test_session.query(Model).all()) == 1
    assert len(test_session.query(PredictionVector).all()) == 1
    assert len(test_session.query(SceneClassificationVector).all()) == 1
