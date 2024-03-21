import datetime

import psycopg2
import pytest
from geoalchemy2.shape import from_shape
from shapely.geometry.polygon import Polygon
from sqlalchemy import create_engine
from sqlalchemy.exc import DataError, IntegrityError
from sqlalchemy.orm import Session, create_session
from sqlalchemy_utils import create_database, database_exists

from plastic_detection_service.config import DB_NAME
from plastic_detection_service.database.insert import Insert
from plastic_detection_service.database.models import (
    Base,
    Image,
    Model,
    PredictionRaster,
    PredictionVector,
    SceneClassificationVector,
)
from plastic_detection_service.models import DownloadResponse, Raster, Vector
from plastic_detection_service.types import HeightWidth

DB_NAME = "oew_test"
DB_USER = "postgres"
DB_PW = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

TEST_DB_URL = f"postgresql://{DB_USER}:{DB_PW}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


@pytest.fixture
def create_test_db():
    engine = create_engine(TEST_DB_URL)
    if not database_exists(engine.url):
        create_database(engine.url)
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PW,
            host=DB_HOST,
            port=DB_PORT,
        )
        cursor = conn.cursor()
        cursor.execute("CREATE EXTENSION postgis")
        conn.commit()
        cursor.close()
        conn.close()
    yield engine
    engine.dispose()


@pytest.fixture
def mock_session():
    class _MockSession:
        def __init__(self):
            self.queries = []

        def add(self, obj):
            self.queries.append(obj)

        def commit(self):
            pass

    return _MockSession()


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
def mock_session():
    class MockSession:
        def __init__(self):
            self.queries = []

        def add(self, obj):
            self.queries.append(obj)

        def commit(self):
            pass

        def bulk_save_objects(self, objs):
            self.queries.extend(objs)

    return MockSession()


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


def test_insert_mock_session(mock_session, download_response, db_raster, db_vectors):
    insert = Insert(mock_session)
    image = insert.insert_image(download_response, db_raster, "test_image_url")

    model = insert.insert_model("test_model_id", "test_model_url")
    raster = insert.insert_prediction_raster(
        db_raster, image.id, model.id, "test_raster_url"
    )
    prediction_vectors = insert.insert_prediction_vectors(db_vectors, raster.id)

    scls_vectors = insert.insert_scls_vectors(db_vectors, image.id)

    assert image.image_id == download_response.image_id
    assert image.image_url == "test_image_url"
    assert image.timestamp == download_response.timestamp
    assert image.dtype == db_raster.dtype
    assert image.image_width == db_raster.size.width
    assert image.image_height == db_raster.size.height
    assert image.bands == len(db_raster.bands)
    assert image.provider == download_response.data_collection
    assert image.bbox == from_shape(db_raster.geometry)

    assert model.model_id == "test_model_id"
    assert model.model_url == "test_model_url"

    assert raster.raster_url == "test_raster_url"
    assert raster.dtype == db_raster.dtype
    assert raster.image_id == image.id
    assert raster.model_id == model.id

    assert len(prediction_vectors) == 1
    assert prediction_vectors[0].prediction_raster_id == raster.id

    assert len(scls_vectors) == 1
    assert scls_vectors[0].image_id == image.id

    assert len(mock_session.queries) == 5


@pytest.mark.integration
def test_image_invalid_dtype(test_session):
    image = Image(
        image_id="test_image_id",
        image_url="test_image_url",
        timestamp=datetime.datetime.now(),
        dtype="INVALID",
        image_width=10,
        image_height=10,
        bands=3,
        provider="test_data_collection",
        bbox=from_shape(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])),
    )
    test_session.add(image)
    with pytest.raises(DataError):
        test_session.commit()


@pytest.mark.integration
def test_image_invalid_band(test_session):
    image = Image(
        image_id="test_image_id",
        image_url="test_image_url",
        timestamp=datetime.datetime.now(),
        dtype="uint8",
        image_width=10,
        image_height=10,
        bands=0,
        provider="test_data_collection",
        bbox=from_shape(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])),
    )
    test_session.add(image)
    with pytest.raises(IntegrityError):
        test_session.commit()


@pytest.mark.integration
def test_image_unique_constraint(test_session):
    image = Image(
        image_id="test_image_id",
        image_url="test_image_url",
        timestamp=datetime.datetime(2021, 1, 1, 0, 0, 0),  # same timestamp
        dtype="uint8",
        image_width=10,
        image_height=10,
        bands=3,
        provider="test_data_collection",
        bbox=from_shape(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])),
    )
    duplicate_image = Image(
        image_id="test_image_id",
        image_url="other_test_image_url",
        timestamp=datetime.datetime(2021, 1, 1, 0, 0, 0),  # same timestamp
        dtype="uint8",
        image_width=50,
        image_height=50,
        bands=5,
        provider="other_test_data_collection",
        bbox=from_shape(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])),
    )

    test_session.add(image)
    test_session.commit()

    test_session.add(duplicate_image)
    with pytest.raises(IntegrityError):
        test_session.commit()


@pytest.mark.integration
def test_insert_db(
    db_raster: Raster,
    download_response: DownloadResponse,
    db_vectors: list[Vector],
    test_session: Session,
):
    insert = Insert(test_session)
    image = insert.insert_image(download_response, db_raster, "test_image_url")
    model = insert.insert_model("test_model_id", "test_model_url")
    raster = insert.insert_prediction_raster(
        db_raster, image.id, model.id, "test_raster_url"
    )
    insert.insert_prediction_vectors(
        db_vectors,
        raster.id,
    )
    insert.insert_scls_vectors(db_vectors, image.id)

    assert len(test_session.query(Image).all()) == 1
    assert len(test_session.query(Model).all()) == 1
    assert len(test_session.query(PredictionRaster).all()) == 1
    assert len(test_session.query(PredictionVector).all()) == 1
    assert len(test_session.query(SceneClassificationVector).all()) == 1
    assert test_session.query(Image).first().id == image.id
    assert test_session.query(Model).first().id == model.id
    assert test_session.query(PredictionRaster).first().id == raster.id
    assert (
        test_session.query(PredictionVector).first().prediction_raster_id == raster.id
    )
    assert test_session.query(SceneClassificationVector).first().image_id == image.id

    assert all(
        [
            raster.image_id == image.id
            for raster in test_session.query(PredictionRaster).all()
        ]
    )
    assert all(
        [
            vector.prediction_raster_id == raster.id
            for vector in test_session.query(PredictionVector).all()
        ]
    )

    assert all(
        [
            scls.image_id == image.id
            for scls in test_session.query(SceneClassificationVector).all()
        ]
    )
