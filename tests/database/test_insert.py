import datetime

import psycopg2
import pytest
from geoalchemy2.shape import from_shape
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from sqlalchemy import create_engine
from sqlalchemy.exc import DataError, IntegrityError
from sqlalchemy.orm import Session, create_session
from sqlalchemy_utils import create_database, database_exists, drop_database

from src.database.insert import Insert
from src.database.models import (
    AOI,
    Base,
    Image,
    Job,
    JobStatus,
    Model,
    PredictionRaster,
    PredictionVector,
    SceneClassificationVector,
)
from src.models import DownloadResponse, Raster, Vector
from src.types import HeightWidth
from tests.conftest import TEST_AOI_POLYGON

DB_NAME = "oew_test"
DB_USER = "postgres"
DB_PW = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

TEST_DB_URL = f"postgresql://{DB_USER}:{DB_PW}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


@pytest.fixture
def create_test_db():
    engine = create_engine(TEST_DB_URL)
    if database_exists(engine.url):
        drop_database(engine.url)
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
        resolution=10.0,
        geometry=Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
    )


@pytest.fixture
def db_vectors():
    return [Vector(geometry=Point(0, 0), pixel_value=1, crs=4326)]


@pytest.fixture
def db_scls_vectors():
    return [
        Vector(
            geometry=Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]), pixel_value=1, crs=4326
        )
    ]


@pytest.fixture
def aoi(test_session):
    aoi = AOI(
        name="test_aoi",
        created_at=datetime.datetime.now(),
        geometry=from_shape(TEST_AOI_POLYGON, srid=4326),
    )
    test_session.add(aoi)
    test_session.commit()
    return aoi


@pytest.fixture
def model(test_session):
    model = Model(model_id="test_model_id", model_url="test_model_url")
    test_session.add(model)
    test_session.commit()
    return model


@pytest.fixture
def job(aoi, model, test_session):
    job = Job(JobStatus.PENDING, datetime.datetime.now(), aoi.id, model.id)
    test_session.add(job)
    test_session.commit()
    return job


def test_insert_mock_session(
    mock_session, download_response, db_raster, db_vectors, db_scls_vectors
):
    insert = Insert(mock_session)
    model = insert.insert_model("test_model_id", "test_model_url")
    aoi = insert.insert_aoi(
        name="test_aoi",
        created_at=datetime.datetime.now(),
        geometry=Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
    )
    job = insert.insert_job(aoi.id, model.id)
    image = insert.insert_image(download_response, db_raster, "test_image_url", job.id)

    raster = insert.insert_prediction_raster(
        db_raster, image.id, model.id, "test_raster_url"
    )
    prediction_vectors = insert.insert_prediction_vectors(db_vectors, raster.id)

    scls_vectors = insert.insert_scls_vectors(db_scls_vectors, image.id)

    assert aoi.name == "test_aoi"
    assert aoi.geometry == from_shape(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]))
    assert job.aoi_id == aoi.id
    assert job.model_id == model.id
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


@pytest.mark.integration
def test_image_invalid_dtype(test_session, aoi, model, job):
    test_session.add(aoi)
    test_session.add(model)
    test_session.add(job)

    image = Image(
        image_id="test_image_id",
        image_url="test_image_url",
        timestamp=datetime.datetime.now(),
        dtype="INVALID",
        resolution=10.0,
        image_width=10,
        image_height=10,
        bands=3,
        provider="test_data_collection",
        bbox=from_shape(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])),
        job_id=job.id,
    )
    test_session.add(image)
    with pytest.raises(DataError):
        test_session.commit()


@pytest.mark.integration
def test_image_invalid_band(test_session, aoi, model, job):
    image = Image(
        image_id="test_image_id",
        image_url="test_image_url",
        timestamp=datetime.datetime.now(),
        dtype="uint8",
        resolution=10.0,
        image_width=10,
        image_height=10,
        bands=0,
        provider="test_data_collection",
        bbox=from_shape(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])),
        job_id=job.id,
    )
    test_session.add(image)
    with pytest.raises(IntegrityError):
        test_session.commit()


@pytest.mark.integration
def test_image_unique_constraint(test_session, aoi, model, job):
    image = Image(
        image_id="test_image_id",
        image_url="test_image_url",
        timestamp=datetime.datetime(2021, 1, 1, 0, 0, 0),  # same timestamp
        dtype="uint8",
        resolution=10.0,
        image_width=10,
        image_height=10,
        bands=3,
        provider="test_data_collection",
        bbox=from_shape(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])),
        job_id=job.id,
    )
    duplicate_image = Image(
        image_id="test_image_id",
        image_url="other_test_image_url",
        timestamp=datetime.datetime(2021, 1, 1, 0, 0, 0),  # same timestamp
        dtype="uint8",
        resolution=10.0,
        image_width=50,
        image_height=50,
        bands=5,
        provider="other_test_data_collection",
        bbox=from_shape(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])),
        job_id=job.id,
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
    db_scls_vectors: list[Vector],
    test_session: Session,
):
    insert = Insert(test_session)
    aoi = insert.insert_aoi(
        "test_aoi", created_at=datetime.datetime.now(), geometry=db_raster.geometry
    )
    model = insert.insert_model("test_model_id", "test_model_url")
    job = insert.insert_job(aoi.id, model.id)
    image = insert.insert_image(download_response, db_raster, "test_image_url", job.id)
    raster = insert.insert_prediction_raster(
        db_raster, image.id, model.id, "test_raster_url"
    )
    insert.insert_prediction_vectors(
        db_vectors,
        raster.id,
    )
    insert.insert_scls_vectors(db_scls_vectors, image.id)

    assert len(test_session.query(AOI).all()) == 1
    assert len(test_session.query(Job).all()) == 1
    assert len(test_session.query(Image).all()) == 1
    assert len(test_session.query(Model).all()) == 1
    assert len(test_session.query(PredictionRaster).all()) == 1
    assert len(test_session.query(PredictionVector).all()) == 1
    assert len(test_session.query(SceneClassificationVector).all()) == 1
    assert test_session.query(AOI).first().id == aoi.id
    assert test_session.query(Job).first().id == job.id
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
