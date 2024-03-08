import datetime
import os

import psycopg2
from geoalchemy2 import Geometry
from geoalchemy2.elements import WKBElement
from geoalchemy2.shape import from_shape
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    inspect,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy_utils import create_database, database_exists

from plastic_detection_service.config import POSTGIS_URL
from plastic_detection_service.models import DownloadResponse, Raster, Vector

Base = declarative_base()


def get_db_engine():
    return create_engine(POSTGIS_URL, echo=False)


def create_postgis_db(engine):
    create_database(url=engine.url)
    conn = psycopg2.connect(
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PW"],
        host=os.environ["DB_HOST"],
        port=os.environ["DB_PORT"],
    )
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION postgis")
    conn.commit()
    cursor.close()
    conn.close()


def check_tables_exists(engine):
    ins = inspect(engine)
    for _t in ins.get_table_names():
        print(_t)


def create_tables(engine, base):
    base.metadata.drop_all(engine)
    base.metadata.create_all(engine)

    check_tables_exists(engine)
    engine.dispose()


def create_triggers():
    custom_trigger_function_sql = """
    CREATE OR REPLACE FUNCTION prevent_duplicate_image_insert()
    RETURNS TRIGGER AS $$
    BEGIN
        IF EXISTS (
            SELECT 1
            FROM images
            WHERE timestamp = NEW.timestamp
            AND ST_Equals(bbox, NEW.bbox)
            AND image_id = NEW.image_id
        ) THEN
            RAISE EXCEPTION 'Image with the same timestamp, bbox and sentinel_hub_id already exists';
        END IF;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    """

    trigger_sql = """
    CREATE TRIGGER prevent_duplicate_image_insert
    BEFORE INSERT ON images
    FOR EACH ROW
    EXECUTE FUNCTION prevent_duplicate_image_insert();
    """

    conn = psycopg2.connect(
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PW"],
        host=os.environ["DB_HOST"],
        port=os.environ["DB_PORT"],
    )
    cursor = conn.cursor()
    cursor.execute(custom_trigger_function_sql)
    cursor.execute(trigger_sql)
    conn.commit()
    cursor.close()
    conn.close()


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True)
    image_id = Column(String, nullable=False)
    image_url = Column(String, nullable=False, unique=True)
    timestamp = Column(DateTime, nullable=False)
    image_width = Column(Integer, nullable=False)
    image_height = Column(Integer, nullable=False)
    bands = Column(Integer, nullable=False)
    provider = Column(String, nullable=False)
    bbox = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=False)

    __table_args__ = (UniqueConstraint("image_id", "timestamp", "bbox"),)

    def __init__(
        self,
        image_id: str,
        image_url: str,
        timestamp: datetime.datetime,
        image_width: int,
        image_height: int,
        bands: int,
        provider: str,
        bbox: WKBElement,
    ):
        self.image_id = image_id
        self.image_url = image_url
        self.timestamp = timestamp
        self.image_width = image_width
        self.image_height = image_height
        self.bands = bands
        self.provider = provider
        self.bbox = bbox

    @classmethod
    def from_response_and_raster(
        cls, response: DownloadResponse, raster: Raster, image_url: str
    ):
        return cls(
            image_id=response.image_id,
            image_url=image_url,
            timestamp=response.timestamp,
            image_width=raster.size[0],
            image_height=raster.size[1],
            bands=len(raster.bands),
            provider=response.data_collection,
            bbox=from_shape(raster.geometry),
        )


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    model_id = Column(String, nullable=False)
    model_url = Column(String, nullable=False, unique=True)
    model_name = Column(String, nullable=False)

    __table_args__ = (UniqueConstraint("model_id", "model_url"),)

    def __init__(self, model_id: str, model_url: str, model_name: str):
        self.model_id = model_id
        self.model_url = model_url
        self.model_name = model_name


class PredictionVector(Base):
    __tablename__ = "prediction_vectors"

    id = Column(Integer, primary_key=True)
    pixel_value = Column(Integer, nullable=False)
    geometry = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=False)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)

    image = relationship("Image", backref="prediction_vectors")
    model = relationship("Model", backref="prediction_vectors")

    def __init__(
        self, pixel_value: int, geometry: WKBElement, image_id: int, model_id: int
    ):
        self.pixel_value = pixel_value
        self.geometry = geometry
        self.image_id = image_id
        self.model_id = model_id

    @classmethod
    def from_vector(cls, vector: Vector, image_id: int, model_id: int):
        return cls(
            pixel_value=vector.pixel_value,
            geometry=from_shape(vector.geometry),
            image_id=image_id,
            model_id=model_id,
        )


class SceneClassificationVector(Base):
    __tablename__ = "scene_classification_vectors"

    id = Column(Integer, primary_key=True)
    pixel_value = Column(Integer, nullable=False)
    geometry = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=False)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)

    image = relationship("Image", backref="scene_classification_vectors")

    def __init__(self, pixel_value: int, geometry: WKBElement, image_id: int):
        self.pixel_value = pixel_value
        self.geometry = geometry
        self.image_id = image_id

    @classmethod
    def from_vector(cls, vector: Vector, image_id: int):
        return cls(
            pixel_value=vector.pixel_value,
            geometry=from_shape(vector.geometry),
            image_id=image_id,
        )


if __name__ == "__main__":
    engine = get_db_engine()
    if not database_exists(engine.url):
        create_postgis_db(engine)
    create_tables(engine, Base)
    create_triggers()
