import datetime
import os

import psycopg2
from geoalchemy2 import Geometry
from geoalchemy2.elements import WKBElement
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
    CREATE OR REPLACE FUNCTION prevent_duplicate_sh_response_insert()
    RETURNS TRIGGER AS $$
    BEGIN
        IF EXISTS (
            SELECT 1
            FROM sentinel_hub_responses
            WHERE timestamp = NEW.timestamp
            AND ST_Equals(bbox, NEW.bbox)
            AND sentinel_hub_id = NEW.sentinel_hub_id
        ) THEN
            RAISE EXCEPTION 'Duplicate sentinel_hub_response insert not allowed.';
        END IF;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    """

    trigger_sql = """
    CREATE TRIGGER check_duplicate_sh_response_insert
    BEFORE INSERT ON sentinel_hub_responses
    FOR EACH ROW
    EXECUTE FUNCTION prevent_duplicate_sh_response_insert();
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


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    model_id = Column(String, nullable=False)
    model_url = Column(String, nullable=False, unique=True)
    model_name = Column(String, nullable=False)
    model_description = Column(String, nullable=False)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)

    __table_args__ = (UniqueConstraint("model_id", "model_url"),)

    image = relationship("Image", backref="models")

    def __init__(
        self, model_id: str, model_url: str, model_name: str, model_description: str
    ):
        self.model_id = model_id
        self.model_url = model_url
        self.model_name = model_name
        self.model_description = model_description


class PredictionVector(Base):
    __tablename__ = "prediction_vectors"

    id = Column(Integer, primary_key=True)
    pixel_value = Column(Integer, nullable=False)
    geometry = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=False)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)

    image = relationship("Image", backref="prediction_vectors")

    def __init__(
        self, pixel_value: int, geometry: WKBElement, sentinel_hub_response_id: int
    ):
        self.pixel_value = pixel_value
        self.geometry = geometry
        self.sentinel_hub_response_id = sentinel_hub_response_id


class SceneClassificationVector(Base):
    __tablename__ = "scene_classification_vectors"

    id = Column(Integer, primary_key=True)
    pixel_value = Column(Integer, nullable=False)
    geometry = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=False)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)

    image = relationship("Image", backref="scene_classification_vectors")

    def __init__(
        self, pixel_value: int, geometry: WKBElement, sentinel_hub_response_id: int
    ):
        self.pixel_value = pixel_value
        self.geometry = geometry
        self.sentinel_hub_response_id = sentinel_hub_response_id


if __name__ == "__main__":
    engine = get_db_engine()
    if not database_exists(engine.url):
        create_postgis_db(engine)
    create_tables(engine, Base)
    create_triggers()
