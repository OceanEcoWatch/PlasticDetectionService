import datetime
import os

import psycopg2
from geoalchemy2 import Geometry
from geoalchemy2.elements import WKBElement
from sqlalchemy import (
    Column,
    DateTime,
    Float,
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


class SentinelHubResponse(Base):
    __tablename__ = "sentinel_hub_responses"

    id = Column(Integer, primary_key=True)
    sentinel_hub_id = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    bbox = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=False)
    image_width = Column(Integer, nullable=False)
    image_height = Column(Integer, nullable=False)
    max_cc = Column(Integer, nullable=False)
    data_collection = Column(String, nullable=False)
    mime_type = Column(String, nullable=False)
    evalscript = Column(String, nullable=False)
    request_datetime = Column(DateTime, nullable=False)
    processing_units_spent = Column(Float, nullable=False)

    __table_args__ = (UniqueConstraint("sentinel_hub_id", "timestamp", "bbox"),)

    def __init__(
        self,
        sentinel_hub_id: str,
        timestamp: datetime.datetime,
        bbox: WKBElement,
        image_width: int,
        image_height: int,
        max_cc: float,
        data_collection: str,
        mime_type: str,
        evalscript: str,
        request_datetime: datetime.datetime,
        processing_units_spent: float,
    ):
        self.sentinel_hub_id = sentinel_hub_id
        self.timestamp = timestamp
        self.bbox = bbox
        self.image_width = image_width
        self.image_height = image_height
        self.max_cc = max_cc
        self.data_collection = data_collection
        self.mime_type = mime_type
        self.evalscript = evalscript
        self.request_datetime = request_datetime
        self.processing_units_spent = processing_units_spent


class PredictionVector(Base):
    __tablename__ = "prediction_vectors"

    id = Column(Integer, primary_key=True)
    pixel_value = Column(Integer, nullable=False)
    geometry = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=False)
    sentinel_hub_response_id = Column(Integer, ForeignKey("sentinel_hub_responses.id"), nullable=False)
    sentinel_hub_response = relationship("SentinelHubResponse", backref="prediction_vectors")

    def __init__(self, pixel_value: int, geometry: WKBElement, sentinel_hub_response_id: int):
        self.pixel_value = pixel_value
        self.geometry = geometry
        self.sentinel_hub_response_id = sentinel_hub_response_id


class ClearWaterVector(Base):
    __tablename__ = "clear_water_vectors"

    id = Column(Integer, primary_key=True)
    geometry = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=False)
    sentinel_hub_response_id = Column(Integer, ForeignKey("sentinel_hub_responses.id"), nullable=False)
    sentinel_hub_response = relationship("SentinelHubResponse", backref="clear_water_vectors")

    def __init__(self, geometry: WKBElement, sentinel_hub_response_id: int):
        self.geometry = geometry
        self.sentinel_hub_response_id = sentinel_hub_response_id


if __name__ == "__main__":
    engine = get_db_engine()
    if not database_exists(engine.url):
        create_postgis_db(engine)
    create_tables(engine, Base)
    create_triggers()
