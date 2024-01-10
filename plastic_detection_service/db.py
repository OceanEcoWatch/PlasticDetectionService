import datetime
import os

import psycopg2
from geoalchemy2 import Geometry
from geoalchemy2.types import WKBElement
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


def create_tables(engine, base):
    base.metadata.drop_all(engine)
    base.metadata.create_all(engine)

    # check tables exists
    ins = inspect(engine)
    for _t in ins.get_table_names():
        print(_t)
    engine.dispose()


def create_triggers():
    custom_trigger_function_sql = """
    CREATE OR REPLACE FUNCTION prevent_duplicate_bbox_insert()
    RETURNS TRIGGER AS $$
    BEGIN
        IF EXISTS (
            SELECT 1
            FROM prediction_bbox
            WHERE timestamp = NEW.timestamp
            AND ST_Equals(bbox, NEW.bbox)
        ) THEN
            RAISE EXCEPTION 'Duplicate bbox and timestamp insert not allowed.';
        END IF;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    """

    trigger_sql = """
    CREATE TRIGGER check_duplicate_bbox_insert
    BEFORE INSERT ON prediction_bbox
    FOR EACH ROW
    EXECUTE FUNCTION prevent_duplicate_bbox_insert();
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


class PredictionBbox(Base):
    __tablename__ = "prediction_bbox"

    id = Column(Integer, primary_key=True)
    sentinel_hub_id = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    bbox = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=False)

    __table_args__ = (UniqueConstraint("timestamp", "bbox", name="_timestamp_bbox_uc"),)

    def __init__(
        self,
        timestamp: datetime.datetime,
        bbox: WKBElement,
    ):
        self.timestamp = timestamp
        self.bbox = bbox


class PredictionVector(Base):
    __tablename__ = "prediction_vectors"

    id = Column(Integer, primary_key=True)
    pixel_value = Column(Integer, nullable=False)
    geometry = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=False)
    prediction_bbox_id = Column(Integer, ForeignKey("prediction_bbox.id"), nullable=False)
    prediction_bbox = relationship("PredictionBbox", backref="prediction_bbox")

    def __init__(self, sentinel_hub_id, pixel_value: int, geometry: WKBElement, prediction_bbox_id: int):
        self.sentinel_hub_id = sentinel_hub_id
        self.pixel_value = pixel_value
        self.geometry = geometry
        self.prediction_bbox_id = prediction_bbox_id


class ClearWaterVector(Base):
    __tablename__ = "clear_water_vectors"

    id = Column(Integer, primary_key=True)
    geometry = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=False)
    prediction_bbox_id = Column(Integer, ForeignKey("prediction_bbox.id"), nullable=False)
    prediction_bbox = relationship("PredictionBbox", backref="clear_water_vectors")

    def __init__(self, geometry: WKBElement, prediction_bbox_id: int):
        self.geometry = geometry
        self.prediction_bbox_id = prediction_bbox_id


if __name__ == "__main__":
    engine = get_db_engine()
    if not database_exists(engine.url):
        create_postgis_db(engine)
    create_tables(engine, Base)
    create_triggers()
