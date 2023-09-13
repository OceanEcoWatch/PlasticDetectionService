import datetime
import os
from typing import Optional

import psycopg2
from geoalchemy2 import Geometry, Raster, RasterElement
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
    cursor.execute("CREATE EXTENSION postgis_raster")
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
    CREATE OR REPLACE FUNCTION prevent_duplicate_raster_insert()
    RETURNS TRIGGER AS $$
    BEGIN
        IF EXISTS (
            SELECT 1
            FROM prediction_rasters
            WHERE timestamp = NEW.timestamp
            AND ST_Equals(bbox, NEW.bbox)
        ) THEN
            RAISE EXCEPTION 'Duplicate raster insert not allowed.';
        END IF;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    """

    trigger_sql = """
    CREATE TRIGGER check_duplicate_raster_insert
    BEFORE INSERT ON prediction_rasters
    FOR EACH ROW
    EXECUTE FUNCTION prevent_duplicate_raster_insert();
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


class PredictionRaster(Base):
    __tablename__ = "prediction_rasters"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    bbox = Column(Geometry(geometry_type="POLYGON", srid=4326))
    dtype = Column(String)
    width = Column(Integer)
    height = Column(Integer)
    bands = Column(Integer)
    prediction_mask = Column(Raster, nullable=False)
    clear_water_mask = Column(Raster, nullable=True)

    __table_args__ = (UniqueConstraint("timestamp", "bbox", name="uq_timestamp_bbox"),)

    def __init__(
        self,
        timestamp: datetime.datetime,
        bbox: WKBElement,
        dtype: str,
        width: int,
        height: int,
        bands: int,
        prediction_mask: RasterElement,
        clear_water_mask: Optional[RasterElement] = None,
    ):
        self.timestamp = timestamp
        self.bbox = bbox
        self.dtype = dtype
        self.width = width
        self.height = height
        self.bands = bands
        self.prediction_mask = prediction_mask
        self.clear_water_mask = clear_water_mask


class PredictionVector(Base):
    __tablename__ = "prediction_vectors"

    id = Column(Integer, primary_key=True)
    pixel_value = Column(Integer)
    geometry = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=False)
    prediction_raster_id = Column(
        Integer, ForeignKey("prediction_rasters.id"), nullable=False
    )
    prediction_raster = relationship("PredictionRaster", backref="prediction_vectors")

    def __init__(
        self, pixel_value: int, geometry: WKBElement, prediction_raster_id: int
    ):
        self.pixel_value = pixel_value
        self.geometry = geometry
        self.prediction_raster_id = prediction_raster_id


if __name__ == "__main__":
    engine = get_db_engine()
    if not database_exists(engine.url):
        create_postgis_db(engine)
    create_tables(engine, Base)
    create_triggers()
    create_triggers()
