import datetime
import os

import psycopg2
from geoalchemy2 import Geometry, Raster, RasterElement
from geoalchemy2.types import WKBElement
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    create_engine,
    inspect,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy_utils import create_database, database_exists

from plastic_detection_service.config import POSTGIS_URL

Base = declarative_base()


def get_db_engine():
    return create_engine(POSTGIS_URL, echo=False)


def get_db_session():
    engine = get_db_engine()
    session = sessionmaker(bind=engine)
    return session()


session = get_db_session()


def sql_alch_commit(model):
    session = get_db_session()
    session.add(model)
    session.commit()
    session.close()


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


class PredictionRaster(Base):
    __tablename__ = "prediction_rasters"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    bbox = Column(Geometry(geometry_type="POLYGON", srid=4326))
    dtype = Column(String)
    width = Column(Integer)
    height = Column(Integer)
    bands = Column(Integer)
    prediction_mask = Column(Raster)

    def __init__(
        self,
        timestamp: datetime.datetime,
        bbox: WKBElement,
        dtype: str,
        width: int,
        height: int,
        bands: int,
        prediction_mask: RasterElement,
    ):
        self.timestamp = timestamp
        self.bbox = bbox
        self.dtype = dtype
        self.width = width
        self.height = height
        self.bands = bands
        self.prediction_mask = prediction_mask


class PredictionVector(Base):
    __tablename__ = "prediction_vectors"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    geometry = Column(Geometry(geometry_type="MULTIPOINT", srid=4326))
    prediction_raster_id = Column(
        Integer, ForeignKey("prediction_rasters.id"), nullable=False
    )
    prediction_raster = relationship("PredictionRaster", backref="prediction_vectors")

    def __init__(self, geometry):
        self.geometry = geometry


if __name__ == "__main__":
    engine = get_db_engine()
    if not database_exists(engine.url):
        create_postgis_db(engine)
    create_tables(engine, Base)
