import datetime

from geoalchemy2 import Geometry
from geoalchemy2.elements import WKBElement
from geoalchemy2.shape import from_shape
from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

from plastic_detection_service.models import (
    DownloadResponse,
    Raster,
    Vector,
)
from plastic_detection_service.types import IMAGE_DTYPES

Base = declarative_base()

CONSTRAINT_STR = String(255)


class Image(Base):
    __tablename__ = "images"
    __table_args__ = (UniqueConstraint("image_id", "timestamp", "bbox"),)

    id = Column(Integer, primary_key=True)
    image_id = Column(CONSTRAINT_STR, nullable=False)
    image_url = Column(CONSTRAINT_STR, nullable=False, unique=True)
    timestamp = Column(DateTime, nullable=False)
    dtype = Column(
        Enum(*IMAGE_DTYPES, name="image_dtype"),
        nullable=False,
    )
    image_width = Column(Integer, nullable=False)
    image_height = Column(Integer, nullable=False)
    bands = Column(Integer, CheckConstraint("bands>0 AND bands<=100"), nullable=False)
    provider = Column(CONSTRAINT_STR, nullable=False)
    bbox = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=False)

    def __init__(
        self,
        image_id: str,
        image_url: str,
        timestamp: datetime.datetime,
        dtype: str,
        image_width: int,
        image_height: int,
        bands: int,
        provider: str,
        bbox: WKBElement,
    ):
        self.image_id = image_id
        self.image_url = image_url
        self.timestamp = timestamp
        self.dtype = dtype
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
            dtype=raster.dtype,
            image_width=raster.size[0],
            image_height=raster.size[1],
            bands=len(raster.bands),
            provider=response.data_collection,
            bbox=from_shape(raster.geometry),
        )


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    model_id = Column(CONSTRAINT_STR, nullable=False, unique=True)
    model_url = Column(CONSTRAINT_STR, nullable=False, unique=True)

    def __init__(self, model_id: str, model_url: str):
        self.model_id = model_id
        self.model_url = model_url


class PredictionRaster(Base):
    __tablename__ = "prediction_rasters"

    id = Column(Integer, primary_key=True)
    raster_url = Column(CONSTRAINT_STR, nullable=False, unique=True)
    dtype = Column(
        Enum(*IMAGE_DTYPES, name="image_dtype"),
        nullable=False,
    )
    image_width = Column(Integer, nullable=False)
    image_height = Column(Integer, nullable=False)
    bbox = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=False)

    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    image = relationship("Image", backref="prediction_rasters")
    model = relationship("Model", backref="prediction_rasters")

    def __init__(
        self,
        raster_url: str,
        dtype: str,
        image_width: int,
        image_height: int,
        bbox: WKBElement,
        image_id: int,
        model_id: int,
    ):
        self.raster_url = raster_url
        self.dtype = dtype
        self.image_width = image_width
        self.image_height = image_height
        self.bbox = bbox
        self.image_id = image_id
        self.model_id = model_id

    @classmethod
    def from_raster(cls, raster: Raster, image_id: int, model_id: int, raster_url: str):
        return cls(
            raster_url=raster_url,
            dtype=raster.dtype,
            image_width=raster.size[0],
            image_height=raster.size[1],
            bbox=from_shape(raster.geometry),
            image_id=image_id,
            model_id=model_id,
        )


class PredictionVector(Base):
    __tablename__ = "prediction_vectors"

    id = Column(Integer, primary_key=True)
    pixel_value = Column(Integer, nullable=False)
    geometry = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=False)

    prediction_raster_id = Column(
        Integer, ForeignKey("prediction_rasters.id"), nullable=False
    )

    prediction_raster = relationship("PredictionRaster", backref="prediction_vectors")

    def __init__(self, pixel_value: int, geometry: WKBElement, raster_id: int):
        self.pixel_value = pixel_value
        self.geometry = geometry

        self.prediction_raster_id = raster_id

    @classmethod
    def from_vector(cls, vector: Vector, raster_id: int):
        return cls(
            pixel_value=vector.pixel_value,
            geometry=from_shape(vector.geometry),
            raster_id=raster_id,
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
