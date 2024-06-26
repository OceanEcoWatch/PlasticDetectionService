import datetime
import io
import logging
from typing import Iterable, Optional

from geoalchemy2 import WKBElement
from geoalchemy2.shape import from_shape
from shapely.geometry import Polygon, box
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session

from src import config
from src.aws import s3
from src.database.models import (
    AOI,
    Image,
    Job,
    JobStatus,
    Model,
    PredictionRaster,
    PredictionVector,
    SceneClassificationVector,
)
from src.geo_utils import reproject_geometry
from src.models import DownloadResponse, Raster, Vector

LOGGER = logging.getLogger(__name__)


class Insert:
    def __init__(self, session: Session):
        self.session = session

    def insert_aoi(
        self,
        name: str,
        created_at: datetime.datetime,
        geometry: Polygon,
        is_deleted: bool = False,
    ) -> AOI:
        aoi = AOI(
            name=name,
            created_at=created_at,
            geometry=from_shape(geometry),
            is_deleted=is_deleted,
        )
        self.session.add(aoi)
        self.session.commit()
        return aoi

    def insert_model(self, model_id: str, model_url: str) -> Model:
        model = Model(model_id=model_id, model_url=model_url)
        self.session.add(model)
        self.session.commit()
        return model

    def insert_job(
        self,
        aoi_id: int,
        model_id: int,
        time_range: tuple[datetime.datetime, datetime.datetime],
        maxcc: float,
        status: JobStatus = JobStatus.PENDING,
        created_at: datetime.datetime = datetime.datetime.now(),
    ) -> Job:
        job = Job(
            aoi_id=aoi_id,
            model_id=model_id,
            status=status,
            created_at=created_at,
            start_date=time_range[0],
            end_date=time_range[1],
            maxcc=maxcc,
        )
        self.session.add(job)
        self.session.commit()
        return job

    def insert_image(
        self,
        download_response: DownloadResponse,
        raster: Raster,
        image_url: str,
        job_id: int,
    ) -> Image:
        target_crs = 4326
        transformed_geometry = reproject_geometry(
            raster.geometry, raster.crs, target_crs
        )
        image = Image(
            image_id=download_response.image_id,
            image_url=image_url,
            timestamp=download_response.timestamp,
            dtype=str(raster.dtype),
            crs=raster.crs,
            resolution=raster.resolution,
            image_width=raster.size[0],
            image_height=raster.size[1],
            bands=len(raster.bands),
            provider=download_response.data_collection,
            bbox=from_shape(transformed_geometry, srid=target_crs),
            job_id=job_id,
        )
        self.session.add(image)
        self.session.commit()
        return image

    def insert_prediction_raster(
        self, raster: Raster, image_id: int, raster_url: str
    ) -> PredictionRaster:
        prediction_raster = PredictionRaster(
            raster_url=raster_url,
            dtype=str(raster.dtype),
            image_width=raster.size[0],
            image_height=raster.size[1],
            bbox=from_shape(raster.geometry, srid=raster.crs),
            image_id=image_id,
        )
        self.session.add(prediction_raster)
        self.session.commit()
        return prediction_raster

    def insert_prediction_vectors(
        self, vectors: Iterable[Vector], raster_id: int
    ) -> list[PredictionVector]:
        prediction_vectors = [
            PredictionVector(
                v.pixel_value, from_shape(v.geometry, srid=v.crs), raster_id
            )
            for v in vectors
        ]
        self.session.bulk_save_objects(prediction_vectors)
        self.session.commit()
        return prediction_vectors

    def insert_scls_vectors(
        self, vectors: Iterable[Vector], image_id: int
    ) -> list[SceneClassificationVector]:
        scls_vectors = [
            SceneClassificationVector(
                v.pixel_value, from_shape(v.geometry, srid=v.crs), image_id
            )
            for v in vectors
        ]
        self.session.bulk_save_objects(scls_vectors)
        self.session.commit()
        return scls_vectors


class InsertJob:
    def __init__(self, insert: Insert):
        self.insert = insert

    def insert_all(
        self,
        job_id: int,
        download_response: DownloadResponse,
        image: Raster,
        pred_raster: Raster,
        vectors: Iterable[Vector],
        scl_vectors: Iterable[Vector],
    ) -> Optional[
        tuple[Image, PredictionRaster, PredictionVector, SceneClassificationVector]
    ]:
        unique_id = f"{download_response.bbox}/{download_response.image_id}"
        image_url = s3.stream_to_s3(
            io.BytesIO(download_response.content),
            config.S3_BUCKET_NAME,
            f"images/{unique_id}.tif",
        )

        image_db = self.insert.insert_image(download_response, image, image_url, job_id)
        LOGGER.info(f"Inserted image {unique_id} into database")
        pred_raster_url = s3.stream_to_s3(
            io.BytesIO(pred_raster.content),
            config.S3_BUCKET_NAME,
            f"predictions/{unique_id}.tif",
        )
        prediction_raster_db = self.insert.insert_prediction_raster(
            pred_raster, image_db.id, pred_raster_url
        )
        LOGGER.info(f"Inserted prediction raster for image {unique_id} into database")
        prediction_vectors_db = self.insert.insert_prediction_vectors(
            vectors, prediction_raster_db.id
        )
        LOGGER.info(f"Inserted prediction vectors for image {unique_id} into database")
        scl_vectors_db = self.insert.insert_scls_vectors(scl_vectors, image_db.id)
        LOGGER.info(
            f"Inserted scene classification vectors for image {unique_id} into database"
        )
        return image_db, prediction_raster_db, prediction_vectors_db, scl_vectors_db


def set_init_job_status(db_session: Session, job_id: int) -> Job:
    job = db_session.query(Job).filter(Job.id == job_id).first()

    if job is None:
        update_job_status(db_session, job_id, JobStatus.FAILED)
        raise NoResultFound("Job not found")

    else:
        LOGGER.info(f"Updating job {job_id} to in progress")
        update_job_status(db_session, job_id, JobStatus.IN_PROGRESS)
    return job


def update_job_status(db_session: Session, job_id: int, status: JobStatus):
    db_session.query(Job).filter(Job.id == job_id).update({"status": status})
    db_session.commit()


def image_in_db(
    db_session: Session, download_response: DownloadResponse, job_id: int
) -> bool:
    bbox = box(*download_response.bbox)
    bbox_geom_4326 = reproject_geometry(bbox, download_response.crs, 4326)
    bbox_geom = WKBElement(bbox_geom_4326.wkb, srid=4326)

    image = (
        db_session.query(Image)
        .filter(Image.image_id == download_response.image_id)
        .filter(Image.timestamp == download_response.timestamp)
        .filter(Image.bbox.ST_Equals(bbox_geom))
        .filter(Image.job_id == job_id)
        .first()
    )

    return image is not None
