import datetime
import io
import logging
from typing import Iterable, Optional

from geoalchemy2.shape import from_shape
from pyproj import Transformer
from shapely.geometry import Polygon, shape
from shapely.ops import transform
from sqlalchemy.exc import IntegrityError, NoResultFound
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
        status: JobStatus = JobStatus.PENDING,
        created_at: datetime.datetime = datetime.datetime.now(),
    ) -> Job:
        job = Job(
            aoi_id=aoi_id,
            model_id=model_id,
            status=status,
            created_at=created_at,
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
        transformer = Transformer.from_crs(raster.crs, target_crs, always_xy=True)
        transformed_geometry = transform(transformer.transform, shape(raster.geometry))
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
    ) -> tuple[
        Optional[Image], Optional[PredictionRaster], Optional[list[PredictionVector]]
    ]:
        unique_id = f"{download_response.bbox}/{download_response.image_id}"
        image_url = s3.stream_to_s3(
            io.BytesIO(download_response.content),
            config.S3_BUCKET_NAME,
            f"images/{unique_id}.tif",
        )
        try:
            image_db = self.insert.insert_image(
                download_response, image, image_url, job_id
            )
        except IntegrityError:
            LOGGER.warning(f"Image {unique_id} already exists. Skipping")
            return None, None, None

        pred_raster_url = s3.stream_to_s3(
            io.BytesIO(pred_raster.content),
            config.S3_BUCKET_NAME,
            f"predictions/{unique_id}.tif",
        )
        prediction_raster_db = self.insert.insert_prediction_raster(
            pred_raster, image_db.id, pred_raster_url
        )
        prediction_vectors_db = self.insert.insert_prediction_vectors(
            vectors, prediction_raster_db.id
        )
        return image_db, prediction_raster_db, prediction_vectors_db


def set_init_job_status(db_session: Session, job_id: int, model_id: int):
    model = db_session.query(Model).filter(Model.id == model_id).first()
    if model is None:
        update_job_status(db_session, job_id, JobStatus.FAILED)
        raise NoResultFound("Model not found")
    job = db_session.query(Job).filter(Job.id == job_id).first()

    if job is None:
        update_job_status(db_session, job_id, JobStatus.FAILED)
        raise NoResultFound("Job not found")

    else:
        LOGGER.info(f"Updating job {job_id} to in progress")
        update_job_status(db_session, job_id, JobStatus.IN_PROGRESS)


def update_job_status(db_session: Session, job_id: int, status: JobStatus):
    db_session.query(Job).filter(Job.id == job_id).update({"status": status})
    db_session.commit()
