import datetime
import io
from typing import Iterable

from geoalchemy2.shape import from_shape
from shapely.geometry import Polygon
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
        image = Image.from_response_and_raster(
            job_id, download_response, raster, image_url
        )
        self.session.add(image)
        self.session.commit()
        return image

    def insert_prediction_raster(
        self, raster: Raster, image_id: int, model_id: int, raster_url: str
    ) -> PredictionRaster:
        prediction_raster = PredictionRaster.from_raster(
            raster, image_id, model_id, raster_url
        )
        self.session.add(prediction_raster)
        self.session.commit()
        return prediction_raster

    def insert_prediction_vectors(
        self, vectors: Iterable[Vector], raster_id: int
    ) -> list[PredictionVector]:
        prediction_vectors = [
            PredictionVector.from_vector(vector, raster_id) for vector in vectors
        ]
        self.session.bulk_save_objects(prediction_vectors)
        self.session.commit()
        return prediction_vectors

    def insert_scls_vectors(
        self, vectors: Iterable[Vector], image_id: int
    ) -> list[SceneClassificationVector]:
        scls_vectors = [
            SceneClassificationVector.from_vector(vector, image_id)
            for vector in vectors
        ]
        self.session.bulk_save_objects(scls_vectors)
        self.session.commit()
        return scls_vectors

    def commit_all(
        self,
        download_response: DownloadResponse,
        raster: Raster,
        model_id: str,
        model_url: str,
        vectors: Iterable[Vector],
    ) -> tuple[Image, Model, PredictionRaster, list[PredictionVector]]:
        image_url = s3.stream_to_s3(
            io.BytesIO(download_response.content),
            config.S3_BUCKET_NAME,
            f"images/{download_response.bbox}/{download_response.image_id}.tif",
        )
        image = self.insert_image(download_response, raster, image_url)
        model = self.insert_model(model_id, model_url)
        raster_url = s3.stream_to_s3(
            io.BytesIO(raster.content),
            config.S3_BUCKET_NAME,
            f"predictions/{image.id}/{model.id}.tif",
        )
        prediction_raster = self.insert_prediction_raster(
            raster, image.id, model.id, raster_url
        )
        prediction_vectors = self.insert_prediction_vectors(
            vectors, prediction_raster.id
        )
        return image, model, prediction_raster, prediction_vectors
