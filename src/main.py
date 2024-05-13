import io
import itertools
import logging
from typing import Iterable, Optional

import click
import rasterio
from sentinelhub.constants import MimeType
from sentinelhub.data_collections import DataCollection
from sqlalchemy.exc import IntegrityError, NoResultFound

from src import config
from src._types import BoundingBox, HeightWidth, TimeRange
from src.aws import s3
from src.database.connect import create_db_session
from src.database.insert import Insert
from src.database.models import (
    Image,
    Job,
    JobStatus,
    Model,
    PredictionRaster,
    PredictionVector,
)
from src.inference.inference_callback import RunpodInferenceCallback
from src.models import Raster, Vector
from src.raster_op.band import RasterioRemoveBand
from src.raster_op.composite import CompositeRasterOperation
from src.raster_op.convert import RasterioDtypeConversion
from src.raster_op.inference import RasterioInference
from src.raster_op.merge import RasterioRasterMerge
from src.raster_op.padding import RasterioRasterPad, RasterioRasterUnpad
from src.raster_op.reproject import RasterioRasterReproject
from src.raster_op.split import RasterioRasterSplit
from src.raster_op.vectorize import RasterioRasterToVector

from .download.abstractions import DownloadResponse
from .download.evalscripts import L2A_12_BANDS_SCL
from .download.sh import (
    SentinelHubDownload,
    SentinelHubDownloadParams,
)
from .raster_op.utils import create_raster

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _create_raster(image: DownloadResponse) -> Raster:
    with rasterio.open(io.BytesIO(image.content)) as src:
        np_image = src.read().copy()
        meta = src.meta.copy()
        bounds = BoundingBox(*src.bounds)
    return create_raster(
        content=image.content,
        image=np_image,
        bounds=bounds,
        meta=meta,
        padding_size=HeightWidth(0, 0),
    )


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


def set_init_job_status(db_session, job_id, model_id):
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


def update_job_status(db_session, job_id, status):
    db_session.query(Job).filter(Job.id == job_id).update({"status": status})
    db_session.commit()


@click.command()
@click.option(
    "--bbox",
    nargs=4,
    type=float,
    help="Bounding box of the area to be processed. Format: min_lon min_lat max_lon max_lat",
)
@click.option(
    "--time-range",
    nargs=2,
    help="Time interval to be processed. Format: YYYY-MM-DD YYYY-MM-DD",
)
@click.option("--maxcc", type=float, required=True)
@click.option("--job-id", type=int, required=True)
@click.option("--model-id", type=int, required=True)
def main(
    bbox: BoundingBox,
    time_range: tuple[str, str],
    maxcc: float,
    job_id: int,
    model_id: int,
):
    with create_db_session() as db_session:
        set_init_job_status(db_session, job_id, model_id)

    downloader = SentinelHubDownload(
        SentinelHubDownloadParams(
            bbox=bbox,
            time_interval=TimeRange(*time_range),
            maxcc=maxcc,
            config=config.SH_CONFIG,
            evalscript=L2A_12_BANDS_SCL,
            data_collection=DataCollection.SENTINEL2_L2A,
            mime_type=MimeType.TIFF,
        )
    )

    download_generator = downloader.download_images()
    try:
        first_response = next(download_generator)
    except StopIteration:
        with create_db_session() as db_session:
            update_job_status(db_session, job_id, JobStatus.FAILED)
        raise ValueError("No images found for given parameters")

    try:
        for download_response in itertools.chain([first_response], download_generator):
            comp_op = CompositeRasterOperation()
            comp_op.add(RasterioRasterSplit())
            comp_op.add(RasterioRasterPad())
            comp_op.add(RasterioRemoveBand(band=13))
            comp_op.add(RasterioInference(inference_func=RunpodInferenceCallback()))
            comp_op.add(RasterioRasterUnpad())
            comp_op.add(RasterioRasterMerge())
            comp_op.add(RasterioRasterReproject(target_crs=4326, target_bands=[1]))
            comp_op.add(RasterioDtypeConversion(dtype="uint8"))
            image = _create_raster(download_response)

            LOGGER.info(f"Processing raster for image {download_response.image_id}")
            pred_raster = next(comp_op.execute([image]))

            LOGGER.info(f"Got prediction raster for image {download_response.image_id}")
            pred_vectors = RasterioRasterToVector().execute(pred_raster)
            LOGGER.info(
                f"Got prediction vectors for image {download_response.image_id}"
            )

            with create_db_session() as db_session:
                insert_job = InsertJob(insert=Insert(db_session))
                insert_job.insert_all(
                    job_id=job_id,
                    download_response=download_response,
                    image=image,
                    pred_raster=pred_raster,
                    vectors=pred_vectors,
                )
            LOGGER.info(f"Inserted image {download_response.image_id}")
    except Exception as e:
        with create_db_session() as db_session:
            update_job_status(db_session, job_id, JobStatus.FAILED)
        LOGGER.error(f"Job {job_id} failed with error {e}")
        raise e

    with create_db_session() as db_session:
        update_job_status(db_session, job_id, JobStatus.COMPLETED)
    LOGGER.info(f"Job {job_id} completed {JobStatus.COMPLETED}")


if __name__ == "__main__":
    main()
