import io
import itertools
import logging
from typing import Generator, Iterable, Optional

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

from .download.abstractions import DownloadResponse, DownloadStrategy
from .download.evalscripts import L2A_12_BANDS_SCL
from .download.sh import (
    SentinelHubDownload,
    SentinelHubDownloadParams,
)
from .raster_op.utils import create_raster

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class MainHandler:
    def __init__(
        self, downloader: DownloadStrategy, raster_ops: CompositeRasterOperation
    ) -> None:
        self.downloader = downloader
        self.raster_ops = raster_ops

    def download(self) -> Generator[DownloadResponse, None, None]:
        for image in self.downloader.download_images():
            yield image

    def create_raster(self, image: DownloadResponse) -> Raster:
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

    def get_prediction_raster(self, image: Raster) -> Raster:
        return next(self.raster_ops.execute([image]))


class InsertJob:
    def __init__(self, insert: Insert):
        self.insert = insert

    def insert_all(
        self,
        job_id: int,
        download_response: DownloadResponse,
        raster: Raster,
        vectors: Iterable[Vector],
    ) -> tuple[Optional[Image], Optional[PredictionRaster], Optional[PredictionVector]]:
        image_url = s3.stream_to_s3(
            io.BytesIO(download_response.content),
            config.S3_BUCKET_NAME,
            f"images/{download_response.bbox}/{download_response.image_id}.tif",
        )
        try:
            image = self.insert.insert_image(
                download_response, raster, image_url, job_id
            )
        except IntegrityError:
            LOGGER.warning(
                f"Image {download_response.image_id} already exists. Skipping"
            )
            return None, None, None

        raster_url = s3.stream_to_s3(
            io.BytesIO(raster.content),
            config.S3_BUCKET_NAME,
            f"predictions/{download_response.bbox}/{download_response.image_id}.tif",
        )
        prediction_raster = self.insert.insert_prediction_raster(
            raster, image.id, raster_url
        )
        prediction_vectors = self.insert.insert_prediction_vectors(
            vectors, prediction_raster.id
        )
        return image, prediction_raster, prediction_vectors


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
        model = db_session.query(Model).filter(Model.id == model_id).first()
        if model is None:
            db_session.query(Job).filter(Job.id == job_id).update(
                {"status": JobStatus.FAILED}
            )
            raise NoResultFound("Model not found")
        job = db_session.query(Job).filter(Job.id == job_id).first()

        if job is None:
            db_session.query(Job).filter(Job.id == job_id).update(
                {"status": JobStatus.FAILED}
            )
            raise NoResultFound("Job not found")

        else:
            LOGGER.info(f"Updating job {job_id} to in progress")
            db_session.query(Job).filter(Job.id == job_id).update(
                {"status": JobStatus.IN_PROGRESS}
            )

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
    comp_op = CompositeRasterOperation()
    comp_op.add(RasterioRasterSplit())
    comp_op.add(RasterioRasterPad())
    comp_op.add(RasterioRemoveBand(band=13))
    comp_op.add(RasterioInference(inference_func=RunpodInferenceCallback()))
    comp_op.add(RasterioRasterUnpad())
    comp_op.add(RasterioRasterMerge())
    comp_op.add(RasterioRasterReproject(target_crs=4326, target_bands=[1]))
    comp_op.add(RasterioDtypeConversion(dtype="uint8"))

    handler = MainHandler(downloader, raster_ops=comp_op)
    download_generator = handler.download()
    try:
        first_response = next(download_generator)
    except StopIteration:
        with create_db_session() as db_session:
            db_session.query(Job).filter(Job.id == job_id).update(
                {"status": JobStatus.FAILED}
            )
        raise ValueError("No images found for given parameters")

    try:
        for download_response in itertools.chain([first_response], download_generator):
            LOGGER.info(f"Processing image {download_response.image_id}")

            raster = handler.create_raster(download_response)
            LOGGER.info(f"Processing raster for image {download_response.image_id}")
            pred_raster = handler.get_prediction_raster(raster)
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
                    raster=pred_raster,
                    vectors=pred_vectors,
                )
            LOGGER.info(f"Inserted image {download_response.image_id}")
    except Exception as e:
        with create_db_session() as db_session:
            db_session.query(Job).filter(Job.id == job_id).update(
                {"status": JobStatus.FAILED}
            )
        LOGGER.error(f"Job {job_id} failed with error {e}")
        raise e

    with create_db_session() as db_session:
        db_session.query(Job).filter(Job.id == job_id).update(
            {"status": JobStatus.COMPLETED}
        )
    LOGGER.info(f"Job {job_id} completed {JobStatus.COMPLETED}")


if __name__ == "__main__":
    main()
