import io
import itertools
import logging

import click
from geoalchemy2.shape import to_shape
from sentinelhub.constants import MimeType
from sqlalchemy.orm import joinedload

from src import config
from src._types import BoundingBox
from src.aws import s3
from src.database.connect import create_db_session
from src.database.insert import (
    Insert,
    set_init_job_status,
)
from src.database.models import (
    AOI,
    Band,
    JobStatus,
    Model,
    ModelBand,
    Satellite,
)
from src.raster_op.utils import create_raster_from_download_response

from .._types import TimeRange
from ..download.evalscripts import generate_evalscript
from ..download.sh import DataCollection, SentinelHubDownload, SentinelHubDownloadParams

logging.basicConfig(level=logging.INFO)
logging.getLogger("rasterio").setLevel(logging.ERROR)
logging.getLogger("rasterio.env").setLevel(logging.ERROR)
logging.getLogger("rasterio._io").setLevel(logging.ERROR)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

LOGGER = logging.getLogger(__name__)


def get_data_collection(satellite: str) -> DataCollection:
    _satellite = satellite.upper().replace(" ", "_")
    try:
        return getattr(DataCollection, _satellite)
    except AttributeError:
        raise ValueError(
            "Could not find a matching DataCollection" f" for {_satellite}"
        )


@click.command()
@click.option("--job-id", type=int, required=True)
def main(job_id: int):
    with create_db_session() as db_session:
        aoi = db_session.query(AOI).filter(AOI.jobs.any(id=job_id)).one()
        aoi_geometry = to_shape(aoi.geometry)
        bbox = BoundingBox(*aoi_geometry.bounds)
        job = set_init_job_status(db_session, job_id)
        model = (
            db_session.query(Model)
            .options(joinedload(Model.expected_bands))
            .filter(Model.id == job.model_id)
            .one()
        )
        satellite = (
            db_session.query(Satellite)
            .join(Band)
            .join(ModelBand)
            .join(Model)
            .filter(Model.id == model.id)
            .one()
        )
        sat_id = satellite.id

        expected_bands = (
            db_session.query(Band.name)
            .join(ModelBand)
            .filter(ModelBand.model_id == model.id)
            .all()
        )
        band_names = [band.name for band in expected_bands]

        LOGGER.info(
            f"Starting job {job_id} with model:{model.model_id} for AOI: {aoi.name}. "
            f"Satellite: {satellite.name} with bands: {band_names}. Timestamps: {job.start_date} - {job.end_date}"
        )
        downloader = SentinelHubDownload(
            SentinelHubDownloadParams(
                bbox=bbox,
                time_interval=TimeRange(job.start_date, job.end_date),
                maxcc=job.maxcc,
                config=config.SH_CONFIG,
                evalscript=generate_evalscript(band_names),
                data_collection=get_data_collection(satellite.name),
                mime_type=MimeType.TIFF,
            )
        )

        download_generator = downloader.download_images()
        try:
            first_response = next(download_generator)
        except StopIteration:
            job.status = JobStatus.FAILED
            db_session.commit()
            LOGGER.error(f"No images found for job {job_id}")
            return

        for download_response in itertools.chain([first_response], download_generator):
            LOGGER.info(
                f"Downloaded image {download_response.image_id} for job {job_id}"
            )
            image = create_raster_from_download_response(download_response)
            unique_id = f"{download_response.bbox}/{download_response.image_id}"
            image_url = s3.stream_to_s3(
                io.BytesIO(download_response.content),
                config.S3_BUCKET_NAME,
                f"images/{unique_id}.tif",
            )
            insert = Insert(db_session)
            _ = insert.insert_image(download_response, image, image_url, job_id, sat_id)


if __name__ == "__main__":
    main()
