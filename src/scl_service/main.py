import logging

import click
from sentinelhub import DataCollection, MimeType

from src import config
from src._types import BoundingBox, TimeRange
from src.database.connect import create_db_session
from src.database.insert import Insert
from src.database.models import Image, Job, JobStatus, Satellite
from src.download.evalscripts import L2A_SCL
from src.download.sh import SentinelHubDownload, SentinelHubDownloadParams
from src.raster_op.utils import create_raster_from_download_response
from src.raster_op.vectorize import RasterioRasterToPolygon

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--job-id", type=int, required=True)
def main(job_id: int):
    with create_db_session() as db:
        try:
            job = db.query(Job).filter(Job.id == job_id).one()
            if job.status != JobStatus.COMPLETED:
                raise ValueError(f"Job {job_id} is not completed")

            satellite = (
                db.query(Satellite)
                .join(Satellite.images)
                .join(Image.job)
                .filter(Job.id == job_id)
                .one()
            )
            if satellite.name.capitalize() != "SENTINEL_2_L2A":
                raise ValueError(f"Job {job_id} is not for Sentinel 2 L2A")

            images = job.images
            if not images:
                raise ValueError(f"Job {job_id} has no images")

            job.status = JobStatus.IN_PROGRESS
            for image in images:
                downloader = SentinelHubDownload(
                    SentinelHubDownloadParams(
                        bbox=BoundingBox(*image.geometry.bounds),
                        time_interval=TimeRange(image.timestamp, image.timestamp),
                        maxcc=job.maxcc,
                        config=config.SH_CONFIG,
                        evalscript=L2A_SCL,
                        data_collection=DataCollection.SENTINEL2_L2A,
                        mime_type=MimeType.TIFF,
                    )
                )
                download_generator = downloader.download_images()

                for download_response in download_generator:
                    if download_response is None:
                        continue
                    scl_raster = create_raster_from_download_response(download_response)
                    vectors = RasterioRasterToPolygon(band=1).execute(scl_raster)
                    inserter = Insert(db)
                    inserter.insert_scls_vectors(vectors=vectors, image_id=image.id)

            job.status = JobStatus.COMPLETED
        except Exception as e:
            job.status = JobStatus.FAILED
            LOGGER.error(f"Failed to process job {job_id}")
            db.rollback()
            raise e
