import logging

import click
from geoalchemy2.shape import to_shape
from sentinelhub import DataCollection, MimeType
from sqlalchemy.exc import NoResultFound

from src import config
from src._types import BoundingBox, TimeRange
from src.database.connect import create_db_session
from src.database.insert import Insert
from src.database.models import Image, Job
from src.download.evalscripts import L2A_SCL
from src.download.sh import SentinelHubDownload, SentinelHubDownloadParams
from src.raster_op.clip import RasterioClip
from src.raster_op.composite import CompositeRasterOperation
from src.raster_op.reproject import RasterioRasterReproject
from src.raster_op.utils import create_raster_from_download_response
from src.raster_op.vectorize import RasterioRasterToPolygon

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--job-id", type=int, required=True)
def main(job_id: int):
    with create_db_session() as db:
        job = db.query(Job).filter(Job.id == job_id).one_or_none()
        if job is None:
            LOGGER.error(f"Job {job_id} not found.")
            return
        try:
            # if job.status != JobStatus.COMPLETED:
            #     LOGGER.warning(f"Job {job_id} is not in COMPLETED status. Skipping...")
            #     return

            # satellite = (
            #     db.query(Satellite)
            #     .join(Satellite.images)
            #     .join(Image.job)
            #     .filter(Job.id == job_id)
            #     .one()
            # )
            # if satellite.name.capitalize().strip() != "SENTINEL2_L2A":
            #     LOGGER.warning(f"Job {job_id} is not for Sentinel-2 L2A. Skipping...")
            #     return

            downloader = SentinelHubDownload(
                SentinelHubDownloadParams(
                    bbox=BoundingBox(*to_shape(job.aoi.geometry).bounds),
                    time_interval=TimeRange(job.start_date, job.end_date),
                    maxcc=job.maxcc,
                    config=config.SH_CONFIG,
                    evalscript=L2A_SCL,
                    data_collection=DataCollection.SENTINEL2_L2A,
                    mime_type=MimeType.TIFF,
                )
            )
            download_generator = downloader.download_images()

            for download_response in download_generator:
                image = (
                    db.query(Image)
                    .filter(Image.image_id == download_response.image_id)
                    .filter(Image.timestamp == download_response.timestamp)
                    .filter(Image.job_id == job_id)
                    .one_or_none()
                )
                if image is None:
                    raise NoResultFound(
                        f"Image with id {download_response.image_id, download_response.timestamp} not found."
                    )

                if download_response is None:
                    continue
                scl_raster = create_raster_from_download_response(download_response)
                comp_op = CompositeRasterOperation()
                comp_op.add(RasterioRasterReproject(target_crs=4326))
                comp_op.add(RasterioClip(to_shape(job.aoi.geometry)))
                clipped_scl_raster = next(comp_op.execute([scl_raster]))
                scl_vectors = RasterioRasterToPolygon(band=1).execute(
                    clipped_scl_raster
                )

                inserter = Insert(db)
                inserter.insert_scls_vectors(vectors=scl_vectors, image_id=image.id)

        except NoResultFound as e:
            LOGGER.error(f"Failed to process job {job_id}: {str(e)}")
            db.rollback()
        except Exception as e:
            LOGGER.error(f"Failed to process job {job_id}: {str(e)}")
            db.rollback()
            raise e


if __name__ == "__main__":
    main()
