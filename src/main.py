import io
import itertools
import logging
from concurrent.futures import ThreadPoolExecutor

import click
import rasterio
from geoalchemy2.shape import to_shape
from sentinelhub.constants import MimeType
from sentinelhub.data_collections import DataCollection

from src import config
from src._types import BoundingBox
from src.database.connect import create_db_session
from src.database.insert import (
    Insert,
    InsertJob,
    image_in_db,
    set_init_job_status,
    update_job_status,
)
from src.database.models import (
    AOI,
    JobStatus,
)
from src.inference.inference_callback import RunpodInferenceCallback
from src.raster_op.band import RasterioRemoveBand
from src.raster_op.composite import CompositeRasterOperation
from src.raster_op.convert import RasterioDtypeConversion
from src.raster_op.inference import RasterioInference
from src.raster_op.merge import RasterioRasterMerge
from src.raster_op.padding import RasterioRasterPad, RasterioRasterUnpad
from src.raster_op.reproject import RasterioRasterReproject
from src.raster_op.split import RasterioRasterSplit
from src.raster_op.utils import create_raster
from src.raster_op.vectorize import RasterioRasterToPoint
from src.scl import get_scl_vectors
from src.vector_op import probability_to_pixelvalue

from ._types import HeightWidth, TimeRange
from .download.abstractions import DownloadResponse
from .download.evalscripts import L2A_12_BANDS_SCL
from .download.sh import (
    SentinelHubDownload,
    SentinelHubDownloadParams,
)
from .models import Raster

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


def process_response(
    download_response: DownloadResponse, job_id: int, probability_threshold: float
):
    with create_db_session() as db_session:
        if image_in_db(db_session, download_response):
            LOGGER.warning(
                f"Image {download_response.bbox}/{download_response.image_id} already in db"
            )
            return
    image = _create_raster(download_response)

    scl_vectors = list(get_scl_vectors(image, band=13))
    comp_op = CompositeRasterOperation()
    comp_op.add(RasterioRasterSplit())
    comp_op.add(RasterioRasterPad())
    comp_op.add(RasterioRemoveBand(band=13))
    comp_op.add(RasterioInference(inference_func=RunpodInferenceCallback()))
    comp_op.add(RasterioRasterUnpad())
    comp_op.add(RasterioRasterMerge())
    comp_op.add(RasterioRasterReproject(target_crs=4326, target_bands=[1]))
    comp_op.add(RasterioDtypeConversion(dtype="uint8"))

    LOGGER.info(f"Processing raster for image {download_response.image_id}")
    pred_raster = next(comp_op.execute([image]))

    LOGGER.info(f"Got prediction raster for image {download_response.image_id}")
    pred_vectors = RasterioRasterToPoint(
        threshold=probability_to_pixelvalue(probability_threshold)
    ).execute(pred_raster)

    LOGGER.info(f"Got prediction vectors for image {download_response.image_id}")

    with create_db_session() as db_session:
        insert_job = InsertJob(insert=Insert(db_session))
        insert_job.insert_all(
            job_id=job_id,
            download_response=download_response,
            image=image,
            pred_raster=pred_raster,
            vectors=pred_vectors,
            scl_vectors=scl_vectors,
        )


@click.command()
@click.option("--job-id", type=int, required=True)
@click.option("--probability-threshold", type=float, required=True)
def main(
    job_id: int,
    probability_threshold: float,
):
    with create_db_session() as db_session:
        aoi = db_session.query(AOI).filter(AOI.jobs.any(id=job_id)).one()
        bbox = BoundingBox(*to_shape(aoi.geometry).bounds)
        job = set_init_job_status(db_session, job_id)
        maxcc = job.maxcc
        time_range = TimeRange(job.start_date, job.end_date)

    downloader = SentinelHubDownload(
        SentinelHubDownloadParams(
            bbox=bbox,
            time_interval=time_range,
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
        return LOGGER.info(f"No images found for job {job_id}")

    try:
        with ThreadPoolExecutor() as executor:
            executor.map(
                lambda response: process_response(
                    response, job_id, probability_threshold
                ),
                itertools.chain([first_response], download_generator),
            )

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
