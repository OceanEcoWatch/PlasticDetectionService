import itertools
import logging

import click
from geoalchemy2.shape import to_shape
from sentinelhub.constants import MimeType
from sentinelhub.data_collections import DataCollection
from shapely.geometry import Polygon
from sqlalchemy.orm import joinedload

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
    Band,
    JobStatus,
    Model,
    ModelBand,
    ModelType,
    Satellite,
)
from src.inference.inference_callback import RunpodInferenceCallback
from src.raster_op.clip import RasterioClip
from src.raster_op.composite import CompositeRasterOperation
from src.raster_op.convert import RasterioDtypeConversion
from src.raster_op.inference import RasterioInference
from src.raster_op.merge import RasterioRasterMerge
from src.raster_op.padding import RasterioRasterPad, RasterioRasterUnpad
from src.raster_op.reproject import RasterioRasterReproject
from src.raster_op.split import RasterioRasterSplit
from src.raster_op.utils import create_raster_from_download_response
from src.raster_op.vectorize import RasterioRasterToPoint
from src.vector_op import probability_to_pixelvalue

from .._types import HeightWidth, TimeRange
from ..download.abstractions import DownloadResponse
from ..download.evalscripts import generate_evalscript
from ..download.sh import (
    SentinelHubDownload,
    SentinelHubDownloadParams,
)

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


def is_segmentation(model: Model) -> bool:
    return model.type.value == ModelType.SEGMENTATION.value


def process_response(
    download_response: DownloadResponse,
    job_id: int,
    probability_threshold: float,
    aoi_geometry: Polygon,
    model: Model,
    satellite_id: int,
):
    with create_db_session() as db_session:
        if image_in_db(db_session, download_response, job_id):
            LOGGER.warning(
                f"Image {model.model_id}/{download_response.bbox}/{download_response.image_id} already in db"
            )
            return

    image = create_raster_from_download_response(download_response)

    comp_op = CompositeRasterOperation()
    comp_op.add(
        RasterioRasterSplit(
            HeightWidth(model.expected_image_height, model.expected_image_width)
        )
    )
    comp_op.add(RasterioRasterPad())
    comp_op.add(
        RasterioInference(
            inference_func=RunpodInferenceCallback(endpoint_url=model.model_url),
            output_dtype=model.output_dtype,
        )
    )
    comp_op.add(RasterioRasterUnpad())
    comp_op.add(RasterioRasterMerge())
    comp_op.add(RasterioRasterReproject(target_crs=4326, target_bands=[1]))
    comp_op.add(RasterioDtypeConversion(dtype="uint8", scale=is_segmentation(model)))
    comp_op.add(RasterioClip(aoi_geometry))

    LOGGER.info(f"Processing raster for image {download_response.image_id}")
    pred_raster = next(comp_op.execute([image]))

    LOGGER.info(f"Got prediction raster for image {download_response.image_id}")
    threshold = (
        probability_to_pixelvalue(probability_threshold)
        if is_segmentation(model)
        else None
    )
    print(threshold)
    pred_vectors = list(RasterioRasterToPoint(threshold=threshold).execute(pred_raster))

    LOGGER.info(
        f"Got {len(pred_vectors)} prediction vectors for image {download_response.image_id}"
    )

    with create_db_session() as db_session:
        insert_job = InsertJob(insert=Insert(db_session))
        insert_job.insert_all(
            job_id=job_id,
            satellite_id=satellite_id,
            model_name=model.model_id,
            download_response=download_response,
            image=image,
            pred_raster=pred_raster,
            vectors=pred_vectors,
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
        with create_db_session() as db_session:
            update_job_status(db_session, job_id, JobStatus.FAILED)
        return LOGGER.info(f"No images found for job {job_id}")

    try:
        for response in itertools.chain([first_response], download_generator):
            process_response(
                response,
                job_id,
                probability_threshold,
                aoi_geometry,
                model,
                sat_id,
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
