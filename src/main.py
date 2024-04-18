import io
from typing import Generator, Iterable

import click
import rasterio
from sentinelhub.constants import MimeType
from sentinelhub.data_collections import DataCollection
from wandb import Image

from src import config
from src.aws import s3
from src.database.connect import create_db_session
from src.database.insert import Insert
from src.database.models import Job, JobStatus, PredictionRaster, PredictionVector
from src.inference.inference_callback import RunpodInferenceCallback
from src.models import Raster, Vector
from src.raster_op.band import RasterioRemoveBand
from src.raster_op.composite import RasterOpHandler
from src.raster_op.convert import RasterioDtypeConversion
from src.raster_op.inference import RasterioInference
from src.raster_op.merge import RasterioRasterMerge, copy_smooth
from src.raster_op.padding import RasterioRasterPad, RasterioRasterUnpad
from src.raster_op.reproject import RasterioRasterReproject
from src.raster_op.split import RasterioRasterSplit
from src.raster_op.vectorize import RasterioRasterToVector
from src.types import BoundingBox, HeightWidth, TimeRange

from .download.abstractions import DownloadResponse, DownloadStrategy
from .download.evalscripts import L2A_12_BANDS_SCL
from .download.sh import (
    SentinelHubDownload,
    SentinelHubDownloadParams,
)
from .raster_op.utils import create_raster


class MainHandler:
    def __init__(
        self, downloader: DownloadStrategy, raster_ops: RasterOpHandler
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
        return self.raster_ops.execute(image)


class InsertJob:
    def __init__(self, insert: Insert):
        self.insert = insert

    def insert_all(
        self,
        job_id: int,
        download_response: DownloadResponse,
        raster: Raster,
        vectors: Iterable[Vector],
    ) -> tuple[Image, PredictionRaster, list[PredictionVector]]:
        image_url = s3.stream_to_s3(
            io.BytesIO(download_response.content),
            config.S3_BUCKET_NAME,
            f"images/{download_response.bbox}/{download_response.image_id}.tif",
        )
        image = self.insert.insert_image(download_response, raster, image_url, job_id)

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
    "--timestamp",
    type=str,
    help="Time interval to be processed. Format: YYYY-MM-DD YYYY-MM-DD",
)
@click.option("--maxcc", type=float, required=True, default=0.05)
@click.option("--job-id", type=int, required=True)
@click.option("--model-id", type=int, required=True)
def main(
    bbox: BoundingBox,
    time_interval: TimeRange,
    maxcc: float,
    job_id: int,
    model_id: int,
):
    with create_db_session() as db_session:
        # change status of job to processing
        db_session.query(Job).filter(Job.id == job_id).update(
            {"status": JobStatus.IN_PROGRESS}
        )
    downloader = SentinelHubDownload(
        SentinelHubDownloadParams(
            bbox=bbox,
            time_interval=time_interval,
            maxcc=maxcc,
            config=config.SH_CONFIG,
            evalscript=L2A_12_BANDS_SCL,
            data_collection=DataCollection.SENTINEL2_L2A,
            mime_type=MimeType.TIFF,
        )
    )

    raster_handler = RasterOpHandler(
        split=RasterioRasterSplit(),
        pad=RasterioRasterPad(),
        band=RasterioRemoveBand(band=13),
        inference=RasterioInference(inference_func=RunpodInferenceCallback()),
        unpad=RasterioRasterUnpad(),
        merge=RasterioRasterMerge(merge_method=copy_smooth),
        convert=RasterioDtypeConversion(dtype="uint8"),
        reproject=RasterioRasterReproject(target_crs=4326, target_bands=[1]),
    )

    handler = MainHandler(downloader, raster_ops=raster_handler)

    try:
        for download_response in handler.download():
            db_session = create_db_session()

            raster = handler.create_raster(download_response)
            pred_raster = handler.get_prediction_raster(raster)
            pred_vectors = RasterioRasterToVector().execute(pred_raster)

            insert_job = InsertJob(insert=Insert(db_session))
            insert_job.insert_all(
                job_id=job_id,
                model_id=model_id,
                download_response=download_response,
                raster=raster,
                vectors=pred_vectors,
            )
    except Exception as e:
        with create_db_session() as db_session:
            db_session.query(Job).filter(Job.id == job_id).update(
                {"status": JobStatus.FAILED}
            )
        raise e

    with create_db_session() as db_session:
        db_session.query(Job).filter(Job.id == job_id).update(
            {"status": JobStatus.COMPLETED}
        )


if __name__ == "__main__":
    main()
