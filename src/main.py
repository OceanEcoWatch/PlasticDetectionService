import io
from typing import Generator

import httpx
import rasterio
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, HttpUrl
from sentinelhub.constants import MimeType
from sentinelhub.data_collections import DataCollection

from src import config
from src.database.connect import create_db_session
from src.database.insert import Insert
from src.inference.inference_callback import RunpodInferenceCallback
from src.models import Raster
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


class JobRequest(BaseModel):
    bbox: tuple[float, float, float, float]
    time_interval: tuple[str, str]
    maxcc: float
    job_id: int
    callback_url: HttpUrl


app = FastAPI()


@app.post("/process-job/")
async def process_job(request: JobRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        main,
        bbox=BoundingBox(*request.bbox),
        time_interval=TimeRange(*request.time_interval),
        maxcc=request.maxcc,
        callback_url=request.callback_url,
        job_id=request.job_id,
    )
    return {"message": "Job received, processing started."}


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


async def main(
    bbox: BoundingBox,
    time_interval: TimeRange,
    maxcc: float,
    callback_url: HttpUrl,
    job_id: int,
):
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
    for download_response in handler.download():
        db_session = create_db_session()

        raster = handler.create_raster(download_response)
        pred_raster = handler.get_prediction_raster(raster)
        pred_vectors = RasterioRasterToVector().execute(pred_raster)

        db_insert = Insert(db_session)
        db_insert.commit_all(
            download_response,
            pred_raster,
            config.RUNDPOD_MODEL_ID,
            config.RUNPOD_ENDPOINT_ID,
            pred_vectors,
        )
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    callback_url, json={"status": "completed", "job_id": job_id}
                )
                response.raise_for_status()
                print(f"Callback successful: {response.json()}")

            except httpx.HTTPStatusError as e:
                print(f"Callback failed: {e.response.status_code}, {e.response.text}")
