import io
from typing import Generator

import click
import rasterio
import requests
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


def send_notification(url: str, status: str, details: str):
    payload = {"status": status, "details": details}
    response = requests.post(url, json=payload)  # add auth headers
    response.raise_for_status()
    return response.status_code, response.text


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


@click.command()
@click.option(
    "--bbox",
    nargs=4,
    type=float,
    help="Bounding box of the area to be processed. Format: min_lon min_lat max_lon max_lat",
)
@click.option(
    "--time-interval",
    nargs=2,
    type=str,
    help="Time interval to be processed. Format: YYYY-MM-DD YYYY-MM-DD",
)
@click.option("--maxcc", type=float, required=True, default=0.05)
@click.option("--callback-url", type=str, required=True)
def main(bbox: BoundingBox, time_interval: TimeRange, maxcc: float, callback_url: str):
    send_notification(
        callback_url, "received", "Request received and is being processed."
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

            db_insert = Insert(db_session)
            db_insert.commit_all(
                download_response,
                pred_raster,
                config.RUNDPOD_MODEL_ID,
                config.RUNPOD_ENDPOINT_ID,
                pred_vectors,
            )
    except Exception as e:
        send_notification(callback_url, "failed", str(e))
        raise e

    send_notification(callback_url, "success", "Request completed successfully.")


if __name__ == "__main__":
    main()
