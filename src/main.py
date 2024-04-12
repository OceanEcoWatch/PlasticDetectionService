import io
from typing import Generator

import click
import rasterio
from sentinelhub import DataCollection, MimeType

from src import config
from src.aws import s3
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

    def upload_image_to_s3(self, image: DownloadResponse) -> str:
        return s3.stream_to_s3(
            io.BytesIO(image.content),
            config.S3_BUCKET_NAME,
            f"{image.bbox}/{image.image_id}.tif",
        )

    def download(self) -> Generator[DownloadResponse, None, None]:
        for image in self.downloader.download_images():
            yield image

    def create_raster(self, image: DownloadResponse) -> Raster:
        with rasterio.open(io.BytesIO(image.content)) as src:
            np_image = src.read().copy()
            bounds = src.bounds.copy()
            meta = src.meta.copy()
        return create_raster(image.content, np_image, bounds, meta, HeightWidth(0, 0))

    def process(self, image: Raster) -> Generator[Vector, None, None]:
        return self.raster_ops.execute(image)


@click.command()
@click.option(
    "--bbox",
    nargs=4,
    type=float,
    help="Bounding box of the area to be processed. Format: min_lon min_lat max_lon max_lat",
    default=config.AOI,
)
@click.option(
    "--time-interval",
    nargs=2,
    type=str,
    help="Time interval to be processed. Format: YYYY-MM-DD YYYY-MM-DD",
    default=config.TIME_INTERVAL,
)
@click.option(
    "--maxcc",
    type=float,
    default=config.MAX_CC,
    help="Maximum cloud cover of the images to be processed.",
)
def main(
    bbox: tuple[float, float, float, float],
    time_interval: tuple[str, str],
    maxcc: float,
):
    downloader = SentinelHubDownload(
        SentinelHubDownloadParams(
            bbox=BoundingBox(*bbox),
            time_interval=TimeRange(*time_interval),
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
        to_vector=RasterioRasterToVector(band=1),
    )
    handler = MainHandler(downloader, raster_ops=raster_handler)
    for image in handler.download():
        raster = handler.create_raster(image)
        for vector in handler.process(raster):
            print(vector)


if __name__ == "__main__":
    main()
