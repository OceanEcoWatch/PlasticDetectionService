import io
import logging

import click
from sentinelhub import DataCollection, MimeType

from plastic_detection_service import config
from plastic_detection_service.aws import s3

from .evalscripts import L2A_12_BANDS_SCL
from .models import DownloadResponse
from .sh import (
    SentinelHubDownload,
    SentinelHubDownloadParams,
)

LOGGER = logging.getLogger(__name__)


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
            bbox=bbox,
            time_interval=time_interval,
            maxcc=maxcc,
            config=config.SH_CONFIG,
            evalscript=L2A_12_BANDS_SCL,
            data_collection=DataCollection.SENTINEL2_L2A,
            mime_type=MimeType.TIFF,
        )
    )
    for image in downloader.download_images():
        upload_image_to_s3(image)


def upload_image_to_s3(image: DownloadResponse) -> str:
    s3_url = s3.stream_to_s3(
        io.BytesIO(image.content),
        config.S3_BUCKET_NAME,
        f"{image.bbox}/{image.image_id}.tif",
    )
    return s3_url


if __name__ == "__main__":
    main()
