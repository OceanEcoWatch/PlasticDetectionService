from typing import Generator, Optional

from sentinelhub import (
    BBox,
    DataCollection,
    MimeType,
    SentinelHubCatalog,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)
from sentinelhub.download.models import DownloadResponse

from plastic_detection_service.config import SH_CONFIG


class TimestampResponse(DownloadResponse):
    def __init__(self, timestamp: str, bbox: BBox, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestamp = timestamp
        self.bbox = bbox


def search_images(
    config: SHConfig,
    bbox: BBox,
    time_interval: tuple[str, str],
    maxcc: float,
    data_collection: DataCollection = DataCollection.SENTINEL2_L2A,
):
    catalog = SentinelHubCatalog(config=config)
    images = catalog.search(
        bbox=bbox, time=time_interval, collection=data_collection, filter=f"eo:cloud_cover<={maxcc * 100}"
    )

    return images


def stream_in_images(
    config: SHConfig,
    bbox: BBox,
    time_interval: tuple[str, str],
    evalscript: str,
    maxcc: float,
    data_collection: DataCollection = DataCollection.SENTINEL2_L2A,
    mime_type: MimeType = MimeType.TIFF,
    output_folder: str = "images",
) -> Optional[list[TimestampResponse]]:
    images_search = list(search_images(config, bbox, time_interval, maxcc, data_collection))
    if not images_search:
        return None
    bbox_size = bbox_to_dimensions(bbox, resolution=10)

    timestamp_responses = []
    for image in images_search:
        _ti = image["properties"]["datetime"], image["properties"]["datetime"]
        request = SentinelHubRequest(
            evalscript=evalscript,
            size=bbox_size,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=data_collection,
                    time_interval=_ti,
                    maxcc=maxcc,
                )
            ],
            responses=[SentinelHubRequest.output_response("default", mime_type)],
            bbox=bbox,
            config=config,
            data_folder=output_folder,
        )
        response_list = request.get_data(decode_data=False)

        if len(response_list) != 1:
            raise ValueError("Expected only one image to be returned.")
        response = response_list[0]
        timestamp_responses.append(
            TimestampResponse(
                image["properties"]["datetime"],
                bbox,
                request=response.request,
                content=response.content,
                status_code=response.status_code,
                headers=response.headers,
                elapsed=response.elapsed,
            )
        )

    return timestamp_responses


def image_generator(
    bbox_list: list[BBox],
    time_interval: tuple[str, str],
    evalscript: str,
    maxcc: float,
    data_collection: DataCollection = DataCollection.SENTINEL2_L2A,
    mime_type: MimeType = MimeType.TIFF,
) -> Generator[list[TimestampResponse], None, None]:
    """Generator that streams in images from sentinel hub.

    :param bbox_list: list of bounding boxes
    :param time_interval: time interval to be processed. Format: YYYY-MM-DD YYYY-MM-DD
    :param evalscript: sentinel hub evalscript
    :param maxcc: maximum cloud cover of the images to be processed.
    :param data_collection: sentinel hub data collection
    :param mime_type: sentinel hub mime type
    :param output_folder: directory where the images will be saved.

    :return: generator that yields sentinel hub data for each bbox in bbox_list
    """
    for bbox in bbox_list:
        data = stream_in_images(
            SH_CONFIG,
            bbox,
            time_interval,
            evalscript=evalscript,
            maxcc=maxcc,
            data_collection=data_collection,
            mime_type=mime_type,
        )

        if data is not None:
            yield data
