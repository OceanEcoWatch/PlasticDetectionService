import os
from typing import Optional

from sentinelhub import (
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)
from sentinelhub.download.models import DownloadResponse


def stream_in_images(
    config: SHConfig,
    bbox: BBox,
    time_interval: tuple[str, str],
    evalscript: str,
    maxcc: Optional[float] = None,
    data_collection: DataCollection = DataCollection.SENTINEL2_L2A,
    mime_type: MimeType = MimeType.TIFF,
    output_folder: str = "images",
) -> Optional[list[DownloadResponse]]:
    os.makedirs(output_folder, exist_ok=True)
    bbox_size = bbox_to_dimensions(bbox, resolution=10)
    request = SentinelHubRequest(
        evalscript=evalscript,
        size=bbox_size,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=data_collection,
                time_interval=time_interval,
                maxcc=maxcc,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", mime_type)],
        bbox=bbox,
        config=config,
        data_folder=output_folder,
    )
    data = request.get_data(decode_data=False, show_progress=True)

    return data
