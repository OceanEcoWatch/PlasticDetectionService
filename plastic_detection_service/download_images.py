import os
from typing import Optional

import numpy as np
from sentinelhub import (
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)


def stream_in_images(
    config: SHConfig,
    bbox: BBox,
    time_interval: tuple[str, str],
    evalscript: str,
    maxcc: Optional[float] = None,
    data_collection: DataCollection = DataCollection.SENTINEL2_L2A,
    mime_type: MimeType = MimeType.TIFF,
    output_folder: str = "images",
) -> Optional[list[np.ndarray]]:
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
    data = request.get_data(save_data=True, max_threads=4)

    # check if data is empty by checking if first element has 0 values
    if np.count_nonzero(data[0]) == 0:
        return None

    return data
