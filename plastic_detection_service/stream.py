from typing import Optional

import numpy as np
from sentinelhub import BBox, DataCollection, MimeType, SentinelHubRequest, SHConfig


def stream_in_images(
    config: SHConfig,
    bbox: BBox,
    time_interval: tuple,
    evalscript: str,
    maxcc: Optional[float] = None,
    data_collection: DataCollection = DataCollection.SENTINEL2_L2A,
    mime_type: MimeType = MimeType.TIFF,
) -> Optional[list[np.ndarray]]:
    request = SentinelHubRequest(
        evalscript=evalscript,
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
    )
    data = request.get_data()

    # check if data is empty by checking if first element has 0 values
    if np.count_nonzero(data[0]) == 0:
        return None

    return data
