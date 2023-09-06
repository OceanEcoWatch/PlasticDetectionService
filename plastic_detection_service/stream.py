import numpy as np
from sentinelhub import BBox, DataCollection, MimeType, SentinelHubRequest, SHConfig


def stream_in_images(
    config: SHConfig,
    bbox: BBox,
    time_interval: tuple,
    evalscript: str,
    data_collection: DataCollection = DataCollection.SENTINEL2_L2A,
    mime_type: MimeType = MimeType.TIFF,
) -> list[np.ndarray]:
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=data_collection,
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", mime_type)],
        bbox=bbox,
        config=config,
    )
    return request.get_data()
