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
    """
    Stream in an image from SentinelHub.
    :param bbox: Bounding box of the image.
    :param time_interval: Time interval of the image.
    :param evalscript: Evalscript to use for the image.
    :param resolution: Resolution of the image.
    :return: List of images.
    """

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
