import matplotlib.pyplot as plt
import numpy as np
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    bbox_to_dimensions,
)

from plastic_detection_service import evalscripts
from plastic_detection_service.config import config


def stream_in_image(
    bbox: BBox,
    time_interval: tuple,
    evalscript: str,
    resolution: int,
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

    size = bbox_to_dimensions(bbox, resolution=resolution)
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
        size=size,
        config=config,
    )
    return request.get_data()


if __name__ == "__main__":
    # manilla bay
    bbox = BBox(
        bbox=(
            120.53058253709094,
            14.384463071206468,
            120.99038315968619,
            14.812423505754381,
        ),
        crs=CRS.WGS84,
    )
    time_interval = ("2020-01-01", "2020-01-31")
    resolution = 60
    images = stream_in_image(
        bbox=bbox,
        time_interval=time_interval,
        evalscript=evalscripts.EVALSCRIPT_ALL_BANDS,
        resolution=resolution,
        data_collection=DataCollection.SENTINEL2_L1C,
    )
    plt.imshow(images[0][:, :, 3])
    plt.show()
