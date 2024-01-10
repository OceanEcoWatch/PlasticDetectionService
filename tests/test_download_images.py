from io import BytesIO
from pprint import pprint

import pytest
import rasterio
from sentinelhub import CRS, BBox

from plastic_detection_service.config import SH_CONFIG
from plastic_detection_service.download_images import search_images, stream_in_images
from plastic_detection_service.evalscripts import L2A_12_BANDS_CLEAR_WATER_MASK


@pytest.mark.skip(reason="Need to mock the request")
def test_stream_in_images():
    bbox = BBox(
        bbox=(
            120.82481750015815,
            14.619576605802296,
            120.82562856620629,
            14.66462165084734,
        ),
        crs=CRS.WGS84,
    )
    time_interval = "2024-01-01", "2024-01-01"
    evalscript = L2A_12_BANDS_CLEAR_WATER_MASK
    download_responses = stream_in_images(
        config=SH_CONFIG, bbox=bbox, time_interval=time_interval, maxcc=0.1, evalscript=evalscript
    )
    if download_responses is not None:
        for image in download_responses:
            print(image)

        assert len(download_responses) == 1
        assert download_responses[0].status_code == 200
        assert download_responses[0].content is not None

        with rasterio.open(BytesIO(download_responses[0].content)) as src:
            assert src.count == 13
            tags = src.tags()
            assert "TIFFTAG_DATETIME" in tags.keys()
    else:
        assert False


def test_search_images():
    bbox = BBox(bbox=(120.82481750015815, 14.619576605802296, 120.82562856620629, 14.66462165084734), crs=CRS.WGS84)
    time_interval = "2023-01-01", "2024-01-10"
    images = list(search_images(SH_CONFIG, bbox, time_interval, 0.1))
    if images:
        for image in images:
            pprint(image)
            assert image["properties"]["eo:cloud_cover"] <= 0.1 * 100
            assert image["properties"]["datetime"] >= time_interval[0]
            assert image["properties"]["datetime"] <= time_interval[1]
    else:
        assert False
