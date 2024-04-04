import datetime
import json
from unittest.mock import MagicMock, patch

import pytest
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    UtmZoneSplitter,
    bbox_to_dimensions,
)

from src.config import SH_CONFIG
from src.download.evalscripts import L2A_12_BANDS_SCL
from src.download.sh import (
    SentinelHubDownload,
    SentinelHubDownloadParams,
)
from src.types import BoundingBox, HeightWidth, TimeRange

TIME_INTERVAL = TimeRange("2023-11-01", "2024-01-01")


@pytest.fixture
def bbox():
    return BBox(
        bbox=(
            120.82481750015815,
            14.619576605802296,
            120.82562856620629,
            14.66462165084734,
        ),
        crs=CRS.WGS84,
    )


@pytest.fixture
def bbox_utm(bbox):
    return UtmZoneSplitter([bbox], crs=bbox.crs, bbox_size=5000).get_bbox_list()[0]


@pytest.fixture
def sh_download_params():
    return SentinelHubDownloadParams(
        bbox=BoundingBox(
            120.82481750015815,
            14.619576605802296,
            120.82562856620629,
            14.66462165084734,
        ),
        time_interval=TIME_INTERVAL,
        maxcc=0.1,
        config=SH_CONFIG,
        evalscript=L2A_12_BANDS_SCL,
        data_collection=DataCollection.SENTINEL2_L2A,
        mime_type=MimeType.TIFF,
    )


@pytest.fixture
def sh_download(sh_download_params):
    return SentinelHubDownload(params=sh_download_params)


@pytest.fixture
def sh_request(bbox_utm):
    bbox_size = bbox_to_dimensions(bbox_utm, resolution=10)
    return SentinelHubRequest(
        evalscript=L2A_12_BANDS_SCL,
        size=bbox_size,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=TIME_INTERVAL,
                maxcc=0.1,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox_utm,
        config=SH_CONFIG,
    )


@pytest.fixture
def catalog_search() -> dict:
    with open("tests/assets/catalog_search.json") as f:
        return json.load(f)


@pytest.fixture
def sh_request_payload():
    with open("tests/assets/sh_request.json") as f:
        return json.load(f)


@patch("src.download.sh.SentinelHubRequest.get_data")
def test_download_image(
    mock_get_data,
    sh_download: SentinelHubDownload,
    sh_request: SentinelHubRequest,
    catalog_search: dict,
    bbox_utm: BBox,
):
    mock_response = MagicMock()
    mock_response.content = b"test content"
    mock_response.headers = {"Date": "Mon, 01 Jan 2000 00:00:00 GMT"}
    mock_get_data.return_value = [mock_response]
    image = sh_download._download_image(
        search_response=catalog_search, request=sh_request, bbox=bbox_utm
    )

    assert image.content is not None
    assert isinstance(image.content, bytes)
    assert image.timestamp >= datetime.datetime.fromisoformat(
        sh_download.params.time_interval[0]
    )
    assert image.timestamp <= datetime.datetime.fromisoformat(
        sh_download.params.time_interval[1]
    )
    assert image.bbox == (
        bbox_utm.min_x,
        bbox_utm.min_y,
        bbox_utm.max_x,
        bbox_utm.max_y,
    )
    assert image.image_size == (500, 500)
    assert image.data_collection == DataCollection.SENTINEL2_L2A.value.api_id
    assert isinstance(image.request_timestamp, datetime.datetime)


def test_create_request(
    catalog_search: dict,
    sh_download: SentinelHubDownload,
    bbox_utm: BBox,
    sh_request_payload: dict,
):
    request = sh_download._create_request(search_response=catalog_search, bbox=bbox_utm)

    assert request.payload == sh_request_payload


@patch("src.download.sh.SentinelHubCatalog.search")
def test_download_images(
    mock_search,
    sh_download: SentinelHubDownload,
    catalog_search,
):
    mock_search.return_value = [catalog_search]
    with patch("src.download.sh.SentinelHubRequest.get_data") as mock_get_data:
        mock_response = MagicMock()
        mock_response.content = b"test content"
        mock_response.headers = {"Date": "Mon, 01 Jan 2000 00:00:00 GMT"}
        mock_get_data.return_value = [mock_response]

        images = list(sh_download.download_images())

        assert len(images) == 2
        assert images[0].content is not None
        assert isinstance(images[0].content, bytes)
        assert images[0].timestamp >= datetime.datetime.fromisoformat(
            sh_download.params.time_interval[0]
        )
        assert images[0].timestamp <= datetime.datetime.fromisoformat(
            sh_download.params.time_interval[1]
        )

        # UTM
        assert images[0].bbox == (264000.0, 1612800.0, 268800.0, 1617600.0)
        assert images[0].image_size == HeightWidth(480, 480)
        assert images[0].data_collection == DataCollection.SENTINEL2_L2A.value.api_id
        assert isinstance(images[0].request_timestamp, datetime.datetime)


@pytest.mark.integration
def test_search_images_integration(sh_download: SentinelHubDownload, bbox_utm: BBox):
    images = sh_download._search_images(bbox=bbox_utm)
    if images:
        for image in images:
            assert image["properties"]["eo:cloud_cover"] <= 0.1 * 100
            assert (
                image["properties"]["datetime"] >= sh_download.params.time_interval[0]
            )
            assert (
                image["properties"]["datetime"] <= sh_download.params.time_interval[1]
            )

    else:
        assert False


@pytest.mark.integration
def test_download_images_integration(sh_download: SentinelHubDownload):
    time_interval = sh_download.params.time_interval
    expected_bboxes = [
        (264000.0, 1612800.0, 268800.0, 1617600.0),
        (264000.0, 1617600.0, 268800.0, 1622400.0),
    ]
    download_responses = list(sh_download.download_images())
    assert len(download_responses) == 2

    for res, _bbox in zip(download_responses, expected_bboxes):
        assert res.content is not None
        assert isinstance(res.content, bytes)
        assert res.timestamp >= datetime.datetime.fromisoformat(time_interval[0])
        assert res.timestamp <= datetime.datetime.fromisoformat(time_interval[1])

        # UTM
        assert res.bbox == _bbox
        assert res.image_size == HeightWidth(480, 480)
        assert res.data_collection == DataCollection.SENTINEL2_L2A.value.api_id
        assert isinstance(res.request_timestamp, datetime.datetime)
