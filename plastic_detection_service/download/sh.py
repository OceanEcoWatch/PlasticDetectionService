import datetime
from dataclasses import dataclass
from typing import Generator

from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubCatalog,
    SentinelHubRequest,
    SHConfig,
    UtmZoneSplitter,
    bbox_to_dimensions,
)
from sentinelhub.api.catalog import CatalogSearchIterator

from plastic_detection_service.types import BoundingBox

from .abstractions import DownloadParams, DownloadResponse, DownloadStrategy


@dataclass
class SentinelHubDownloadParams(DownloadParams):
    config: SHConfig
    evalscript: str
    data_collection: DataCollection
    mime_type: MimeType


class SentinelHubDownload(DownloadStrategy):
    def __init__(self, params: SentinelHubDownloadParams):
        self.params = params

    def _split_bbox(self, bbox: BoundingBox) -> list[BBox]:
        bbox_crs = BBox(bbox, crs=CRS.WGS84)
        return UtmZoneSplitter(
            [bbox_crs], crs=bbox_crs.crs, bbox_size=5000
        ).get_bbox_list()

    def _search_images(
        self,
        bbox: BBox,
    ) -> CatalogSearchIterator:
        catalog = SentinelHubCatalog(config=self.params.config)
        return catalog.search(
            bbox=bbox,
            time=self.params.time_interval,
            collection=self.params.data_collection,
            filter=f"eo:cloud_cover<={self.params.maxcc * 100}",
        )

    def _create_request(self, search_response: dict, bbox: BBox) -> SentinelHubRequest:
        time_interval = (
            search_response["properties"]["datetime"],
            search_response["properties"]["datetime"],
        )
        bbox_size = bbox_to_dimensions(bbox, resolution=10)
        return SentinelHubRequest(
            evalscript=self.params.evalscript,
            size=bbox_size,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=self.params.data_collection,
                    time_interval=time_interval,
                    maxcc=self.params.maxcc,
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", self.params.mime_type)
            ],
            bbox=bbox,
            config=self.params.config,
        )

    def _download_image(
        self, search_response: dict, request: SentinelHubRequest, bbox: BBox
    ) -> DownloadResponse:
        bbox_size = bbox_to_dimensions(bbox, resolution=10)
        response_list = request.get_data(decode_data=False, save_data=False)
        if len(response_list) != 1:
            raise ValueError("Expected only one image to be returned.")
        response = response_list[0]

        return DownloadResponse(
            image_id=search_response["id"],
            timestamp=datetime.datetime.fromisoformat(
                search_response["properties"]["datetime"].rstrip("Z")
            ),
            bbox=(bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y),
            crs=int(bbox.crs.value),
            image_size=bbox_size,
            maxcc=self.params.maxcc,
            data_collection=self.params.data_collection.value.api_id,
            request_timestamp=datetime.datetime.strptime(
                response.headers["Date"], "%a, %d %b %Y %H:%M:%S GMT"
            ),
            content=response.content,
            headers=response.headers,
        )

    def _download_for_bbox(
        self,
        bbox: BBox,
    ) -> Generator[DownloadResponse, None, None]:
        search_iterator = list(self._search_images(bbox=bbox))
        for search_response in search_iterator:
            yield self._download_image(
                search_response, self._create_request(search_response, bbox), bbox
            )

    def download_images(
        self,
    ) -> Generator[DownloadResponse, None, None]:
        for _bbox in self._split_bbox(self.params.bbox):
            yield from self._download_for_bbox(_bbox)
