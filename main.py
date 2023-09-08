from sentinelhub import CRS, BBox, UtmZoneSplitter

from plastic_detection_service.config import config
from plastic_detection_service.constants import MANILLA_BAY_BBOX
from plastic_detection_service.download_images import stream_in_images
from plastic_detection_service.evalscripts import L2A_12_BANDS


def main():
    bbox = BBox(MANILLA_BAY_BBOX, crs=CRS.WGS84)
    time_interval = ("2023-08-01", "2023-09-01")
    maxcc = 0.5

    bbox_list = UtmZoneSplitter([bbox], crs=CRS.WGS84, bbox_size=5000).get_bbox_list()

    for bbox in bbox_list:
        data = stream_in_images(
            config, bbox, time_interval, evalscript=L2A_12_BANDS, maxcc=maxcc
        )
        print(data)


if __name__ == "__main__":
    main()
