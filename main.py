
import io
import ssl
from sentinelhub import CRS, BBox, UtmZoneSplitter

from plastic_detection_service.config import config
from plastic_detection_service.constants import MANILLA_BAY_BBOX
from plastic_detection_service.download_images import stream_in_images
from plastic_detection_service.evalscripts import L2A_12_BANDS


def image_generator(bbox_list, time_interval, evalscript, maxcc):
    for bbox in bbox_list:
        data = stream_in_images(
            config, bbox, time_interval, evalscript=evalscript, maxcc=maxcc
        )

        if data is not None:
            yield data


def main():
    ssl._create_default_https_context = (
        ssl._create_unverified_context
    )  # fix for SSL error on Mac

    bbox = BBox(MANILLA_BAY_BBOX, crs=CRS.WGS84)
    time_interval = ("2023-08-01", "2023-09-01")
    maxcc = 0.5
    out_dir = "images"
    bbox_list = UtmZoneSplitter([bbox], crs=CRS.WGS84, bbox_size=5000).get_bbox_list()

    data_gen = image_generator(bbox_list, time_interval, L2A_12_BANDS, maxcc)

    for data in data_gen:
        detector = SegmentationModel.load_from_checkpoint(
            CHECKPOINTS["unet++1"], map_location="cpu", trust_repo=True
        )
        for _d in data:
            if _d.content is not None:
                predictor = ScenePredictor(device="cpu")
                predictor.predict(
                    detector, data=io.BytesIO(_d.content), out_dir=out_dir
                )



if __name__ == "__main__":
    main()
