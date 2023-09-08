import datetime
import ssl

from sentinelhub import CRS, BBox, UtmZoneSplitter

from marinedebrisdetector.checkpoints import CHECKPOINTS
from marinedebrisdetector.model.segmentation_model import SegmentationModel
from marinedebrisdetector.predictor import ScenePredictor
from plastic_detection_service.config import config
from plastic_detection_service.constants import MANILLA_BAY_BBOX
from plastic_detection_service.download_images import stream_in_images
from plastic_detection_service.evalscripts import L2A_12_BANDS


def main():
    ssl._create_default_https_context = (
        ssl._create_unverified_context
    )  # fix for SSL error

    bbox = BBox(MANILLA_BAY_BBOX, crs=CRS.WGS84)
    time_interval = ("2023-08-01", "2023-09-01")
    maxcc = 0.5
    output_folder = "images"
    out_path = f"{output_folder}/prediction_{datetime.datetime.now()}.tif"

    bbox_list = UtmZoneSplitter([bbox], crs=CRS.WGS84, bbox_size=5000).get_bbox_list()

    data_list = []
    for bbox in bbox_list:
        data = stream_in_images(
            config, bbox, time_interval, evalscript=L2A_12_BANDS, maxcc=maxcc
        )
        print(data)
        data_list.append(data)

        detector = SegmentationModel.load_from_checkpoint(
            CHECKPOINTS["unet++1"], map_location="cpu", trust_repo=True
        )
        predictor = ScenePredictor(device="cpu")
        predictor.predict(detector, data=data[0], out_path=out_path)


if __name__ == "__main__":
    main()
