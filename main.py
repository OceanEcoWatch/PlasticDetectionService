import torch
from marinedebrisdetector import CHECKPOINTS, SegmentationModel
from marinedebrisdetector.predictor import ScenePredictor
from sentinelhub import CRS, BBox, DataCollection

from plastic_detection_service import config, evalscripts, stream


def unet(seed=1):
    assert seed in [1, 2, 3]
    return SegmentationModel.load_from_checkpoint(
        CHECKPOINTS[f"unet{seed}"], trust_repo=True
    )


def main():
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

    images = stream.stream_in_image(
        config=config.config,
        bbox=bbox,
        time_interval=("2020-01-01", "2020-01-10"),
        evalscript=evalscripts.EVALSCRIPT_ALL_BANDS,
        resolution=60,
        data_collection=DataCollection.SENTINEL2_L1C,
    )

    detector = unet()
    predictor = ScenePredictor(device="cuda" if torch.cuda.is_available() else "cpu")
    for image in images:
        predictor.predict(detector, image, "test.tif")


if __name__ == "__main__":
    main()
