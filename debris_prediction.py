from datetime import datetime

from marinedebrisdetector.checkpoints import CHECKPOINTS
from marinedebrisdetector.model.segmentation_model import SegmentationModel
from marinedebrisdetector.predictor import ScenePredictor


def main(scene_path):  # ~ 14 min to predict on 11th Gen Intel® Core™ i5
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    print(f"start {datetime.now()}")

    detector = SegmentationModel.load_from_checkpoint(
        CHECKPOINTS["unet++1"], map_location="cpu", trust_repo=True
    )
    predictor = ScenePredictor(device="cpu")
    predictor.predict(
        detector, scene_path, scene_path.replace(".tif", "_prediction.tif")
    )

    print(f"finished {datetime.now()}")


if __name__ == "__main__":
    main("images/d95da6d3c9b9f66a8cd7a17d311beb0d/response.tiff")
