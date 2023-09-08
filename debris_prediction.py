from marinedebrisdetector.model.segmentation_model import SegmentationModel
from marinedebrisdetector.checkpoints import CHECKPOINTS

from datetime import datetime

from marinedebrisdetector.predictor import ScenePredictor


def main(scene_path): # ~ 14 min to predict on 11th Gen Intel® Core™ i5
    print(f'start {datetime.now()}')

    detector = SegmentationModel.load_from_checkpoint(CHECKPOINTS["unet++1"], map_location='cpu', trust_repo=True)
    predictor = ScenePredictor(device='cpu')
    predictor.predict(detector, scene_path, scene_path.replace(".tif", "_prediction.tif"))

    print(f'finished {datetime.now()}')

#main('marinedebrisdetector/durban_20190424.tif')
