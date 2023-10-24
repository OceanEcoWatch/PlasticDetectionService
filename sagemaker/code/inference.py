import io

import ssh_util
import torch

from marinedebrisdetector.checkpoints import CHECKPOINTS
from marinedebrisdetector.model.segmentation_model import SegmentationModel
from marinedebrisdetector.predictor import ScenePredictor


def define_device():
    if torch.cuda.is_available():
        processing_unit = "cuda"
    else:
        processing_unit = "cpu"
    return processing_unit


def model_fn(model_dir):
    """
    Args:
      model_dir: the directory where model is saved.
    Returns:
      SegmentationModel from unet++ checkpoint
    """
    ssh_util.create_unverified_https_context()
    processing_unit = define_device()

    detector = SegmentationModel.load_from_checkpoint(
        CHECKPOINTS["unet++1"], map_location=processing_unit, trust_repo=True
    )
    return detector


def input_fn(request_body, request_content_type):
    if request_content_type == "application/octet-stream":
        return io.BytesIO(request_body)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """
    Args:
      input_data: Returned input data from input_fn
      model: Returned model from model_fn
    Returns:
      The predictions
    """
    processing_unit = define_device()
    predictor = ScenePredictor(device=processing_unit)
    return predictor.predict(model, input_data)


def output_fn(prediction, content_type):
    if content_type == "application/octet-stream":
        return prediction
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
