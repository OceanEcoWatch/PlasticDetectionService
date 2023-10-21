import os

import torch


def model_fn(model_dir):
    """
    Args:
      model_dir: the directory where model is saved.
    Returns:
      The model
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(
        os.path.join(model_dir, "epoch=73-val_loss=0.60-auroc=0.984.ckpt"), "rb"
    ) as f:
        model = torch.load(f, map_location=device)
    return model.to(device)


def input_fn(request_body, request_content_type):
    """
    handle image byte inputs

    Args:
      request_body: the request body
      request_content_type: the request content type
    """
    if request_content_type == "application/x-image":
        image = request_body.read()

        return image
    else:
        raise ValueError("Content type {} not supported.".format(request_content_type))


def predict_fn(input_data, model):
    """
    Args:
      input_data: Returned input data from input_fn
      model: Returned model from model_fn
    Returns:
      The predictions
    """
    return model.predict(input_data)
