
def local_inference_callback(payload: bytes) -> bytes:
    from .sagemaker_model.code.inference import input_fn, model_fn, output_fn, predict_fn
    model = model_fn(".")
    array = input_fn(payload, "application/octet-stream")
    pred_array = predict_fn(array, model)
    return output_fn(pred_array, "application/octet-stream")
