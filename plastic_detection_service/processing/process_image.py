import logging

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage.filters import gaussian_filter

LOGGER = logging.getLogger(__name__)


def unpad(y_score: np.ndarray, window: Window, dh: float, dw: float):
    y_score = y_score[
        int(np.ceil(dh)) : y_score.shape[0] - int(np.floor(dh)),
        int(np.ceil(dw)) : y_score.shape[1] - int(np.floor(dw)),
    ]
    if y_score.shape[0] != window.height:
        raise ValueError(
            f"unpadding size mismatch: {y_score.shape[0]} != {window.height}"
        )
    if y_score.shape[1] != window.width:
        raise ValueError(
            f"unpadding size mismatch: {y_score.shape[1]} != {window.width}"
        )
    return y_score


def post_process_image(
    predictions: list[np.ndarray],
    images: list[bytes],
    windows: list[Window],
    meta: dict,
    offset=64,
    window_size=(480, 480),
) -> bytes:
    H, W = window_size
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**meta) as dst:
            for pred, image, window in zip(predictions, images, windows):
                H, W = window_size
                H, W = H + offset * 2, W + offset * 2
                band, h, w = image.shape
                dh = (H - h) / 2
                dw = (W - w) / 2
                pred = unpad(pred, window, dh, dw)

                data = dst.read(window=window)[0] / 255

                overlap = data > 0

                if overlap.any():
                    LOGGER.info("Overlap detected")
                    dx, dy = np.gradient(overlap.astype(float))
                    g = np.abs(dx) + np.abs(dy)
                    transition = gaussian_filter(g, sigma=offset / 2)
                    transition /= transition.max()
                    transition[~overlap] = 1.0  # normalize to 1

                    y_score = transition * pred + (1 - transition) * data

                    writedata = (
                        np.expand_dims(y_score, 0).astype(np.float32) * 255
                    ).astype(np.uint8)
                    dst.write(writedata, window=window)
            return memfile.read()
