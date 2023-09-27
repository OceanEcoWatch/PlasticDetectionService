import io

import numpy as np
from rasterio.io import MemoryFile


def filter_pixels_below_threshold(
    input_bytes: io.BytesIO, threshold: float = 30
) -> bytes:
    with MemoryFile(input_bytes) as memfile:
        with memfile.open() as src:
            image = src.read(1)
            image[image < threshold] = 0

            with MemoryFile() as memfile:
                with memfile.open(**src.profile) as dst:
                    dst.write(image, 1)
                return memfile.read()


def scale_pixel_values(
    input_bytes: io.BytesIO,
    source_max: int = 255,
    target_max: int = 100,
) -> bytes:
    with MemoryFile(input_bytes) as memfile:
        with memfile.open() as src:
            image = src.read(1)
            image = (image / source_max) * target_max

            with MemoryFile() as memfile:
                with memfile.open(**src.profile) as dst:
                    dst.write(image, 1)
                return memfile.read()


def round_to_nearest_5_int(input_bytes: io.BytesIO) -> bytes:
    with MemoryFile(input_bytes) as memfile:
        with memfile.open() as src:
            image = src.read(1)
            rounded_image = np.round(image / 5) * 5
            rounded_image = rounded_image.astype(np.int8)

            with MemoryFile() as memfile:
                with memfile.open(**src.profile) as dst:
                    dst.write(rounded_image, 1)
                return memfile.read()


if __name__ == "__main__":
    with open(
        "images/5cb12a6cbd6df0865947f21170bc432a/response_wgs84_test.tiff", "rb"
    ) as f:
        rounded = round_to_nearest_5_int(io.BytesIO(f.read()))
        with open("rounded.tif", "wb") as f2:
            f2.write(rounded)
            print("done")
            print("done")
            print("done")
