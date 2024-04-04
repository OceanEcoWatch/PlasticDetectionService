import io

import numpy as np
import rasterio

from src.raster_op.split import RasterioRasterSplit
from src.types import HeightWidth


def test_split_raster(s2_l2a_raster):
    image_size = HeightWidth(480, 480)
    offset = 64
    processor = RasterioRasterSplit(image_size=image_size, offset=offset)

    split_raster = next(processor.execute(s2_l2a_raster))

    assert split_raster.size == image_size
    assert split_raster.crs == s2_l2a_raster.crs
    assert split_raster.bands == s2_l2a_raster.bands
    assert split_raster.content != s2_l2a_raster.content
    assert isinstance(split_raster.content, bytes)

    with rasterio.open("tests/assets/test_exp_split.tif") as exp_src:
        exp_image = exp_src.read()
        with rasterio.open(io.BytesIO(split_raster.content)) as src:
            image = src.read()
            assert image.shape == exp_image.shape
            assert image.dtype == exp_image.dtype
            assert src.meta["height"] == exp_src.meta["height"]
            assert src.meta["width"] == exp_src.meta["width"]
            assert src.meta["crs"] == exp_src.meta["crs"]
            assert src.meta["count"] == exp_src.meta["count"]
            assert src.meta["transform"] == exp_src.meta["transform"]
            assert src.meta["dtype"] == exp_src.meta["dtype"]
            assert src.meta["nodata"] == exp_src.meta["nodata"]

            assert np.array_equal(image, exp_image)


def test_split_rasters_are_in_bounds_original(s2_l2a_raster):
    image_size = HeightWidth(480, 480)
    offset = 64
    processor = RasterioRasterSplit(image_size=image_size, offset=offset)

    rasters = list(processor.execute(s2_l2a_raster))

    for i, raster in enumerate(rasters):
        assert raster.geometry.bounds[0] >= s2_l2a_raster.geometry.bounds[0]
        assert raster.geometry.bounds[1] >= s2_l2a_raster.geometry.bounds[1]
        assert raster.geometry.bounds[2] <= s2_l2a_raster.geometry.bounds[2]
        assert raster.geometry.bounds[3] <= s2_l2a_raster.geometry.bounds[3]

        with rasterio.open(io.BytesIO(s2_l2a_raster.content)) as exp_src:
            exp_image = exp_src.read()
            exp_bounds = exp_src.bounds
        with rasterio.open(io.BytesIO(raster.content)) as src:
            image = src.read()
            bounds = src.bounds

        assert image.dtype == exp_image.dtype
        assert bounds[0] >= exp_bounds[0]
        assert bounds[1] >= exp_bounds[1]
        assert bounds[2] <= exp_bounds[2]
        assert bounds[3] <= exp_bounds[3]
