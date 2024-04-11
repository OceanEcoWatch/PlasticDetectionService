import io

import numpy as np
import pytest
import rasterio

from src.raster_op.merge import (
    RasterioRasterMerge,
    copy_smooth,
    smooth_overlap_callable,
)
from src.raster_op.split import (
    RasterioRasterSplit,
)
from src.types import HeightWidth


@pytest.mark.parametrize(
    "merge_method", ["first", smooth_overlap_callable, copy_smooth]
)
def test_merge_rasters(s2_l2a_raster, merge_method):
    merge_strategy = RasterioRasterMerge(offset=64, merge_method=merge_method)
    split_strategy = RasterioRasterSplit(image_size=HeightWidth(480, 480), offset=64)
    rasters = list(split_strategy.execute(s2_l2a_raster))
    merged = merge_strategy.execute(rasters)
    merged.to_file(
        f"tests/assets/test_out_merge_{merge_method if isinstance(merge_method, str) else merge_method.__name__}.tif"
    )
    assert merged.size == s2_l2a_raster.size
    assert merged.dtype == s2_l2a_raster.dtype
    assert merged.crs == s2_l2a_raster.crs
    assert merged.bands == s2_l2a_raster.bands
    assert isinstance(merged.content, bytes)
    assert merged.padding_size == (0, 0)
    assert merged.geometry == s2_l2a_raster.geometry
    assert np.array_equal(merged.to_numpy(), s2_l2a_raster.to_numpy())

    with rasterio.open(io.BytesIO(s2_l2a_raster.content)) as src:
        exp_image = src.read()
        exp_meta = src.meta.copy()

    with rasterio.open(io.BytesIO(merged.content)) as src:
        image = src.read()
        meta = src.meta.copy()

    assert image.shape == exp_image.shape
    assert image.dtype == exp_image.dtype
    assert meta["height"] == exp_meta["height"]
    assert meta["width"] == exp_meta["width"]
    assert meta["crs"] == exp_meta["crs"]
    assert meta["count"] == exp_meta["count"]
    assert meta["transform"] == exp_meta["transform"]
    assert meta["dtype"] == exp_meta["dtype"]

    assert np.array_equal(image, exp_image)

    assert merged.size == s2_l2a_raster.size
    assert merged.crs == s2_l2a_raster.crs
    assert merged.bands == s2_l2a_raster.bands
    assert merged.padding_size == (0, 0)
    assert merged.geometry == s2_l2a_raster.geometry
