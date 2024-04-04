import io

import numpy as np
import pytest
import rasterio

from src.processing.raster_operations import (
    RasterioRasterPad,
    RasterioRasterSplit,
    RasterioRasterUnpad,
)
from src.types import HeightWidth


@pytest.mark.parametrize("padding", [0, 16, 32, 64, 128])
@pytest.mark.parametrize("divisible_by", [16, 32, 64, 128])
def test_pad_raster(s2_l2a_raster, padding, divisible_by):
    processor = RasterioRasterPad(padding=padding, divisible_by=divisible_by)

    padded_raster = processor.execute(s2_l2a_raster)

    assert padded_raster.size >= s2_l2a_raster.size + (padding, padding)

    assert padded_raster.crs == s2_l2a_raster.crs
    assert padded_raster.bands == s2_l2a_raster.bands
    assert padded_raster.content != s2_l2a_raster.content
    assert isinstance(padded_raster.content, bytes)

    # check image size is a multiple of divisible_by
    assert padded_raster.size[0] % divisible_by == 0
    assert padded_raster.size[1] % divisible_by == 0

    # check padding_size is as expected
    exp_padding_size = processor._calculate_padding_size(
        s2_l2a_raster.to_numpy(), padding
    )
    assert padded_raster.padding_size == exp_padding_size[0]


@pytest.mark.slow
@pytest.mark.parametrize("padding", [32])
@pytest.mark.parametrize("image_size", [HeightWidth(480, 480)])
@pytest.mark.parametrize("divisible_by", [32])
def test_pad_raster_with_split_full_durban_scene(
    durban_full_raster, padding, image_size, divisible_by
):
    split_processor = RasterioRasterSplit(image_size=image_size, offset=padding)

    for split_raster in split_processor.execute(durban_full_raster):
        padded_raster = RasterioRasterPad(
            padding=padding, divisible_by=divisible_by
        ).execute(split_raster)

        assert padded_raster.crs == split_raster.crs
        assert padded_raster.bands == split_raster.bands
        assert padded_raster.content != split_raster.content
        assert isinstance(padded_raster.content, bytes)

        # check image size is a multiple of divisible_by
        assert padded_raster.size[0] % divisible_by == 0
        assert padded_raster.size[1] % divisible_by == 0

        # check padding_size is as expected
        exp_padding_size = RasterioRasterPad(
            padding, divisible_by=divisible_by
        )._calculate_padding_size(split_raster.to_numpy(), padding)
        assert padded_raster.padding_size == exp_padding_size[0]


def test_unpad_split_rasters(s2_l2a_raster):
    split_strategy = RasterioRasterSplit(image_size=HeightWidth(480, 480), offset=64)
    pad_strategy = RasterioRasterPad(padding=64)
    unpad_strategy = RasterioRasterUnpad()

    rasters = list(split_strategy.execute(s2_l2a_raster))
    padded_rasters = [pad_strategy.execute(raster) for raster in rasters]
    unpadded_rasters = [unpad_strategy.execute(raster) for raster in padded_rasters]

    for exp, result in zip(rasters, unpadded_rasters):
        with rasterio.open(io.BytesIO(exp.content)) as exp_src:
            exp_image = exp_src.read()
            exp_meta = exp_src.meta.copy()

        with rasterio.open(io.BytesIO(result.content)) as src:
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
        assert result.size == exp.size
        assert result.crs == exp.crs
        assert result.bands == exp.bands
        assert result.padding_size == (0, 0)
        assert result.geometry == exp.geometry
