import io

import numpy as np
import rasterio

from src.raster_op.convert import (
    RasterioDtypeConversion,
)


def test_dtype_conversion(s2_l2a_raster):
    dtype = rasterio.uint8
    strategy = RasterioDtypeConversion(dtype=dtype)
    converted_raster = next(strategy.execute([s2_l2a_raster]))
    assert converted_raster.size == s2_l2a_raster.size
    assert converted_raster.crs == s2_l2a_raster.crs
    assert converted_raster.bands == s2_l2a_raster.bands
    assert converted_raster.dtype == dtype
    assert isinstance(converted_raster.content, bytes)
    assert converted_raster.geometry == s2_l2a_raster.geometry

    with rasterio.open(io.BytesIO(s2_l2a_raster.content)) as src:
        exp_image = strategy._scale(src.read()).astype(dtype)
        exp_meta = src.meta.copy()
        exp_meta["dtype"] = dtype

    with rasterio.open(io.BytesIO(converted_raster.content)) as src:
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


def test_uint8_to_uint16_scaling():
    target_dtype = rasterio.uint16
    strategy = RasterioDtypeConversion(dtype=target_dtype)
    image = np.array([0, 128, 255], dtype=rasterio.uint8)
    scaled_image = strategy._scale(image).astype(target_dtype)
    assert scaled_image.dtype == target_dtype
    assert scaled_image.min() >= np.iinfo(target_dtype).min
    assert scaled_image.max() <= np.iinfo(target_dtype).max


def test_uint16_to_uint8_scaling():
    target_dtype = rasterio.uint8
    strategy = RasterioDtypeConversion(dtype=target_dtype)
    image = np.array([0, 128, 255], dtype=rasterio.uint16)
    scaled_image = strategy._scale(image).astype(target_dtype)
    assert scaled_image.dtype == target_dtype
    assert scaled_image.min() >= np.iinfo(target_dtype).min
    assert scaled_image.max() <= np.iinfo(target_dtype).max


def test_integer_to_float_scaling():
    target_dtype = rasterio.float32
    strategy = RasterioDtypeConversion(dtype=target_dtype)
    image = np.array([0, 128, 255], dtype=rasterio.uint8)
    scaled_image = strategy._scale(image).astype(target_dtype)
    assert scaled_image.dtype == target_dtype
    assert scaled_image.min() >= 0
    assert scaled_image.max() <= 1
    assert all(
        isinstance(x, np.floating) for x in scaled_image.flat
    ), "Not all elements are floats"


def test_float_to_integer_scaling():
    target_dtype = rasterio.uint8
    strategy = RasterioDtypeConversion(dtype=target_dtype)
    image = np.array([0, 0.5, 1], dtype=rasterio.float32)
    scaled_image = strategy._scale(image).astype(target_dtype)
    assert scaled_image.dtype == target_dtype
    assert scaled_image.min() >= np.iinfo(target_dtype).min
    assert scaled_image.max() <= np.iinfo(target_dtype).max
    assert all(
        isinstance(x, np.integer) or isinstance(x, np.uint8)  # type: ignore
        for x in scaled_image.flat  # Fix: np.uint8
    ), "Not all elements are integers"
