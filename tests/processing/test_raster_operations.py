import io

import numpy as np
import pytest
import rasterio
from shapely.geometry import Point

from plastic_detection_service.inference.inference_callback import (
    local_inference_callback,
)
from plastic_detection_service.models import Raster
from plastic_detection_service.processing.merge_callable import (
    copy_smooth,
    smooth_overlap_callable,
)
from plastic_detection_service.processing.raster_operations import (
    CompositeRasterOperation,
    RasterInference,
    RasterioDtypeConversion,
    RasterioRasterMerge,
    RasterioRasterPad,
    RasterioRasterReproject,
    RasterioRasterSplit,
    RasterioRasterToVector,
    RasterioRasterUnpad,
    RasterioRemoveBand,
)
from plastic_detection_service.types import HeightWidth


def _mock_inference_func(_raster_bytes) -> bytes:
    with rasterio.open(io.BytesIO(_raster_bytes)) as src:
        image = src.read()
        band1 = image[0, :, :].astype(np.float32)
        return band1.tobytes()


@pytest.mark.parametrize(
    "strategy",
    [
        RasterioRasterReproject(target_crs=4326, target_bands=[1]),
        CompositeRasterOperation(
            [RasterioRasterReproject(target_crs=4326, target_bands=[1])]
        ),
    ],
)
def test_reproject_raster(raster, rasterio_ds, strategy):
    target_crs = 4326
    target_bands = [1]

    reprojected_raster = strategy.execute(raster)

    assert reprojected_raster.crs == target_crs
    assert reprojected_raster.bands == target_bands
    assert isinstance(reprojected_raster, Raster)

    original_mean = rasterio_ds.read().mean()
    reprojected_mean = np.mean(reprojected_raster.to_numpy())
    assert np.isclose(
        original_mean, reprojected_mean, rtol=0.05
    )  # Allow a relative tolerance of 5%

    # check if the reprojected geometry coordinates are in degrees
    assert reprojected_raster.geometry.bounds[0] > -180
    assert reprojected_raster.geometry.bounds[2] < 180
    assert reprojected_raster.geometry.bounds[1] > -90
    assert reprojected_raster.geometry.bounds[3] < 90


def test_to_vector(raster: Raster):
    strategy = RasterioRasterToVector(band=1)
    vectors = strategy.execute(
        raster=raster,
    )

    for vec in vectors:
        assert isinstance(vec.pixel_value, int), "Pixel value is not an integer"
        assert isinstance(vec.geometry, Point), "Geometry is not a Point"
        assert vec.crs == raster.crs

        # test if geometry is within the bounds of the raster
        assert vec.geometry.bounds[0] >= raster.geometry.bounds[0]
        assert vec.geometry.bounds[1] >= raster.geometry.bounds[1]
        assert vec.geometry.bounds[2] <= raster.geometry.bounds[2]
        assert vec.geometry.bounds[3] <= raster.geometry.bounds[3]


def test_split_raster(s2_l2a_raster):
    image_size = (480, 480)
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


def test_remove_band(s2_l2a_raster, caplog):
    band = 1
    remove_band_strategy = RasterioRemoveBand(band=band)
    removed_band_raster = remove_band_strategy.execute(s2_l2a_raster)
    assert "does not exist in raster, skipping" not in caplog.text
    assert removed_band_raster.size == s2_l2a_raster.size
    assert removed_band_raster.crs == s2_l2a_raster.crs
    assert band not in removed_band_raster.bands
    assert isinstance(removed_band_raster.content, bytes)

    assert removed_band_raster.geometry == s2_l2a_raster.geometry

    with rasterio.open(io.BytesIO(removed_band_raster.content)) as src:
        image = src.read()
        assert image.shape[0] == len(s2_l2a_raster.bands) - 1


def test_remove_band_skips_nonexistent_band(s2_l2a_raster, caplog):
    remove_band_strategy = RasterioRemoveBand(band=13)
    removed_band_raster = remove_band_strategy.execute(s2_l2a_raster)
    assert removed_band_raster.size == s2_l2a_raster.size
    assert removed_band_raster.crs == s2_l2a_raster.crs
    assert removed_band_raster.bands == s2_l2a_raster.bands
    assert isinstance(removed_band_raster.content, bytes)

    assert removed_band_raster.geometry == s2_l2a_raster.geometry
    assert "does not exist in raster, skipping" in caplog.text


def test_dtype_conversion(s2_l2a_raster):
    dtype = rasterio.uint8
    strategy = RasterioDtypeConversion(dtype=dtype)
    converted_raster = strategy.execute(s2_l2a_raster)
    assert converted_raster.size == s2_l2a_raster.size
    assert converted_raster.crs == s2_l2a_raster.crs
    assert converted_raster.bands == s2_l2a_raster.bands
    assert converted_raster.dtype == dtype
    assert isinstance(converted_raster.content, bytes)
    assert converted_raster.geometry == s2_l2a_raster.geometry

    with rasterio.open(io.BytesIO(s2_l2a_raster.content)) as src:
        exp_image = strategy._scale(src.read())
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
    scaled_image = strategy._scale(image)
    assert scaled_image.dtype == target_dtype
    assert scaled_image.min() >= np.iinfo(target_dtype).min
    assert scaled_image.max() <= np.iinfo(target_dtype).max


def test_uint16_to_uint8_scaling():
    target_dtype = rasterio.uint8
    strategy = RasterioDtypeConversion(dtype=target_dtype)
    image = np.array([0, 128, 255], dtype=rasterio.uint16)
    scaled_image = strategy._scale(image)
    assert scaled_image.dtype == target_dtype
    assert scaled_image.min() >= np.iinfo(target_dtype).min
    assert scaled_image.max() <= np.iinfo(target_dtype).max


def test_integer_to_float_scaling():
    target_dtype = rasterio.float32
    strategy = RasterioDtypeConversion(dtype=target_dtype)
    image = np.array([0, 128, 255], dtype=rasterio.uint8)
    scaled_image = strategy._scale(image)
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
    scaled_image = strategy._scale(image)
    assert scaled_image.dtype == target_dtype
    assert scaled_image.min() >= np.iinfo(target_dtype).min
    assert scaled_image.max() <= np.iinfo(target_dtype).max
    assert all(
        isinstance(x, np.integer) or isinstance(x, np.uint8) for x in scaled_image.flat
    ), "Not all elements are integers"


def test_inference_raster_mock(s2_l2a_raster):
    operation = RasterInference(inference_func=_mock_inference_func)

    result = operation.execute(s2_l2a_raster)

    assert isinstance(result, Raster)

    assert result.size == s2_l2a_raster.size
    assert result.dtype == "float32"
    assert result.crs == s2_l2a_raster.crs
    assert result.bands == [1]
    assert result.geometry == s2_l2a_raster.geometry

    assert isinstance(result.content, bytes)


@pytest.mark.slow
def test_inference_raster_real(s2_l2a_raster, pred_durban_first_split_raster):
    raster = next(
        RasterioRasterSplit(image_size=HeightWidth(480, 480), offset=64).execute(
            s2_l2a_raster
        )
    )
    raster = RasterioRasterPad(padding=64).execute(raster)
    raster = RasterioRemoveBand(band=13).execute(raster)
    inference_raster = RasterInference(inference_func=local_inference_callback).execute(
        raster
    )

    inference_raster = RasterioRasterUnpad().execute(inference_raster)
    inference_raster.to_file("tests/assets/test_out_inference.tif")
    assert isinstance(inference_raster, Raster)

    # assert pixel values are within the expected range
    assert np.all(inference_raster.to_numpy() >= 0)
    assert np.any(inference_raster.to_numpy() > 0)
    assert np.all(inference_raster.to_numpy() <= 1)

    assert inference_raster.size == pred_durban_first_split_raster.size
    assert inference_raster.crs == pred_durban_first_split_raster.crs


def test_composite_raster_operation(s2_l2a_raster):
    split_op = RasterioRasterSplit(image_size=HeightWidth(480, 480), offset=64)
    comp_op = CompositeRasterOperation(
        [
            RasterioRasterPad(padding=64),
            RasterioRemoveBand(band=13),
            RasterInference(inference_func=_mock_inference_func),
            RasterioRasterUnpad(),
        ]
    )
    results = []
    for raster in split_op.execute(s2_l2a_raster):
        result = comp_op.execute(raster)
        results.append(result)

    merge_op = RasterioRasterMerge(offset=64, merge_method=copy_smooth)

    merged = merge_op.execute(results)

    merged.to_file(
        f"tests/assets/test_out_composite_{_mock_inference_func.__name__}.tif"
    )
    assert merged.size == s2_l2a_raster.size
    assert merged.dtype == "float32"
    assert merged.crs == s2_l2a_raster.crs
    assert merged.bands == [1]
    assert isinstance(merged.content, bytes)
    assert merged.padding_size == (0, 0)
    assert merged.geometry == s2_l2a_raster.geometry


@pytest.mark.slow
def test_composite_raster_real_inference(s2_l2a_raster, raster):
    split_op = RasterioRasterSplit(image_size=HeightWidth(480, 480), offset=64)
    comp_op = CompositeRasterOperation(
        [
            RasterioRasterPad(padding=64),
            RasterioRemoveBand(band=13),
            RasterInference(inference_func=local_inference_callback),
            RasterioRasterUnpad(),
        ]
    )
    results = []
    for r in split_op.execute(s2_l2a_raster):
        result = comp_op.execute(r)
        results.append(result)

    merge_op = RasterioRasterMerge(offset=64, merge_method=copy_smooth)

    merged = merge_op.execute(results)
    merged = RasterioDtypeConversion(dtype="uint8").execute(merged)
    merged.to_file(
        f"tests/assets/test_out_composite_{local_inference_callback.__name__}.tif"
    )

    assert merged.size == raster.size
    assert merged.dtype == raster.dtype
    assert merged.crs == raster.crs
    assert merged.bands == [1]
    assert isinstance(merged.content, bytes)
    assert merged.padding_size == (0, 0)
    assert merged.geometry == raster.geometry
    assert np.isclose(merged.to_numpy().mean(), raster.to_numpy().mean(), rtol=0.05)
