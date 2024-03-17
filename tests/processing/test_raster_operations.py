import io

import numpy as np
import pytest
import rasterio
from shapely.geometry import Polygon

from plastic_detection_service.models import Raster
from plastic_detection_service.processing.raster_operations import (
    CompositeRasterOperation,
    RasterInferenceOperation,
    RasterioRasterMerge,
    RasterioRasterPad,
    RasterioRasterReproject,
    RasterioRasterSplit,
    RasterioRasterToVector,
    RasterioRasterUnpad,
)


def _calculate_padding_size(
    image: np.ndarray, target_image_size: tuple[int, int], padding: int
) -> tuple[int, int]:
    _, input_image_height, input_image_width = image.shape

    target_height_with_padding = target_image_size[0] + padding * 2
    target_width_with_padding = target_image_size[1] + padding * 2

    padding_height = round(target_height_with_padding - input_image_height) / 2
    padding_width = round(target_width_with_padding - input_image_width) / 2

    return int(padding_height), int(padding_width)


@pytest.mark.parametrize(
    "strategy",
    [
        RasterioRasterReproject(target_crs=4326, target_bands=[1]),
        CompositeRasterOperation(
            [RasterioRasterReproject(target_crs=4326, target_bands=[1])]
        ),
    ],
)
def test_reproject_raster(raster: Raster, rasterio_ds, strategy):
    target_crs = 4326
    target_bands = [1]

    out_file = f"tests/assets/test_out_reprojected_{strategy.__class__.__name__}.tif"
    reprojected_raster = strategy.execute(raster)

    reprojected_raster.to_file(out_file)
    assert reprojected_raster.crs == target_crs
    assert reprojected_raster.bands == target_bands
    assert isinstance(reprojected_raster, Raster)

    original_mean = rasterio_ds.read().mean()
    reprojected_mean = np.mean(reprojected_raster.to_numpy())
    assert np.isclose(
        original_mean, reprojected_mean, rtol=0.03
    )  # Allow a relative tolerance of 3%

    # check if the reprojected geometry coordinates are in degrees
    assert reprojected_raster.geometry.bounds[0] > -180
    assert reprojected_raster.geometry.bounds[2] < 180
    assert reprojected_raster.geometry.bounds[1] > -90
    assert reprojected_raster.geometry.bounds[3] < 90


def test_to_vector(raster: Raster):
    strategy = RasterioRasterToVector(band=1)
    vectors = list(
        strategy.execute(
            raster=raster,
        )
    )
    assert len(vectors) == 14311

    for vec in vectors:
        assert isinstance(vec.pixel_value, int)
        assert isinstance(vec.geometry, Polygon)
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
    out_file = f"tests/assets/test_out_split_{processor.__class__.__name__}.tif"
    split_raster = next(processor.execute(s2_l2a_raster))

    split_raster.to_file(out_file)

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


def test_pad_raster(s2_l2a_raster):
    padding = 64
    processor = RasterioRasterPad(padding=64)
    out_file = f"tests/assets/test_out_pad_{processor.__class__.__name__}.tif"

    image_size = (s2_l2a_raster.size[0], s2_l2a_raster.size[1])
    padded_raster = processor.execute(s2_l2a_raster)

    padded_raster.to_file(out_file)

    assert padded_raster.size == (
        image_size[0] + 2 * padding,
        image_size[1] + 2 * padding,
    )
    assert padded_raster.crs == s2_l2a_raster.crs
    assert padded_raster.bands == s2_l2a_raster.bands
    assert padded_raster.content != s2_l2a_raster.content
    assert isinstance(padded_raster.content, bytes)

    # check padding_size is as expected
    exp_padding_size = _calculate_padding_size(
        s2_l2a_raster.to_numpy(), image_size, padding
    )
    assert padded_raster.padding_size == exp_padding_size

    with rasterio.open("tests/assets/test_exp_pad.tif") as exp_src:
        exp_image = exp_src.read()
        with rasterio.open(io.BytesIO(padded_raster.content)) as src:
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


def test_unpad_raster(s2_l2a_raster):
    pad_strategy = RasterioRasterPad(padding=64)
    unpad_strategy = RasterioRasterUnpad()
    out_file = f"tests/assets/test_out_unpad_{unpad_strategy.__class__.__name__}.tif"
    padded_raster = pad_strategy.execute(s2_l2a_raster)
    unpadded_raster = unpad_strategy.execute(padded_raster)

    unpadded_raster.to_file(out_file)

    # check if the unpadded raster is the same as the original raster
    assert unpadded_raster.size == s2_l2a_raster.size
    assert unpadded_raster.crs == s2_l2a_raster.crs
    assert unpadded_raster.bands == s2_l2a_raster.bands

    assert isinstance(unpadded_raster.content, bytes)
    assert unpadded_raster.padding_size == (0, 0)
    assert unpadded_raster.geometry == s2_l2a_raster.geometry
    assert np.array_equal(unpadded_raster.to_numpy(), s2_l2a_raster.to_numpy())


def test_merge_rasters(s2_l2a_raster):
    merge_strategy = RasterioRasterMerge(
        target_raster=s2_l2a_raster, offset=64, smooth_overlap=False
    )
    split_strategy = RasterioRasterSplit(image_size=(480, 480), offset=64)
    rasters = list(split_strategy.execute(s2_l2a_raster))
    merged = merge_strategy.execute(rasters)
    merged.to_file(
        f"tests/assets/test_out_merge_{merge_strategy.__class__.__name__}.tif"
    )
    assert merged.size == s2_l2a_raster.size
    assert merged.dtype == s2_l2a_raster.dtype
    assert merged.crs == s2_l2a_raster.crs
    assert merged.bands == s2_l2a_raster.bands
    assert isinstance(merged.content, bytes)
    assert merged.padding_size == (0, 0)
    assert merged.geometry == s2_l2a_raster.geometry
    assert np.array_equal(merged.to_numpy(), s2_l2a_raster.to_numpy())


def _inference_func(content: bytes) -> bytes:
    with rasterio.open(io.BytesIO(content)) as src:
        image = src.read()
    band1 = image[0, :, :]
    return band1.tobytes()


def test_inference_raster(raster):
    operation = RasterInferenceOperation(_inference_func)

    result = operation.execute(raster)

    assert isinstance(result, Raster)

    assert result.size == raster.size
    assert result.dtype == raster.dtype
    assert result.crs == raster.crs
    assert result.bands == raster.bands
    assert result.geometry == raster.geometry

    assert isinstance(result.content, bytes)

    assert result.content == raster.content


def test_composite_raster_operation(s2_l2a_raster):
    reproject_op = RasterioRasterSplit(image_size=(480, 480), offset=64)
    results = []
    for r in reproject_op.execute(s2_l2a_raster):
        comp_op = CompositeRasterOperation(
            [
                RasterioRasterPad(padding=64),
                RasterInferenceOperation(inference_func=_inference_func),
                RasterioRasterUnpad(),
            ]
        )
        results.append(comp_op.execute(r))

    merge_op = RasterioRasterMerge(
        target_raster=s2_l2a_raster, offset=64, smooth_overlap=True
    )
    merged = merge_op.execute(results)
    merged.to_file("tests/assets/test_out_composite.tif")
    assert merged.size == s2_l2a_raster.size
    assert merged.dtype == s2_l2a_raster.dtype
    assert merged.crs == s2_l2a_raster.crs
    assert merged.bands == s2_l2a_raster.bands
    assert isinstance(merged.content, bytes)
    assert merged.padding_size == (0, 0)
    assert merged.geometry == s2_l2a_raster.geometry
