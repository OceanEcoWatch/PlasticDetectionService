import io

import rasterio

from src.raster_op.band import RasterioRemoveBand


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
