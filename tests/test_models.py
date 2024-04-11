import os

import rasterio

from src.models import Raster


def test_raster_to_numpy(raster: Raster):
    numpy_arr = raster.to_numpy()
    assert numpy_arr.shape == (len(raster.bands), raster.size[1], raster.size[0])


def test_raster_to_numpy_same_as_rasterio(raster: Raster, rasterio_ds):
    numpy_arr = raster.to_numpy()
    for i in range(len(raster.bands)):
        assert (numpy_arr[i] == rasterio_ds.read(i + 1)).all()


def test_raster_to_file(raster: Raster):
    file = "tests/assets/test_raster_to_file.tif"
    try:
        raster.to_file(file)
        assert os.path.exists(file)

        with rasterio.open(file) as src:
            assert src.crs.to_epsg() == raster.crs
            assert src.bounds == raster.geometry.bounds
            assert src.shape == (raster.size[1], raster.size[0])
            for i in range(len(raster.bands)):
                assert (src.read(i + 1) == raster.to_numpy()[i]).all()
    finally:
        os.remove(file)


def test_vector_geojson(vector):
    geojson = vector.geojson
    assert geojson["type"] == "Feature"
    assert geojson["geometry"] == vector.geometry.__geo_interface__
    assert geojson["properties"]["pixel_value"] == vector.pixel_value
