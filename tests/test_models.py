from plastic_detection_service.models import Raster


def test_raster_to_numpy(raster: Raster):
    numpy_arr = raster.to_numpy()
    assert numpy_arr.shape == (len(raster.bands), raster.size[1], raster.size[0])


def test_raster_to_numpy_same_as_gdal(raster: Raster, ds):
    numpy_arr = raster.to_numpy()
    for i in range(len(raster.bands)):
        assert (numpy_arr[i] == ds.GetRasterBand(i + 1).ReadAsArray()).all()
