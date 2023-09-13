from osgeo import gdal


def get_gdal_ds_from_memory(input_raster: bytes) -> gdal.Dataset:
    gdal.FileFromMemBuffer("/vsimem/input_raster.tif", input_raster)
    input_ds = gdal.Open("/vsimem/input_raster.tif")
    return input_ds
