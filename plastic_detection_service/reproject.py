from osgeo import gdal, osr


def raster_to_wgs84(input_raster: str) -> bytes:
    input_ds = gdal.Open(input_raster)

    srs_utm = osr.SpatialReference()
    srs_utm.ImportFromWkt(input_ds.GetProjection())

    srs_wgs84 = osr.SpatialReference()
    srs_wgs84.ImportFromEPSG(4326)

    # Create a coordinate transformation from UTM to WGS 84
    osr.CoordinateTransformation(srs_utm, srs_wgs84)

    out_path_memory = "/vsimem/temp.tif"
    out_ds = gdal.Warp(out_path_memory, input_ds, dstSRS=srs_wgs84)
    del input_ds
    del out_path_memory
    return out_ds.ReadRaster()


if __name__ == "__main__":
    reproj = raster_to_wgs84(
        "../images/120.53058253709094_14.42725911466126_120.57656259935047_14.47005515811605.tif"
    )
    print(reproj)
