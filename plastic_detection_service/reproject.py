from osgeo import gdal, osr


def raster_to_wgs84(input_raster):
    input_ds = gdal.Open(input_raster)

    srs_utm = osr.SpatialReference()
    srs_utm.ImportFromWkt(input_ds.GetProjection())

    srs_wgs84 = osr.SpatialReference()
    srs_wgs84.ImportFromEPSG(4326)

    # Create a coordinate transformation from UTM to WGS 84
    osr.CoordinateTransformation(srs_utm, srs_wgs84)

    # Create a new raster dataset with the same dimensions but in WGS 84
    out_path = input_raster.replace(".tiff", "_wgs84.tiff")
    return gdal.Warp(out_path, input_ds, dstSRS=srs_wgs84)


if __name__ == "__main__":
    raster_to_wgs84(
        "../images/4df92568740fcdb7e339d7e5e2848ad0/response_prediction.tiff"
    )
