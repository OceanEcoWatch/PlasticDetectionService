from osgeo import gdal, ogr, osr


def raster2points(input_raster: bytes, output_vector, pixel_value_threshold=0):
    gdal.FileFromMemBuffer("/vsimem/pred_raster.tif", input_raster)

    raster_ds = gdal.Open("/vsimem/pred_raster.tif")
    raster_layer = raster_ds.GetRasterBand(1)

    srs = osr.SpatialReference()
    srs.ImportFromWkt(raster_ds.GetProjection())

    driver = ogr.GetDriverByName("GeoJSON")
    vector_ds = driver.CreateDataSource(output_vector)

    vector_layer = vector_ds.CreateLayer("points", srs, ogr.wkbPoint)

    field_defn = ogr.FieldDefn("pixel_value", ogr.OFTInteger)
    vector_layer.CreateField(field_defn)

    for x in range(raster_layer.XSize):
        for y in range(raster_layer.YSize):
            value = raster_layer.ReadAsArray(x, y, 1, 1)[0, 0]
            if value != pixel_value_threshold:
                point = ogr.Geometry(ogr.wkbPoint)
                geotransform = raster_ds.GetGeoTransform()
                x_coord = geotransform[0] + x * geotransform[1]
                y_coord = geotransform[3] + y * geotransform[5]
                point.AddPoint(x_coord, y_coord)

                feature = ogr.Feature(vector_layer.GetLayerDefn())
                feature.SetGeometry(point)

                feature.SetField("pixel_value", int(value))

                vector_layer.CreateFeature(feature)

                del feature

    del raster_ds
    del vector_ds

    print("Raster to vector conversion completed.")


def polygonize_raster(in_path, out_path, layer_name):
    src_ds = gdal.Open(in_path)

    srcband = src_ds.GetRasterBand(1)
    drv = ogr.GetDriverByName("GeoJSON")
    dst_ds = drv.CreateDataSource(out_path)

    sp_ref = osr.SpatialReference()
    sp_ref.SetFromUserInput("EPSG:4326")

    dst_layer = dst_ds.CreateLayer(layer_name, srs=sp_ref)

    fld = ogr.FieldDefn("HA", ogr.OFTInteger)
    dst_layer.CreateField(fld)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("HA")

    gdal.Polygonize(srcband, None, dst_layer, dst_field, [], callback=None)

    del dst_ds
    del src_ds


if __name__ == "__main__":
    in_path = (
        "../images/4df92568740fcdb7e339d7e5e2848ad0/response_prediction_wgs84.tiff"
    )
    out_path = "prediction.geojson"
    layer_name = "prediction"
    raster2points(in_path, out_path)
