import io
import json

import rasterio
from osgeo import gdal, ogr, osr
from rasterio.features import shapes
from shapely.geometry import MultiPolygon, shape


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


def polygonize_raster(input_gdal_ds: gdal.Dataset, crs: int = 4326) -> gdal.Dataset:
    driver = ogr.GetDriverByName("Memory")
    output_vector_ds = driver.CreateDataSource("")

    srs = osr.SpatialReference()

    srs.ImportFromEPSG(crs)
    output_layer = output_vector_ds.CreateLayer(
        "polygons", srs=srs, geom_type=ogr.wkbPolygon
    )

    fld = ogr.FieldDefn("pixel_value", ogr.OFTInteger)
    output_layer.CreateField(fld)
    dst_field = output_layer.GetLayerDefn().GetFieldIndex("pixel_value")
    band = input_gdal_ds.GetRasterBand(1)
    gdal.Polygonize(band, None, output_layer, dst_field, [], callback=None)

    return output_vector_ds


def vectorize_raster(input_raster: io.BytesIO) -> MultiPolygon:
    with rasterio.open(input_raster) as src:
        mask = src.read_masks(1)
        shapes_generator = shapes(mask, mask=mask, transform=src.transform)
        geometries = [shape(geometry) for geometry, _ in shapes_generator]
        return MultiPolygon(geometries)


if __name__ == "__main__":
    src = gdal.Open(
        "../images/120.53058253709094_14.42725911466126_120.57656259935047_14.47005515811605.tif"
    )
    ds = polygonize_raster(src)

    for feature in ds.GetLayer():
        print(feature.ExportToJson())
        print(shape(json.loads(feature.ExportToJson())["geometry"]))
