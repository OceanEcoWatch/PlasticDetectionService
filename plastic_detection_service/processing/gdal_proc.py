from typing import Generator

from osgeo import gdal, ogr, osr

from plastic_detection_service.models import Vector

from .abstractions import Raster, RasterProcessor


class GdalRasterProcessor(RasterProcessor):
    _temp_file = "/vsimem/temp.tif"

    def _get_gdal_ds_from_memory(
        self, content: bytes
    ) -> Generator[gdal.Dataset, None, None]:
        gdal.FileFromMemBuffer(self._temp_file, content)
        yield gdal.Open(self._temp_file)

        gdal.Unlink(self._temp_file)

    def _get_epsg_from_ds(self, ds: gdal.Dataset) -> int:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjection())
        return int(srs.GetAttrValue("AUTHORITY", 1))

    def _ds_to_raster(self, ds: gdal.Dataset) -> Raster:
        gdal.GetDriverByName("GTiff").CreateCopy(self._temp_file, ds)
        f = gdal.VSIFOpenL(self._temp_file, "rb")
        gdal.VSIFSeekL(f, 0, 2)
        size = gdal.VSIFTellL(f)
        gdal.VSIFSeekL(f, 0, 0)
        content = gdal.VSIFReadL(1, size, f)

        return Raster(
            content=content,
            width=ds.RasterXSize,
            height=ds.RasterYSize,
            crs=self._get_epsg_from_ds(ds),
            bands=[i for i in range(1, ds.RasterCount + 1)],
        )

    def reproject_raster(
        self,
        raster: Raster,
        target_crs: int,
        target_bands: list[int],
        resample_alg: str = gdal.GRA_NearestNeighbour,
    ) -> Raster:
        srs_utm = osr.SpatialReference()
        srs_utm.ImportFromEPSG(raster.crs)

        srs_wgs84 = osr.SpatialReference()
        srs_wgs84.ImportFromEPSG(target_crs)

        osr.CoordinateTransformation(srs_utm, srs_wgs84)
        in_ds = next(self._get_gdal_ds_from_memory(raster.content))
        out_ds: gdal.Dataset = gdal.Warp(
            self._temp_file,
            in_ds,
            dstSRS=srs_wgs84,
            resampleAlg=resample_alg,
            srcBands=target_bands,
        )  # type: ignore

        return self._ds_to_raster(out_ds)

    def to_vector(self, raster: Raster, field: str, band: int) -> Vector:
        driver = ogr.GetDriverByName("Memory")
        output_vector_ds: ogr.DataSource = driver.CreateDataSource("")

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(raster.crs)
        output_layer = output_vector_ds.CreateLayer(
            "polygons", srs=srs, geom_type=ogr.wkbPolygon
        )

        fld = ogr.FieldDefn(field, ogr.OFTInteger)
        output_layer.CreateField(fld)
        dst_field = output_layer.GetLayerDefn().GetFieldIndex(field)

        in_ds = next(self._get_gdal_ds_from_memory(raster.content))
        band = in_ds.GetRasterBand(band)

        gdal.Polygonize(band, None, output_layer, dst_field, [], callback=None)

        return Vector(geometry=output_vector_ds, field=field)
