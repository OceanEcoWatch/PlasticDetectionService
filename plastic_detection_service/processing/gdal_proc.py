import uuid
from typing import Generator, Iterable, Union

import numpy as np
from osgeo import gdal, ogr, osr
from shapely import wkt
from shapely.geometry import Polygon, box

from plastic_detection_service.models import Vector

from .abstractions import Raster, RasterProcessor, VectorsProcessor


class GdalRasterProcessor(RasterProcessor):
    def __init__(self):
        self.TEMP_FILE = f"/vsimem/{uuid.uuid4()}.tif"

    def _get_gdal_ds_from_memory(self, content: bytes) -> gdal.Dataset:
        try:
            gdal.FileFromMemBuffer(self.TEMP_FILE, content)
            return gdal.Open(self.TEMP_FILE, gdal.GA_Update)
        finally:
            gdal.Unlink(self.TEMP_FILE)

    def _get_epsg_from_ds(self, ds: gdal.Dataset) -> int:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjection())
        return int(srs.GetAttrValue("AUTHORITY", 1))

    def _get_raster_geometry(self, ds: gdal.Dataset) -> Polygon:
        gt = ds.GetGeoTransform()

        xmin = gt[0]
        ymax = gt[3]
        xmax = xmin + gt[1] * ds.RasterXSize
        ymin = ymax + gt[5] * ds.RasterYSize
        return box(xmin, ymin, xmax, ymax)

    def _srs_from_epsg(self, epsg: int) -> osr.SpatialReference:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        return srs

    def _ds_to_raster(self, ds: gdal.Dataset) -> Raster:
        gdal.GetDriverByName("GTiff").CreateCopy(self.TEMP_FILE, ds)
        f = gdal.VSIFOpenL(self.TEMP_FILE, "rb")
        gdal.VSIFSeekL(f, 0, 2)
        size = gdal.VSIFTellL(f)
        gdal.VSIFSeekL(f, 0, 0)
        content = gdal.VSIFReadL(1, size, f)

        raster = Raster(
            content=content,
            width=ds.RasterXSize,
            height=ds.RasterYSize,
            crs=self._get_epsg_from_ds(ds),
            bands=[i for i in range(1, ds.RasterCount + 1)],
            geometry=self._get_raster_geometry(ds),
        )
        ds = None  # type: ignore
        return raster

    def reproject_raster(
        self,
        raster: Raster,
        target_crs: int,
        target_bands: list[int],
        resample_alg: str = "nearest",
    ) -> Raster:
        srs_utm = self._srs_from_epsg(raster.crs)
        srs_wgs84 = self._srs_from_epsg(target_crs)

        osr.CoordinateTransformation(srs_utm, srs_wgs84)
        in_ds = self._get_gdal_ds_from_memory(raster.content)
        out_ds: gdal.Dataset = gdal.Warp(
            self.TEMP_FILE,
            in_ds,
            dstSRS=srs_wgs84,
            resampleAlg=resample_alg,
            srcBands=target_bands,
        )  # type: ignore

        return self._ds_to_raster(out_ds)

    def to_vector(
        self, raster: Raster, field: str, band: int = 1
    ) -> Generator[Vector, None, None]:
        driver = ogr.GetDriverByName("Memory")
        output_vector_ds: ogr.DataSource = driver.CreateDataSource("")

        srs = self._srs_from_epsg(raster.crs)
        output_layer = output_vector_ds.CreateLayer(
            "polygons", srs=srs, geom_type=ogr.wkbPolygon
        )

        fld = ogr.FieldDefn(field, ogr.OFTInteger)
        output_layer.CreateField(fld)
        dst_field = output_layer.GetLayerDefn().GetFieldIndex(field)

        in_ds = self._get_gdal_ds_from_memory(raster.content)
        band = in_ds.GetRasterBand(band)

        gdal.Polygonize(band, None, output_layer, dst_field, [], callback=None)

        for feature in output_vector_ds.GetLayer():
            yield Vector(
                geometry=wkt.loads(feature.GetGeometryRef().ExportToWkt()),
                pixel_value=feature.GetField(field),
            )

    def round_pixel_values(
        self, raster: Raster, round_to: Union[int, float] = 1
    ) -> Raster:
        ds = self._get_gdal_ds_from_memory(raster.content)

        for i in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(i)
            array = band.ReadAsArray()
            rounded_array = np.round(array / round_to) * round_to
            band.WriteArray(rounded_array)

        return self._ds_to_raster(ds)


class GdalVectorsProcessor(VectorsProcessor):
    def to_raster(self, vectors: Iterable[Vector]) -> Raster:
        raise NotImplementedError
