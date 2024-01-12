import io
from pathlib import Path
from typing import Optional

import rasterio
from osgeo import gdal, osr
from rasterio.warp import Resampling, calculate_default_transform, reproject


def reproject_raster(raster: io.BytesIO, dst_crs: str) -> bytes:
    with rasterio.open(raster) as src:
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({"crs": dst_crs, "transform": transform, "width": width, "height": height})

        reprojected_raster = Path("reprojected_raster.tif")

        with rasterio.open(reprojected_raster, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )

    return reprojected_raster.read_bytes()


def raster_to_wgs84(
    input_raster: gdal.Dataset,
    target_bands: Optional[list] = None,
    resample_alg=gdal.GRA_NearestNeighbour,
) -> gdal.Dataset:
    srs_utm = osr.SpatialReference()
    srs_utm.ImportFromWkt(input_raster.GetProjection())

    srs_wgs84 = osr.SpatialReference()
    srs_wgs84.ImportFromEPSG(4326)

    osr.CoordinateTransformation(srs_utm, srs_wgs84)

    out_path_memory = "/vsimem/temp.tif"

    out_ds: gdal.Dataset = gdal.Warp(
        out_path_memory,
        input_raster,
        dstSRS=srs_wgs84,
        resampleAlg=resample_alg,
        srcBands=target_bands,
    )  # type: ignore

    return out_ds


if __name__ == "__main__":
    # Example usage
    with open("images/230000.0_1590000.0_235000.0_1595000.0.tif", "rb") as f:
        wgs84_raster = raster_to_wgs84(gdal.Open(f.name), resample_alg=gdal.GRA_Cubic)

    # save
    gdal.Translate(
        "response_wgs84_test.tiff",
        wgs84_raster,
    )
