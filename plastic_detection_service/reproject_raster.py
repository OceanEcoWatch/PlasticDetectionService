import io
from pathlib import Path

import rasterio
from osgeo import gdal, osr
from rasterio.warp import Resampling, calculate_default_transform, reproject


def reproject_raster(raster: io.BytesIO, dst_crs: str) -> bytes:
    with rasterio.open(raster) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )

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


def raster_to_wgs84(input_raster: bytes) -> gdal.Dataset:
    gdal.FileFromMemBuffer("/vsimem/input_raster.tif", input_raster)

    input_ds = gdal.Open("/vsimem/input_raster.tif")
    srs_utm = osr.SpatialReference()
    srs_utm.ImportFromWkt(input_ds.GetProjection())

    srs_wgs84 = osr.SpatialReference()
    srs_wgs84.ImportFromEPSG(4326)

    # Create a coordinate transformation from UTM to WGS 84
    osr.CoordinateTransformation(srs_utm, srs_wgs84)

    out_path_memory = "/vsimem/temp.tif"
    out_ds: gdal.Dataset = gdal.Warp(
        out_path_memory,
        input_ds,
        dstSRS=srs_wgs84,
        resampleAlg=gdal.GRA_Cubic,
    )  # type: ignore
    return out_ds


if __name__ == "__main__":
    # Example usage
    with open(
        "../images/4df92568740fcdb7e339d7e5e2848ad0/response_prediction.tiff", "rb"
    ) as f:
        wgs84_raster = raster_to_wgs84(f.read())

    # save
    gdal.Translate(
        "../images/4df92568740fcdb7e339d7e5e2848ad0/response_prediction_wgs84_cubic.tiff",
        wgs84_raster,
    )
