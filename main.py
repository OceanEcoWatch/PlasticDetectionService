import datetime
import io
import json
import os
import ssl
import tempfile

import numpy as np
import rasterio
from geoalchemy2.shape import from_shape
from geoalchemy2.types import RasterElement
from osgeo import gdal
from sentinelhub import CRS, BBox, UtmZoneSplitter
from shapely.geometry import box, shape
from sqlalchemy.orm import Session

from marinedebrisdetector.checkpoints import CHECKPOINTS
from marinedebrisdetector.model.segmentation_model import SegmentationModel
from marinedebrisdetector.predictor import ScenePredictor
from plastic_detection_service.config import config
from plastic_detection_service.constants import MANILLA_BAY_BBOX
from plastic_detection_service.db import (
    ClearWaterVector,
    PredictionRaster,
    PredictionVector,
    get_db_engine,
)
from plastic_detection_service.download_images import stream_in_images
from plastic_detection_service.evalscripts import L2A_12_BANDS_CLEAR_WATER_MASK
from plastic_detection_service.gdal_ds import get_gdal_ds_from_memory
from plastic_detection_service.reproject_raster import raster_to_wgs84
from plastic_detection_service.to_vector import (
    filter_out_no_data_polygons,
    polygonize_raster,
)


def image_generator(bbox_list, time_interval, evalscript, maxcc):
    for bbox in bbox_list:
        data = stream_in_images(
            config, bbox, time_interval, evalscript=evalscript, maxcc=maxcc
        )

        if data is not None:
            yield data


def inspect_raster_data(raster_bytes: bytes):
    # Create a temporary file to write the bytes
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmpfile:
        tmpfile.write(raster_bytes)
        tmp_filepath = tmpfile.name

    # Use rasterio to open the temporary file
    with rasterio.open(tmp_filepath, "r") as src:
        # Read the raster data
        data = src.read(
            1
        )  # Read the first band. Change the index if you have multiple bands.

    # Inspect the data values
    print(data)

    # Optionally, delete the temporary file after use
    os.remove(tmp_filepath)


def round_to_nearest_5(input_raster: gdal.Dataset) -> gdal.Dataset:
    band = input_raster.GetRasterBand(1)  # Assuming a single band raster
    data = band.ReadAsArray() * 100
    rounded_data = np.round(data / 5) * 5
    rounded_data = rounded_data.astype(np.int8)
    band.WriteArray(rounded_data)
    return input_raster


def round_to_int(input_raster: gdal.Dataset) -> gdal.Dataset:
    band = input_raster.GetRasterBand(1)  # Assuming a single band raster
    data = band.ReadAsArray() * 100
    rounded_data = np.round(data, 0).astype(np.int8)
    band.WriteArray(rounded_data)
    return input_raster


def main():
    bbox = BBox(MANILLA_BAY_BBOX, crs=CRS.WGS84)
    time_interval = ("2022-08-01", "2023-09-01")
    maxcc = 0.5
    out_dir = "images"

    ssl._create_default_https_context = (
        ssl._create_unverified_context
    )  # fix for SSL error on Mac

    bbox_list = UtmZoneSplitter([bbox], crs=CRS.WGS84, bbox_size=5000).get_bbox_list()

    data_gen = image_generator(
        bbox_list, time_interval, L2A_12_BANDS_CLEAR_WATER_MASK, maxcc
    )
    detector = SegmentationModel.load_from_checkpoint(
        CHECKPOINTS["unet++1"], map_location="cpu", trust_repo=True
    )
    predictor = ScenePredictor(device="cpu")

    for data in data_gen:
        for _d in data:
            if _d.content is not None:
                raster_ds = get_gdal_ds_from_memory(_d.content)
                clear_water_mask = raster_to_wgs84(
                    raster_ds, target_bands=[13], resample_alg=gdal.GRA_NearestNeighbour
                )

                clear_water_ds = polygonize_raster(clear_water_mask)
                clear_water_ds = filter_out_no_data_polygons(clear_water_ds)

                pred_raster = predictor.predict(
                    detector, data=io.BytesIO(_d.content), out_dir=out_dir
                )
                timestamp = datetime.datetime.strptime(
                    _d.headers["Date"], "%a, %d %b %Y %H:%M:%S %Z"
                )
                pred_raster_ds = get_gdal_ds_from_memory(pred_raster)
                pred_wgs84_raster = raster_to_wgs84(
                    pred_raster_ds, resample_alg=gdal.GRA_Cubic
                )

                pred_rounded_poly = round_to_nearest_5(pred_wgs84_raster)
                round_to_int(pred_wgs84_raster)

                bands = pred_wgs84_raster.RasterCount
                height = pred_wgs84_raster.RasterYSize
                width = pred_wgs84_raster.RasterXSize
                transform = pred_wgs84_raster.GetGeoTransform()
                bbox = (
                    transform[0],
                    transform[3],
                    transform[0] + transform[1] * width,
                    transform[3] + transform[5] * height,
                )

                wkb_geometry = from_shape(box(*bbox), srid=4326)
                band = pred_rounded_poly.GetRasterBand(1)
                dtype = gdal.GetDataTypeName(band.DataType)

                with Session(get_db_engine()) as session:
                    db_raster = PredictionRaster(
                        timestamp=timestamp,
                        dtype=dtype,
                        bbox=wkb_geometry,
                        prediction_mask=RasterElement(pred_wgs84_raster.ReadRaster()),
                        clear_water_mask=RasterElement(clear_water_mask.ReadRaster()),
                        width=width,
                        height=height,
                        bands=bands,
                    )
                    session.add(db_raster)
                    session.commit()
                    print("Successfully added prediction raster to database.")

                    ds = polygonize_raster(pred_rounded_poly)
                    db_vectors = []
                    for feature in ds.GetLayer():
                        pixel_value = int(feature.GetField("pixel_value"))
                        geom = from_shape(
                            shape(json.loads(feature.ExportToJson())["geometry"]),
                            srid=4326,
                        )
                        db_vector = PredictionVector(
                            pixel_value=pixel_value,
                            geometry=geom,
                            prediction_raster_id=db_raster.id,  # type: ignore
                        )
                        db_vectors.append(db_vector)
                    session.bulk_save_objects(db_vectors)
                    session.commit()
                    print("Successfully added prediction vector to database.")

                    clear_waters = []
                    for feature in clear_water_ds.GetLayer():
                        geom = from_shape(
                            shape(json.loads(feature.ExportToJson())["geometry"]),
                            srid=4326,
                        )
                        clear_water_db = ClearWaterVector(
                            geometry=geom,
                            prediction_raster_id=db_raster.id,  # type: ignore
                        )
                        clear_waters.append(clear_water_db)
                    session.bulk_save_objects(clear_waters)
                    session.commit()
                    print("Successfully added clear water vector to database.")


if __name__ == "__main__":
    main()
