import datetime
import io
import json
import logging

import click
from geoalchemy2.shape import from_shape
from osgeo import gdal
from sentinelhub import CRS, BBox, UtmZoneSplitter
from shapely.geometry import box, shape
from sqlalchemy.orm import Session

from plastic_detection_service import config, sagemaker_endpoint
from plastic_detection_service.db import (
    ClearWaterVector,
    PredictionVector,
    SentinelHubResponse,
    get_db_engine,
)
from plastic_detection_service.download_images import image_generator
from plastic_detection_service.evalscripts import L2A_12_BANDS_CLEAR_WATER_MASK
from plastic_detection_service.gdal_ds import get_gdal_ds_from_memory
from plastic_detection_service.reproject_raster import raster_to_wgs84
from plastic_detection_service.scaling import round_to_nearest_5_int, scale_pixel_values
from plastic_detection_service.to_vector import (
    filter_out_no_data_polygons,
    polygonize_raster,
)

logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--bbox",
    nargs=4,
    type=float,
    help="Bounding box of the area to be processed. Format: min_lon min_lat max_lon max_lat",
    default=config.AOI,
)
@click.option(
    "--time-interval",
    nargs=2,
    type=str,
    help="Time interval to be processed. Format: YYYY-MM-DD YYYY-MM-DD",
    default=config.TIME_INTERVAL,
)
@click.option(
    "--maxcc",
    type=float,
    default=config.MAX_CC,
    help="Maximum cloud cover of the images to be processed.",
)
def main(
    bbox: tuple[float, float, float, float],
    time_interval: tuple[str, str],
    maxcc: float,
):
    bbox_crs = BBox(bbox, crs=CRS.WGS84)
    bbox_list = UtmZoneSplitter([bbox_crs], crs=CRS.WGS84, bbox_size=5000).get_bbox_list()

    images = list(image_generator(bbox_list, time_interval, L2A_12_BANDS_CLEAR_WATER_MASK, maxcc))
    images = [i for i in images if i is not None]
    LOGGER.info(f"Found {len(images)} images.")
    if len(images) == 0:
        LOGGER.info("No images found.")
        return

    for data in images:
        for _d in data:
            if _d.content is not None:
                date_str = _d.timestamp.rstrip("Z")
                timestamp = datetime.datetime.fromisoformat(date_str)
                timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)
                LOGGER.info(f"Processing image from {timestamp} at {_d.bbox}.")
                raster_ds = get_gdal_ds_from_memory(_d.content)
                clear_water_mask = raster_to_wgs84(raster_ds, target_bands=[13], resample_alg=gdal.GRA_NearestNeighbour)

                clear_water_ds = polygonize_raster(clear_water_mask)
                clear_water_ds = filter_out_no_data_polygons(clear_water_ds)

                LOGGER.info("Sending image to sagemaker endpoint...")
                pred_raster = sagemaker_endpoint.invoke(
                    endpoint_name=config.ENDPOINT_NAME,
                    content_type=config.CONTENT_TYPE,
                    payload=_d.content,
                )
                LOGGER.info("Received prediction raster from sagemaker endpoint.")
                scaled_pred_raster = scale_pixel_values(io.BytesIO(pred_raster))

                LOGGER.info("Postprocessing prediction raster...")
                pred_rounded = round_to_nearest_5_int(io.BytesIO(scaled_pred_raster))
                pred_raster_ds = get_gdal_ds_from_memory(pred_rounded)

                LOGGER.info("Reprojecting prediction raster...")
                wgs84_raster = raster_to_wgs84(pred_raster_ds, resample_alg=gdal.GRA_Cubic)
                LOGGER.info("Transforming prediction raster to vector...")
                pred_polys_ds = polygonize_raster(wgs84_raster)
                LOGGER.info("Filtering out no data polygons...")
                pred_polys_ds = filter_out_no_data_polygons(pred_polys_ds, threshold=30)

                wkb_geometry = from_shape(
                    box(*_d.bbox, ccw=True),
                    srid=4326,
                )
                LOGGER.info("Saving in DB...")
                with Session(get_db_engine()) as session:
                    db_raster = SentinelHubResponse(
                        timestamp=timestamp,
                        bbox=wkb_geometry,
                    )
                    session.add(db_raster)
                    session.commit()
                    LOGGER.info("Successfully added prediction raster to database.")

                    db_vectors = []
                    for feature in pred_polys_ds.GetLayer():
                        pixel_value = int(json.loads(feature.ExportToJson())["properties"]["pixel_value"])
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
                    LOGGER.info("Successfully added prediction vector to database.")

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
                    LOGGER.info("Successfully added clear water vector to database.")
            else:
                LOGGER.info("No image data found.")


if __name__ == "__main__":
    main()
