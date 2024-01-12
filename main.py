import io
import json
import logging

import click
from geoalchemy2.elements import WKBElement
from geoalchemy2.shape import from_shape
from osgeo import gdal, ogr
from sentinelhub import CRS, BBox, UtmZoneSplitter
from shapely.geometry import box, shape
from sqlalchemy.orm import Session

from plastic_detection_service import config, sagemaker_endpoint
from plastic_detection_service.db import (
    PredictionVector,
    SceneClassificationVector,
    SentinelHubResponse,
    get_db_engine,
)
from plastic_detection_service.download_images import TimestampResponse, image_generator
from plastic_detection_service.evalscripts import L2A_12_BANDS_SCL
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

    image_gen = list(image_generator(bbox_list, time_interval, L2A_12_BANDS_SCL, maxcc))
    image_list = [i for i in image_gen if i is not None]
    LOGGER.info(f"Found {len(image_list)} images.")
    if len(image_list) == 0:
        LOGGER.info("No images found.")
        return

    for data in image_list:
        for _d in data:
            if _d.content is not None:
                process_image(_d)
            else:
                LOGGER.info("No image data found.")


def process_image(image: TimestampResponse) -> None:
    LOGGER.info(f"Processing image from {image.timestamp} at {image.bbox}.")
    raster_ds = get_gdal_ds_from_memory(image.content)
    scl_mask = raster_to_wgs84(raster_ds, target_bands=[13], resample_alg=gdal.GRA_NearestNeighbour)

    scl_ds = polygonize_raster(scl_mask)
    scl_ds = filter_out_no_data_polygons(scl_ds)

    LOGGER.info("Sending image to sagemaker endpoint...")
    pred_raster = sagemaker_endpoint.invoke(
        endpoint_name=config.ENDPOINT_NAME,
        content_type=config.CONTENT_TYPE,
        payload=image.content,
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

    width = wgs84_raster.RasterXSize
    height = wgs84_raster.RasterYSize
    transform = wgs84_raster.GetGeoTransform()
    bbox = (
        transform[0],
        transform[3],
        transform[0] + transform[1] * width,
        transform[3] + transform[5] * height,
    )

    wkb_geometry = from_shape(box(*bbox), srid=4326)

    return insert_db(image, pred_polys_ds, scl_ds, wkb_geometry)


def insert_db(
    image: TimestampResponse,
    pred_polys_ds: ogr.DataSource,
    scl_ds: ogr.DataSource,
    wkb_geometry: WKBElement,
) -> None:
    LOGGER.info("Saving in DB...")

    with Session(get_db_engine()) as session:
        db_sh_resp = SentinelHubResponse(
            sentinel_hub_id=image.sentinel_hub_id,
            timestamp=image.timestamp,
            bbox=wkb_geometry,
            image_width=image.image_size[0],
            image_height=image.image_size[1],
            max_cc=image.max_cc,
            data_collection=image.data_collection.value.api_id,
            mime_type=image.mime_type.value,
            evalscript=image.evalscript,
            request_datetime=image.request_datetime,
            processing_units_spent=image.processing_units_spent,
        )
        session.add(db_sh_resp)
        session.commit()
        LOGGER.info("Successfully added prediction raster to database.")

        prediction_vectors = []
        for feature in pred_polys_ds.GetLayer():
            db_vector = PredictionVector(
                pixel_value=int(json.loads(feature.ExportToJson())["properties"]["pixel_value"]),
                geometry=from_shape(shape(json.loads(feature.ExportToJson())["geometry"]), srid=4326),
                sentinel_hub_response_id=db_sh_resp.id,  # type: ignore
            )
            prediction_vectors.append(db_vector)
        session.bulk_save_objects(prediction_vectors)
        session.commit()
        LOGGER.info("Successfully added prediction vector to database.")

        scl_vectors = []
        for feature in scl_ds.GetLayer():
            scl_db = SceneClassificationVector(
                pixel_value=int(json.loads(feature.ExportToJson())["properties"]["pixel_value"]),
                geometry=from_shape(shape(json.loads(feature.ExportToJson())["geometry"]), srid=4326),
                sentinel_hub_response_id=db_sh_resp.id,  # type: ignore
            )
            scl_vectors.append(scl_db)
        session.bulk_save_objects(scl_vectors)
        session.commit()
        LOGGER.info("Successfully added clear water vector to database.")


if __name__ == "__main__":
    main()
