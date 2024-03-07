import io
import json
import logging

from geoalchemy2.shape import from_shape
from osgeo import gdal, ogr
from shapely.geometry import box, shape
from sqlalchemy.orm import Session

from plastic_detection_service import config, sagemaker_endpoint
from plastic_detection_service.db import (
    PredictionVector,
    SceneClassificationVector,
    SentinelHubResponse,
    get_db_engine,
)
from plastic_detection_service.download.models import TimestampResponse
from plastic_detection_service.gdal_ds import get_gdal_ds_from_memory
from plastic_detection_service.reproject_raster import raster_to_wgs84
from plastic_detection_service.scaling import round_to_nearest_5_int, scale_pixel_values
from plastic_detection_service.to_vector import (
    filter_out_no_data_polygons,
    polygonize_raster,
)

logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger(__name__)


def process_image(image: TimestampResponse) -> None:
    LOGGER.info(
        f"Processing image from {image.timestamp} at {image.bbox}, id: {image.image_id}"
    )
    raster_ds = get_gdal_ds_from_memory(image.content)
    scl_mask = raster_to_wgs84(
        raster_ds, target_bands=[13], resample_alg=gdal.GRA_NearestNeighbour
    )

    scl_ds = polygonize_raster(scl_mask)
    scl_ds = filter_out_no_data_polygons(scl_ds)

    LOGGER.info("Sending image to sagemaker endpoint...")
    pred_raster = sagemaker_endpoint.invoke(
        endpoint_name=config.ENDPOINT_NAME,
        content_type=config.CONTENT_TYPE,
        payload=image.content,
    )
    LOGGER.info("Received prediction raster from sagemaker endpoint.")
    LOGGER.info("Postprocessing prediction raster...")
    scaled_pred_raster = scale_pixel_values(io.BytesIO(pred_raster))
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
    proj_bbox = (
        transform[0],
        transform[3],
        transform[0] + transform[1] * width,
        transform[3] + transform[5] * height,
    )

    return insert_db(image, pred_polys_ds, scl_ds, proj_bbox)


def insert_db(
    image: TimestampResponse,
    pred_polys_ds: ogr.DataSource,
    scl_ds: ogr.DataSource,
    proj_bbox: tuple[float, float, float, float],
) -> None:
    LOGGER.info("Storing_raw_image_in_s3...")

    LOGGER.info("Saving in DB...")
    with Session(get_db_engine()) as session:
        db_sh_resp = SentinelHubResponse(
            sentinel_hub_id=image.image_id,
            timestamp=image.timestamp,
            bbox=from_shape(box(*proj_bbox), srid=4326),
            image_width=image.image_size[0],
            image_height=image.image_size[1],
            max_cc=image.max_cc,
            data_collection=image.data_collection,
            request_datetime=image.request_datetime,
            processing_units_spent=image.processing_units_spent,
            image_url=s3_url,
        )
        session.add(db_sh_resp)
        session.commit()
        LOGGER.info("Successfully added prediction raster to database.")

        prediction_vectors = []
        for feature in pred_polys_ds.GetLayer():
            db_vector = PredictionVector(
                pixel_value=int(
                    json.loads(feature.ExportToJson())["properties"]["pixel_value"]
                ),
                geometry=from_shape(
                    shape(json.loads(feature.ExportToJson())["geometry"]), srid=4326
                ),
                sentinel_hub_response_id=db_sh_resp.id,  # type: ignore
            )
            prediction_vectors.append(db_vector)
        session.bulk_save_objects(prediction_vectors)
        session.commit()
        LOGGER.info("Successfully added prediction vector to database.")

        scl_vectors = []
        for feature in scl_ds.GetLayer():
            scl_db = SceneClassificationVector(
                pixel_value=int(
                    json.loads(feature.ExportToJson())["properties"]["pixel_value"]
                ),
                geometry=from_shape(
                    shape(json.loads(feature.ExportToJson())["geometry"]), srid=4326
                ),
                sentinel_hub_response_id=db_sh_resp.id,  # type: ignore
            )
            scl_vectors.append(scl_db)
        session.bulk_save_objects(scl_vectors)
        session.commit()
        LOGGER.info("Successfully added clear water vector to database.")


if __name__ == "__main__":
    main()
