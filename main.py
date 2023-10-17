import datetime
import io
import json

import click
from geoalchemy2.shape import from_shape
from geoalchemy2.types import RasterElement
from osgeo import gdal
from sentinelhub import CRS, BBox, UtmZoneSplitter
from shapely.geometry import box, shape
from sqlalchemy.orm import Session

from marinedebrisdetector.checkpoints import CHECKPOINTS
from marinedebrisdetector.model.segmentation_model import SegmentationModel
from marinedebrisdetector.predictor import ScenePredictor
from plastic_detection_service.constants import AOI
from plastic_detection_service.db import (
    ClearWaterVector,
    PredictionRaster,
    PredictionVector,
    get_db_engine,
)
from plastic_detection_service.download_images import image_generator
from plastic_detection_service.dt_util import get_past_date, get_today_str
from plastic_detection_service.evalscripts import L2A_12_BANDS_CLEAR_WATER_MASK
from plastic_detection_service.gdal_ds import get_gdal_ds_from_memory
from plastic_detection_service.reproject_raster import raster_to_wgs84
from plastic_detection_service.scaling import round_to_nearest_5_int, scale_pixel_values
from plastic_detection_service.ssh_util import create_unverified_https_context
from plastic_detection_service.to_vector import (
    filter_out_no_data_polygons,
    polygonize_raster,
)


@click.command()
@click.option(
    "--bbox",
    nargs=4,
    type=float,
    help="Bounding box of the area to be processed. Format: min_lon min_lat max_lon max_lat",
    default=AOI,
)
@click.option(
    "--time-interval",
    nargs=2,
    type=str,
    help="Time interval to be processed. Format: YYYY-MM-DD YYYY-MM-DD",
    default=(get_past_date(7), get_today_str()),
)
@click.option(
    "--maxcc",
    type=float,
    default=0.5,
    help="Maximum cloud cover of the images to be processed.",
)
@click.option(
    "--processing-unit",
    type=enumerate(["cpu", "gpu"]),
    default="gpu",
    help="Processing unit to be used. gpu or cpu.",
)
@click.option(
    "--model_checkpoint",
    type=enumerate(CHECKPOINTS.keys()),
    default="unet++1",
    help=f"Model checkpoint to be used. Choose from {CHECKPOINTS.keys()}",
)
def main(
    bbox: tuple[float, float, float, float],
    time_interval: tuple[str, str],
    maxcc: float,
    processing_unit: str,
    model_checkpoint: str,
):
    create_unverified_https_context()
    detector = SegmentationModel.load_from_checkpoint(
        CHECKPOINTS[model_checkpoint], map_location=processing_unit, trust_repo=True
    )
    predictor = ScenePredictor(device=processing_unit)

    bbox_crs = BBox(bbox, crs=CRS.WGS84)
    bbox_list = UtmZoneSplitter(
        [bbox_crs], crs=CRS.WGS84, bbox_size=5000
    ).get_bbox_list()

    for data in image_generator(
        bbox_list, time_interval, L2A_12_BANDS_CLEAR_WATER_MASK, maxcc
    ):
        for _d in data:
            if _d.content is not None:
                timestamp = datetime.datetime.strptime(
                    _d.headers["Date"], "%a, %d %b %Y %H:%M:%S %Z"
                )
                print(f"Processing image from {timestamp}...")
                raster_ds = get_gdal_ds_from_memory(_d.content)
                clear_water_mask = raster_to_wgs84(
                    raster_ds, target_bands=[13], resample_alg=gdal.GRA_NearestNeighbour
                )

                clear_water_ds = polygonize_raster(clear_water_mask)
                clear_water_ds = filter_out_no_data_polygons(clear_water_ds)

                pred_raster = predictor.predict(detector, data=io.BytesIO(_d.content))
                scaled_pred_raster = scale_pixel_values(io.BytesIO(pred_raster))

                pred_rounded = round_to_nearest_5_int(io.BytesIO(scaled_pred_raster))
                pred_raster_ds = get_gdal_ds_from_memory(pred_rounded)

                wgs84_raster = raster_to_wgs84(
                    pred_raster_ds, resample_alg=gdal.GRA_Cubic
                )
                pred_polys_ds = polygonize_raster(wgs84_raster)
                pred_polys_ds = filter_out_no_data_polygons(pred_polys_ds, threshold=30)

                bands = wgs84_raster.RasterCount
                height = wgs84_raster.RasterYSize
                width = wgs84_raster.RasterXSize
                transform = wgs84_raster.GetGeoTransform()
                bbox = (
                    transform[0],
                    transform[3],
                    transform[0] + transform[1] * width,
                    transform[3] + transform[5] * height,
                )

                wkb_geometry = from_shape(box(*bbox), srid=4326)
                band = wgs84_raster.GetRasterBand(1)
                dtype = gdal.GetDataTypeName(band.DataType)

                with Session(get_db_engine()) as session:
                    db_raster = PredictionRaster(
                        timestamp=timestamp,
                        dtype=dtype,
                        bbox=wkb_geometry,
                        prediction_mask=RasterElement(wgs84_raster.ReadRaster()),
                        clear_water_mask=RasterElement(clear_water_mask.ReadRaster()),
                        width=width,
                        height=height,
                        bands=bands,
                    )
                    session.add(db_raster)
                    session.commit()
                    print("Successfully added prediction raster to database.")

                    db_vectors = []
                    print(pred_polys_ds.GetLayer().GetFeatureCount())
                    for feature in pred_polys_ds.GetLayer():
                        pixel_value = int(
                            json.loads(feature.ExportToJson())["properties"][
                                "pixel_value"
                            ]
                        )
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
