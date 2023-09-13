import datetime
import io
import json
import ssl

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
    PredictionRaster,
    PredictionVector,
    get_db_engine,
)
from plastic_detection_service.download_images import stream_in_images
from plastic_detection_service.evalscripts import L2A_12_BANDS_CLEAR_WATER_MASK
from plastic_detection_service.reproject_raster import raster_to_wgs84
from plastic_detection_service.to_vector import polygonize_raster


def image_generator(bbox_list, time_interval, evalscript, maxcc):
    for bbox in bbox_list:
        data = stream_in_images(
            config, bbox, time_interval, evalscript=evalscript, maxcc=maxcc
        )

        if data is not None:
            yield data


def main():
    bbox = BBox(MANILLA_BAY_BBOX, crs=CRS.WGS84)
    time_interval = ("2023-08-01", "2023-09-01")
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
                pred_raster = predictor.predict(
                    detector, data=io.BytesIO(_d.content), out_dir=out_dir
                )
                timestamp = datetime.datetime.strptime(
                    _d.headers["Date"], "%a, %d %b %Y %H:%M:%S %Z"
                )
                wgs84_raster = raster_to_wgs84(pred_raster, gdal.GRA_Cubic)
                unclear_pixels = raster_to_wgs84(_d.content, gdal.GRA_NearestNeighbour)

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
                        width=width,
                        height=height,
                        bands=bands,
                    )
                    session.add(db_raster)
                    session.commit()
                    print("Successfully added prediction raster to database.")

                    ds = polygonize_raster(wgs84_raster)
                    polygonize_raster(unclear_pixels)
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
                        session.add(db_vector)
                        session.commit()
                        print("Successfully added prediction vector to database.")

                print("Successfully added all prediction vector polygons to database.")


if __name__ == "__main__":
    main()
