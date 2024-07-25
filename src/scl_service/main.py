import logging

from geoalchemy2.shape import to_shape
from sentinelhub import DataCollection, MimeType

from src import config
from src._types import BoundingBox, TimeRange
from src.database.connect import create_db_session
from src.database.insert import Insert
from src.database.models import AOI, Image, Job, Satellite, SceneClassificationVector
from src.download.evalscripts import L2A_SCL
from src.download.sh import SentinelHubDownload, SentinelHubDownloadParams
from src.raster_op.clip import RasterioClip
from src.raster_op.composite import CompositeRasterOperation
from src.raster_op.reproject import RasterioRasterReproject
from src.raster_op.utils import create_raster_from_download_response
from src.raster_op.vectorize import RasterioRasterToPolygon

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def main():
    with create_db_session() as db:
        images_without_scl_vectors = (
            db.query(Image)
            .join(Satellite, Image.satellite_id == Satellite.id)
            .outerjoin(
                SceneClassificationVector,
                Image.id == SceneClassificationVector.image_id,
            )
            .filter(Satellite.name == "SENTINEL2_L2A")
            .filter(
                SceneClassificationVector.id == None  # noqa: E711
            )  # Check where no related SCL vectors exist
            .all()
        )

        if not images_without_scl_vectors:
            LOGGER.info("No images without SCL vectors found.")
            return

        for image in images_without_scl_vectors:
            image_geom = to_shape(image.bbox)
            aoi = db.query(AOI).join(Job).filter(Job.id == image.job_id).one_or_none()

            if aoi:
                aoi_geom = to_shape(aoi.geometry)
                if not aoi_geom.intersects(image_geom):
                    raise ValueError(
                        f"Image {image.id} does not intersect with its AOI {aoi.id}"
                    )
            else:
                raise ValueError(f"AOI not found for image {image.id}")

            downloader = SentinelHubDownload(
                SentinelHubDownloadParams(
                    bbox=BoundingBox(*image_geom.bounds),
                    time_interval=TimeRange(image.timestamp, image.timestamp),
                    maxcc=1.0,
                    config=config.SH_CONFIG,
                    evalscript=L2A_SCL,
                    data_collection=DataCollection.SENTINEL2_L2A,
                    mime_type=MimeType.TIFF,
                )
            )
            download_response_list = list(downloader.download_images())
            if len(download_response_list) == 0:
                LOGGER.error(f"No images found for image {image.id}")
                continue
            for download_response in download_response_list:
                LOGGER.info(
                    f"Downloaded SCL image {download_response.image_id} for image {image.id}"
                )
                scl_raster = create_raster_from_download_response(download_response)
                comp_op = CompositeRasterOperation()
                comp_op.add(RasterioRasterReproject(target_crs=4326))
                comp_op.add(RasterioClip(aoi_geom))
                clipped_scl_raster = next(comp_op.execute([scl_raster]))
                scl_vectors = list(
                    RasterioRasterToPolygon(band=1).execute(clipped_scl_raster)
                )

                inserter = Insert(db)

                inserter.insert_scls_vectors(vectors=scl_vectors, image_id=image.id)
                LOGGER.info(
                    f"Inserted {len(scl_vectors)} SCL vectors for image {image.id}"
                )


if __name__ == "__main__":
    main()
