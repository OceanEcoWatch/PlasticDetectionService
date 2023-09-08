import datetime
import os

import numpy as np
from eolearn.core import (
    EOExecutor,
    EOTask,
    EOWorkflow,
    FeatureType,
    OutputTask,
    SaveTask,
    WorkflowResults,
    linearly_connect_tasks,
)
from eolearn.features import NormalizedDifferenceIndexTask
from eolearn.io import SentinelHubInputTask
from sentinelhub import CRS, DataCollection, UtmZoneSplitter
from shapely.geometry import box


class SentinelHubValidDataTask(EOTask):
    """
    Combine Sen2Cor's classification map with `IS_DATA` to define a `VALID_DATA_SH` mask
    The SentinelHub's cloud mask is asumed to be found in eopatch.mask['CLM']
    """

    def __init__(self, output_feature):
        self.output_feature = output_feature

    def execute(self, eopatch):
        eopatch[self.output_feature] = eopatch.mask["IS_DATA"].astype(bool) & (
            ~eopatch.mask["CLM"].astype(bool)
        )
        return eopatch


class AddValidCountTask(EOTask):
    """
    The task counts number of valid observations in time-series and stores
    the results in the timeless mask.
    """

    def __init__(self, count_what, feature_name):
        self.what = count_what
        self.name = feature_name

    def execute(self, eopatch):
        eopatch[FeatureType.MASK_TIMELESS, self.name] = np.count_nonzero(
            eopatch.mask[self.what], axis=0
        )
        return eopatch


def create_patches(
    bbox: tuple[float, float, float, float],
    time_interval: tuple[float],
    maxcc: float,
    output_folder: str,
    resolution: int,
    bbox_size: int,
) -> list[WorkflowResults]:
    """Download Sentinel-2 L2A data and masks and save them as EOPatches.

    :param bbox: Bounding box of the area to download.
        Format: min_lon min_lat max_lon max_lat
    :param time_interval: Time interval to download. Format: YYYY-MM-DD YYYY-MM-DD
    :param maxcc: Maximum cloud coverage allowed. Float number from 0.0 to 1.0
    :param output_folder: Folder where to save downloaded EOPatches
    :param resolution: Resolution of the data in meters
    :param bbox_size: The size of generated bounding boxes in meters
    :return: List of WorkflowResults containing EOPatches
    """

    os.makedirs(output_folder, exist_ok=True)
    manilla_polygon = box(*bbox)
    bbox_splitter = UtmZoneSplitter(
        [manilla_polygon], crs=CRS.WGS84, bbox_size=bbox_size
    )

    bbox_list = np.array(bbox_splitter.get_bbox_list())

    patch_ids = np.arange(len(bbox_list))

    band_names = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B09",
        "B11",
        "B12",
    ]

    time_difference = datetime.timedelta(hours=2)

    add_l2a_and_scl = SentinelHubInputTask(
        data_collection=DataCollection.SENTINEL2_L2A,
        bands=band_names,
        bands_feature=(FeatureType.DATA, "L2A_data"),
        additional_data=[
            (FeatureType.MASK, "SCL"),
            (FeatureType.MASK, "dataMask", "IS_DATA"),
            (FeatureType.MASK, "CLM"),
            (FeatureType.DATA, "CLP"),
        ],
        maxcc=maxcc,
        resolution=resolution,
        time_difference=time_difference,
        max_threads=3,
    )
    ndvi = NormalizedDifferenceIndexTask(
        (FeatureType.DATA, "L2A_data"),
        (FeatureType.DATA, "NDVI"),
        (band_names.index("B08"), band_names.index("B04")),
    )
    ndwi = NormalizedDifferenceIndexTask(
        (FeatureType.DATA, "L2A_data"),
        (FeatureType.DATA, "NDWI"),
        (band_names.index("B03"), band_names.index("B08")),
    )
    ndbi = NormalizedDifferenceIndexTask(
        (FeatureType.DATA, "L2A_data"),
        (FeatureType.DATA, "NDBI"),
        (band_names.index("B11"), band_names.index("B08")),
    )

    add_sh_validmask = SentinelHubValidDataTask((FeatureType.MASK, "IS_VALID"))

    add_valid_count = AddValidCountTask("IS_VALID", "VALID_COUNT")

    save = SaveTask(output_folder)
    output = OutputTask("eopatch")

    workflow_nodes = linearly_connect_tasks(
        add_l2a_and_scl,
        ndvi,
        ndwi,
        ndbi,
        add_sh_validmask,
        add_valid_count,
        save,
        output,
    )
    workflow = EOWorkflow(workflow_nodes)

    input_node = workflow_nodes[0]
    save_node = workflow_nodes[-2]
    execution_args = []
    for idx, bbox in enumerate(bbox_list[patch_ids]):
        execution_args.append(
            {
                input_node: {"bbox": bbox, "time_interval": time_interval},
                save_node: {"eopatch_folder": f"eopatch_{idx}"},
            }
        )
        break

    executor = EOExecutor(workflow, execution_args)
    results = executor.run(workers=4, multiprocess=True)

    failed_ids = executor.get_failed_executions()
    if failed_ids:
        raise RuntimeError(
            f"Execution failed EOPatches with IDs:\n{failed_ids}\n"
            f"For more info check report at {executor.get_report_path()}"
        )

    return results
