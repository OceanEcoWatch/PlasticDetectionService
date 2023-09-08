import datetime

import matplotlib.pyplot as plt
import numpy as np
from aenum import MultiValueEnum
from eolearn.core import (
    EOExecutor,
    EOPatch,
    EOTask,
    EOWorkflow,
    FeatureType,
    OutputTask,
    SaveTask,
    WorkflowResults,
    linearly_connect_tasks,
)
from eolearn.features import NormalizedDifferenceIndexTask
from eolearn.io import ExportToTiffTask, SentinelHubInputTask
from matplotlib.colors import BoundaryNorm, ListedColormap
from sentinelhub import BBox, DataCollection


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


class SCL(MultiValueEnum):
    """Enum class containing basic LULC types"""

    NO_DATA = "no data", 0, "#000000"
    SATURATED_DEFECTIVE = "saturated / defective", 1, "#ff0004"
    DARK_AREA_PIXELS = "dark area pixels", 2, "#868686"
    CLOUD_SHADOWS = "cloud shadows", 3, "#774c0b"
    VEGETATION = "vegetation", 4, "#10d32d"
    BARE_SOILS = "bare soils", 5, "#ffff53"
    WATER = "water", 6, "#0000ff"
    CLOUDS_LOW_PROBA = "clouds low proba.", 7, "#818181"
    CLOUDS_MEDIUM_PROBA = "clouds medium proba.", 8, "#c0c0c0"
    CLOUDS_HIGH_PROBA = "clouds high proba.", 9, "#f2f2f2"
    CIRRUS = "cirrus", 10, "#bbc5ec"
    SNOW_ICE = "snow / ice", 11, "#53fffa"

    @property
    def rgb(self):
        return [c / 255.0 for c in self.rgb_int]

    @property
    def rgb_int(self):
        hex_val = self.values[2][1:]
        return [int(hex_val[i : i + 2], 16) for i in (0, 2, 4)]


def create_patches(
    bbox_list: list[BBox],
    time_interval: tuple[str, str],
    maxcc: float,
    output_folder: str,
    resolution: int,
) -> list[WorkflowResults]:
    """Download Sentinel-2 L2A data and masks and save them as EOPatches.

    :param bbox_list: List of bounding boxes to download.
    :param time_interval: Time interval to download. Format: YYYY-MM-DD YYYY-MM-DD
    :param maxcc: Maximum cloud coverage allowed. Float number from 0.0 to 1.0
    :param output_folder: Folder where to save downloaded EOPatches
    :param resolution: Resolution of the data in meters

    :return: List of WorkflowResults containing EOPatches
    """
    np_bbox_list = np.array(bbox_list)
    patch_ids = np.arange(len(np_bbox_list))

    band_names = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
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

    export_task = ExportToTiffTask(
        (FeatureType.DATA, "L2A_data"),
        folder=output_folder,
        date_indices=[0],
    )
    save = SaveTask(output_folder)
    output = OutputTask("eopatch")

    workflow_nodes = linearly_connect_tasks(
        add_l2a_and_scl,
        ndvi,
        ndwi,
        ndbi,
        add_sh_validmask,
        add_valid_count,
        export_task,
        save,
        output,
    )
    workflow = EOWorkflow(workflow_nodes)

    input_node = workflow_nodes[0]
    save_node = workflow_nodes[-2]
    export_node = workflow_nodes[-3]
    execution_args = []
    for idx, bbox in enumerate(np_bbox_list[patch_ids]):
        execution_args.append(
            {
                input_node: {"bbox": bbox, "time_interval": time_interval},
                save_node: {"eopatch_folder": f"eopatch_{idx}"},
                export_node: {"filename": f"eopatch_{idx}.tif"},
            }
        )

    executor = EOExecutor(workflow, execution_args)

    results = executor.run(workers=4, multiprocess=False)

    failed_ids = executor.get_failed_executions()
    if failed_ids:
        raise RuntimeError(
            f"Execution failed EOPatches with IDs:\n{failed_ids}\n"
            f"For more info check report at {executor.get_report_path()}"
        )

    return results


def scl_plot(eopatch):
    scl_bounds = [-0.5 + i for i in range(len(SCL) + 1)]
    scl_cmap = ListedColormap([x.rgb for x in SCL], name="scl_cmap")
    scl_norm = BoundaryNorm(scl_bounds, scl_cmap.N)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = plt.imshow(eopatch.mask["SCL"][0].squeeze(), cmap=scl_cmap, norm=scl_norm)

    cb = fig.colorbar(im, orientation="horizontal", pad=0.01, aspect=100)
    cb.ax.tick_params(labelsize=20)
    cb.set_ticks([entry.values[1] for entry in SCL])
    cb.ax.set_xticklabels(
        [entry.values[0] for entry in SCL], rotation=45, fontsize=15, ha="right"
    )
    plt.show()


def valid_count_plot(eopatch):
    """Plot the VALID_COUNT timeless mask."""
    vmin, vmax = None, None
    data = eopatch.mask_timeless["VALID_COUNT"].squeeze()
    vmin = (
        np.min(data)
        if vmin is None
        else (np.min(data) if np.min(data) < vmin else vmin)
    )
    vmax = (
        np.max(data)
        if vmax is None
        else (np.max(data) if np.max(data) > vmax else vmax)
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        eopatch.mask_timeless["VALID_COUNT"].squeeze(),
        vmin=vmin,
        vmax=vmax,
        cmap=plt.cm.inferno,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("auto")
    cb = fig.colorbar(
        im, ax=ax, orientation="horizontal", pad=0.01, aspect=100, shrink=0.8
    )
    cb.ax.tick_params(labelsize=20)
    plt.show()


def plot_patches(eopatch: EOPatch):
    eopatch.plot((FeatureType.DATA, "L2A_data"))
    eopatch.plot((FeatureType.DATA, "NDVI"))
    eopatch.plot((FeatureType.DATA, "NDWI"))
    eopatch.plot((FeatureType.DATA, "NDBI"))
    eopatch.plot((FeatureType.MASK, "IS_DATA"))
    eopatch.plot((FeatureType.MASK, "IS_VALID"))
    scl_plot(eopatch)
    valid_count_plot(eopatch)
