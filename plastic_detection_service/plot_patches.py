import matplotlib.pyplot as plt
import numpy as np
from aenum import MultiValueEnum
from eolearn.core import EOPatch, FeatureType
from matplotlib.colors import BoundaryNorm, ListedColormap


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


def plot_patches(eopatch):
    eopatch.plot((FeatureType.DATA, "L2A_data"))
    eopatch.plot((FeatureType.DATA, "NDVI"))
    eopatch.plot((FeatureType.DATA, "NDWI"))
    eopatch.plot((FeatureType.DATA, "NDBI"))
    eopatch.plot((FeatureType.MASK, "IS_DATA"))
    eopatch.plot((FeatureType.MASK, "IS_VALID"))
    scl_plot(eopatch)
    valid_count_plot(eopatch)


if __name__ == "__main__":
    eopatch = EOPatch.load("eopatches/eopatch_0")
    print(eopatch)
    plot_patches(eopatch)
