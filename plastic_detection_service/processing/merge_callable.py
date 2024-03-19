import numpy as np
from scipy.ndimage import gaussian_filter


def pre121_max(old_data, new_data, old_nodata, new_nodata, **kwargs):
    mask = np.logical_and(~old_nodata, ~new_nodata)
    old_data[mask] = np.maximum(old_data[mask], new_data[mask])
    mask = np.logical_and(old_nodata, ~new_nodata)
    old_data[mask] = new_data[mask]


def smooth_overlap_callable(
    merged_data,
    new_data,
    merged_mask,
    new_mask,
    index=None,
    roff=None,
    coff=None,
    sigma=64,
):
    overlap = merged_mask & new_mask

    if overlap.any():
        # Calculate the gradient of the overlap mask
        dx, dy = np.gradient(overlap.astype(float))
        g = np.abs(dx) + np.abs(dy)

        # Smooth the gradient to create a transition mask
        transition = gaussian_filter(g, sigma=sigma)
        transition /= transition.max()
        transition[overlap] = 1.0

        # Applying the blending logic correctly
        for band in range(merged_data.shape[0]):
            # Only blend where there's overlap
            blend_area = (transition < 1) & overlap
            merged_data[band][blend_area] = (
                transition[blend_area] * new_data[band][blend_area]
                + (1 - transition[blend_area]) * merged_data[band][blend_area]
            )

    else:
        # Simplified approach for cases without overlap
        for band in range(merged_data.shape[0]):
            # Considering new_mask to directly replace data in areas without overlap
            # Ensure this direct replacement logic matches your requirements
            replace_area = ~new_mask
            merged_data[band][replace_area] = new_data[band][replace_area]
