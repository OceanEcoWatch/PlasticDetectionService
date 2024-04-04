import io

import numpy as np
import rasterio

from src.models import Raster
from src.processing.raster_operations import (
    _create_raster,
)


def test__create_raster(raster):
    content = raster.content
    image = raster.to_numpy()
    bounds = raster.geometry.bounds

    with rasterio.open(io.BytesIO(content)) as src:
        meta = src.meta.copy()

        new_raster = _create_raster(
            content=content,
            image=image,
            bounds=bounds,
            meta=meta,
            padding_size=raster.padding_size,
        )

        assert isinstance(new_raster, Raster)
        assert isinstance(new_raster.content, bytes)
        assert np.array_equal(new_raster.to_numpy(), image)
        assert new_raster.geometry.bounds == bounds
        assert new_raster.size == (meta["width"], meta["height"])
        assert new_raster.crs == meta["crs"].to_epsg()
        assert new_raster.bands == [i for i in range(1, meta["count"] + 1)]
        assert new_raster.resolution == src.res[0]
        assert new_raster.dtype == meta["dtype"]
        assert new_raster.padding_size == raster.padding_size
