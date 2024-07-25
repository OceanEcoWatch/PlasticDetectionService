from src.raster_op.clip import RasterioClip


def test_clip_raster(raster):
    clip_geometry = raster.geometry.buffer(-1000)
    org_size = raster.size[0] * raster.size[1]
    clip = RasterioClip(geometry=clip_geometry, crop=True)

    clipped = list(clip.execute([raster]))
    assert len(clipped) == 1
    clipped = clipped[0]
    clipped.to_file("tests/assets/test_out_clip.tif")
    assert clipped.size[0] * clipped.size[1] < org_size
    assert clipped.crs == raster.crs
    assert clipped.resolution == raster.resolution
    assert clipped.bands == raster.bands
    assert clipped.dtype == raster.dtype
    assert clipped.padding_size == raster.padding_size
