import struct
from sys import byteorder

import numpy as np

_DTYPE = {
    "?": [0, "?", 1],
    "u1": [2, "B", 1],
    "i1": [3, "b", 1],
    "B": [4, "B", 1],
    "i2": [5, "h", 2],
    "u2": [6, "H", 2],
    "i4": [7, "i", 4],
    "u4": [8, "I", 4],
    "f4": [10, "f", 4],
    "f8": [11, "d", 8],
}


def write_wkb_raster(dataset):
    """Creates a WKB raster from the given raster file with rasterio.
    :dataset: Rasterio dataset
    :returns: binary: Binary raster in WKB format

    This function was imported from
    https://github.com/nathancahill/wkb-raster/blob/master/wkb_raster.py
    and slightly adapted.
    """

    # Define format, see https://docs.python.org/3/library/struct.html
    format_string = "bHHddddddIHH"

    if byteorder == "big":
        endian = ">"
        endian_byte = 0
    elif byteorder == "little":
        endian = "<"
        endian_byte = 1

    transform = dataset.transform.to_gdal()

    version = 0
    nBands = int(dataset.count)
    scaleX = transform[1]
    scaleY = transform[5]
    ipX = transform[0]
    ipY = transform[3]
    skewX = 0
    skewY = 0
    srid = int(dataset.crs.to_string().split("EPSG:")[1])
    width = int(dataset.meta.get("width"))
    height = int(dataset.meta.get("height"))

    fmt = f"{endian}{format_string}"

    header = struct.pack(
        fmt,
        endian_byte,
        version,
        nBands,
        scaleX,
        scaleY,
        ipX,
        ipY,
        skewX,
        skewY,
        srid,
        width,
        height,
    )

    bands = []

    # Create band header data

    # not used - always False
    isOffline = False
    hasNodataValue = False

    if "nodata" in dataset.meta:
        hasNodataValue = True

    # not used - always False
    isNodataValue = False

    # unset
    reserved = False

    # # Based on the pixel type, determine the struct format, byte size and
    # # numpy dtype
    rasterio_dtype = dataset.meta.get("dtype")
    dt_short = np.dtype(rasterio_dtype).str[1:]
    pixtype, nodata_fmt, _ = _DTYPE[dt_short]

    # format binary -> :b
    binary_str = (
        f"{isOffline:b}{hasNodataValue:b}{isNodataValue:b}{reserved:b}{pixtype:b}"
    )
    # convert to int
    binary_decimal = int(binary_str, 2)

    # pack to 1 byte
    # 4 bits for ifOffline, hasNodataValue, isNodataValue, reserved
    # 4 bit for pixtype
    # -> 8 bit = 1 byte
    band_header = struct.pack("<b", binary_decimal)

    # Write the nodata value
    nodata = struct.pack(nodata_fmt, int(dataset.meta.get("nodata") or 0))

    for i in range(1, nBands + 1):
        band_array = dataset.read(i)

        # # Write the pixel values: width * height * size

        # numpy tobytes() method instead of packing with struct.pack()
        band_binary = band_array.reshape(width * height).tobytes()

        bands.append(band_header + nodata + band_binary)

    # join all bands
    allbands = bytes()
    for b in bands:
        allbands += b

    wkb = allbands + header

    return wkb
