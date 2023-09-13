L2A_12_BANDS = """
    //VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B09", "B11", "B12"],
      units: "DN"
    }],
    output: {
      id: "default",
      bands: 12,
      sampleType: SampleType.UINT16
    }
  }
}

function evaluatePixel(sample) {
    return [sample.B01, sample.B02, sample.B03, sample.B04, sample.B05, sample.B06,
            sample.B07, sample.B08, sample.B8A, sample.B09, sample.B11, sample.B12]
}
"""

L2A_12_BANDS_CLOUD_MASK = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B09", "B11", "B12"],
      units: "DN"
    }],
    output: {
      id: "default",
      bands: 13,  // 12 original bands + 1 for cloud mask
      sampleType: SampleType.UINT16
    }
  }
}

function evaluatePixel(sample) {
  // Your cloud mask logic here. 0 = no cloud, 1 = cloud.
  // For demonstration, we'll set it to 0 (no cloud).
  var cloudMask = 0;

  return [
    sample.B01, sample.B02, sample.B03, sample.B04,
    sample.B05, sample.B06, sample.B07, sample.B08,
    sample.B8A, sample.B09, sample.B11, sample.B12,
    cloudMask  // Adding cloud mask as the 13th band
  ];
}
"""

NDVI_NDWI = """
    //VERSION=3

    function setup() {
        return {
            input: ["B03","B04","B08","dataMask"],
            output:[{
                id: "indices",
                bands: 2,
                sampleType: SampleType.FLOAT32
            }]
        }
    }

    function evaluatePixel(sample) {
        let ndvi = index(sample.B08, sample.B04);
        let ndwi = index(sample.B03, sample.B08);
        return {
           indices: [ndvi, ndwi]
        };
    }
"""
