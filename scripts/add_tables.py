import datetime

from geoalchemy2.shape import from_shape
from shapely.geometry import shape

from src.database.connect import create_db_session
from src.database.models import (
    AOI,
    Band,
    ClassificationClass,
    Job,
    JobStatus,
    Model,
    ModelBand,
    ModelType,
    Satellite,
)

session = create_db_session()
sat = Satellite(
    name="SENTINEL2_L2A",
)
session.add(sat)
session.commit()

band1 = Band(
    satellite_id=sat.id,
    index=1,
    name="B01",
    description="Coastal aerosol",
    resolution=60,
    wavelength="443nm",
)
band2 = Band(
    satellite_id=sat.id,
    index=2,
    name="B02",
    description="Blue",
    resolution=10,
    wavelength="492nm",
)
band3 = Band(
    satellite_id=sat.id,
    index=3,
    name="B03",
    description="Green",
    resolution=10,
    wavelength="560nm",
)
band4 = Band(
    satellite_id=sat.id,
    index=4,
    name="B04",
    description="Red",
    resolution=10,
    wavelength="665nm",
)
band5 = Band(
    satellite_id=sat.id,
    index=5,
    name="B05",
    description="Vegetation red edge 1",
    resolution=20,
    wavelength="704nm",
)
band6 = Band(
    satellite_id=sat.id,
    index=6,
    name="B06",
    description="Vegetation red edge 2",
    resolution=20,
    wavelength="740nm",
)
band7 = Band(
    satellite_id=sat.id,
    index=7,
    name="B07",
    description="Vegetation red edge 3",
    resolution=20,
    wavelength="783nm",
)
band8 = Band(
    satellite_id=sat.id,
    index=8,
    name="B08",
    description="NIR",
    resolution=10,
    wavelength="832nm",
)
band8a = Band(
    satellite_id=sat.id,
    index=9,
    name="B8A",
    description="Narrow NIR",
    resolution=20,
    wavelength="864nm",
)
band9 = Band(
    satellite_id=sat.id,
    index=10,
    name="B09",
    description="Water vapour",
    resolution=60,
    wavelength="945nm",
)

band11 = Band(
    satellite_id=sat.id,
    index=12,
    name="B11",
    description="SWIR 1",
    resolution=20,
    wavelength="1613nm",
)
band12 = Band(
    satellite_id=sat.id,
    index=13,
    name="B12",
    description="SWIR 2",
    resolution=20,
    wavelength="2202nm",
)

session.add(band1)
session.add(band2)
session.add(band3)
session.add(band4)
session.add(band5)
session.add(band6)
session.add(band7)
session.add(band8)
session.add(band8a)
session.add(band9)
session.add(band11)
session.add(band12)
session.commit()

model1 = Model(
    model_id="oceanecowatch/plasticdetectionmodel:1.0.1",
    model_url="2qzxx0ljdkepts",
    created_at=datetime.datetime.now(),
    version=1,
    expected_image_height=480,
    expected_image_width=480,
    type=ModelType.SEGMENTATION,
    output_dtype="float32",
)

model2 = Model(
    model_id="oceanecowatch/marinext:2",
    model_url="64hvcppe4m24z8",
    created_at=datetime.datetime.now(),
    version=1,
    expected_image_height=240,
    expected_image_width=240,
    type=ModelType.CLASSIFICATION,
    output_dtype="int64",
)
session.add(model1)
session.add(model2)
session.commit()
model1_bands = [
    ModelBand(model_id=model1.id, band_id=band1.id),
    ModelBand(model_id=model1.id, band_id=band2.id),
    ModelBand(model_id=model1.id, band_id=band3.id),
    ModelBand(model_id=model1.id, band_id=band4.id),
    ModelBand(model_id=model1.id, band_id=band5.id),
    ModelBand(model_id=model1.id, band_id=band6.id),
    ModelBand(model_id=model1.id, band_id=band7.id),
    ModelBand(model_id=model1.id, band_id=band8.id),
    ModelBand(model_id=model1.id, band_id=band8a.id),
    ModelBand(model_id=model1.id, band_id=band9.id),
    ModelBand(model_id=model1.id, band_id=band11.id),
    ModelBand(model_id=model1.id, band_id=band12.id),
]
model2_bands = [
    ModelBand(model_id=model2.id, band_id=band1.id),
    ModelBand(model_id=model2.id, band_id=band2.id),
    ModelBand(model_id=model2.id, band_id=band3.id),
    ModelBand(model_id=model2.id, band_id=band4.id),
    ModelBand(model_id=model2.id, band_id=band5.id),
    ModelBand(model_id=model2.id, band_id=band6.id),
    ModelBand(model_id=model2.id, band_id=band7.id),
    ModelBand(model_id=model2.id, band_id=band8.id),
    ModelBand(model_id=model2.id, band_id=band8a.id),
    ModelBand(model_id=model2.id, band_id=band11.id),
    ModelBand(model_id=model2.id, band_id=band12.id),
]

session.add_all(model1_bands)
session.add_all(model2_bands)
session.commit()


model_1_classes = [
    ClassificationClass(model_id=model1.id, name="Marine Debris", index=0),
]


model_2_classes = [
    ClassificationClass(model_id=model2.id, name="Non-annotated", index=0),
    ClassificationClass(model_id=model2.id, name="Marine Debris", index=1),
    ClassificationClass(model_id=model2.id, name="Dense Sargassum", index=2),
    ClassificationClass(model_id=model2.id, name="Sparse Floating Algae", index=3),
    ClassificationClass(model_id=model2.id, name="Natural Organic Material", index=4),
    ClassificationClass(model_id=model2.id, name="Ship", index=5),
    ClassificationClass(model_id=model2.id, name="Oil Spill", index=6),
    ClassificationClass(model_id=model2.id, name="Marine Water", index=7),
    ClassificationClass(model_id=model2.id, name="Sediment-Laden Water", index=8),
    ClassificationClass(model_id=model2.id, name="Foam", index=9),
    ClassificationClass(model_id=model2.id, name="Turbid Water", index=10),
    ClassificationClass(model_id=model2.id, name="Shallow Water", index=11),
    ClassificationClass(model_id=model2.id, name="Waves & Wakes", index=12),
    ClassificationClass(model_id=model2.id, name="Oil Platform", index=13),
    ClassificationClass(model_id=model2.id, name="Jellyfish", index=14),
    ClassificationClass(model_id=model2.id, name="Sea snot", index=15),
]
session.add_all(model_1_classes)
session.add_all(model_2_classes)
session.commit()

aoi = {
    "name": "manilla bay",
    "geometry": {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "coordinates": [
                [
                    [120.8617410392963, 14.622900808887422],
                    [120.8617410392963, 14.56223679040346],
                    [120.94056794296893, 14.56223679040346],
                    [120.94056794296893, 14.622900808887422],
                    [120.8617410392963, 14.622900808887422],
                ]
            ],
            "type": "Polygon",
        },
    },
}


aoi = AOI(
    name=aoi["name"],
    geometry=from_shape(shape(aoi["geometry"]["geometry"]), srid=4326),
)
session.add(aoi)
session.commit()


job = Job(
    aoi_id=aoi.id,
    status=JobStatus.PENDING,
    start_date=datetime.datetime.now() - datetime.timedelta(days=100),
    end_date=datetime.datetime.now(),
    maxcc=0.1,
    model_id=model2.id,
)
session.add(job)
session.commit()
