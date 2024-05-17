from geoalchemy2.shape import from_shape
from shapely.geometry import shape

from src.database.connect import create_db_session
from src.database.models import AOI

polygon = {
    "coordinates": [
        [
            [120.77316160215673, 14.696728156230293],
            [120.77316160215673, 14.47197556914594],
            [121.0516526209322, 14.47197556914594],
            [121.0516526209322, 14.696728156230293],
            [120.77316160215673, 14.696728156230293],
        ]
    ],
    "type": "Polygon",
}
shape_poly = shape(polygon)
wkb = from_shape(shape_poly)
with create_db_session() as sess:
    aoi = sess.query(AOI).filter(AOI.id == 1).update({"geometry": wkb})

    sess.commit()
