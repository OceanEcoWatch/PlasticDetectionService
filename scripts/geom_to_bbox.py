from shapely.geometry import Polygon

geom = {
    "type": "FeatureCollection",
    "features": [
        {
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
        }
    ],
}


def geom_to_bbox(geom: dict) -> tuple:
    coords = geom["features"][0]["geometry"]["coordinates"][0]
    return Polygon(coords).bounds


print(
    geom_to_bbox(geom)
)  # (120.53947145910576, 14.438016273402596, 120.9926404870738, 14.79964156981643)
