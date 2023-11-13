import json
from enum import Enum, auto
from typing import Optional

import jsonschema
from jsonschema import ValidationError


class NotSupportedGeometryType(Exception):
    """Exception for not supported geometry type"""


class GeometryType(Enum):
    POINT = auto()
    LINESTRING = auto()
    LINEARRING = auto()
    POLYGON = auto()
    MULTIPOINT = auto()
    MULTILINESTRING = auto()
    MULTIPOLYGON = auto()
    GEOMETRYCOLLECTION = auto()

    @classmethod
    def has_name(cls, name: str) -> bool:
        return name.upper() in [g.name.upper() for g in cls]

    @classmethod
    def check_name(cls, name: str) -> None:
        if not cls.has_name(name):
            raise ValueError(f"Not supported geometry type: {name}")


def _read_geojson(geojson_path: str) -> str:
    try:
        with open(geojson_path, encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Geojson file not found: {geojson_path}") from err


def _load_geojson(geojson_str: str) -> dict:
    return json.loads(geojson_str)


SCHEMA = _load_geojson(_read_geojson("assets/geojson_schema.json"))


def get_geojson(
    geojson_path: str,
    schema: dict = SCHEMA,
    not_allowed_geometry_types: Optional[list[str]] = None,
) -> dict:
    """Reads geojson file and returns a geojson dictionary.

    :param geojson_path: Path to geojson file
    :param schema: Geojson schema.
        Default is schema from 'https://geojson.org/schema/FeatureCollection.json'
    :param not_allowed_geometry_types: List of not allowed geometry types.
        Defaults to None, which means all geometry types are allowed.

    :return: Geojson dictionary

    :raises FileNotFoundError: If geojson file not found.
    :raises JSONDecodeError: If geojson file is not valid json.
    :raises jsonschema.exceptions.ValidationError: If geojson is not valid.
    :raises jsonschema.exceptions.SchemaError: If schema is not valid.

    """
    geojson = _load_geojson(_read_geojson(geojson_path))
    validate_geojson(geojson, schema, not_allowed_geometry_types)

    return geojson


def validate_geojson(
    geojson: dict,
    schema: dict,
    not_allowed_geometry_types: Optional[list[str]] = None,
) -> None:
    jsonschema.validate(geojson, schema)

    if not geojson["features"]:
        raise ValidationError("No features in geojson")

    if not_allowed_geometry_types:
        if not all([GeometryType.has_name(g) for g in not_allowed_geometry_types]):
            raise NotSupportedGeometryType(
                "Not supported geometry type in not_allowed_geometry_types"
            )

        upper_na_geom_types = [g.upper() for g in not_allowed_geometry_types]

        for feature in geojson["features"]:
            geom_type = str(feature["geometry"]["type"]).upper()
            if geom_type in upper_na_geom_types:
                raise ValueError(
                    f"Geometry type {feature['geometry']['type']} not allowed"
                )


def list_geojson_geometries(geojson_path: str) -> list[dict]:
    """returns list of geometries from geojson file"""
    val_geojson = get_geojson(geojson_path)
    return [feature["geometry"] for feature in val_geojson["features"]]


def write_geojson(geojson: dict, path: str) -> None:
    """Writes geojson dictionary to file. Validates geojson before writing.

    :param geojson: Geojson dictionary
    :param path: Path to file

    """
    validate_geojson(geojson, SCHEMA)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(geojson, file, indent=4)
