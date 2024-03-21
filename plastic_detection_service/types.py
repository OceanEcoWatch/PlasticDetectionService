from typing import NamedTuple


class BoundingBox(NamedTuple):
    min_x: float
    min_y: float
    max_x: float
    max_y: float


class TimeRange(NamedTuple):
    """Time range in ISO 8601 format."""

    start: str
    end: str


class HeightWidth(NamedTuple):
    height: int
    width: int
