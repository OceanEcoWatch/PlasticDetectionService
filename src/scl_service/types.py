import enum


class SCL(enum.Enum):
    NO_DATA = 0
    SATURATED = 1
    SHADOWS = 2
    CLOUD_SHADOWS = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM_PROB = 8
    CLOUD_HIGH_PROB = 9
    THIN_CIRRUS = 10
    SNOW_ICE = 11

    @classmethod
    def max(cls) -> int:
        return 11

    @classmethod
    def min(cls) -> int:
        return 0
