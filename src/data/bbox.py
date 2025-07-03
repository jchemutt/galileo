from dataclasses import dataclass
from math import cos, radians, sin
from typing import List, Optional, Tuple


@dataclass
class BBox:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    name: Optional[str] = None

    def __post_init__(self):
        if self.max_lon < self.min_lon:
            raise ValueError("max_lon should be larger than min_lon")
        if self.max_lat < self.min_lat:
            raise ValueError("max_lat should be larger than min_lat")

        self.url = (
            f"http://bboxfinder.com/#{self.min_lat},{self.min_lon},{self.max_lat},{self.max_lon}"
        )

    def contains(self, lat: float, lon: float) -> bool:
        return (
            (lat >= self.min_lat)
            & (lat <= self.max_lat)
            & (lon >= self.min_lon)
            & (lon <= self.max_lon)
        )

    def contains_bbox(self, bbox: "BBox") -> bool:
        return (
            (bbox.min_lat >= self.min_lat)
            & (bbox.max_lat <= self.max_lat)
            & (bbox.min_lon >= self.min_lon)
            & (bbox.max_lon <= self.max_lon)
        )

    @property
    def three_dimensional_points(self) -> List[float]:
        r"""
        If we are passing the central latitude and longitude to
        an ML model, we want it to know the extremes are close together.
        Mapping them to 3d space allows us to do that
        """
        lat, lon = self.get_centre(in_radians=True)
        return [cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)]

    def get_centre(self, in_radians: bool = True) -> Tuple[float, float]:
        # roughly calculate the centres
        lat = self.min_lat + ((self.max_lat - self.min_lat) / 2)
        lon = self.min_lon + ((self.max_lon - self.min_lon) / 2)
        if in_radians:
            return radians(lat), radians(lon)
        else:
            return lat, lon

    def get_identifier(self, start_date, end_date) -> str:
        # Identifier is rounded to the nearest ~10m
        min_lon = round(self.min_lon, 4)
        min_lat = round(self.min_lat, 4)
        max_lon = round(self.max_lon, 4)
        max_lat = round(self.max_lat, 4)
        return (
            f"min_lat={min_lat}_min_lon={min_lon}_max_lat={max_lat}_max_lon={max_lon}_"
            f"dates={start_date}_{end_date}"
        )

    def __add__(self, other_box: "BBox") -> "BBox":
        return BBox(
            min_lat=min([self.min_lat, other_box.min_lat]),
            min_lon=min([self.min_lon, other_box.min_lon]),
            max_lon=max([self.max_lon, other_box.max_lon]),
            max_lat=max([self.max_lat, other_box.max_lat]),
            name="_".join([x for x in [self.name, other_box.name] if x is not None]),
        )
