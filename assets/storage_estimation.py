import geopandas as gpd

from geo_helper.reproject import reproject_geometry

coastal_water = gpd.read_file("assets/coastal_water.geojson", crs="EPSG:3395")

# reproject to WGS84
coastal_water = coastal_water.to_crs("EPSG:4326")

coastal_water.to_file("assets/coastal_water_wgs84.geojson", driver="GeoJSON")

coastal_water["geometry"] = coastal_water.geometry.apply(
    lambda x: reproject_geometry(x)
)
# measure area
coastal_water["area"] = coastal_water.geometry.area
print(coastal_water["area"].sum() / 1e6, "km^2")


# reproject to WGS84
coastal_water = coastal_water.to_crs("EPSG:4326")

coastal_water.to_file("assets/coastal_water_with_area.geojson", driver="GeoJSON")
