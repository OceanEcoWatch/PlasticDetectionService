import geopandas as gpd

from geo_helper.area import calculate_area

coastline = gpd.read_file(
    "assets/ne_10m_coastline/ne_10m_coastline.shp", crs="EPSG:4326"
)
ocean = gpd.read_file("assets/ne_10m_ocean/ne_10m_ocean.shp", crs="EPSG:4326")

# reproject to metric
coastline = coastline.to_crs("EPSG:3395")
ocean = ocean.to_crs("EPSG:3395")

# Step 2: Buffer the coastline by 50 km
# Assuming the CRS is in meters
buffered_coast = coastline.geometry.buffer(100000)
buffered_coast = gpd.GeoDataFrame(geometry=buffered_coast, crs="EPSG:3395")

# Step 3: Clip water bodies within 50 km of the coastline
coastal_water = gpd.overlay(ocean, buffered_coast, how="intersection")

# get union of all polygons
coastal_water = coastal_water.geometry.unary_union

# convert to geodataframe
coastal_water = gpd.GeoDataFrame(geometry=[coastal_water], crs="EPSG:3395")

# reproject to WGS84
coastal_water = coastal_water.to_crs("EPSG:4326")

# calculate area in km^2
coastal_water["area_sqkm"] = coastal_water.geometry.apply(lambda x: calculate_area(x))

# Step 4: Save to GeoJSON
coastal_water.to_file("coastal_water.geojson", driver="GeoJSON")
