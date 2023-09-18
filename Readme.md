# Plastic Detection Server

The purpose of the plastic detection server is to fetch sentinel-2 imagery from sentinel hub and detect marine debris.

### Useful SQL queries

- convert bbox from 'well-known-binary' into Polygon
    - `SELECT ST_AsText(bbox) FROM prediction_rasters;`
- query prediction_rasters for bounding box:
    - `SELECT * 
      FROM prediction_rasters
      WHERE ST_Within(bbox, ST_GeomFromText('POLYGON((120.53058253709094 14.384463071206468,
      120.99038315968619 14.384463071206468,
      120.99038315968619 14.812423505754381,
      120.53058253709094 14.812423505754381,
      120.53058253709094 14.384463071206468))', 4326));
      `