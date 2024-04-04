# PlasticDetectionService

[![Format Lint Test](https://github.com/OceanEcoWatch/PlasticDetectionService/actions/workflows/lint_test.yml/badge.svg)](https://github.com/OceanEcoWatch/PlasticDetectionService/actions/workflows/lint_test.yml)

This repository contains a cloud-based pipeline for predicting marine debris from Sentinel-2 L2A images using machine learning. The predictions are made using the [PlasticDetectionModel](https://github.com/OceanEcoWatch/PlasticDetectionModel). The output is stored in a PostGIS database hosted on AWS RDS.
Ultimately, the predictions will be displayed on our [mapping application](https://github.com/OceanEcoWatch/website), deployed here: https://oceanecowatch.org/en


## Dependencies
- AWS account with configured access to RDS and S3 services.
- Rundpod account with a serverless endpoint.
- Access to [Sentinel-Hub Processing API](https://sentinelhub-py.readthedocs.io/en/latest/examples/process_request.html)

## Architecture Diagram

![architecture_diagram](https://github.com/OceanEcoWatch/PlasticDetectionService/blob/main/docs/geom_based_architecture.png?raw=true)

## Cloud Architecture Diagram

![cloud_architecture_diagram](https://github.com/OceanEcoWatch/PlasticDetectionService/blob/main/docs/PlasticDetectionService.png?raw=true)

