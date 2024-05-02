# PlasticDetectionService

[![Format Lint Test](https://github.com/OceanEcoWatch/PlasticDetectionService/actions/workflows/lint_test.yml/badge.svg)](https://github.com/OceanEcoWatch/PlasticDetectionService/actions/workflows/lint_test.yml)

This repository contains a cloud-based pipeline for predicting marine debris from Sentinel-2 L2A images using machine learning. The predictions are made using the [PlasticDetectionModel](https://github.com/OceanEcoWatch/PlasticDetectionModel). The output is stored in a PostGIS database hosted on AWS RDS.
Ultimately, the predictions will be displayed on our [mapping application](https://github.com/OceanEcoWatch/website), deployed here: https://oceanecowatch.org/en

This repository is triggered by the [OceanEcoMapServer](https://github.com/OceanEcoWatch/OceanEcoMapServer) via a Github Actions workflow endpoint.

## Dependencies

- AWS account with configured access to RDS and S3 services.
- Rundpod account with a serverless endpoint.
- Access to [Sentinel-Hub Processing API](https://sentinelhub-py.readthedocs.io/en/latest/examples/process_request.html)

## Running the service manually

The service can be run manually by triggering the workflow_dispatch event [here](https://github.com/OceanEcoWatch/PlasticDetectionService/actions/workflows/job.yml)

## Local installation

1. Clone the repository
2. Install the dependencies

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with the following variables:

```bash
DB_USER=
DB_PW=
DB_NAME=
DB_HOST=
DB_PORT=
SH_INSTANCE_ID=
SH_CLIENT_ID=
SH_CLIENT_SECRET=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=
RUNPOD_API_KEY=
RUNPOD_ENDPOINT_ID=
```

3. Greate the database and a aoi, job and model row in the database

```bash
python -m scripts.reset_db
```

```bash
python -m scripts.add_job_to_db
```
4. Run the service locally
   The job and model id will be 1 if you have run the reset_db and add_job_to_db scripts

```bash
python -m src.main --bbox <max_lat> <min_lat> <max_lon> <min_lon> --time <start_date> <end_date> --maxcc <cloudcover_float> --job-id <job_id> --model-id <model_id>
```

## Development environment and testing

```bash
pip install -r requirements-dev.txt
```
1. Run quick unit tests

```bash
pytest -m 'not integration and not slow'
```

2. Run integration tests

```bash
pytest -m 'integration'
```

3. Run slow (real inference) tests

```bash
pytest -m 'slow and not integration'
```

## Software Design Documentation

[software_design_documentation](software_design_documentation.md)
