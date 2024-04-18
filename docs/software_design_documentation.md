# Software Design Documentation

## Project Summary

Ocean Eco Watch is a open-source project that aims to provide a web mapping application that displays marine debris in coastal areas. The project is funded by the Prototype fund.

Our mission is to support clean-up operations, research, and development by identifying and mapping marine debris, providing critical insights to combat this pressing challenge and foster a more sustainable relationship with our oceans.

You can view the project and a demo of the web mapping application [here](https://www.oceanecowatch.org/).

## Use cases 

The target users of the project are:

- Cleanup Organizations
- Researchers
- General Public

The use cases are:

- Users can view marine debris predictions on a web mapping application

Not implemented yet:

- Users cam view available areas for prediction
- Users can filter the predictions by timestamp, cloud cover, and prediction probability
- Users can request a prediction for a specific area and timestamp

The parts of the system that I worked on are:

- [PlasticDetectionModel](https://github.com/OceanEcoWatch/PlasticDetectionModel)
- [PlasticDetecionService](https://github.com/OceanEcoWatch/PlasticDetectionService)
- PostGIS Database

## Quality Attributes:

This is a early stage project so the following quality attributes are most important to the project in order of importance:

### Extensibility

- The project should be easy to extend with new features
- The project should be easy to extend with new prediction models
- The project should be easy to extend with new data sources

### Deployability

- The project should be containerized well for portability
- The project should be easy to update
- The project should be easy to deploy

### Maintainability

- The code should be easy to understand and modify
- The code should be well documented
- The code should be well tested

## Architectural Style and Patterns

## Blueprint

Domain
models.py
abstractions for rasterop
abstractions for download
abstractions for inference

Use Cases
implementations for rasterop
implementations for download
implementations for inference
Utilities

MVC

Ports and Adapters

Clean Architecture

Note
pydeps src -show-deps
enforce contract with lint-imports

## Diagrams:

### System Context Diagram

![system_context_diagram](diagrams/system_context_diagram.png?raw=true)

### Container Diagram

![container_diagram](diagrams/container_diagram.png?raw=true)

### Component Diagram

![component_diagram](diagrams/compontent_diagram.png?raw=true)

### Code Diagram

![code_diagram_raster_operations](diagrams/code_diagram_raster_operations.png?raw=true)

## Design Patterns

To allow for easy extensibility and maintainability, a mix of the strategy and composite design patterns are used in the `raster_op` module. The `RasterOperationStrategy`, `RasterSplitStrategy`, `RasterMergeStrategy` and `RasterToVectorStrategy` classes are used to define the interface for the different raster operations. The `RasterOperationComposite` class is used to combine the different raster operations into a single operation (See Code Diagram). This allows for easy extension of the raster operations and easy modification of the raster operation pipeline.
In combination with the custom Raster and Vector dataclasses I made the implementation of the raster operations independent of the external library used, which is useful for future extensions with potentially different raster libraries. This already proved valuable in my refactoring from GDAL to Rasterio.
The code for this you can find [here]:(..src/raster_op/)

The `download` module implements the strategy pattern as well to allow for changing to another satellite data provider in the future. The `DownloadStrategy` class defines the interface for the different download strategies. The `SentinelHubDownload` class implements the interface for downloading data from SentinelHub. The code for this you can find [here]:(..src/download/)
