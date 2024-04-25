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

- Users can view available areas for prediction
- Users can filter the predictions by timestamp, cloud cover, and prediction probability
- Users can request a prediction for a specific area and timestamp

The parts of the system that I worked on are:

- [PlasticDetectionModel](https://github.com/OceanEcoWatch/PlasticDetectionModel)
- [PlasticDetecionService](https://github.com/OceanEcoWatch/PlasticDetectionService)
- [PostGIS Database](https://github.com/OceanEcoWatch/PlasticDetectionService/blob/main/src/database/models.py)

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

### Scalability

- The project should be able to handle a large number of users
- The project should be able to handle a large number of predictions

## Architectural Style and Patterns

The whole system is designed as a serverless, service-based architecture. We currently only have two services, the PlasticDetectionService and the PlasticDetectionModel. They are loosely-coupled and can easily be interchanged. The OceanEcoMapServer acts as a API gateway between the PlasticDetectionService and the frontend web mapping application. All components are containerized and deployed on AWS as serverless containers. This allows for easy scaling, updating, and deployment. The services read and write to the same database, therefore this is a service-based architecture and not a microservices architecture. Architectures that were considered as well are the modular monolith for its simplicity and the microkernel architecture for its flexibility and cost-effectiveness. Both options are not easily scalable and deployable as the serverless architecture, therefore the service-based architecture was chosen. A microservices architecture was also not a good fit at this stage due to the high costs and complexity.

## C4 Diagrams:

### System Context Diagram Ocean Eco Watch

![system_context_diagram](diagrams/system_context_diagram.png?raw=true)

### Container Diagram Ocean Eco Watch

![container_diagram](diagrams/container_diagram.png?raw=true)

### Component Diagram PlasticDetectionService

![component_diagram](diagrams/compontent_diagram.png?raw=true)

### Code Diagram raster_op

![code_diagram_raster_operations](diagrams/code_diagram_raster_operations.png?raw=true)

### Dependency graph PlasticDetectionService

![see](diagrams/dependency_graph.png?raw=true)

### Cloud Architecture

![cloud_architecture_diagram](diagrams/cloud_diagram.png?raw=true)

## Design Patterns

Machine Learning pipelines are usually a series of data processing steps that are run in a sequence. Therefore, the pipeline pattern is a good fit. Key elements of the pipeline design patterns are:

- Steps: Steps are the individual processing units in the pipeline. Each step is a specific task that is performed on the data.
- Input and Output: Each step takes input data and produces output data. The output of one step is the input to the next step.
- Order: The steps are executed in a specific order.
- Context: Context is a shared object that is passed between the steps. It contains the data that is processed by the pipeline or state information that is shared between the steps.
  [source](https://levelup.gitconnected.com/design-patterns-implementing-pipeline-design-pattern-824bd2d42bab)

The main advantage of the pipeline pattern is that it allows for easy extensibility and maintainability. New steps can be added to the pipeline without changing the existing steps. The pipeline pattern also allows for easy testing of individual steps and the pipeline as a whole.

To make the code extensible and maintainable, I've implemented a version of the pipeline pattern in the `PlasticDetectionService` with a combination of the strategy and composite pattern. The steps of the pipeline all adhere to the `RasterOperationStrategy` and `RasterToVectorStrategy` interfaces [here](..src/raster_op/abstractions.py). The input and output of each step are the `Raster` and `Vector` dataclasses . The context is saved in the `Raster` and `Vector` dataclasses as well [here](...src/models.py). The pipeline is executed in the `execute` method of the `CompositeRasterOperation` class. The code for this you can find [here](..src/raster_op/composite.py)

The strategy pattern in combination with the `Raster` and `Vector` dataclasses also enable the abstraction of external libraries used for raster operations. This allows for easy extension of the raster operations and easy modification of the raster operation pipeline. This already proved valuable in my refactoring from GDAL to Rasterio.

The `download` module implements the strategy pattern as well to allow for changing to another satellite data provider in the future. The `DownloadStrategy` class defines the interface for the different download strategies. The `SentinelHubDownload` class implements the interface for downloading data from SentinelHub. The code for this you can find [here](..src/download/)
