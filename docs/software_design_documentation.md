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

## Diagrams:

### System Context Diagram

![system_context_diagram](diagrams/system_context_diagram.png?raw=true)

### Container Diagram

![container_diagram](diagrams/container_diagram.png?raw=true)

### Component Diagram

![component_diagram](diagrams/compontent_diagram.png?raw=true)

### Code Diagram

![code_diagram_raster_operations](diagrams/code_diagram_raster_operations.png?raw=true)
