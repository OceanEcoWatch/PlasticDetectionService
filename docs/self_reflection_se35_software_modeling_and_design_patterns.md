# Self-reflection

## Marc Leerink, SE_35 Software Modeling and Design Patterns, 23 April 2024, Semester 6

At the start of this semester my team had build a working prototype for Ocean Eco Watch. My contribution was a deployed PlasticDetectionModel on sagemaker and a PlasticDetectionService running on Github Actions. Both services were build very quickly without tests and with a annoying external library GDAL, which takes 5min to compile which isn't ideal for serverless instances.

### Initial Steps: Testing and Refactoring

I started this semester by adding tests to both services and refactoring the code to make it more maintainable and adding abstraction to slowly remove the dependency on GDAL. I added the domain specific dataclasses `Raster`, `Vector` and `DownloadResponse` which hold the context and data of the pipeline. Secondly, I added the `RasterOperationStrategy` and `RasterToVectorStrategy` interfaces to abstract the raster operations. I then refactored the `PlasticDetectionService` to use these abstractions and added a `CompositeRasterOperation` class to execute the pipeline. Similar principles were used in the download and inference components.

### Design Principles

In refactoring and expanding I applied the SOLID principles to ensure a maintainable and extensible design. The main principles I applied are:

- Single Responsibility Principle: I restructured classes to ensure each handles a single functionality.
- Open/Closed Principle: I designed our classes to be open for extension but closed for modification, which allows for extension without altering existing code.
- Dependency Inversion Principle: High-level modules were made to depend on abstractions rather than concrete implementations, easing the integration of different libraries.

### Achieving Maintainability

To achive a maintainable codebase I've thoroughly tested the PlasticDetectionModel and PlasticDetectionService. CI/CD pipelines were implemented to ensure the code is always in a deployable state. The main components depend on abstractions, not on implementation, making the codebase easy to maintain.

### Design Patterns

- Pipeline Pattern:
  The overcoupling design pattern i've used is the pipeline pattern in the `PlasticDetectionService`. This is pattern often used in ML pipelines.
  The main advantage of the pipeline pattern is that it allows for easy extensibility and maintainability. I've implemented the pipeline pattern in the `PlasticDetectionService` with a combination of the strategy and composite pattern. The steps of the pipeline all adhere to the `RasterOperationStrategy` and `RasterToVectorStrategy` interfaces. The input and output of each step are the `Raster` and `Vector` dataclasses. The context is saved in the `Raster` and `Vector` dataclasses as well. The pipeline is executed in the execute method of the `CompositeRasterOperation` class.

- Strategy Pattern:
  I've implemented the Strategy Pattern to abstract the raster operations and download operations. This allows me to easily swap out the implementation of the operations without changing the code that uses the operations. This already proved valuable in my refactoring from out the GDAL library to the Rasterio library.

- Composite Pattern:
  I've implemented a Composite Pattern to execute the pipeline of raster operations. This allows me to easily add new operations to the pipeline.

It was a challenge to fit all the raster operations in one interface. Initially I had 3 interfaces to deal with single and multiple raster inputs and outputs. I've managed to combine them into one interface by using a collection of `Raster` and `Vector` as input and output of the `RasterOperationStrategy`. This allows me to easily add new operations to the pipeline, even for operations like merge and split which require multiple rasters as input and output respectively.

### Architectural Style and Patterns

It is essential that we can extend/change our system to changing requirements. We want to be able to add more features/services when needed. For example we want to add different prediciton models. Our architectural design needs to make it possible to switch out models with ease. Because of that, we've chosen a service-oriented architecture for Ocean Eco Watch. This allows us easily scale the services and add new features/services in the future. The services are containerized and deployed on AWS/Runpod as serverless containers. All services are easy to deploy with a CI/CD pipeline.
Architectures that were considered as well are the modular monolith for its simplicity and the microkernel architecture for its flexibility and cost-effectiveness. Especially the micro-kernel is a architecture that fits our use case right now. However, this architecture is not easy to scale. A move to a microservices architecture was also not chosen due to the high costs and added complexity.

## Self Assessment

I've implemented design principles, multiple design patterns and a advanced architectural pattern for a complex/real-world project. The implementation of the principles and patterns was done based on the quality attributes most important to the stage our project is in. Therefore, I self-assess myself to be at level 2-3.
