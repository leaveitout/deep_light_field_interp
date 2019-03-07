# Synthesising Light Field Volumetric Visualisations in Real-time Using a Compressed Representation

This project can be used to synthesise a full light field for volume rendering applications, at real-time rates.
A convolutional auto-encoder architecture is used to interpolate from volume renderings of four corner viewpoints.
For more details about the techniques used please see our paper "Synthesising Light Field Volumetric Visualizations in Real-time Using a Compressed Volume Representation".

If you use or extend this work, please cite this publication.

## Getting Started

To use this code, a dataset of light field volume renders for training and testing must be created. 
To do this, our [custom fork of Inviwo](http://github.com/leaveitout/inviwo.git) must be used.
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

```
git clone --recurse-submodules https://github.com/leaveitout/inviwo.git
```

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

To use and adapt the 
A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
