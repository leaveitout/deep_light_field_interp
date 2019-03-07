# Synthesising Light Field Volumetric Visualisations in Real-time Using a Compressed Representation

This project can be used to synthesise a full light field for volume rendering applications, at real-time rates.
A convolutional auto-encoder architecture is used to interpolate from volume renderings of four corner viewpoints.
For more details about the techniques used please see our paper "Synthesising Light Field Volumetric Visualizations in Real-time Using a Compressed Volume Representation".

If you use or extend this work, please cite this publication.

```
@inproceedings{bruton_synthesising_2019,
	author = {Bruton, Se\'{a}n and Ganter, David and Manzke, Michael},
	title = {Synthesising Light Field Volumetric Visualizations in Real-time
        Using A Compressed Volume Representation},
	month = feb,
	year = {2019},
        booktitle = {{IVAPP} 2019 - Proceedings of the International Conference on Information Visualization Theory and Applications, Volume 1, Prague, Czech Republic, 25-27 February, 2019.},
}
```


## Getting Started

To use this code, a dataset of light field volume renders for training and testing must be created. 
To do this, our [custom fork of Inviwo](http://github.com/leaveitout/inviwo.git) must be used.
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

```
git clone --recurse-submodules https://github.com/leaveitout/inviwo.git
```


### Prerequisites

What things you need to install the software and how to install them

Python 3, Pytorch, numpy, matplotlib.


## Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
python3 setup.py --install [--user] 
```

## Running the scripts

Create a valid .json file in line with the scripts and run scripts/run_experiment.py from the same environment in which the package is installed.

