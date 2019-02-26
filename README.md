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

## Prerequisites

Python 3, Pytorch, numpy, matplotlib.


## Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
python3 setup.py --install [--user] 
```

## Running the scripts

Create a valid .json file in line with the scripts and run scripts/run_experiment.py from the same environment in which the package is installed.

