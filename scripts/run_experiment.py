#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017. 
Contact sbruton[á]tcd.ie.
"""
import argparse
import sys
import os
import json

from deeplfinterp.experiments import LFBaseNetExperiment
from deeplfinterp.experiments import LFVolNetExperiment
from deeplfinterp.experiments import LFBaseNetExperimentFullLF
from deeplfinterp.experiments import LFBaseNetExperimentFullLFSmall

experiments = {
    'lf_base_net': LFBaseNetExperiment,
    'lf_base_net_full_lf': LFBaseNetExperimentFullLF,
    'lf_base_net_full_lf_small': LFBaseNetExperimentFullLFSmall,
    'lf_vol_net': LFVolNetExperiment
}


def run_experiment(experiment_config: dict):
    experiment_type = experiments[experiment_config['experiment_type']]
    print("Experiment Type: {}".format(experiment_type))

    experiment = experiment_type(experiment_config)

    experiment.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run an experiment."
    )

    parser.add_argument("config",
                        help="The json config file for the experiment.",
                        type=str)
    args = parser.parse_args()

    if os.path.isfile(args.config):
        with open(args.config) as json_config:
            exp_config = json.load(json_config)
            run_experiment(exp_config)
    else:
        print("Config file does not exist at location, exiting...")
        sys.exit(1)

