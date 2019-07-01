#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2019.
Contact sbruton[á]tcd.ie.
"""
import argparse
import sys
import os
import json
import inspect

from deeplfinterp.experiments import *


def run_experiment(experiment_config: dict):
    class_members = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    class_members_dict = dict(class_members)
    experiment_type = class_members_dict[experiment_config['experiment_type']]
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
