#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017.
Contact sbruton[á]tcd.ie.
"""
from torch import optim

from .experiment import Experiment
from ..datasets import LightFieldFullLFDataset
from ..models import LFVolBaseNetFullLFSmall


class LFBaseNetExperimentFullLFSmall(Experiment):
    def __init__(self, config):
        super().__init__(config=config)

        self.train_set = LightFieldFullLFDataset(
            dataset_loc=config['dataset_loc'],
            training=True
        )

        self.test_set = LightFieldFullLFDataset(
            dataset_loc=config['dataset_loc'],
            training=False
        )

        self.model = LFVolBaseNetFullLFSmall()

        self.model.apply(self.model.weights_init)

        self.optimizer = optim.Adamax(
            params=self.model.parameters()
        )
