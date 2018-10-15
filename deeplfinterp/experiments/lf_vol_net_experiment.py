#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017.
Contact sbruton[á]tcd.ie.
"""
from torch import optim

from .experiment import Experiment
from ..datasets.light_field_volume_dataset import LightFieldVolumeDataset
from ..models import LFRankVolBaseNet


class LFVolNetExperiment(Experiment):
    def __init__(self, config):
        super().__init__(config=config)

        self.train_set = LightFieldVolumeDataset(
            dataset_loc=config['dataset_loc'],
            entropy=config['entropy'],
            training=True
        )

        self.test_set = LightFieldVolumeDataset(
            dataset_loc=config['dataset_loc'],
            entropy=config['entropy'],
            training=False
        )

        self.model = LFRankVolBaseNet()

        self.optimizer = optim.Adamax(
            params=self.model.parameters()
        )
