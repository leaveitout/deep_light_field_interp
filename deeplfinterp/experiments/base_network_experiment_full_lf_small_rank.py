#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017.
Contact sbruton[á]tcd.ie.
"""
from torch import optim

from .experiment import Experiment
from ..datasets import LightFieldFullLFVolumeDataset
from ..models import LFVolBaseNetFullLFSmallRank


class LFBaseNetExperimentFullLFSmallRank(Experiment):
    def __init__(self, config):
        super().__init__(config=config)

        self.train_set = LightFieldFullLFVolumeDataset(
            dataset_loc=config['dataset_loc'],
            rank_reps_loc=config['rank_reps_loc'],
            training=True
        )

        self.test_set = LightFieldFullLFVolumeDataset(
            dataset_loc=config['dataset_loc'],
            rank_reps_loc=config['rank_reps_loc'],
            training=False
        )

        self.model = LFVolBaseNetFullLFSmallRank()

        self.optimizer = optim.Adamax(
            params=self.model.parameters()
        )
