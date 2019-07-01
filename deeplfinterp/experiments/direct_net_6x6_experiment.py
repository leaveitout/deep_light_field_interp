#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2019.
Contact sbruton[á]tcd.ie.
"""
from .experiment import Experiment
from ..datasets import LightField6x6Dataset
from ..models import DirectNet6x6


class DirectNet6x6Experiment(Experiment):
    def __init__(self, config):
        super().__init__(config)
        self.train_set = LightField6x6Dataset(
            dataset_loc=config['dataset_loc'],
            testing_set_idx_start=config['testing_set_idx_start'],
            testing_set_idx_stop=config['testing_set_idx_stop'],
            is_training=True,
            is_x_cached=config['is_x_cached_training'],
            is_y_cached=config['is_y_cached_training'],
            data_key=config['data_key'],
        )

        self.test_set = LightField6x6Dataset(
            dataset_loc=config['dataset_loc'],
            testing_set_idx_start=config['testing_set_idx_start'],
            testing_set_idx_stop=config['testing_set_idx_stop'],
            is_training=False,
            is_x_cached=config['is_x_cached_testing'],
            is_y_cached=config['is_y_cached_testing'],
            data_key=config['data_key'],
        )

        self.model = DirectNet6x6()
