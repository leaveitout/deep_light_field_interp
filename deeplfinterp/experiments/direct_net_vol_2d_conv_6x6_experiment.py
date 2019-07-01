#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2019.
Contact sbruton[á]tcd.ie.
"""
from .experiment import Experiment
from ..datasets import LightField6x6Dataset
from ..models import DirectNetVolImages6x6


class DirectNetLegacy6x6Experiment(Experiment):
    def __init__(self, config):
        super().__init__(config)
        self.train_set = LightField6x6Dataset(
            dataset_loc=config['dataset_loc'],
            testing_set_idx_start=config['testing_set_idx_start'],
            testing_set_idx_stop=config['testing_set_idx_stop'],
            downscale_x=config['downscale_x'],
            downscale_y=config['downscale_y'],
            is_training=True,
            is_x_cached=config['is_x_cached_training'],
            is_y_cached=config['is_y_cached_training'],
            data_key=config['data_key'],
        )

        self.test_set = LightField6x6Dataset(
            dataset_loc=config['dataset_loc'],
            testing_set_idx_start=config['testing_set_idx_start'],
            testing_set_idx_stop=config['testing_set_idx_stop'],
            downscale_x=config['downscale_x'],
            downscale_y=config['downscale_y'],
            is_training=False,
            is_x_cached=config['is_x_cached_testing'],
            is_y_cached=config['is_y_cached_testing'],
            data_key=config['data_key'],
        )

        self.model = DirectUNet6x6(
            num_convs=config['num_convs'],
            use_bias=config['use_bias'],
            output_full_lf=False,
            scale_outputs=True
        )
