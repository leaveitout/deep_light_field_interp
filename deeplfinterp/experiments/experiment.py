#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017.
Contact sbruton[á]tcd.ie.
"""
import time
from functools import partial

import torch

from ..util import train_tools, analytics
from ..util import custom_losses


class Experiment:
    def __init__(self, config):
        self.config = config
        self.train_set = None
        self.valid_set = None
        self.test_set = None
        self.model = None
        self.lr_scheduler = None

        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer()

    def _get_criterion(self) -> torch.nn.Module:
        loss_str = self.config['loss']
        if loss_str == 'l1':
            return torch.nn.L1Loss()
        elif loss_str == 'LPIPS':
            return custom_losses.LPIPSLoss(
                to_zero_data=self.config['loss_to_zero_data'],
                to_scale_data=self.config['loss_to_scale_data']
            )
        else:
            raise ValueError(
                "Loss config key: {}, not implemented".format(loss_str))

    def _get_optimizer(self):
        optimizer_str = self.config['optimizer']

        if optimizer_str == 'Adamax':
            learning_rate = self.config['optimizer_learning_rate']
            eps = self.config['optimizer_eps']
            weight_decay = self.config['optimizer_weight_decay']
            return partial(
                torch.optim.Adamax,
                lr=learning_rate,
                eps=eps,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(
                "Optimizer config key: {}, "
                "not implemented".format(optimizer_str))

    def run(self):
        start_time = time.clock()

        if self.valid_set is None:
            self.valid_set = self.test_set

        torch.manual_seed(int(round(time.time() * 1000)))
        history = train_tools.train(
            model=self.model,
            train_set=self.train_set,
            valid_set=self.valid_set,
            test_set=self.test_set,
            save_path=self.config['save_path'],
            optimizer=self.optimizer,
            criterion=self.criterion,
            num_epochs=self.config['num_epochs'],
            evaluate_epoch_freq=self.config['evaluate_epoch_freq'],
            batch_size=self.config['batch_size']
        )

        print("Saving history...")
        analytics.save_history(train_loss_history=history['train'],
                               valid_loss_history=history['valid'],
                               output_path=self.config['save_path'])

        print("Saving config...")
        analytics.save_training_config(train_config=self.config,
                                       output_path=self.config['save_path'])

        train_tools.print_time_taken(time.clock() - start_time)
