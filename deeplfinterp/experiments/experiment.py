#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017. 
Contact sbruton[á]tcd.ie.
"""
import time

from ..util import train_tools, analytics

from apex import amp


class Experiment:
    def __init__(self, config):
        self.config = config
        self.train_set = None
        self.test_set = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None

    def run(self):
        start_time = time.clock()

        if self.config['fp16']:
            print("Using fp16")
            amp_handle = amp.init(enabled=True)
        else:
            amp_handle = amp.init(enabled=False)

        history = train_tools.train(
            model=self.model,
            train_set=self.train_set,
            valid_set=self.test_set,
            save_path=self.config['save_path'],
            optimizer=self.optimizer,
            num_epochs=self.config['num_epochs'],
            batch_size=self.config['batch_size'],
            amp_handle=amp_handle
        )

        print("Saving history...")
        analytics.save_history(train_loss_history=history['train'],
                               valid_loss_history=history['valid'],
                               output_path=self.config['save_path'])

        print("Saving config...")
        analytics.save_training_config(train_config=self.config,
                                       output_path=self.config['save_path'])

        train_tools.print_time_taken(time.clock() - start_time)
