#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017. 
Contact sbruton[á]tcd.ie.
"""
import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler


class DataLoaderMaker:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.size = len(self.dataset)
        self.batch_size = batch_size

    def get_loader(self):
        while True:
            loader = torch.utils.data.DataLoader(
                dataset=self.dataset,
                pin_memory=True,
                batch_size=self.batch_size,
                sampler=RandomSampler(self.size),
                num_workers=2
            )

            yield loader

    def get_full_loader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            pin_memory=True,
            batch_size=self.batch_size,
            sampler=SequentialSampler(self.size),
            num_workers=2
        )

        return loader
