#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2019.
Contact sbruton[á]tcd.ie.
"""
import os
from typing import Union, List

import h5py
import numpy as np
import torch

from .light_field_dataset import LightFieldDataset


class LightField6x6Dataset(LightFieldDataset):
    """
    Dataset wrapping images and targets for the low res inputs
    """

    def __init__(self,
                 dataset_loc: os.path,
                 testing_set_idx_stop: int,
                 testing_set_idx_start: int,
                 is_training: bool = True,
                 is_timing: bool = False,
                 is_x_cached: bool = False,
                 is_y_cached: bool = False,
                 data_key: str = 'images'):
        super(LightField6x6Dataset, self).__init__()
        self.is_x_cached = is_x_cached
        self.is_y_cached = is_y_cached
        self.dataset_loc = dataset_loc
        self.is_training = is_training
        self.is_timing = is_timing

        self.h5_file = h5py.File(self.dataset_loc, 'r')

        self.x_images = self.h5_file[data_key + '_x']

        self.y_images = self.h5_file[data_key + '_y']

        print("Found x dataset with shape {}".format(self.x_images.shape))
        print("Found y dataset with shape {}".format(self.y_images.shape))

        self.num_samples = self.x_images.shape[-5]
        self.num_channels = self.x_images.shape[-3]

        self.num_views_x = self.x_images.shape[-4]
        self.height_x = self.x_images.shape[-2]
        self.width_x = self.x_images.shape[-1]

        self.num_views_y = self.y_images.shape[-4]
        self.height_y = self.y_images.shape[-2]
        self.width_y = self.y_images.shape[-1]

        self.all_indices = np.arange(self.num_samples, dtype=np.int32)

        self.testing_indices = np.arange(testing_set_idx_start,
                                         testing_set_idx_stop,
                                         dtype=np.int32)

        self.training_indices = \
            np.array(list(set(self.all_indices) - set(self.testing_indices)))

        if self.is_training:
            self.dataset_type = "Train"
            self.image_indices = self.training_indices
        else:
            self.dataset_type = "Test"
            self.image_indices = self.testing_indices

        print("{} idxs {} ... {}".format(self.dataset_type,
                                         self.image_indices[:10],
                                         self.image_indices[-10:]))

        if self.is_x_cached:
            print("Caching x dataset please wait...")
            self.x_images = self.x_images[self.image_indices.tolist()]

        if self.is_y_cached and not self.is_timing:
            print("Caching y dataset please wait...")
            self.y_images = self.y_images[self.image_indices.tolist()]

        self.item_fn = self.calc_item_fn()

    def calc_item_fn(self):
        if self.is_x_cached and self.is_y_cached:
            item_fn = self._get_sample_x_y_cached
        elif self.is_x_cached:
            item_fn = self._get_sample_x_cached
        elif self.is_y_cached:
            item_fn = self._get_sample_y_cached
        else:
            item_fn = self._get_sample_no_cached

        return item_fn

    def set_only_x_dataset(self):
        if not self.is_x_cached:
            print("Caching x dataset please wait...")
            self.x_images = self.x_images[self.image_indices.tolist()]
            self.is_x_cached = True

        self.item_fn = self._get_only_x

    def revert_only_x_dataset(self):
        self.item_fn = self.calc_item_fn()

    def _get_only_x(self, idx):
        return self.x_images[idx]

    def get_only_y(self, idx):
        if self.is_y_cached:
            image_idx = idx
        else:
            image_idx = self.image_indices[idx]

        return self.y_images[image_idx]

    def __len__(self):
        return self.image_indices.shape[0]

    def _get_sample_x_cached(self, idx):
        image_idx_y = self.image_indices[idx]

        return self.x_images[idx], self.y_images[image_idx_y]

    def _get_sample_y_cached(self, idx):
        image_idx_x = self.image_indices[idx]

        return self.x_images[image_idx_x], self.y_images[idx]

    def _get_sample_x_y_cached(self, idx):
        return self.x_images[idx], self.y_images[idx]

    def _get_sample_no_cached(self, idx):
        image_idx = self.image_indices[idx]

        return self.x_images[image_idx], self.y_images[image_idx]

    def _get_sample_timing_cached(self, idx):
        return self.x_images[idx]

    def __getitem__(self, idx):
        return self.item_fn(idx=idx)

    def get_x_images(self, to_list=True):
        if self.is_x_cached:
            if to_list:
                return self.x_images.tolist()
            else:
                return self.x_images
        else:
            print("Caching x dataset please wait...")
            if to_list:
                return self.x_images[self.image_indices.tolist()].tolist()
            else:
                return self.x_images[self.image_indices.tolist()]

    def get_y_images(self, to_list: bool = True) -> Union[np.ndarray, List]:
        if self.is_x_cached:
            if to_list:
                return self.x_images.tolist()
            else:
                return self.x_images
        else:
            print("Caching y dataset please wait...")
            if to_list:
                return self.x_images[self.image_indices.tolist()].tolist()
            else:
                return self.x_images[self.image_indices.tolist()]

    def fast_collate(self, batch):
        input_lfs = [lf[0] for lf in batch]
        output_lfs = [lf[1] for lf in batch]

        input_batch_tensor = torch.empty(
            (len(input_lfs),
             self.num_views_x,
             self.num_channels,
             self.height_x,
             self.width_x),
            dtype=torch.uint8,
            requires_grad=False
        )

        for idx, input_lf in enumerate(input_lfs):
            input_batch_tensor[idx] = torch.from_numpy(input_lf)

        output_batch_tensor = torch.empty(
            (len(output_lfs),
             self.num_views_y,
             self.num_channels,
             self.height_y,
             self.width_y),
            dtype=torch.uint8,
            requires_grad=False
        )

        for idx, output_lf in enumerate(output_lfs):
            output_batch_tensor[idx] = torch.from_numpy(output_lf)

        return input_batch_tensor, output_batch_tensor
