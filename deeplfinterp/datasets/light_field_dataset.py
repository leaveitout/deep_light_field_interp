#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017. 
Contact sbruton[á]tcd.ie.
"""
import os

import numpy as np
import h5py
from torch.utils.data.dataset import Dataset


class LightFieldDataset(Dataset):
    """
    Dataset wrapping images and targets for the lfvishead dataset
    """

    def __init__(self,
                 dataset_loc: os.path,
                 training: bool = True):
        super(LightFieldDataset, self).__init__()
        self.dataset_loc = dataset_loc
        self.training = training

        # TODO: Parameterize this
        if self.training:
            self.start_idx = 0
            self.stop_idx = 1600
        else:
            self.start_idx = 1600
            self.stop_idx = 2000

        self.h5_file = h5py.File(dataset_loc, 'r')
        images_shape = self.h5_file['images'].shape
        self.lf_images = self.h5_file['images']
        print("Found images dataset with shape {}".format(self.lf_images.shape))

        self.x_light_field_idxs = [0, 7, 56, 63]
        self.channels = images_shape[-3]
        self.total_channels = self.channels
        self.height = images_shape[-2]
        self.width = images_shape[-1]
        self.total_samples = self.stop_idx - self.start_idx

        self.image_indices = np.arange(self.start_idx,
                                       self.stop_idx,
                                       dtype=np.int32)

        if self.training:
            dataset_type = "Train"
        else:
            dataset_type = "Test"

        print("{} idxs {} ... {}".format(dataset_type,
                                         self.image_indices[:10],
                                         self.image_indices[-10:]))

        # Cache y to ram
        if not self.training:
            self.y_images = \
                np.squeeze(
                    self.lf_images[self.start_idx:self.stop_idx, 37, :, :, :]
                )

        self.y = np.zeros(shape=(len(self.image_indices),
                                 self.channels,
                                 self.height,
                                 self.width),
                          dtype=np.float32)

        self.mean = self.h5_file['mean'][:]

        self.mean = np.zeros(self.mean.shape, self.mean.dtype)

        # TODO: Test the actual variance
        self.var = self.h5_file['scale'][:]
        self.std_dev = np.sqrt(self.var)
        self.std_dev = 255.0 * np.ones(self.std_dev.shape, self.std_dev.dtype)

        self.y_mean = self.mean[37]
        self.y_std_dev = self.std_dev[37]

        self.x_mean = np.array(
            [self.mean[0],
             self.mean[7],
             self.mean[56],
             self.mean[63]]
        )
        self.x_mean = np.concatenate(self.x_mean)

        self.x_std_dev = np.array(
            [self.std_dev[0],
             self.std_dev[7],
             self.std_dev[56],
             self.std_dev[63]]
        )

        self.x_std_dev = np.concatenate(self.x_std_dev)

        for y_idx, image_idx in enumerate(self.image_indices):
            y_image = self.lf_images[image_idx, 37].astype(np.float32)
            y_image -= self.y_mean
            y_image /= self.y_std_dev
            self.y[y_idx] = y_image

        self.sample_shape = (4, self.channels, self.height, self.width)

        if self.training:
            self.get_item_fn = self._get_sample_train
        else:
            self.get_item_fn = self._get_sample_test

    def __len__(self):
        return self.image_indices.shape[0]

    def _get_sample_train(self, idx):
        y = self.y[idx]

        x = np.zeros(shape=(len(self.x_light_field_idxs) * self.channels,
                            self.height,
                            self.width),
                     dtype=np.float32)

        image_idx = self.image_indices[idx]
        for x_idx, sub_img_idx in enumerate(self.x_light_field_idxs):
            x[x_idx * self.channels: (x_idx * self.channels) + self.channels] = \
                self.lf_images[image_idx, sub_img_idx].astype(np.float32)

        x -= self.x_mean
        x /= self.x_std_dev

        return x, y

    def _get_sample_test(self, idx):
        y = self.y[idx]

        x = np.zeros(shape=(len(self.x_light_field_idxs) * self.channels,
                            self.height,
                            self.width),
                     dtype=np.float32)

        image_idx = self.image_indices[idx]
        for x_idx, sub_img_idx in enumerate(self.x_light_field_idxs):
            x[x_idx * self.channels: (x_idx * self.channels) + self.channels] = \
                self.lf_images[image_idx, sub_img_idx].astype(np.float32)

        x -= self.x_mean
        x /= self.x_std_dev

        return x, y

    def __getitem__(self, idx):
        return self.get_item_fn(idx=idx)
