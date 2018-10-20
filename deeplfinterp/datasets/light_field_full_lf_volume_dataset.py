#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2018.
Contact sbruton[á]tcd.ie.
"""
import os

import numpy as np
import h5py
from torch.utils.data.dataset import Dataset


class LightFieldFullLFVolumeDataset(Dataset):
    """
    Dataset wrapping images and targets for the lfvishead dataset
    """

    def __init__(self,
                 dataset_loc: os.path,
                 rank_reps_loc: os.path,
                 training: bool = True):
        super(LightFieldFullLFVolumeDataset, self).__init__()
        self.dataset_loc = dataset_loc
        self.training = training

        self.rank_rep = np.load(rank_reps_loc)

        # TODO: Parametrize this
        if self.rank_rep.shape[-1] == 512:
            self.rank_rep = self.rank_rep[:, 2, :2]

        # Center the ranking representation on (-1, 1)
        self.rank_rep = self.rank_rep / (255 / 2)
        self.rank_rep = self.rank_rep - 1.0

        # TODO: Parameterize this
        if self.training:
            self.start_idx = 0
            self.stop_idx = 1600
        else:
            self.start_idx = 1600
            self.stop_idx = 2000

        self.h5_file = h5py.File(dataset_loc, 'r')
        self.lf_images = self.h5_file['images']
        images_shape = self.h5_file['images'].shape

        print("Found images dataset with shape {}".format(self.lf_images.shape))

        self.x_light_field_idxs = [0, 5, 40, 45]
        self.y_light_field_idxs = [1, 2, 3, 4,
                                   8, 9, 10, 11, 12, 13,
                                   16, 17, 18, 19, 20, 21,
                                   24, 25, 26, 27, 28, 29,
                                   32, 33, 34, 35, 36, 37,
                                   41, 42, 43, 44]
        # self.x_light_field_idxs = [0, 7, 56, 63]
        # all_idxs = list(range(64))
        # self.y_light_field_idxs = list(set(all_idxs) -
        #                                set(self.x_light_field_idxs))
        # self.x_light_field_idxs = [0, 6, 48, 54]
        # self.y_light_field_idxs = [2, 4,
        #                            16, 18, 20, 22,
        #                            32, 34, 36, 38,
        #                            50, 52]

        self.channels = images_shape[-3]
        self.total_channels = self.channels * len(self.y_light_field_idxs)
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
            self.x_images = self.lf_images[
                            self.start_idx:self.stop_idx,
                            self.x_light_field_idxs, :, :, :]

            self.x_images = self.x_images.reshape(
                self.stop_idx - self.start_idx,
                len(self.x_light_field_idxs) * self.channels,
                self.height,
                self.width
            )

            self.y_images = self.lf_images[
                            self.start_idx:self.stop_idx,
                            self.y_light_field_idxs, :, :, :]

            self.y_images = self.y_images.reshape(
                self.stop_idx - self.start_idx,
                len(self.y_light_field_idxs) * self.channels,
                self.height,
                self.width
            )

        # self.y = np.zeros(shape=(len(self.image_indices),
        #                          self.channels,
        #                          self.height,
        #                          self.width),
        #                   dtype=np.float32)

        self.mean = self.h5_file['mean'][:]

        self.mean = np.zeros(self.mean.shape, self.mean.dtype)

        # TODO: Test the actual variance
        self.var = self.h5_file['scale'][:]
        self.std_dev = np.sqrt(self.var)
        self.std_dev = 255.0 * np.ones(self.std_dev.shape, self.std_dev.dtype)

        self.y_mean = np.array(self.mean[self.y_light_field_idxs])
        self.y_mean = np.concatenate(self.y_mean)

        self.y_std_dev = np.array(self.std_dev[self.y_light_field_idxs])
        self.y_std_dev = np.concatenate(self.y_std_dev)

        self.x_mean = np.array(self.mean[self.x_light_field_idxs])
        self.x_mean = np.concatenate(self.x_mean)

        self.x_std_dev = np.array(self.std_dev[self.x_light_field_idxs])
        self.x_std_dev = np.concatenate(self.x_std_dev)

        self.viewpoint_mean = np.mean(self.x_mean, axis=0)
        self.viewpoint_std_dev = np.mean(self.x_std_dev, axis=0)

        # for y_idx, image_idx in enumerate(self.image_indices):
        #     y_image = self.lf_images[image_idx, 37].astype(np.float32)
        #     y_image -= self.y_mean
        #     y_image /= self.y_std_dev
        #     self.y[y_idx] = y_image

        self.sample_shape = (4, self.channels, self.height, self.width)

        if self.training:
            self.get_item_fn = self._get_sample_train
        else:
            self.get_item_fn = self._get_sample_test

    def get_x_mean(self):
        return self.x_mean

    def get_x_std_dev(self):
        return self.x_std_dev

    def get_y_mean(self):
        return self.y_mean

    def get_y_std_dev(self):
        return self.y_std_dev

    def get_viewpoint_mean(self):
        return self.viewpoint_mean

    def get_viewpoint_std_dev(self):
        return self.viewpoint_std_dev

    def __len__(self):
        return self.image_indices.shape[0]

    def _get_sample_train(self, idx):
        image_idx = self.image_indices[idx]

        x = np.zeros(shape=(len(self.x_light_field_idxs) * self.channels,
                            self.height,
                            self.width),
                     dtype=np.uint8)

        for x_idx, sub_img_idx in enumerate(self.x_light_field_idxs):
            x[x_idx * self.channels: (x_idx * self.channels) + self.channels] = \
                self.lf_images[image_idx, sub_img_idx]
        #
        # x -= self.x_mean
        # x /= self.x_std_dev
        x = np.concatenate([x, self.rank_rep], axis=0)

        y = np.zeros(shape=(len(self.y_light_field_idxs) * self.channels,
                            self.height,
                            self.width),
                     dtype=np.uint8)

        for y_idx, sub_img_idx in enumerate(self.y_light_field_idxs):
            y[y_idx * self.channels: (y_idx * self.channels) + self.channels] = \
                self.lf_images[image_idx, sub_img_idx]
        #
        # y -= self.y_mean
        # y /= self.y_std_dev

        return x, y

    def _get_sample_test(self, idx):
        # image_idx = self.image_indices[idx]

        x = self.x_images[idx]

        x = np.concatenate([x, self.rank_rep], axis=0)

        y = self.y_images[idx]

        # x = np.zeros(shape=(len(self.x_light_field_idxs) * self.channels,
        #                     self.height,
        #                     self.width),
        #              dtype=np.float32)
        #
        # for x_idx, sub_img_idx in enumerate(self.x_light_field_idxs):
        #     x[x_idx * self.channels: (x_idx * self.channels) + self.channels] = \
        #         self.lf_images[image_idx, sub_img_idx].astype(np.float32)

        # x -= self.x_mean
        # x /= self.x_std_dev
        #
        # y -= self.y_mean
        # y /= self.y_std_dev

        return x, y

    def __getitem__(self, idx):
        return self.get_item_fn(idx=idx)
