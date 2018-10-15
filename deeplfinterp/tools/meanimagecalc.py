#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017. 
Contact sbruton[á]tcd.ie.
"""
import argparse
import os
import sys

import welford
import progressbar
import numpy as np
import h5py


def run_script(h5_location):
    with h5py.File(h5_location, "r+") as dataset_file:
        images = dataset_file["images"]

        # images of interest
        # TODO: Redo for all images
        # sub_image_idxs = [0, 7, 37, 56, 63]
        sub_image_idxs = list(range(64))
        train_range = (0, 1600)

        accumulators = [welford.WelfordAccumulator(min_valid_samples=100)
                        for _ in sub_image_idxs]

        print("Adding frames to accumulators...")
        bar = progressbar.ProgressBar()
        for frame_idx in bar(np.arange(train_range[0], train_range[1])):
            for accum_index, sub_image_idx in enumerate(sub_image_idxs):
                float_image = images[frame_idx, sub_image_idx].astype(
                    np.float32)
                accumulators[accum_index].add(float_image)

        mu = np.zeros(images[0].shape, dtype=np.float32)
        sigma = np.zeros(images[0].shape, dtype=np.float32)
        scale = np.zeros(images[0].shape, dtype=np.float32)

        for accum_index, sub_image_idx in enumerate(sub_image_idxs):
            sub_image_mu = accumulators[accum_index].get_mean()
            sub_image_mu = np.nan_to_num(sub_image_mu)
            mu[sub_image_idx] = sub_image_mu

            sub_image_sigma = accumulators[accum_index].get_variance()
            sub_image_sigma = np.nan_to_num(sub_image_sigma)
            sigma[sub_image_idx] = sub_image_sigma

            sub_image_scale = (np.max(mu, axis=(0, 1)) - np.min(mu)) / 2
            sub_image_scale = np.nan_to_num(sub_image_scale)
            scale[sub_image_idx] = sub_image_scale

        dataset_file.create_dataset(
            name="mean",
            shape=mu.shape,
            dtype=mu.dtype,
            data=mu
        )

        dataset_file.create_dataset(
            name="var",
            shape=sigma.shape,
            dtype=sigma.dtype,
            data=sigma
        )

        dataset_file.create_dataset(
            name="scale",
            shape=scale.shape,
            dtype=scale.dtype,
            data=scale
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate the mean and variance "
                    "for the images in the dataset"
    )
    parser.add_argument("input_h5",
                        help="The input h5 file.",
                        type=str)
    args = parser.parse_args()

    try:
        input_h5_location = os.path.abspath(args.input_h5)

        if not os.path.isfile(input_h5_location):
            raise ValueError("Invalid input h5 dataset argument.")

        run_script(input_h5_location)
    except ValueError as err:
        print("Value Error: {}".format(err))
        sys.exit(1)
    except OSError as err:
        print("Value Error: {}".format(err))
        sys.exit(1)
