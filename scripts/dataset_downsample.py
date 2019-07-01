#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2019.
Contact sbruton[á]tcd.ie.
"""
import argparse
import sys
import os
from typing import List
import h5py
import progressbar
import numpy as np


def downsample_dataset(h5_path: os.path, images_key: str):
    with h5py.File(h5_path, 'r+') as h5_file:
        low_res_images_key = images_key + '_lr'
        if low_res_images_key in list(h5_file.keys()):
            print("Low res dataset already exists, skipping.")
            return

        images_dataset = h5_file[images_key]
        num_images = images_dataset.shape[0]
        num_views = images_dataset.shape[1]
        num_channels = images_dataset.shape[2]
        full_height = images_dataset.shape[3]
        half_height = full_height // 2
        full_width = images_dataset.shape[4]
        half_width = full_width // 2

        low_res_dataset_shape = (
            num_images,
            num_views,
            num_channels,
            half_height,
            half_width
        )

        image_lr_chunk_shape = (1, 1, num_channels, half_height, half_width)

        low_res_dataset = h5_file.create_dataset(
            name=low_res_images_key,
            shape=low_res_dataset_shape,
            dtype=images_dataset.dtype,
            compression='lzf',
            chunks=image_lr_chunk_shape
        )

        print("Creating downsampled dataset.")
        bar = progressbar.ProgressBar()
        for idx in bar(np.arange(num_images)):
            low_res_dataset[idx] = images_dataset[idx, :, :, ::2, ::2]


def extract_indexed_dataset(h5_path: os.path,
                            input_images_key: str,
                            indices: List[int],
                            new_suffix: str):
    with h5py.File(h5_path, 'r+') as h5_file:
        new_images_key = input_images_key + new_suffix
        if new_images_key in list(h5_file.keys()):
            print("{} dataset already exists, skipping.".format(new_images_key))

        lr_images_dataset = h5_file[input_images_key]
        num_images = lr_images_dataset.shape[0]
        num_views = len(indices)
        num_channels = lr_images_dataset.shape[2]
        height = lr_images_dataset.shape[3]
        width = lr_images_dataset.shape[4]

        new_dataset_shape = (
            num_images,
            num_views,
            num_channels,
            height,
            width
        )

        new_chunk_shape = (1, num_views, num_channels, height, width)

        new_dataset = h5_file.create_dataset(
            name=new_images_key,
            shape=new_dataset_shape,
            dtype=lr_images_dataset.dtype,
            compression='lzf',
            chunks=new_chunk_shape
        )

        print("Creating new dataset: {}.".format(new_images_key))
        bar = progressbar.ProgressBar()
        for idx in bar(np.arange(num_images)):
            new_dataset[idx] = lr_images_dataset[idx, indices, :, :, :]


def get_y_indices(h5_path: os.path,
                  input_images_key: str,
                  x_indices: List[int]) -> List[int]:
    with h5py.File(h5_path) as h5_file:
        lr_images_dataset = h5_file[input_images_key]
        all_indices = np.arange(lr_images_dataset.shape[1], dtype=np.int32)
        return list(set(all_indices) - set(x_indices))


def create_new_datasets(h5_path: os.path,
                        data_key: str,
                        x_indices: List[int]):
    downsample_dataset(h5_path, data_key)

    y_indices = get_y_indices(h5_path, data_key, x_indices)
    data_key_hr = data_key
    data_key_lr = data_key + '_lr'

    extract_indexed_dataset(h5_path, data_key_hr, x_indices, '_x')
    extract_indexed_dataset(h5_path, data_key_hr, y_indices, '_y')

    extract_indexed_dataset(h5_path, data_key_lr, x_indices, '_x')
    extract_indexed_dataset(h5_path, data_key_lr, y_indices, '_y')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Downsample the images and save to the h5 file."
    )
    parser.add_argument("input",
                        help="The input h5 file.",
                        type=str)
    parser.add_argument("key",
                        help="The images dataset key.",
                        type=str,
                        default='images')
    parser.add_argument("-i", "--indices",
                        nargs='+',
                        help="The x indices.",
                        type=int,
                        required=True)
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)

    if not os.path.isfile(input_path):
        print("The specified input file {} does not exist.".format(input_path))
        sys.exit(1)

    try:
        create_new_datasets(input_path, args.key, args.indices)
    except ValueError as e:
        print("Error occurred: {}".format(e))
        sys.exit(1)
