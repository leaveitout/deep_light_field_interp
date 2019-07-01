#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017. 
Contact sbruton[á]tcd.ie.
"""
import json
import os
from typing import Type

# Need to ensure can run on server, i.e. no X session
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import OneHotEncoder
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
from skimage.measure import compare_nrmse as nrmse
from skimage.measure import compare_psnr as psnr
import numpy as np
import torch
from torch import nn
import h5py

from .meters import CustomProgressBar, AverageMeter
from ..datasets import LightFieldDataset


def one_hot_to_dense(y: np.array) -> np.ndarray:
    return np.fromiter((np.argmax(row) for row in y), dtype=np.int)


def dense_to_one_hot(y: np.array,
                     n_values: int = "auto"):
    if n_values == 'auto':
        n_values = np.max(y)

    num_samples = y.shape[-1]

    y_to_encode = y

    if len(y.shape) == 1:
        y_to_encode = np.reshape(y, (-1, 1))

    encoder = OneHotEncoder(n_values=n_values, dtype=np.float32, sparse=False)
    y_encoded = encoder.fit_transform(y_to_encode)
    return y_encoded


def save_model(model: nn.Module,
               output_path: os.path):
    print("Saving " + str(output_path) + " model definition...")
    model_output_path = os.path.join(output_path, "model.pkl")
    torch.save(model.state_dict(), model_output_path)


def save_evaluation(model,
                    output_path: os.path,
                    time_per_image: float,
                    final_loss):
    print("Saving the evaluation of model.")
    save_model(model=model, output_path=output_path)

    results = {'loss': float(final_loss), 'time_per_image': time_per_image}

    print(results)

    with open(os.path.join(output_path, "results.json"), 'w') as fp:
        json.dump(results, fp, indent=4, sort_keys=True)


def save_training_config(train_config: dict,
                         output_path: os.path):
    # TODO: This needs to be adapted for pytorch
    print("Saving training config.")
    json_filename = 'train_config.json'
    with open(os.path.join(output_path, json_filename), 'w') as fp:
        json.dump(train_config, fp, indent=4, sort_keys=True)


def save_history(train_loss_history,
                 valid_loss_history,
                 output_path: os.path):
    print("Saving history to pickle file.")

    with open(os.path.join(output_path, 'train_loss_history.pkl'), 'wb') as fp:
        pickle.dump(train_loss_history, fp)

    with open(os.path.join(output_path, 'valid_loss_history.pkl'), 'wb') as fp:
        pickle.dump(valid_loss_history, fp)

    save_history_plots(train_loss_history=train_loss_history,
                       valid_loss_history=valid_loss_history,
                       output_path=output_path)


def calc_and_save_all_metrics(test_set: Type[LightFieldDataset],
                              output_path: os.path,
                              h5_file_loc: os.path = None,
                              h5_dataset_key: str = None) -> dict:
    with h5py.File(h5_file_loc, 'a') as h5_file:
        output_images = h5_file[h5_dataset_key]

        all_targets_shape = (
            len(test_set),
            test_set.num_views_y,
            test_set.num_channels,
            test_set.height_y,
            test_set.width_y
        )

        assert output_images.shape == all_targets_shape

        # TODO: If the images are in an array we need to reshape them
        # TODO: And again when saving.

        num_images = all_targets_shape[0]
        num_views = all_targets_shape[1]
        ssim_results = np.zeros((num_images, num_views), dtype=np.float32)
        psnr_results = np.zeros((num_images, num_views), dtype=np.float32)
        mse_results = np.zeros((num_images, num_views), dtype=np.float32)
        nrmse_results = np.zeros((num_images, num_views), dtype=np.float32)

        ssim_meter = AverageMeter(name='SSIM', cum=False)
        custom = CustomProgressBar(label='SSIM')

        print("Calculating image metrics.")
        for image_idx in custom.bar(range(num_images)):
            target_lf = test_set.get_only_y(image_idx)
            for view_idx in range(num_views):
                target_reshape = np.moveaxis(
                    target_lf[view_idx],
                    -3,
                    -1
                )
                output_reshape = np.moveaxis(
                    output_images[image_idx, view_idx],
                    -3,
                    -1
                )
                ssim_results[image_idx, view_idx] = ssim(
                    target_reshape,
                    output_reshape,
                    multichannel=True
                )
                psnr_results[image_idx, view_idx] = psnr(
                    target_reshape,
                    output_reshape
                )
                mse_results[image_idx, view_idx] = mse(
                    target_reshape,
                    output_reshape
                )
                nrmse_results[image_idx, view_idx] = nrmse(
                    target_reshape,
                    output_reshape
                )

            # Log errors
            ssim_meter.update(float(np.mean(ssim_results[image_idx])))

            custom.format_custom_text.update_mapping(value=ssim_meter.value())

        metrics = {
            'ssim_avg': float(np.mean(ssim_results)),
            'ssim_std': float(np.std(ssim_results)),
            'psnr_avg': float(np.mean(psnr_results)),
            'psnr_std': float(np.std(psnr_results)),
            'mse_avg': float(np.mean(mse_results)),
            'mse_std': float(np.std(mse_results)),
            'nrmse_avg': float(np.mean(nrmse_results)),
            'nrmse_std': float(np.std(nrmse_results))
        }

        # Also save to a json for easy viewing.
        with open(os.path.join(output_path, "metrics.json"), 'w') as fp:
            json.dump(metrics, fp, indent=4, sort_keys=True)

        output_images.attrs.create('ssim', ssim_results)
        output_images.attrs.create('psnr', psnr_results)
        output_images.attrs.create('mse', mse_results)
        output_images.attrs.create('nrmse', nrmse_results)

        output_images.attrs.create('ssim_avg', metrics['ssim_avg'])
        output_images.attrs.create('ssim_std', metrics['ssim_std'])
        output_images.attrs.create('psnr_avg', metrics['psnr_avg'])
        output_images.attrs.create('psnr_std', metrics['psnr_std'])
        output_images.attrs.create('mse_avg', metrics['mse_avg'])
        output_images.attrs.create('mse_std', metrics['mse_std'])
        output_images.attrs.create('nrmse_avg', metrics['nrmse_avg'])
        output_images.attrs.create('nrmse_std', metrics['nrmse_std'])


def save_history_plots(train_loss_history,
                       valid_loss_history,
                       output_path: os.path):
    loss_fig = plt.figure()
    loss_plot = loss_fig.add_subplot(111)
    loss_plot.plot(train_loss_history)
    loss_plot.plot(valid_loss_history)
    loss_plot.set_title('Model Loss')
    loss_plot.set_xlabel('Updates')
    loss_plot.legend(['Train', 'Test'], loc='upper left')
    loss_fig.savefig(os.path.join(output_path, 'loss.png'))
    plt.close(loss_fig)


def model_dump(full_model,
               train_config,
               output_path: os.path):
    # TODO: This needs to be adapted for pytorch
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    save_model(full_model, output_path)

    save_training_config(train_config, output_path)
