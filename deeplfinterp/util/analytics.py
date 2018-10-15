#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017. 
Contact sbruton[á]tcd.ie.
"""
import json
import os

# Need to ensure can run on server, i.e. no X session
import matplotlib;

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
from skimage.measure import compare_nrmse as nrmse
from skimage.measure import compare_psnr as psnr
import numpy as np
import torch
from torch import nn


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
                    time_taken: float,
                    output_images: np.array,
                    final_loss):
    print("Saving the evaluation of model.")
    save_model(model=model, output_path=output_path)

    results = {'loss': float(final_loss), 'time_per_image': time_taken}

    np.save(os.path.join(output_path, 'y_pred_images.npy'), output_images)

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


# def mse(output, target) -> float:
#     assert output.shape == target.shape
#
#     return np.mean(np.square(output - target))

#
# def mse(output, target) -> float:
#     assert output.shape == target.shape
#
#     mse_channels = np.zeros((output.shape[0]))
#
#     for ch in range(output.shape[0]):
#         mse_channels[ch] = mse_mono(output[ch], target[ch])
#
#     return np.mean(mse_channels)
#
#
# def calc_ssim(output, target) -> float:
#     return ssim
#
#
# def calc_psnr(output, target) -> float:
#     return 10.0 * np.log10(255.0**2 / mse(output, target))


def calc_and_save_all_metrics(all_targets,
                              all_outputs,
                              output_path: os.path) -> dict:
    assert all_outputs.shape == all_targets.shape

    # TODO: If the images are in an array we need to reshape them
    # TODO: And again when saving.

    num_images = all_targets.shape[0]
    ssim_results = np.zeros(num_images, dtype=np.float32)
    psnr_results = np.zeros(num_images, dtype=np.float32)
    mse_results = np.zeros(num_images, dtype=np.float32)
    nrmse_results = np.zeros(num_images, dtype=np.float32)

    for idx in range(num_images):
        target_reshape = np.moveaxis(all_targets[idx], -3, -1)
        output_reshape = np.moveaxis(all_outputs[idx], -3, -1)
        ssim_results[idx] = ssim(target_reshape,
                                 output_reshape,
                                 multichannel=True)
        psnr_results[idx] = psnr(target_reshape, output_reshape)
        mse_results[idx] = mse(target_reshape, output_reshape)
        nrmse_results[idx] = nrmse(target_reshape, output_reshape)

    np.save(os.path.join(output_path, "ssim.npy"), ssim_results)
    np.save(os.path.join(output_path, "psnr.npy"), psnr_results)
    np.save(os.path.join(output_path, "mse.npy"), mse_results)
    np.save(os.path.join(output_path, "nrmse.npy"), nrmse_results)

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

    with open(os.path.join(output_path, "metrics.json"), 'w') as fp:
        json.dump(metrics, fp, indent=4, sort_keys=True)


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

    # save_history(history, output_path)
    #
    # save_history_plots(history, output_path)

    # save_test(full_model, output_path, test_steps, test_generator,
    #           predict_generator, y_true)
