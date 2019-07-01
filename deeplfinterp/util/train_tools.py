#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017.
Contact sbruton[á]tcd.ie.
"""
import os
import time
from typing import Type, Callable, Generator

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import h5py

from .analytics import save_evaluation
from .analytics import calc_and_save_all_metrics
from ..datasets.light_field_dataset import LightFieldDataset

from .meters import CustomProgressBar, AverageMeter

cudnn.benchmark = True


def time_model(model: nn.Module,
               test_set: Type[LightFieldDataset],
               device) -> np.ndarray:
    model.eval()

    custom = CustomProgressBar('N/A')

    loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=1,
        sampler=SequentialSampler(test_set),
        num_workers=1,
        pin_memory=True
    )

    test_set.set_only_x_dataset()

    model.eval()
    torch.set_grad_enabled(False)

    print('Timing')

    all_times = []

    for input_data in custom.bar(loader):
        tick = time.time()
        input_var = input_data.to(device).float().div_(255.0)
        _ = model(input_var)
        tock = time.time()
        time_taken = tock - tick
        all_times.append(time_taken)

    test_set.revert_only_x_dataset()

    return np.array(all_times)


def evaluate_model(model: nn.Module,
                   test_set: Type[LightFieldDataset],
                   criterion,
                   device,
                   batch_size: int,
                   save_path: str,
                   all_times_taken: np.ndarray):
    loss_meter = AverageMeter(name='loss', cum=False)

    model.eval()

    custom = CustomProgressBar(label='Loss')

    loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        sampler=SequentialSampler(test_set),
        num_workers=1,
        pin_memory=True
    )

    model.eval()
    torch.set_grad_enabled(False)

    print('Evaluating')

    if not os.path.isdir(save_path):
        try:
            os.mkdir(save_path)
        except FileExistsError:
            print("Save dir exists, potentially overwriting previous results.")

    results_h5_loc = os.path.join(save_path, 'results.h5')

    with h5py.File(results_h5_loc, 'w') as h5_file:
        output_images = h5_file.create_dataset(
            'output_images',
            shape=(
                len(test_set),
                test_set.num_views_y,
                test_set.num_channels,
                test_set.height_y,
                test_set.width_y
            ),
            chunks=(
                1,
                test_set.num_views_y,
                test_set.num_channels,
                test_set.height_y,
                test_set.width_y
            ),
            dtype=np.uint8,
            compression='lzf'
        )

        sample_batch_output_shape = [batch_size,
                                     test_set.num_views_y,
                                     test_set.num_channels,
                                     test_set.height_y,
                                     test_set.width_y]

        start = 0

        for (input_data, target) in custom.bar(loader):
            # Move data to device, cast to float and transform to [0, 1]
            input_var = input_data.to(device).float().div_(255.0)
            target_var = target.to(device).float().div_(255.0)

            output_var = model(input_var)

            loss = criterion(output_var, target_var)

            # Last batch may be of a different size
            curr_batch_size = input_data.size(0)
            loss_meter.update(loss.item(), curr_batch_size)

            custom.format_custom_text.update_mapping(value=loss_meter.value())

            output_data = output_var

            output_data = \
                output_data.mul_(255.0).clamp_(0.0, 255.0).type(torch.uint8)
            output_data_np = output_data.cpu().numpy()

            if curr_batch_size != batch_size:
                sample_batch_output_shape[0] = curr_batch_size

            output_images[start:start + curr_batch_size] = \
                output_data_np.reshape(sample_batch_output_shape)

            start += curr_batch_size

        # Save to the h5 file: all times, mean time, std time, final loss
        output_images.attrs.create('all_times', all_times_taken)
        output_images.attrs.create('mean_time', np.mean(all_times_taken))
        output_images.attrs.create('std_time', np.std(all_times_taken))
        output_images.attrs.create('final_loss', loss_meter.value())

    save_evaluation(
        model=model,
        output_path=save_path,
        time_per_image=float(np.mean(all_times_taken)),
        final_loss=loss_meter.value()
    )

    calc_and_save_all_metrics(
        test_set,
        save_path,
        results_h5_loc,
        'output_images'
    )


def run_epoch(loader,
              model,
              criterion,
              optimizer,
              device,
              training=True):
    loss_meter = AverageMeter(name='Loss', cum=False)
    custom = CustomProgressBar(label='Loss')

    if training:
        model.train()
        torch.set_grad_enabled(True)
        print('Training')
    else:
        model.eval()
        torch.set_grad_enabled(False)
        print('Evaluating')

    for (input_data, target) in custom.bar(loader):
        with torch.autograd.set_detect_anomaly(training):
            # Forward pass
            input_var = input_data.to(device).float().div_(255.0)
            target_var = target.to(device).float().div_(255.0)

            output_var = model(input_var)

            loss = criterion(output_var, target_var)

            # Backward pass
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Log errors
        loss_meter.update(loss.item())

        custom.format_custom_text.update_mapping(value=loss_meter.value())

    return loss_meter.value()


def train(model,
          train_set: Type[LightFieldDataset],
          valid_set: Type[LightFieldDataset],
          test_set: Type[LightFieldDataset],
          save_path: str,
          optimizer,
          criterion: nn.Module,
          num_epochs: int,
          evaluate_epoch_freq: int = 1,
          batch_size: int = 64,
          seed: int = None) -> dict:
    if seed is not None:
        torch.manual_seed(seed)

    if torch.cuda.is_available():
        criterion = criterion.cuda()

        if torch.cuda.device_count() > 1:
            model_wrapper = torch.nn.DataParallel(model)
        else:
            model_wrapper = model

        device = torch.device("cuda")
        model_wrapper = model_wrapper.to(device)

        if callable(optimizer):
            optimizer = optimizer(params=model_wrapper.parameters())
    else:
        model_wrapper = model
        device = torch.device("cpu")
        model_wrapper = model_wrapper.to(device)

    train_losses, valid_losses = [], []

    best_loss = float('inf')

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        sampler=RandomSampler(train_set),
        num_workers=1,
        pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_set,
        batch_size=batch_size,
        sampler=SequentialSampler(valid_set),
        num_workers=1,
        pin_memory=True
    )

    for epoch in range(1, num_epochs + 1):
        print("{}".format(epoch))

        train_loss = run_epoch(
            loader=train_loader,
            model=model_wrapper,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True
        )
        train_losses.append(train_loss)

        if epoch % evaluate_epoch_freq == 0:
            valid_loss = run_epoch(
                loader=valid_loader,
                model=model_wrapper,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                training=False
            )

            valid_losses.append(valid_loss)

            if valid_loss < best_loss:
                best_loss = valid_loss
                print("New best loss: {:.4}".format(best_loss))

    times_test_set = time_model(
        model=model_wrapper,
        test_set=test_set,
        device=device,
    )

    evaluate_model(
        model=model_wrapper,
        test_set=test_set,
        criterion=criterion,
        device=device,
        batch_size=batch_size,
        save_path=save_path,
        all_times_taken=times_test_set
    )

    return {'train': train_losses, 'valid': valid_losses}


def print_time_taken(time_in_secs):
    # Cast to int to ensure that we can print it correctly.
    m, s = divmod(int(time_in_secs), 60)
    h, m = divmod(m, 60)
    print('Done! Time taken {:d}:{:02d}:{:02d}'.format(h, m, s))
