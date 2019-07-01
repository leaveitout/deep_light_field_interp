#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017.
Contact sbruton[á]tcd.ie.
"""
import os
import time

import numpy as np
import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import progressbar

from .analytics import save_evaluation
from .analytics import calc_and_save_all_metrics
from torch.utils.data.dataset import Dataset
from ..datasets.light_field_dataset import LightFieldDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
# import amp

cudnn.benchmark = True


def convert_to_numpy(value):
    if type(value) is not np.ndarray:
        if type(value) is float or type(value) is int:
            return np.array(value)
        if isinstance(value, torch.autograd.Variable):
            if value.is_cuda:
                return value.data.cpu().numpy()
            else:
                return value.data.numpy()
        elif isinstance(value, torch.FloatTensor):
            if value.is_cuda:
                return value.cpu().numpy()
            else:
                return value.numpy()
    else:
        return value


class Meter:
    def __init__(self, name):
        self.name = name
        self.curr_value = np.zeros(1)

    def update(self, next_value, n=1):
        self.curr_value = next_value

    def value(self):
        return self.curr_value


class AverageMeter(Meter):
    def __init__(self, name, cum=False):
        super().__init__(name)
        self.curr_value = np.zeros(1)
        self.sum = np.zeros(1)
        self.count = np.zeros(1, dtype=np.int)
        self.cum = cum

    def update(self, next_value, n=1):
        self.curr_value = convert_to_numpy(next_value)

        self.sum += self.curr_value * n
        self.count += n

    def value(self):
        if self.cum:
            return self.sum
        else:
            return self.sum / self.count


class AverageAccumMeter(Meter):
    def __init__(self, name):
        """
        Mean accumulator using Welford's algorithm, this meter is somewhat
        safer from overflow errors and higher accuracy when running over a large
        number of samples.
        """
        super().__init__(name)
        self.K = np.zeros(1)
        self.N = np.zeros(1)
        self.ex = np.zeros(1)
        self.curr_value = None

    def update(self, next_value, n=1):
        self.curr_value = convert_to_numpy(next_value)

        if self.N is None:
            self.N = np.ones(self.curr_value.shape)
            self.K = self.curr_value
            self.ex = np.zeros(self.curr_value.shape)
        else:
            self.ex += n * (self.curr_value - self.K)
            self.N += n

    def value(self):
        if self.curr_value is None:
            return np.zeros(1, dtype=np.float)
        else:
            return self.K + (self.ex / self.N)


class TrainProgressBar:
    def __init__(self):
        self.format_custom_text = progressbar.FormatCustomText(
            '| Loss: %(loss).5f',
            dict(loss=0.0),
        )
        self.bar = progressbar.ProgressBar(
            widgets=[progressbar.Percentage(),
                     ' (',
                     progressbar.SimpleProgress(),
                     ') ',
                     progressbar.Bar(),
                     ' ',
                     progressbar.ETA(),
                     ' ',
                     self.format_custom_text
                     ])


class EvaluateProgressBar:
    def __init__(self):
        self.format_custom_text = progressbar.FormatCustomText(
            '| Loss: %(loss).5f',
            dict(loss=0.0),
        )
        self.bar = progressbar.ProgressBar(
            widgets=[progressbar.Percentage(),
                     ' (',
                     progressbar.SimpleProgress(),
                     ') ',
                     progressbar.Bar(),
                     ' ',
                     progressbar.ETA(),
                     ' ',
                     self.format_custom_text
                     ])


def accuracy(output, target, top_k=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def time_model(model: nn.Module,
               criterion,
               train_set: Dataset,
               test_set: Dataset,
               device,
               batch_size: int):
    loss_meter = AverageMeter(name='loss', cum=False)

    model.eval()

    custom = EvaluateProgressBar()

    loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        sampler=SequentialSampler(test_set),
        num_workers=1,
        pin_memory=True
    )

    model.eval()
    torch.set_grad_enabled(False)
    x_mean = train_set.get_viewpoint_mean()
    x_std_dev = train_set.get_viewpoint_std_dev()
    # y_mean = test_dataset.get_y_mean()
    # y_std_dev = test_dataset.get_y_std_dev()

    x_mean_var = torch.from_numpy(x_mean).to(device)
    x_std_dev_var = torch.from_numpy(x_std_dev).to(device)

    print('Timing')

    tick = time.time()
    for (input_data, target) in custom.bar(loader):
        # print("Input data shape {}".format(input_data.size()))
        input_var = input_data.to(device)
        target_var = target.to(device).float()

        output_var = model(input_var,
                           x_mean=x_mean_var,
                           x_std_dev=x_std_dev_var)

        loss = criterion(output_var, target_var).cuda()

        loss_meter.update(loss)

        custom.format_custom_text.update_mapping(loss=loss_meter.value()[0])

    tock = time.time()

    time_taken = tock - tick

    time_taken_per_light_field = time_taken / len(test_set.image_indices)

    return time_taken_per_light_field


def evaluate_model(model: nn.Module,
                   train_set: Dataset,
                   test_set: Dataset,
                   criterion,
                   device,
                   batch_size: int,
                   save_path: str,
                   time_taken: float):
    loss_meter = AverageMeter(name='loss', cum=False)

    model.eval()

    custom = EvaluateProgressBar()

    loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        sampler=SequentialSampler(test_set),
        num_workers=1,
        pin_memory=True
    )

    model.eval()
    torch.set_grad_enabled(False)
    x_mean = train_set.get_viewpoint_mean()
    x_std_dev = train_set.get_viewpoint_std_dev()
    # y_mean = test_dataset.get_y_mean()
    # y_std_dev = test_dataset.get_y_std_dev()

    x_mean_var = torch.from_numpy(x_mean).to(device)
    x_std_dev_var = torch.from_numpy(x_std_dev).to(device)

    print('Evaluating')

    output_images = np.zeros(shape=(len(test_set),
                                    test_set.total_channels,
                                    test_set.height,
                                    test_set.width),
                             dtype=np.uint8)
    start = 0

    for (input_data, target) in custom.bar(loader):
        # print("Input data shape {}".format(input_data.size()))
        input_var = input_data.to(device)
        target_var = target.to(device).float()

        output_var = model(input_var,
                           x_mean=x_mean_var,
                           x_std_dev=x_std_dev_var)

        loss = criterion(output_var, target_var).cuda()

        loss_meter.update(loss)

        custom.format_custom_text.update_mapping(loss=loss_meter.value()[0])

        output_var_np = output_var.data.cpu().numpy()

        # Last batch may be of a different size
        curr_batch_size = output_var_np.shape[0]

        loss_meter.update(loss, curr_batch_size)
        output_images_np = np.clip(output_var_np, 0.0, 255.0)

        output_images[start:start + curr_batch_size] = \
            output_images_np.astype(np.uint8)

        start += curr_batch_size

        custom.format_custom_text.update_mapping(loss=loss_meter.value()[0])

    if not os.path.isdir(save_path):
        try:
            os.mkdir(save_path)
        except FileExistsError:
            print("Save dir exists, potentially overwriting previous results.")

    save_evaluation(
        model=model,
        output_path=save_path,
        time_taken=time_taken,
        output_images=output_images,
        final_loss=loss_meter.value()[0]
    )

    calc_and_save_all_metrics(test_set.y_images, output_images, save_path)


def run_epoch(loader,
              model,
              train_dataset,
              test_dataset,
              criterion,
              optimizer,
              device,
              training=True):
    loss_meter = AverageMeter(name='Loss', cum=False)
    custom = TrainProgressBar()
    # loss_meter = Meter(name='Loss')
    # acc_meter = Meter(name='Accuracy')

    if training:
        model.train()
        torch.set_grad_enabled(True)
        x_mean = train_dataset.get_viewpoint_mean()
        x_std_dev = train_dataset.get_viewpoint_std_dev()
        # y_mean = train_dataset.get_y_mean()
        # y_std_dev = train_dataset.get_y_std_dev()
        print('Training')
    else:
        model.eval()
        torch.set_grad_enabled(False)
        x_mean = train_dataset.get_viewpoint_mean()
        x_std_dev = train_dataset.get_viewpoint_std_dev()
        # y_mean = test_dataset.get_y_mean()
        # y_std_dev = test_dataset.get_y_std_dev()
        print('Evaluating')

    x_mean_var = torch.from_numpy(x_mean).to(device)
    x_std_dev_var = torch.from_numpy(x_std_dev).to(device)
    # y_mean_var = torch.from_numpy(y_mean).to(device)
    # y_std_dev_var = torch.from_numpy(y_std_dev).to(device)

    for (input_data, target) in custom.bar(loader):
        # Forward pass
        input_var = input_data.to(device)
        target_var = target.to(device).float()

        output_var = model(input_var,
                           x_mean=x_mean_var,
                           x_std_dev=x_std_dev_var)

        loss = criterion(output_var, target_var)

        # Backward pass
        if training:
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            # else:
                # loss.backward()
            optimizer.step()

        # Log errors
        loss_meter.update(loss)

        custom.format_custom_text.update_mapping(loss=loss_meter.value()[0])

    return loss_meter.value()


def train(model,
          train_set: Dataset,
          valid_set: Dataset,
          save_path: str,
          optimizer: optim.Optimizer,
          num_epochs: int,
          batch_size: int = 64,
          seed: int = None) -> dict:
    if seed is not None:
        torch.manual_seed(seed)

    # Make model, criterion, optimizer, data loaders
    # Note that this function includes a softmax activation
    # TODO: OPT: determine if this should be nn.CrossEntropyLoss().cuda()
    criterion = nn.L1Loss().cuda()

    param_copy = None

    if torch.cuda.is_available():
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

    # Train model
    train_losses, valid_losses = [], []

    # TODO: Max float instead
    best_loss = 1e10

    for epoch in range(1, num_epochs + 1):
        print("{}".format(epoch))

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

        train_loss = run_epoch(
            loader=train_loader,
            model=model_wrapper,
            train_dataset=train_set,
            test_dataset=valid_set,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True
        )

        valid_loss = run_epoch(
            loader=valid_loader,
            model=model_wrapper,
            train_dataset=train_set,
            test_dataset=valid_set,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=False
        )

        train_losses.append(train_loss[0])
        valid_losses.append(valid_loss[0])

        if valid_loss[0] > best_loss:
            best_loss = valid_loss[0]
            print("New best loss: {:.4}".format(best_loss))

    time_per_light_field = time_model(
        model=model_wrapper,
        criterion=criterion,
        train_set=train_set,
        test_set=valid_set,
        device=device,
        batch_size=batch_size
    )

    evaluate_model(
        model=model_wrapper,
        train_set=train_set,
        test_set=valid_set,
        criterion=criterion,
        device=device,
        batch_size=batch_size,
        save_path=save_path,
        time_taken=time_per_light_field
    )

    return {'train': train_losses, 'valid': valid_losses}


def print_time_taken(time_in_secs):
    # Cast to int to ensure that we can print it correctly.
    m, s = divmod(int(time_in_secs), 60)
    h, m = divmod(m, 60)
    print('Done! Time taken {:d}:{:02d}:{:02d}'.format(h, m, s))
