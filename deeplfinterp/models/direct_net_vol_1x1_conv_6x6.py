#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2019.
Contact sbruton[á]tcd.ie.
"""
from .direct_net_6x6 import SubNet, BasicNet

import os

import torch
import torch.nn.functional as nnf
import numpy as np


class DirectNetVol1x1Conv6x6(torch.nn.Module):
    def __init__(self, vol_path: os.path):
        super().__init__()
        print("DirectNetVol1x1Conv")

        vol_np = np.load(vol_path)

        if vol_np.shape[1] == 512:
            vol_np = vol_np[:, ::2, :]

        if vol_np.shape[2] == 512:
            vol_np = vol_np[:, :, :2]

        self.vol = torch.nn.Parameter(torch.from_numpy(vol_np))

        self.vol_conv1 = torch.nn.Conv2d(in_channels=self.vol.size(0),
                                         out_channels=64,
                                         kernel_size=1,
                                         stride=1)

        self.vol_conv2 = torch.nn.Conv2d(in_channels=64,
                                         out_channels=32,
                                         kernel_size=1,
                                         stride=1)

        self.vol_conv3 = torch.nn.Conv2d(in_channels=32,
                                         out_channels=16,
                                         kernel_size=1,
                                         stride=1)

        self.conv1 = BasicNet(16, 32)
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = BasicNet(32, 64)
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = BasicNet(64, 128)
        self.pool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv4 = BasicNet(128, 256)
        self.pool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv1_vol = BasicNet(16, 32)
        self.pool1_vol = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2_vol = BasicNet(32, 64)
        self.pool2_vol = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3_vol = BasicNet(64, 128)
        self.pool3_vol = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv4_vol = BasicNet(128, 256)
        self.pool4_vol = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.deconv4 = BasicNet(256, 256)
        self.upsample4 = torch.nn.Sequential(
            torch.nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            ),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(inplace=False)
        )

        self.deconv3 = BasicNet(256, 128)
        self.upsample3 = torch.nn.Sequential(
            torch.nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            ),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(inplace=False)
        )

        self.deconv2 = BasicNet(128, 128)
        self.upsample2 = torch.nn.Sequential(
            torch.nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            ),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(inplace=False)
        )

        self.deconv1 = BasicNet(128, 128)
        self.upsample1 = torch.nn.Sequential(
            torch.nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            ),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(inplace=False)
        )

        self.deconv4_vol = BasicNet(256, 256)
        self.upsample4_vol = torch.nn.Sequential(
            torch.nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            ),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(inplace=False)
        )

        self.deconv3_vol = BasicNet(256, 128)
        self.upsample3_vol = torch.nn.Sequential(
            torch.nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            ),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(inplace=False)
        )

        self.deconv2_vol = BasicNet(128, 128)
        self.upsample2_vol = torch.nn.Sequential(
            torch.nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            ),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(inplace=False)
        )

        self.deconv1_vol = BasicNet(128, 128)
        self.upsample1_vol = torch.nn.Sequential(
            torch.nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            ),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(inplace=False)
        )
        # self.final_subnet = SubNet()

        # self.weights_init()

    def _add_tiled(self, x, y):
        """
        Tiles y to be the same shape as x, before adding it.

        :param x: The larger tensor.
        :param y: The tensor that needs to be tiled.
        """
        # Get the required multiple
        mult, remainder = divmod(x.size()[1], y.size()[1])

        if remainder != 0:
            raise ValueError("Tensors of sizes {} and {} "
                             "do not divide evenly".format(x.size(), y.size()))

        # Reshape x so that the multiple is second dimension, after batch
        x_new_shape = \
            (x.size()[0],) + (mult, x.size()[1] // mult,) + tuple(x.size()[2:])
        x_singleton = x.view(x_new_shape)
        # Reshape y so that it has a singleton dimension at that location
        y_new_shape = (y.size()[0],) + (1,) + tuple(y.size()[1:])
        y_singleton = y.view(y_new_shape)

        # Expand y to fit the shape of x
        y_singleton.expand_as(x_singleton)

        # Add it to x
        z = x_singleton + y_singleton

        # View as original size
        return z.view_as(x)

    def forward(self, inputs):
        # Need to do this so that we can load the data as np.uint8.
        # It is extremely slow to load a light field as floats.
        # It is also not possible to load as fp16 as pytorch does not have
        # concatenation defined for cpu fp16.
        inputs = inputs.view(
            inputs.size(0),
            inputs.size(1) * inputs.size(2),
            inputs.size(3),
            inputs.size(4)
        )

        # Main branch
        conv1_output = self.conv1(inputs)
        pool1_output = self.pool1(conv1_output)

        conv2_output = self.conv2(pool1_output)
        pool2_output = self.pool2(conv2_output)

        conv3_output = self.conv3(pool2_output)
        pool3_output = self.pool3(conv3_output)

        conv4_output = self.conv4(pool3_output)
        pool4_output = self.pool4(conv4_output)

        # Volume branch
        vol = nnf.relu(self.vol_conv1(self.vol))
        vol = nnf.relu(self.vol_conv2(vol))
        vol = nnf.relu(self.vol_conv2(vol))

        conv1_vol_output = self.conv1_vol(vol)
        pool1_vol_output = self.pool1_vol(conv1_vol_output)

        conv2_vol_output = self.conv2_vol(pool1_vol_output)
        pool2_vol_output = self.pool2_vol(conv2_vol_output)

        conv3_vol_output = self.conv3_vol(pool2_vol_output)
        pool3_vol_output = self.pool3_vol(conv3_vol_output)

        conv4_vol_output = self.conv4_vol(pool3_vol_output)
        pool4_vol_output = self.pool4_vol(conv4_vol_output)

        # Main branch
        deconv4_output = self.deconv4(pool4_output)
        upsample4_output = self.upsample4(deconv4_output)

        combined_output = upsample4_output + conv4_output

        deconv3_output = self.deconv3(combined_output)
        upsample3_output = self.upsample3(deconv3_output)

        combined_output = upsample3_output + conv3_output

        deconv2_output = self.deconv2(combined_output)
        upsample2_output = self.upsample2(deconv2_output)

        combined_output = self._add_tiled(upsample2_output, conv2_output)

        deconv1_output = self.deconv1(combined_output)
        upsample1_output = self.upsample1(deconv1_output)

        # Volume branch
        deconv4_vol_output = self.deconv4_vol(pool4_vol_output)
        upsample4_vol_output = self.upsample4_vol(deconv4_vol_output)

        combined_vol_output = upsample4_vol_output + conv4_output

        deconv3_vol_output = self.deconv3_vol(combined_vol_output)
        upsample3_vol_output = self.upsample3_vol(deconv3_vol_output)

        combined_vol_output = upsample3_vol_output + conv3_output

        deconv2_vol_output = self.deconv2_vol(combined_vol_output)
        upsample2_vol_output = self.upsample2_vol(deconv2_vol_output)

        combined_vol_output = self._add_tiled(upsample2_vol_output,
                                              conv2_output)

        deconv1_vol_output = self.deconv1_vol(combined_vol_output)
        upsample1_vol_output = self.upsample1_vol(deconv1_vol_output)

        outputs = self._add_tiled(upsample1_output, conv1_output)

        outputs = outputs + upsample1_vol_output

        outputs = outputs.view(
            outputs.size(0),
            outputs.size(1) // 4,
            4,
            outputs.size(2),
            outputs.size(3)
        )

        return outputs

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
