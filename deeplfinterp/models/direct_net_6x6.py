#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017.
Contact sbruton[á]tcd.ie.
"""
import torch


class BasicNet(torch.nn.Module):
    def __init__(self, num_in_channels, num_out_channels):
        super(BasicNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=num_in_channels,
                                     out_channels=num_out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.conv2 = torch.nn.Conv2d(in_channels=num_out_channels,
                                     out_channels=num_out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, input_tensor):
        output = self.conv1(input_tensor)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.relu(output)

        return output


class SubNet(torch.nn.Module):
    def __init__(self):
        super(SubNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=128,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.upsample = torch.nn.Upsample(scale_factor=2,
                                          mode='bilinear',
                                          align_corners=True)

        self.conv4 = torch.nn.Conv2d(in_channels=128,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, input_tensor):
        output = self.conv1(input_tensor)
        output = self.relu(output)
        output = self.upsample(output)
        output = self.conv4(output)

        return output


class DirectNet6x6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        print("DirectNet6x6")

        self.conv1 = BasicNet(16, 32)
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = BasicNet(32, 64)
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = BasicNet(64, 128)
        self.pool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv4 = BasicNet(128, 256)
        self.pool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

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
        # self.final_subnet = SubNet()

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

        conv1_output = self.conv1(inputs)
        pool1_output = self.pool1(conv1_output)

        conv2_output = self.conv2(pool1_output)
        pool2_output = self.pool2(conv2_output)

        conv3_output = self.conv3(pool2_output)
        pool3_output = self.pool3(conv3_output)

        conv4_output = self.conv4(pool3_output)
        pool4_output = self.pool4(conv4_output)

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

        outputs = self._add_tiled(upsample1_output, conv1_output)

        return outputs

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)