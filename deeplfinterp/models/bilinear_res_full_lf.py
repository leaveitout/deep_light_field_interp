#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2017.
Contact sbruton[á]tcd.ie.
"""
import torch


class LightFieldGridUpsample(torch.nn.Module):
    def __init__(self, num_horizontal_steps, num_vertical_steps):
        super(LightFieldGridUpsample, self).__init__()

        self.num_horizontal_steps = num_horizontal_steps
        self.num_vertical_steps = num_vertical_steps
        self.horizontal_stride = 1 / num_horizontal_steps
        self.vertical_stride = 1 / num_vertical_steps

        # We assume that the top-left is (0,0), top-right is (1, 0),
        # bottom-left is (0, 1), and bottom-right is (1, 1)
        self.coords = []
        for x in range(self.num_horizontal_steps):
            for y in range(self.num_vertical_steps):
                self.coords.append((x * self.horinzontal_stride,
                                    y * self.vertical_stride))

        self.interp_factors = []

        for coord in self.coord:
            self.interp_factors.append([
                (1 - coord[0]) * (1 - coord[1]),
                coord[0] * (1 - coord[1]),
                (1 - coord[0]) * coord[1],
                coord[0] * coord[1]
            ])

    def forward(self, corner_image_tensors):
        """

        :param corner_image_tensors: Tensors should be in the order (top-left,
        top-right, bottom-left, bottom-right).
        :return: Bilinearly interpolated light field images
        """
        output = self.conv1(input_tensor)
        # output = self.batch_norm1(output)
        output = self.relu(output)
        output = self.conv2(output)
        # output = self.batch_norm2(output)
        output = self.relu(output)
        output = self.conv3(output)
        # output = self.batch_norm3(output)
        output = self.relu(output)

        return output
