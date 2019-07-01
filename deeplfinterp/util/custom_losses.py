#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2019.
Contact sbruton[á]tcd.ie.
"""
import torch
from torch import nn
from ..ext.PerceptualSimilarity import PerceptualLoss


class LPIPSLoss(nn.Module):
    def __init__(self, to_zero_data=True, to_scale_data=False):
        """

        :param to_zero_data: If inputs [0, 1] True, if inputs [-1, 1] False.
        :param to_scale_data: If inputs [0, 255] True
        """
        super().__init__()
        self.perceptual_loss = PerceptualLoss(net='alex')
        self.to_scale_data = to_scale_data

        if self.to_scale_data:
            self.to_zero_data = True
        else:
            self.to_zero_data = to_zero_data

    def forward(self, input_var, target_var):
        input_var_rgb = torch.narrow(
            input_var.view(input_var.size(0) * input_var.size(1),
                           input_var.size(2),
                           input_var.size(3),
                           input_var.size(4)),
            1, 0, 3
        )

        target_var_rgb = torch.narrow(
            target_var.view(target_var.size(0) * target_var.size(1),
                            target_var.size(2),
                            target_var.size(3),
                            target_var.size(4)),
            1, 0, 3
        )

        if self.to_scale_data:
            target_var_rgb = target_var_rgb / 255.0
            input_var_rgb = input_var_rgb / 255.0

        # This returns a loss variable N long, need to reduce it by mean.
        return torch.mean(
            self.perceptual_loss.forward(input_var_rgb,
                                         target_var_rgb,
                                         normalize=self.to_zero_data))
