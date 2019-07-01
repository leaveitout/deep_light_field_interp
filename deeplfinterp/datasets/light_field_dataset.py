#!/usr/bin/env python3
"""
Copyright Seán Bruton, Trinity College Dublin, 2019.
Contact sbruton[á]tcd.ie.
"""

from torch.utils.data.dataset import Dataset
import numpy as np
from typing import Union, List


class LightFieldDataset(Dataset):
    """
    Abstract Dataset class for use in light field training experiments.
    """

    def __init__(self):
        super(LightFieldDataset, self).__init__()
        self.num_channels = None

        self.num_views_y = None
        self.height_y = None
        self.width_y = None

        self.num_views_x = None
        self.height_x = None
        self.width_x = None

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def set_only_x_dataset(self):
        raise NotImplementedError

    def revert_only_x_dataset(self):
        raise NotImplementedError

    def get_only_y(self, idx: int):
        raise NotImplementedError

    def get_y_images(self, to_list: bool = True) -> Union[np.ndarray, List]:
        pass

    def get_x_images(self, to_list: bool = True) -> Union[np.ndarray, List]:
        pass

    def fast_collate(self, batch):
        pass
