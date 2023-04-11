#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 CNES.
#
# This file is part of pandora_plugin_arnn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This module contains functions used for the prediction and retrain steps of plugin_arnn
"""


from itertools import product
import numpy as np


def patch_image(
    data: np.ndarray, patch_size: int, overlaps: int = 0
) -> np.ndarray:
    """
    Extract patches from image with deep copy

    :param data: 2D or 3D np.ndarray
    :type data: np.ndarray
    :param patch_size: Patch size
    :type patch_size: int
    :param overlaps: Overlaps
    :type overlaps: int
    :return: Patches
    :rtype: np.array (number of patches, band, patch_size, patch_size)
    """
    patches = []

    if len(data.shape) == 3:
        _, nb_row, nb_col = data.shape
    else:
        nb_row, nb_col = data.shape

    # Coordinates list
    offset_col = np.arange(0, nb_col - patch_size, patch_size - overlaps)
    offset_col = np.append(offset_col, nb_col - patch_size)
    offset_row = np.arange(0, nb_row - patch_size, patch_size - overlaps)
    offset_row = np.append(offset_row, nb_row - patch_size)

    offset = product(offset_row, offset_col)
    for row_off, col_off in offset:
        if len(data.shape) == 3:
            patches.append(
                np.copy(
                    data[
                        :,
                        row_off : row_off + patch_size,
                        col_off : col_off + patch_size,
                    ]
                )
            )
        else:
            patches.append(
                np.copy(
                    data[
                        row_off : row_off + patch_size,
                        col_off : col_off + patch_size,
                    ]
                )
            )

    patches = np.array(patches)
    return patches
