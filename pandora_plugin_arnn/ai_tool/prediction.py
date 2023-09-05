#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2023 CNES.
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
This module contains all functions to compute prediction in plugin_arnn
"""

from itertools import product
import torch
import xarray as xr
import numpy as np
from .common import extract_patches


def prediction(image_dataset: xr.Dataset, model: torch.nn.Module, device: torch.device) -> None:
    """
    Makes a prediction using neural network model

    :param image_dataset: Input image xarray.DataSet containing the variables :
            - im : 3D (band_im, row, col) xarray.DataArray float32
    :type image_dataset: xr.Dataset
    :param model: Model object
    :type model: torch.nn.Module
    :param device: torch device to use
    :type device: torch.device
    :return: None
    """

    # Load patch size
    patch_size = model.get_patch_size()  # type: ignore

    # Patches image
    overlaps = 0
    patches = extract_patches(image_dataset["im"].data, patch_size, overlaps=overlaps)

    # Apply model's transformation
    patches = model.transform_patches(patches)  # type: ignore
    patches = patches.to(device)

    # Instantiate parameters
    nb_classes = len(model.get_classes())  # type: ignore
    _, nb_row, nb_col = image_dataset["im"].shape

    # Indices of each patch
    offset_col = np.append(
        np.arange(0, nb_col - patch_size, patch_size - overlaps),
        (nb_col - patch_size),
    )
    offset_row = np.append(
        np.arange(0, nb_row - patch_size, patch_size - overlaps),
        (nb_row - patch_size),
    )
    offset = product(offset_row, offset_col)

    # Make prediction and reconstruct initial 2D image
    pred = np.zeros((nb_classes, nb_row, nb_col), dtype=np.float32)
    with torch.no_grad():
        for patch_idx, (row_off, col_off) in enumerate(offset):
            patch_pred = (
                model.predict(patches[patch_idx : patch_idx + 1, :, :, :])  # type: ignore
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            pred[
                :,
                row_off : row_off + patch_size - overlaps,
                col_off : col_off + patch_size - overlaps,
            ] += np.squeeze(patch_pred)

    pred = np.argmax(pred, axis=0).astype(np.float32)

    # Add prediction to xarray image_dataset
    image_dataset["initial_prediction"] = xr.DataArray(
        data=pred,
        coords=[image_dataset.coords["row"], image_dataset.coords["col"]],
        dims=["row", "col"],
    )
