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
This module contains all functions to compute retrain in plugin_arnn
"""

from typing import Tuple

import numpy as np
import torch
import xarray as xr
from torch import nn

from .common import extract_patches
from .prediction import prediction

_REGULARIZATION_COEFF = 0.1


def retrain(
    model: torch.nn.Module,
    image_dataset: xr.Dataset,
    device: torch.device,
    retrain_epoch: int = 10,
    ignore_index: int = -1,
) -> xr.Dataset:
    """
    Retrain network with annotation

    :param model: Model object
    :type model: torch.nn.Module
    :param image_dataset: Input image xarray.DataSet containing the variables :
            - im : 3D (band_im, row, col) xarray.DataArray float32
            - initial_prediction : argmax of the prediction : 2D xarray.DataArray float32
            - annotation : 2D (row, col) xarray.DataArray float32
    :type image_dataset: xr.Dataset
    :param device: torch device to use
    :type device: torch.device
    :param retrain_epoch: number of retrain epoch
    :type retrain_epoch: int
    :param ignore_index: index to ignore in annotations when retraining
    :type ignore_index: int
    :return: Retraining prediction, input image xarray.Dataset containing the variables :
            - im : 3D (band_im, row, col) xarray.DataArray float32
            - initial_prediction : argmax of the prediction : 2D xarray.DataArray float32
            - annotation : 2D (row, col) xarray.DataArray float32
    """
    # Make initial prediction if needed
    if "initial_prediction" not in image_dataset:
        prediction(image_dataset, model, device)

    image_patches, initial_patches, annotations_patches = prepare_retrain_data(model, image_dataset)
    image_patches = image_patches.to(device)

    for _ in range(retrain_epoch):
        prediction_data = model.predict(image_patches)
        loss = retrain_loss(
            prediction_data,
            annotations_patches,
            ignore_index,
            initial_patches,
            device,
        )
        model.backward(loss)

    # Make new prediction
    prediction(image_dataset, model, device)

    return image_dataset


def prepare_retrain_data(
    model: torch.nn.Module, image_dataset: xr.Dataset, ignore_index: int = -1
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Creates normalized patches images with annotations before retrain.
    Save only annotated patches.

    :param model: Model object
    :type model: torch.nn.Module
    :param image_dataset: Input image xarray.DataSet containing the variables :
            - im : 3D (band_im, row, col) xarray.DataArray float32
            - initial_prediction : argmax of the prediction : 2D xarray.DataArray float32
            - annotation : 2D (row, col) xarray.DataArray float32
    :type image_dataset: xr.Dataset
    :param ignore_index: index to ignore in annotations when retraining
    :type ignore_index: int
    :return: Deep copy of image, initial prediction and annotation patches
    :rtype: Tuple[torch.Tensor(nb_patches, band, row, col), np.ndarray, np.ndarray]
    """
    patch_size = model.get_patch_size()

    # Patches image and annotations
    image_patches = extract_patches(image_dataset["im"].data, patch_size)
    image_patches = model.transform_patches(image_patches)

    initial_patches = extract_patches(image_dataset["initial_prediction"].data, patch_size)

    annotations_patches = extract_patches(image_dataset["annotation"].data, patch_size)

    # Remove patches that do not contain annotations
    patchs_id = []
    for index_ in range(annotations_patches.shape[0]):
        if np.sum(annotations_patches[index_, :, :] != ignore_index) != 0:
            patchs_id.append(index_)

    return (
        image_patches[patchs_id, :, :, :],
        initial_patches[patchs_id, :, :],
        annotations_patches[patchs_id, :, :],
    )


def retrain_loss(
    last_prediction: torch.Tensor,
    annotations: np.ndarray,
    ignore_index: int,
    initial_prediction: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute loss from annotations and predictions

    :param last_prediction: last network prediction
    :type last_prediction: Tensor of shape (nb patch, network output, row, col)
    :param annotations: Ground truth sparse annotation
    :type annotations: np array (nb patch, row, col))
    :param ignore_index: index to ignore in annotations when retraining
    :type ignore_index: int
    :param initial_prediction: initial network prediction
    :param initial_prediction: np array (nb patch, network output, row, col)
    :param device: torch device to use
    :type device: torch.device
    :return: loss
    :rtype: torch.Tensor with grad_fn
    """
    tensor_annotations = torch.from_numpy(annotations)
    tensor_annotations = tensor_annotations.type(torch.long)
    tensor_annotations = tensor_annotations.to(device)

    initial_prediction = torch.from_numpy(initial_prediction).type(torch.long)
    initial_prediction = initial_prediction.to(device)

    ce_annotation = nn.CrossEntropyLoss(ignore_index=ignore_index)
    annotation_loss = ce_annotation(prediction, tensor_annotations)

    ce_initial_prediction = nn.CrossEntropyLoss()
    regularization_loss = ce_initial_prediction(last_prediction, initial_prediction)

    loss = annotation_loss + (_REGULARIZATION_COEFF * regularization_loss)

    return loss
