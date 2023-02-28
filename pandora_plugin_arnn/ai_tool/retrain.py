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
This module contains all functions to compute retrain in plugin_arnn
"""

import torch
import xarray as xr


def retrain(model: torch.nn.Module,
            model_dataset: xr.Dataset,
            device: torch.device,
            retrain_epoch: int = 1,
            ignore_index: int = -1
            ):
    """
    Retrain network with user annotation

    :param model: Model object
    :type model: torch.nn.Module
    :param model_dataset: Input image xarray.DataSet containing the variables :
            - im : 3D (band, row, col) xarray.DataArray float32
            - initial_prediction : argmax of the prediction : 2D xarray.DataArray float32
            - annotation : 2D (row, col) xarray.DataArray float32
    :type model_dataset: xr.Dataset
    :param device: torch device to use
    :type device: torch.device
    :param retrain_epoch: number of retrain epoch
    :type retrain_epoch: int
    :param ignore_index: index to ignore in annotations when retraining
    :type ignore_index: int
    :return: Initial prediction after retraining, input image xarray.DataSet containing the variables :
            - im : 3D (band, row, col) xarray.DataArray float32
            - initial_prediction : argmax of the prediction : 2D xarray.DataArray float32
            - annotation : 2D (row, col) xarray.DataArray float32

    """

    return model_dataset
