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
This module contains all functions to compute prediction in plugin_arnn
"""

import torch
import xarray as xr


def prediction(model_dataset: xr.Dataset, model: torch.nn.Module, device: torch.device) -> xr.Dataset:
    """
    Makes a prediction using neural network model

    :param model_dataset: Input image xarray.DataSet containing the variables :
            - im : 3D (band, row, col) xarray.DataArray float32
    :type model_dataset: xr.Dataset
    :param model: Model object
    :type model: torch.nn.Module
    :param device: torch device to use
    :type device: torch.device
    :return: Input image xarray.DataSet containing the variables :
            - im : 3D (band, row, col) xarray.DataArray float32
            - initial_prediction : argmax of the prediction : 2D xarray.DataArray float32
            - confidence : Network confidence : 2D xarray.DataArray float32
    """

    return model_dataset
