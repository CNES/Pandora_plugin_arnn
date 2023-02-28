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
This module contains fixtures
"""

import pytest
import xarray as xr
import numpy as np


@pytest.fixture()
def create_model_dataset():
    """
    This fixture creates a model_dataset
    """

    data = np.zeros((3, 4, 4))
    data[0, :, :] = np.array(
        (
            [1, 1, 1, 3],
            [1, 3, 2, 5],
            [2, 1, 0, 1],
            [1, 5, 4, 3],
        ),
        dtype=np.float64,
    )

    data[1, :, :] = np.array(
        (
            [2, 3, 4, 6],
            [8, 7, 0, 4],
            [4, 9, 1, 5],
            [6, 5, 2, 1],
        ),
        dtype=np.float64,
    )

    data[2, :, :] = np.array(
        (
            [1, 1, 1, 3],
            [1, 3, 2, 5],
            [2, 1, 0, 1],
            [1, 5, 4, 3],
        ),
        dtype=np.float64,
    )

    model_dataset = xr.Dataset(
        {"im": (["band", "row", "col"], data.astype(np.float32))},
        coords={
            "band": ["r", "g", "b"],
            "row": np.arange(data.shape[1]),
            "col": np.arange(data.shape[2]),
        },
    )

    return model_dataset
