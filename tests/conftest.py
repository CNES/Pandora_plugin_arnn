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
This module contains fixtures
"""

import numpy as np
import pytest
import xarray as xr
import pandora
from pandora.state_machine import PandoraMachine


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
        {"im": (["band_im", "row", "col"], data.astype(np.float32))},
        coords={
            "band_im": ["r", "g", "b"],
            "row": np.arange(data.shape[1]),
            "col": np.arange(data.shape[2]),
        },
    )

    return model_dataset


@pytest.fixture()
def load_rgb_data():
    """
    This fixture creates a model_dataset
    """

    # Cones images
    input_config = {
        "left": {
            "img": "tests/inputs/left_rgb.tif",
            "nodata": np.nan,
            "mask": None,
            "disp": [-60, 0],
        },
        "right": {
            "img": "tests/inputs/right_rgb.tif",
            "nodata": np.nan,
            "mask": None,
            "disp": None,
        },
    }
    left_data = pandora.create_dataset_from_inputs(input_config["left"])
    right_data = pandora.create_dataset_from_inputs(input_config["right"])

    return left_data, right_data


@pytest.fixture()
def load_rgb_data_with_classif():
    """
    This fixture creates a model_dataset with classification
    """

    # Cones images
    # fix test_vegetation_band_on_left_and_right_classif_with_validation with updated disp [0,60]
    # otherwise the disparity doesn't exist in right_data and pandora run crashes.
    input_config = {
        "left": {
            "img": "tests/inputs/left_rgb.tif",
            "nodata": np.nan,
            "mask": None,
            "classif": "tests/inputs/left_classif.tif",
            "disp": [-60, 0],
        },
        "right": {
            "img": "tests/inputs/right_rgb.tif",
            "nodata": np.nan,
            "mask": None,
            "classif": "tests/inputs/right_classif.tif",
            "disp": [0, 60],
        },
    }
    left_data = pandora.create_dataset_from_inputs(input_config["left"])
    right_data = pandora.create_dataset_from_inputs(input_config["right"])

    return left_data, right_data


@pytest.fixture()
def load_ground_truth():
    """
    This fixture loads ground truth
    """

    # Cones images
    input_config = {
        "left": {
            "img": "tests/outputs/gt_disp_left.tif",
            "nodata": np.nan,
            "mask": None,
            "disp": [-60, 0],
        },
        "right": {
            "img": "tests/outputs/gt_disp_right.tif",
            "nodata": np.nan,
            "mask": None,
            "disp": None,
        },
    }
    left_data = pandora.create_dataset_from_inputs(input_config["left"])
    right_data = pandora.create_dataset_from_inputs(input_config["right"])

    return left_data, right_data


@pytest.fixture(name="load_pipeline_conf")
def load_pipeline_conf_fixture():
    """
    This fixture loads pipeline conf
    """
    return pandora.read_config_file("tests/conf/pipeline_arnn_basic.json")


@pytest.fixture(name="load_conf")
def load_conf_fixture(load_pipeline_conf):
    """
    This fixture loads input & pipeline conf
    """
    user_cfg = load_pipeline_conf
    # Add inputs
    user_cfg["input"] = {
        "left": {
            "img": "tests/inputs/left_rgb.tif",
            "disp": [-60, 0],
            "nodata": "NaN",
        },
        "right": {"img": "tests/inputs/right_rgb.tif", "nodata": "NaN"},
    }
    return user_cfg


@pytest.fixture()
def load_conf_with_classifs(load_conf):
    """
    This fixture loads input & pipeline conf and add classif mask
    """
    user_cfg = load_conf
    # Add classif
    user_cfg["input"]["left"]["classif"] = "tests/inputs/left_classif.tif"
    user_cfg["input"]["right"]["classif"] = "tests/inputs/right_classif.tif"
    return user_cfg


@pytest.fixture()
def pandora_machine():
    """
    Load Pandora Machine
    """

    # Import pandora plugins
    pandora.import_plugin()
    # Instantiate machine
    pandora_machine_ = PandoraMachine()

    return pandora_machine_
