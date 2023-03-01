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
"""Tests for `pandora_plugin_arnn` package."""

# Third party imports
import pytest

import pandora

from tests import common


def test_arnn_rgb_band_in_config_and_dataset(load_rgb_data, load_ground_truth, pandora_machine):
    """
    Data with RGB bands in configuration and dataset
    """

    user_cfg = pandora.read_config_file("tests/conf/pipeline_arnn_basic.json")

    # Load data
    left_img, right_img = load_rgb_data

    # Load ground_truth
    left_gt, _ = load_ground_truth

    # Run the pandora pipeline
    left, _ = pandora.run(pandora_machine, left_img, right_img, -60, 0, user_cfg["pipeline"])

    # Compares the calculated left disparity map with the ground truth
    # If the percentage of pixel errors is > 0.20, raise an error
    if common.error(left["disparity_map"].data, left_gt, 1, flag_inverse_value=False) > 0.20:
        raise AssertionError

    # Compares the calculated left disparity map with the ground truth
    # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
    if common.error(left["disparity_map"].data, left_gt, 2, flag_inverse_value=False) > 0.15:
        raise AssertionError


def test_arnn_rgb_band_missing_in_config(load_rgb_data, pandora_machine):
    """
    Data with RGB bands missing in configuration
    RGB band needs to be instantiated in configuration with plugin_arnn
    Check that the run function raises an error.
    """

    # Load fake data
    left_img, right_img = load_rgb_data

    # Load config
    user_cfg = pandora.read_config_file("tests/conf/pipeline_arnn_basic.json")
    # Remove RGB config from user_config
    del user_cfg["semantic_segmentation"]["RGB_bands"]
    # Load Pandora Machine
    pandora_machine_ = pandora_machine

    # The pandora pipeline should fail
    with pytest.raises(ValueError):
        _, _ = pandora.run(pandora_machine_, left_img, right_img, -60, 0, user_cfg["pipeline"])


def test_arnn_only_rg_band_in_config(load_rgb_data, pandora_machine):
    """
    Data with RG bands only in configuration
    B band needs to be instantiated in configuration with plugin_arnn
    Check that the run function raises an error.
    """

    # Load fake data
    left_img, right_img = load_rgb_data

    # Load config
    user_cfg = pandora.read_config_file("tests/conf/pipeline_arnn_basic.json")
    # Remove green band from configuration
    user_cfg["semantic_segmentation"]["RGB_bands"] = {"R": "r", "G": "g"}

    # Load Pandora Machine
    pandora_machine_ = pandora_machine

    # The pandora pipeline should fail
    with pytest.raises(ValueError):
        _, _ = pandora.run(pandora_machine_, left_img, right_img, -60, 0, user_cfg["pipeline"])


def test_arnn_rgb_band_missing_in_dataset(load_rgb_data, pandora_machine):
    """
    Data with RGB bands missing in dataset
    RGB band needs to be instantiated in dataset with plugin_arnn
    Check that the run function raises an error.
    """

    # Load fake data
    left_img, right_img = load_rgb_data

    # Load config
    user_cfg = pandora.read_config_file("tests/conf/pipeline_arnn_basic.json")

    # Replace band name by fake one
    left_img.coords["band"] = [0, 1, 2]
    right_img.coords["band"] = [0, 1, 2]

    # Load Pandora Machine
    pandora_machine_ = pandora_machine

    # The pandora pipeline should fail
    with pytest.raises(ValueError):
        _, _ = pandora.run(pandora_machine_, left_img, right_img, -60, 0, user_cfg["pipeline"])
