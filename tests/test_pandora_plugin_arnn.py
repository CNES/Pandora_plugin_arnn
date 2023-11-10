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
"""Tests for `pandora_plugin_arnn` package."""
import json_checker
import numpy as np
import xarray as xr

# Third party imports
import pytest

import pandora
from pandora import check_configuration


from tests import common

from pandora_plugin_arnn.pandora_plugin_arnn import semantic_segmentation


def test_arnn_rgb_band_in_config_and_dataset(
    load_pipeline_conf, load_rgb_data_with_classif, load_ground_truth, pandora_machine
):
    """
    Data with RGB bands in configuration and dataset
    """

    # Load pipeline config
    user_cfg = load_pipeline_conf
    # Import pandora plugins
    pandora.import_plugin()
    # Load data
    left_img, right_img = load_rgb_data_with_classif

    # Load ground_truth
    left_gt, _ = load_ground_truth

    # Because it is not checked at this point
    user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

    # Run the pandora pipeline
    left, _ = pandora.run(pandora_machine, left_img, right_img, user_cfg)

    # Compares the calculated left disparity map with the ground truth
    # If the percentage of pixel errors is > 0.20, raise an error
    if (
        common.error(
            left["disparity_map"].data,
            left_gt["im"].data,
            1,
            flag_inverse_value=False,
        )
        > 0.55
    ):
        raise AssertionError

    # Compares the calculated left disparity map with the ground truth
    # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
    if (
        common.error(
            left["disparity_map"].data,
            left_gt["im"].data,
            2,
            flag_inverse_value=False,
        )
        > 0.55
    ):
        raise AssertionError


def test_arnn_rgb_band_missing_in_config(load_rgb_data_with_classif, load_pipeline_conf, pandora_machine):
    """
    Data with RGB bands missing in configuration
    RGB band needs to be instantiated in configuration with plugin_arnn
    Check that the run function raises an error.
    """

    # Load fake data
    left_img, right_img = load_rgb_data_with_classif
    # Import pandora plugins
    pandora.import_plugin()
    # Load config
    user_cfg = load_pipeline_conf
    # Remove RGB config from user_config
    del user_cfg["pipeline"]["semantic_segmentation"]["RGB_bands"]
    # Load Pandora Machine
    pandora_machine_ = pandora_machine

    # The pandora pipeline should fail
    with pytest.raises(json_checker.MissKeyCheckerError):
        _, _ = pandora.run(pandora_machine_, left_img, right_img, user_cfg)


def test_arnn_only_rg_band_in_config(load_rgb_data_with_classif, load_pipeline_conf, pandora_machine):
    """
    Data with RG bands only in configuration
    B band needs to be instantiated in configuration with plugin_arnn
    Check that the run function raises an error.
    """

    # Load fake data
    left_img, right_img = load_rgb_data_with_classif
    # Import pandora plugins
    pandora.import_plugin()
    # Load config
    user_cfg = load_pipeline_conf
    # Remove green band from configuration
    user_cfg["pipeline"]["semantic_segmentation"]["RGB_bands"] = {
        "R": "r",
        "G": "g",
    }

    # Load Pandora Machine
    pandora_machine_ = pandora_machine

    # The pandora pipeline should fail
    with pytest.raises(json_checker.DictCheckerError):
        _, _ = pandora.run(pandora_machine_, left_img, right_img, user_cfg)


def test_arnn_rgb_band_missing_in_dataset(load_rgb_data, load_pipeline_conf, pandora_machine):
    """
    Data with RGB bands missing in dataset
    RGB band needs to be instantiated in dataset with plugin_arnn
    Check that the run function raises an error.
    """

    # Load fake data
    left_img, right_img = load_rgb_data

    # Load config
    user_cfg = load_pipeline_conf
    # Import pandora plugins
    pandora.import_plugin()
    # Replace band name by fake one
    left_img.coords["band_im"] = [0, 1, 2]
    right_img.coords["band_im"] = [0, 1, 2]

    # Load Pandora Machine
    pandora_machine_ = pandora_machine

    # The pandora pipeline should fail
    with pytest.raises(SystemExit):
        _, _ = pandora.run(pandora_machine_, left_img, right_img, user_cfg)


def test_merge_into_vegetation_map(load_rgb_data_with_classif):
    """
    Test the compute_vegetation_map function
    """
    left, _ = load_rgb_data_with_classif

    data = np.zeros((2, 4, 4))
    img_metadata = xr.Dataset(
        {"classif": (["band_classif", "row", "col"], data.astype(np.float32))},
        coords={
            "band_classif": ["forest", "olive tree"],
            "row": np.arange(data.shape[1]),
            "col": np.arange(data.shape[2]),
        },
        attrs={"classif": True},
    )

    ssgm_ = semantic_segmentation.AbstractSemanticSegmentation(
        img_metadata,
        **{
            "segmentation_method": "ARNN",
            "RGB_bands": {"R": "r", "G": "g", "B": "b"},
            "vegetation_band": {"classes": ["forest", "olive tree"]},
        }
    )

    vegetation_map = ssgm_.merge_into_vegetation_map(left)

    # ground truth
    gt_vegetation_map = np.load("tests/outputs/fused_classif.npy")

    assert gt_vegetation_map.data == vegetation_map.data


def test_vegetation_without_classif(load_conf, pandora_machine):
    """
    Test the check vegetation option without classif
    """
    # Load config
    user_cfg = load_conf

    # Import pandora plugins
    pandora.import_plugin()

    pandora_machine_ = pandora_machine

    # Check configuration
    with pytest.raises(SystemExit):
        _ = check_configuration.check_conf(user_cfg, pandora_machine_)


def test_wrong_vegetation_class(load_conf_with_classifs, pandora_machine):
    """
    Semantic segmentation on wrong band for left classification
    Classes must be the same to classify band names in data.
    Check that the check_conf function raises an error.
    """
    # Load config
    user_cfg = load_conf_with_classifs
    # Replace with wrong configuration
    user_cfg["pipeline"]["semantic_segmentation"]["vegetation_band"]["classes"] = ["grass"]

    # Import pandora plugins
    pandora.import_plugin()

    pandora_machine_ = pandora_machine

    # Check configuration
    with pytest.raises(SystemExit):
        _ = check_configuration.check_conf(user_cfg, pandora_machine_)


def test_vegetation_band_on_left_classif_without_validation(
    load_conf, load_rgb_data_with_classif, pandora_machine, load_ground_truth
):
    """
    Semantic segmentation with left classification and without validation
    """
    # Load config
    user_cfg = load_conf
    # Add left classif
    user_cfg["input"]["left"]["classif"] = "tests/inputs/left_classif.tif"

    # Import pandora plugins
    pandora.import_plugin()

    pandora_machine_ = pandora_machine

    # Check configuration
    user_cfg = check_configuration.check_conf(user_cfg, pandora_machine_)

    left, right = load_rgb_data_with_classif

    left, _ = pandora.run(pandora_machine, left, right, user_cfg)
    left_gt, _ = load_ground_truth

    # Compares the calculated left disparity map with the ground truth
    # If the percentage of pixel errors is > 0.20, raise an error
    if (
        common.error(
            left["disparity_map"].data,
            left_gt["im"].data,
            1,
            flag_inverse_value=False,
        )
        > 0.55
    ):
        raise AssertionError

    # Compares the calculated left disparity map with the ground truth
    # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
    if (
        common.error(
            left["disparity_map"].data,
            left_gt["im"].data,
            2,
            flag_inverse_value=False,
        )
        > 0.55
    ):
        raise AssertionError


def test_vegetation_band_on_right_classif_without_validation(load_conf, pandora_machine):
    """
    Semantic segmentation with right classification and without validation
    Classification must be instantiated with left data.
    Check that the check_conf function raises an error.
    """

    # Load config
    user_cfg = load_conf
    # Add right classif
    user_cfg["input"]["right"]["classif"] = "tests/inputs/right_classif.tif"

    # Import pandora plugins
    pandora.import_plugin()

    pandora_machine_ = pandora_machine

    # Check configuration
    with pytest.raises(SystemExit):
        _ = check_configuration.check_conf(user_cfg, pandora_machine_)


def test_vegetation_band_on_left_and_right_classif_without_validation(
    load_conf_with_classifs, load_rgb_data_with_classif, pandora_machine, load_ground_truth
):
    """
    Semantic segmentation with left and right classification and without validation
    """
    # Load config
    user_cfg = load_conf_with_classifs

    # Import pandora plugins
    pandora.import_plugin()

    pandora_machine_ = pandora_machine

    # Check configuration
    user_cfg = check_configuration.check_conf(user_cfg, pandora_machine_)

    left, right = load_rgb_data_with_classif

    left, _ = pandora.run(pandora_machine, left, right, user_cfg)
    left_gt, _ = load_ground_truth

    # Compares the calculated left disparity map with the ground truth
    # If the percentage of pixel errors is > 0.20, raise an error
    if (
        common.error(
            left["disparity_map"].data,
            left_gt["im"].data,
            1,
            flag_inverse_value=False,
        )
        > 0.55
    ):
        raise AssertionError

    # Compares the calculated left disparity map with the ground truth
    # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
    if (
        common.error(
            left["disparity_map"].data,
            left_gt["im"].data,
            2,
            flag_inverse_value=False,
        )
        > 0.55
    ):
        raise AssertionError


def test_vegetation_band_on_left_classif_with_validation(load_conf, pandora_machine):
    """
    Semantic segmentation with left classification with validation
    Classification must be instantiated with left and right data.
    Check that the check_conf function raises an error.
    """

    # Load config
    user_cfg = load_conf
    # Add left classif
    user_cfg["input"]["left"]["classif"] = "tests/inputs/left_classif.tif"

    # Add validation step
    user_cfg["pipeline"]["validation"] = {
        "validation_method": "cross_checking_accurate",
        "cross_checking_threshold": 1,
    }

    # Import pandora plugins
    pandora.import_plugin()

    pandora_machine_ = pandora_machine

    # Check configuration
    with pytest.raises(SystemExit):
        _ = check_configuration.check_conf(user_cfg, pandora_machine_)


def test_vegetation_band_on_left_and_right_classif_with_validation(
    load_conf_with_classifs, load_rgb_data_with_classif, pandora_machine, load_ground_truth
):
    """
    Semantic segmentation with left and right classification with validation
    """
    # Load config
    user_cfg = load_conf_with_classifs

    # Add validation step
    user_cfg["pipeline"]["validation"] = {
        "validation_method": "cross_checking_accurate",
        "cross_checking_threshold": 1,
    }

    # Import pandora plugins
    pandora.import_plugin()

    pandora_machine_ = pandora_machine

    # Check configuration
    user_cfg = check_configuration.check_conf(user_cfg, pandora_machine_)

    left, right = load_rgb_data_with_classif

    left, right = pandora.run(pandora_machine, left, right, user_cfg)
    left_gt, right_gt = load_ground_truth

    # Compares the calculated left disparity map with the ground truth
    # If the percentage of pixel errors is > 0.20, raise an error
    if (
        common.error(
            left["disparity_map"].data,
            left_gt["im"].data,
            1,
            flag_inverse_value=False,
        )
        > 0.55
    ):
        raise AssertionError

    # Compares the calculated left disparity map with the ground truth
    # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
    if (
        common.error(
            left["disparity_map"].data,
            left_gt["im"].data,
            2,
            flag_inverse_value=False,
        )
        > 0.55
    ):
        raise AssertionError

    # Compares the calculated left disparity map with the ground truth
    # If the percentage of pixel errors is > 0.20, raise an error
    if common.error(right["disparity_map"].data, right_gt["im"].data, 1) > 0.56:
        raise AssertionError

    # Compares the calculated left disparity map with the ground truth
    # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
    if common.error(right["disparity_map"].data, right_gt["im"].data, 2) > 0.55:
        raise AssertionError
