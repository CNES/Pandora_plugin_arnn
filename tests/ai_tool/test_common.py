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
Tests for ai_tool/common.
"""

import rasterio
import numpy as np
import pandora_plugin_arnn.ai_tool.common as common


def test_image_patch(create_model_dataset):
    """
    Test image_patch function
    """

    img_ds = rasterio.open("tests/inputs/left.png")
    data = img_ds.read()
    if data.shape[0] == 1:
        data = data[0, :, :]
    # Image shape is 375, 450
    # We set patch size to 100
    # 375/100 = 3.75 -> 4
    # 450/100 = 4.5 -> 5
    # Hence we should obtain 4 * 5 = 20 patches of size 100*100
    # The corresponding offsets are col : [0, 100, 200, 300, 350], row : [0, 100, 200, 275]
    # Compute patches using the patch_image function
    output_patches = common.patch_image(data, 100)
    # Check that the output_patches size is the same as gt
    gt_shape = (20, 100, 100)
    assert output_patches.shape == gt_shape

    # Manually compute some of the patches
    gt_patch_0 = data[0:100, 0:100]
    gt_patch_2 = data[0:100, 200:300]
    gt_patch_8 = data[100:200, 300:400]
    gt_patch_19 = data[275:375, 350:450]

    # Check that the selected output patches are the same as gt
    np.testing.assert_array_equal(output_patches[0], gt_patch_0)
    np.testing.assert_array_equal(output_patches[2], gt_patch_2)
    np.testing.assert_array_equal(output_patches[8], gt_patch_8)
    np.testing.assert_array_equal(output_patches[19], gt_patch_19)


def test_image_patch_overlaps(create_model_dataset):
    """
    Test image_patch function with overlaps
    """

    img_ds = rasterio.open("tests/inputs/left.png")
    data = img_ds.read()
    if data.shape[0] == 1:
        data = data[0, :, :]
    # Image shape is 375, 450
    # We set patch size to 100 with 5 overlap
    # 375/100 = 3.75 -> 4
    # 450/100 = 4.5 -> 5
    # Hence we should obtain 4 * 5 = 20 patches of size 100*100
    # The corresponding offsets are col : [0, 95, 190, 280, 350], row : [0, 95, 190, 275]
    # Compute patches using the patch_image function
    output_patches = common.patch_image(data, 100, 5)
    # Check that the output_patches size is the same as gt
    gt_shape = (20, 100, 100)
    assert output_patches.shape == gt_shape

    # Manually compute some of the patches
    gt_patch_0 = data[0:100, 0:100]
    gt_patch_2 = data[0:100, 190:290]
    gt_patch_8 = data[95:195, 285:385]
    gt_patch_19 = data[275:375, 350:450]

    # Check that the selected output patches are the same as gt
    np.testing.assert_array_equal(output_patches[0], gt_patch_0)
    np.testing.assert_array_equal(output_patches[2], gt_patch_2)
    np.testing.assert_array_equal(output_patches[8], gt_patch_8)
    np.testing.assert_array_equal(output_patches[19], gt_patch_19)


def test_image_patch_multiband(create_model_dataset):
    """
    Test image_patch function with multiband image
    """

    img_ds = rasterio.open("tests/inputs/left_rgb.tif")
    data = img_ds.read()

    # Image shape is 3, 375, 450
    # We set patch size to 100
    # 375/100 = 3.75 -> 4
    # 450/100 = 4.5 -> 5
    # Hence we should obtain 4 * 5 = 20 patches of size 100*100 for each band
    # The corresponding offsets are col : [0, 100, 200, 300, 350], row : [0, 100, 200, 275]
    # Compute patches using the patch_image function
    output_patches = common.patch_image(data, 100)
    # Check that the output_patches size is the same as gt
    gt_shape = (20, 3, 100, 100)
    assert output_patches.shape == gt_shape

    # Manually compute some of the patches
    gt_patch_0 = data[:, 0:100, 0:100]
    gt_patch_2 = data[:, 0:100, 200:300]
    gt_patch_8 = data[:, 100:200, 300:400]
    gt_patch_19 = data[:, 275:375, 350:450]

    # Check that the selected output patches are the same as gt
    np.testing.assert_array_equal(output_patches[0], gt_patch_0)
    np.testing.assert_array_equal(output_patches[2], gt_patch_2)
    np.testing.assert_array_equal(output_patches[8], gt_patch_8)
    np.testing.assert_array_equal(output_patches[19], gt_patch_19)
