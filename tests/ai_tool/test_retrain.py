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
Tests for ai_tool/retrain.
"""
import pytest

from pandora_plugin_arnn.ai_tool.retrain import retrain


@pytest.mark.skip(reason="Not implemented yet")
def test_retrain(create_model_dataset):
    """
    Test retrain function
    """
    model_dataset = create_model_dataset
    gt_model_dataset = create_model_dataset

    # Parameters to set later
    model = None
    device = None

    out_model_dataset = retrain(model, model_dataset, device)

    # Check that out_model_dataset returned by retrain equals the ground truth
    assert out_model_dataset == gt_model_dataset
