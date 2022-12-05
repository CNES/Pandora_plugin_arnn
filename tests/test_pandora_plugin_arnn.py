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

# pandora_plugin_arnn imports
import pandora_plugin_arnn


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # Example to edit
    return "response"


def test_content(response):  # pylint: disable=redefined-outer-name
    """Sample pytest test function with the pytest fixture as an argument."""
    # Example to edit
    print(response)


def test_pandora_plugin_arnn():
    """Sample pytest pandora_plugin_arnn module test function"""
    assert pandora_plugin_arnn.__author__ == "CNES"
    assert pandora_plugin_arnn.__email__ == "cars@cnes.fr"
