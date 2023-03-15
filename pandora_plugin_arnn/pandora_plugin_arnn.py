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
This module contains all functions to compute and refine a semantic segmentation map for Pandora
"""

from typing import Dict, Union

import numpy as np
import torch
import xarray as xr
from importlib_resources import files
from json_checker import Checker, Or, And
from pandora import constants as cst
from pandora.disparity import AbstractDisparity

# pylint: disable=import-error
# pylint: disable=no-name-in-module
from pandora.semantic_segmentation import semantic_segmentation
from scipy import ndimage

from pandora_plugin_arnn.ai_tool.prediction import prediction
from pandora_plugin_arnn.ai_tool.retrain import retrain
from pandora_plugin_arnn.model.building_segmentation_model import (
    BuildingSegmentation,
)


@semantic_segmentation.AbstractSemanticSegmentation.register_subclass("ARNN")
class ARNN(semantic_segmentation.AbstractSemanticSegmentation):
    """
    ARNN class is a Pandora plugin that perform building segmentation using a pre-trained network on VHR images.
    It also allows to refine the segmentation map
    """

    # Default configuration, do not change these values
    _REFINEMENT_ = False

    def __init__(self, **cfg: Dict[str, Union[str, int]]) -> None:
        """
        :param cfg: optional configuration,  {'window_size': value, 'subpix': value}
        :type cfg: dict
        :return: None
        """
        self.cfg = self.check_conf(**cfg)
        self._refinement = self.cfg["refinement"]
        self._rgbnir_bands = self.cfg["RGBNIR_bands"]

    def check_conf(self, **cfg: Dict[str, Union[str, int]]) -> Dict[str, Union[str, int]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: matching cost configuration
        :type cfg: dict
        :return cfg: matching cost configuration updated
        :rtype: dict
        """
        # Give the default value if the required element is not in the conf
        if "refinement" not in cfg:
            cfg["refinement"] = self._REFINEMENT_  # type: ignore

        schema = {
            "semantic_segmentation_method": And(str, lambda x: x == "ARNN"),
            "RGBNIR_bands": {
                "R": str,
                "G": str,
                "B": str,
                "NIR": Or(str, lambda input: input is None),
            },
            "refinement": bool,
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg  # type: ignore

    def desc(self):
        """
        Describes the method
        """
        print("Building semantic segmentation")

    def compute_semantic_segmentation(self, cv: xr.Dataset, img_left: xr.Dataset, img_right: xr.Dataset) -> xr.Dataset:
        """
        Compute building semantic segmentation

        :param cv: the cost volume, with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure (optional): 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :param img_left: left Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :type img_left: xarray
        :param img_right: right Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :type img_right: xarray
        :return: The building segmentation in the left image dataset with the data variables:

                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
                - initial : 2D (row, col) xarray.DataArray building segmentation
        :rtype: xarray.Dataset
        """
        # Load pretrained building segmentation model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BuildingSegmentation()
        model = model.eval()
        pretrained_path = files("pandora_plugin_arnn.model").joinpath("pretrained_building_segmentation_model.pt")
        model.load_state_dict(torch.load(pretrained_path))
        model = model.to(device)

        row, begin_row = (len(img_left.coords["row"]), 0)
        col, begin_col = (len(img_left.coords["col"]), 0)

        # Padd image to the size of the model if needed
        if row < 1024:
            row, begin_row = (1024, 1024 - row)
        if col < 1024:
            col, begin_col = (1024, 1024 - col)

        # Extract RGB img
        rgb_img = np.full((3, row, col), fill_value=np.nan, dtype=np.float32)

        for indx, band in enumerate(["R", "G", "B"]):
            band_index = img_left.attrs["band_list"].index(self._rgbnir_bands[band])  # type: ignore
            rgb_img[indx, begin_row:, begin_col:] = np.copy(img_left["im"].data[:, :, band_index])

        model_dataset = xr.Dataset(
            {"im": (["band", "row", "col"], rgb_img.astype(np.float32))},
            coords={
                "band": np.arange(rgb_img.shape[0]),
                "row": np.arange(rgb_img.shape[1]),
                "col": np.arange(rgb_img.shape[2]),
            },
        )

        # Initial prediction
        prediction(model_dataset, model, device)

        # Make refinement if needed
        if self._refinement:
            # Computes annotation map
            annotation = self.compute_annotations(
                cv,
                img_left,
                img_right,
                model_dataset["initial_prediction"].data,
            )

            model_dataset["annotation"] = xr.DataArray(
                data=annotation,
                coords=[img_left.height, img_left.width],
                dims=["row", "col"],
            )

            # Retrain the network with the annotation map and make a new prediction
            model_dataset = retrain(model, model_dataset, device, retrain_epoch=1, ignore_index=-1)

        # Recrop the segmentation to original image size
        img_left["internal"] = xr.DataArray(
            data=model_dataset["initial_prediction"].data[begin_row:, begin_col:],
            coords=[img_left.height, img_left.width],
            dims=["row", "col"],
        )

        return img_left

    def compute_annotations(
        self,
        cv: xr.Dataset,
        img_left: xr.Dataset,
        img_right: xr.Dataset,
        initial_prediction: np.ndarray,
    ) -> np.ndarray:
        """
        Create an annotation card to retrain the model

        :param cv: the cost volume, with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure (optional): 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :param img_left: left Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :type img_left: xarray
        :param img_right: right Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :type img_right: xarray
        :param initial_prediction: First prediction of the model
        :type initial_prediction: 2D (row, col) np.array
        :return: The building segmentation in the left image dataset with the data variables:

                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
                - initial : 2D (row, col) xarray.DataArray building segmentation
        :rtype: xarray.Dataset
        """
        # Apply WTA on cost volume
        wta = AbstractDisparity(**{"disparity_method": "wta", "invalid_disparity": -9999})
        disp = wta.to_disp(cv, img_left)
        wta.validity_mask(disp, img_left, img_right, cv)

        # Find the threshold on the disparity map to dissociate the ground from the overground
        ground_threshold = self.ground_extraction(disp)

        # If Pandora has not calculated confidence, we consider that all points are confident
        ambiguity = np.ones(
            (len(img_left.coords["row"]), len(img_left.coords["col"])),
            dtype=np.float32,
        )
        if "confidence_measure" in cv and "ambiguity_confidence" in cv.coords["indicator"]:
            ambiguity = cv["confidence_measure"].loc[:, :, "ambiguity_confidence"].data

        # Create annotation map
        annotation = np.full(initial_prediction.shape, -1, dtype=np.float32)

        # Extract overground pixels
        annotation[np.where((disp["disparity_map"].data <= ground_threshold) & (ambiguity >= 0.90))] = 1
        # Extract non-buildings pixels
        annotation[
            np.where(
                (disp["disparity_map"].data > (ground_threshold + 1)) & (ambiguity >= 0.90) & (initial_prediction == 1)
            )
        ] = 0
        # Remove invalid pixels
        annotation[np.where(disp["disparity_map"].data == -9999)] = -1
        invalids = (
            cst.PANDORA_MSK_PIXEL_LEFT_NODATA_OR_BORDER
            + cst.PANDORA_MSK_PIXEL_RIGHT_NODATA_OR_DISPARITY_RANGE_MISSING
            + cst.PANDORA_MSK_PIXEL_RIGHT_INCOMPLETE_DISPARITY_RANGE
            + cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_LEFT
            + cst.PANDORA_MSK_PIXEL_IN_VALIDITY_MASK_RIGHT
        )
        annotation[np.where((disp["validity_mask"].data & invalids) != 0)] = -1

        # Find vegetation
        vegetation_map = self.compute_vegetation_map(cv, img_left)

        # Dilates the vegetation map and remove vegetation pixel in annotation map
        vegetation_map = ndimage.binary_dilation(vegetation_map, structure=np.ones((10, 10))).astype(
            vegetation_map.dtype
        )
        annotation[vegetation_map == 1] = -1

        return annotation

    @staticmethod
    def ground_extraction(wta: xr.Dataset):
        """
        KMeans for ground/over-ground point classification

        :param wta: Disparities
        :type wta: 2D (row, col) np.array
        """
        # Copy create to avoid pylint error with stub function
        _ = np.copy(wta)
        # Returns null value for now
        return 0

    @staticmethod
    def compute_vegetation_map(cv: xr.Dataset, img_left: xr.Dataset):
        """
        Compute vegetation map

        :param cv: the cost volume, with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure (optional): 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :param img_left: left Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :type img_left: xarray
        :return: Vegetation map (0 = not vegetation, 1 = vegetation)
        :rtype: 2D (row, cool) np.array dtype=bool
        """
        # Copy create to avoid pylint error with stub function
        _ = np.copy(cv)
        # vegetation map (0 = not vegetation, 1 = vegetation)
        vegetation_map = np.zeros((len(img_left.coords["row"]), len(img_left.coords["col"])), dtype=bool)

        return vegetation_map
