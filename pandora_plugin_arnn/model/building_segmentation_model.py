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
This module contains building segmentation neural network.
"""

import torch
import numpy as np
from torch import nn, optim


# pylint: disable=too-many-arguments
class BasicBlock(nn.Module):
    """Block used in ResNet18 and 34"""

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=False,
        conv="classic",
    ):
        super().__init__()
        if conv == "classic":
            self.conv1 = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=bias,
            )
            self.bn1 = nn.BatchNorm2d(out_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(
                out_planes,
                out_planes,
                kernel_size,
                1,
                padding,
                groups=groups,
                bias=bias,
            )
            self.bn2 = nn.BatchNorm2d(out_planes)

        self.downsample = None
        if stride > 1 or in_planes == 12:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )
        self.conv_type = conv

    def forward(self, input_):
        """
        Forward method

        :param input_: torch 4D array (nb patch, 5, 1024, 1024)
        :return: network prediction torch 4D array (nb patch, 2, 1024, 1024)
        """
        residual = input_

        out = self.conv1(input_)
        if self.conv_type == "classic":
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.conv_type == "classic":
            out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(input_)
        out += residual
        out = self.relu(out)

        return out


class Encoder34(nn.Module):
    """Encoder for LinkNet34 (ie ResNet34)"""

    def __init__(
        self,
        layer,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=False,
        conv="classic",
        drop=False,
    ):
        super().__init__()
        self.layer = layer
        if self.layer in (1, 4):
            self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias, conv)
            self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
            self.block3 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
        if self.layer == 2:
            self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias, conv)
            self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
            self.block3 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
            self.block4 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
        if self.layer == 3:
            self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias, conv)
            self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
            self.block3 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
            self.block4 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
            self.block5 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
            self.block6 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias, conv)
        self.drop = drop

    def forward(self, input_):
        """
        Forward method

        :param input_: torch 4D array (nb patch, 5, 1024, 1024)
        :return: network prediction torch 4D array (nb patch, 2, 1024, 1024)
        """
        if self.drop:
            input_ = nn.Dropout2d(p=0.05)(input_)
        input_ = self.block1(input_)
        input_ = self.block2(input_)
        input_ = self.block3(input_)
        if self.layer == 2:
            input_ = self.block4(input_)
        if self.layer == 3:
            input_ = self.block4(input_)
            input_ = self.block5(input_)
            input_ = self.block6(input_)
        return input_


class Decoder(nn.Module):
    """LinkNet(34) decoder"""

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        bias=False,
        drop=False,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 4, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(in_planes // 4),
            nn.ReLU(inplace=True),
        )
        self.tp_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_planes // 4,
                in_planes // 4,
                kernel_size,
                stride,
                padding,
                output_padding,
                bias=bias,
            ),
            nn.BatchNorm2d(in_planes // 4),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_planes // 4, out_planes, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )
        self.drop = drop

    def forward(self, input_):
        """
        Forward method

        :param input_: torch 4D array (nb patch, 5, 1024, 1024)
        :return: network prediction torch 4D array (nb patch, 2, 1024, 1024)
        """
        if self.drop:
            input_ = nn.Dropout2d(p=0.05)(input_)
        input_ = self.conv1(input_)
        input_ = self.tp_conv(input_)
        input_ = self.conv2(input_)

        return input_


# pylint: disable=too-many-instance-attributes
class BuildingSegmentation(nn.Module):
    """
    Generate model architecture. LinkNet34 with interactivity.
    """

    def __init__(self):
        """
        Model initialization
        """
        in_channels = 3
        n_classes = 2

        super().__init__()
        # assume one channel per class
        self.conv1 = nn.Conv2d(in_channels + n_classes, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.encoder1 = Encoder34(1, 64, 64, 3, 1, 1)
        self.encoder2 = Encoder34(2, 64, 128, 3, 2, 1)
        self.encoder3 = Encoder34(3, 128, 256, 3, 2, 1)
        self.encoder4 = Encoder34(4, 256, 512, 3, 2, 1)

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)

        self.optimizer = optim.SGD(params=self.parameters(), lr=5 * 10 ** (-6))

    def forward(self, input_):
        """
        Forward method

        :param input_: torch 4D array (nb patch, 5, 1024, 1024)
        :return: network prediction torch 4D array (nb patch, 2, 1024, 1024)
        """
        # Initial block
        input_ = self.conv1(input_)
        input_ = self.bn1(input_)
        input_ = self.relu(input_)
        input_ = self.maxpool(input_)

        # Encoder blocks
        encoder_1 = self.encoder1(input_)
        encoder_2 = self.encoder2(encoder_1)
        encoder_3 = self.encoder3(encoder_2)
        encoder_4 = self.encoder4(encoder_3)

        # Decoder blocks

        decoder_4 = encoder_3 + self.decoder4(encoder_4)
        decoder_3 = encoder_2 + self.decoder3(decoder_4)
        decoder_2 = encoder_1 + self.decoder2(decoder_3)
        decoder_1 = input_ + self.decoder1(decoder_2)

        # Classifier
        output_ = self.tp_conv1(decoder_1)
        output_ = self.conv2(output_)
        output_ = self.tp_conv2(output_)

        return output_

    def get_classes(self):
        """Return network's labels

        :return: network's labels
        :rtype: list(dict)
        """
        classes = [
            {"name": "Not building", "value": 0, "color": "#f5fff4"},
            {"name": "Building", "value": 1, "color": "#40baac"},
        ]
        return classes

    def predict(self, data):
        """Make prediction

        :param data: Patches
        :type data: Tensor(number_of_patches, band, row, col)
        :return: Argmax prediction
        :rtype: Tensor(number_of_patches, network output, row, col)
        """
        prediction = self.forward(data)

        return prediction

    def transform_patches(self, patches):
        """Apply transformations to patches

        :param patches: list of patches
        :type patches: np.array(number_of_patches, band, row, col)
        :return: Tensor of transformed patches
        :rtype: Tensor(number_of_patches, band, row, col)
        """
        patches_expand = np.zeros((patches.shape[0], 5, patches.shape[2], patches.shape[3]), dtype=patches.dtype)

        patches_expand[:, 0, :, :] = (patches[:, 0, :, :] - np.nanmean(patches[:, 0, :, :])) / np.nanstd(
            patches[:, 0, :, :]
        )
        patches_expand[:, 1, :, :] = (patches[:, 1, :, :] - np.nanmean(patches[:, 1, :, :])) / np.nanstd(
            patches[:, 1, :, :]
        )
        patches_expand[:, 2, :, :] = (patches[:, 2, :, :] - np.nanmean(patches[:, 2, :, :])) / np.nanstd(
            patches[:, 2, :, :]
        )

        patches_expand[np.isnan(patches_expand)] = 0
        patches_expand = torch.from_numpy(patches_expand)

        return patches_expand

    def get_patch_size(self):
        """Return integer patch size

        :return: patch size
        :rtype: int
        """
        return 1024

    def backward(self, loss):
        """Make backward using custom loss

        :param loss: loss to backward
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
